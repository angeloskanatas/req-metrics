"""Compute registered metrics over the layers of one model and return Records.

Inputs are plain tensors keyed by layer index; how they were produced is left
to the caller and is recorded through the labels and the ViewSpec and
ShiftSpec objects. Levels:

- "sequence": layers[l] is (N, D), one vector per sample.
- "sample": layers[l] is a sequence of (T_i, D) token tensors, one per sample; point
  and trajectory metrics run on each sample's tokens and are aggregated.
- "population": layers[l] is a sequence of (T_i, D) token tensors; the tokens of all
  samples form one point cloud, subsampled to n tokens.
- Token-field metrics take the same per-sample inputs; those whose definition pairs
  tokens within a sample (registry per_sample) run per sample at the population level too.
View metrics take layers[l] of shape (q, N, D); PTE takes layers[l] = (z, shifted).
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from torch import Tensor

from req_metrics import __version__
from req_metrics._types import InputKind, MetricResult, Preprocess
from req_metrics.neighbors import Neighbors, _cdist_mode
from req_metrics.preprocess import apply_preprocess, l2_normalize
from req_metrics.records import Record, Records
from req_metrics.registry import MetricSpec, get_metric
from req_metrics.spectrum import Spectrum
from req_metrics.view_construction import ShiftSpec, ViewSpec

# Estimators reading the shared neighbor table: the argument that sizes their table and the
# one that sets the neighbors behind their value (None: TwoNN, which reads two).
_NEIGHBOR_ARGS: dict[str, tuple[str | None, str | None]] = {
    "intrinsic_dimension/twonn": (None, None),
    "intrinsic_dimension/gride": ("range_max", "scale"),
    "intrinsic_dimension/mle": ("k_range", "k_range"),
    "intrinsic_dimension/mlid": ("k", "k"),
    "neighborhood_curvature": ("k", "k"),
}


def _tensor(data: Any) -> Tensor:
    """A torch view of array-like data; MPS tensors move to the CPU, since the estimators compute in float64."""
    t = torch.as_tensor(data)
    return t.cpu() if t.device.type == "mps" else t


def _device(device: str | torch.device | None) -> torch.device | None:
    """The device estimators run on; MPS maps to the CPU, since it has no float64."""
    if device is None:
        return None
    d = torch.device(device)
    return torch.device("cpu") if d.type == "mps" else d


def _to(t: Tensor, device: torch.device | None) -> Tensor:
    return t if device is None else t.to(device)


def _warn_failed(records: Records) -> Records:
    """Warn once per call when estimators failed; their records hold nan and extras["error"]."""
    failed = [r for r in records.rows if "error" in r.extras]
    if failed:
        first = failed[0]
        warnings.warn(
            f"{len(failed)} of {len(records.rows)} records failed and hold nan; first: {first.metric} "
            f"at layer {first.layer}: {first.extras['error']}",
            RuntimeWarning,
            stacklevel=3,
        )
    return records


def choose_indices(n: int, max_items: int | None, seed: int, group_ids: Sequence | None = None) -> np.ndarray:
    """Sorted row indices to keep: one row per group if group_ids is given, then at most max_items at random.

    One-per-group picks a random member of each group (one clip per
    track in Kanatas et al., 2026); the same indices must be applied to every layer, which compute does.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    if group_ids is not None:
        if len(group_ids) != n:
            raise ValueError("group_ids must have one entry per row")
        order = rng.permutation(n)
        seen, keep = set(), []
        for i in order:
            g = group_ids[i]
            if g not in seen:
                seen.add(g)
                keep.append(i)
        idx = np.array(sorted(keep))
    if max_items is not None and max_items < len(idx):
        idx = np.sort(rng.choice(idx, size=max_items, replace=False))
    return idx


def _population_tokens(
    samples: Sequence[Any], sample_idx: np.ndarray, n: int | None, seed: int
) -> tuple[Tensor, Tensor]:
    """At most n tokens drawn at random from the selected samples, and the sample of each token.

    No sample is concatenated with the others before the draw.
    """
    lengths = np.array([len(samples[i]) for i in sample_idx])
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    flat = choose_indices(int(offsets[-1]), n, seed)  # sorted global token indices
    owner = np.searchsorted(offsets, flat, side="right") - 1
    parts = []
    for j in np.unique(owner):
        rows = torch.from_numpy(flat[owner == j] - offsets[j])
        parts.append(_tensor(samples[sample_idx[j]])[rows])
    return torch.cat(parts), torch.from_numpy(owner)


def _normalized_depths(layer_ids: Sequence[int]) -> dict[int, float]:
    top = max(layer_ids) if layer_ids else 0
    return {l: (l / top if top > 0 else 0.0) for l in layer_ids}


def _preprocess_args(spec: MetricSpec, kwargs: Mapping[str, Any]) -> tuple[Preprocess, dict[str, Any]]:
    """The preprocessing a call applies, and the remaining arguments.

    The registry's preprocessing, overridden by the estimator's own center, standardize
    and l2 arguments; arguments the estimator does not take stay in the remainder.
    """
    accepted = inspect.signature(spec.fn).parameters
    own = {f: bool(kwargs[f]) for f in ("center", "standardize", "l2") if f in kwargs and f in accepted}
    rest = {k: v for k, v in kwargs.items() if k not in own}
    return replace(spec.preprocess, **own), rest


def _run(spec: MetricSpec, args: tuple, kwargs: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    try:
        res: MetricResult = spec.fn(*args, **kwargs)
        return res.value, dict(res.extras)
    except Exception as e:  # the pipeline records failures instead of aborting a sweep
        return float("nan"), {"error": f"{type(e).__name__}: {e}"}


def _aggregate(
    values: list[float], extras_list: list[dict[str, Any]], keep_per_sample: bool
) -> tuple[float, dict[str, Any]]:
    vals = np.array([v for v in values if np.isfinite(v)], dtype=float)
    out: dict[str, Any] = {"n_items": len(values), "n_failed": int(len(values) - vals.size)}
    if vals.size:
        out.update(
            {
                "std": float(vals.std(ddof=0)),
                "median": float(np.median(vals)),
                "min": float(vals.min()),
                "max": float(vals.max()),
            }
        )
    if keep_per_sample:
        out["per_sample"] = [float(v) for v in values]
    numeric = [e for e in extras_list if e and "error" not in e]
    if numeric:  # per-sample extras shared by every sample are averaged under their own names
        for key in set.intersection(*(set(e) for e in numeric)):
            vals_k = [e[key] for e in numeric if isinstance(e[key], (int, float)) and not isinstance(e[key], bool)]
            if len(vals_k) == len(numeric) and key not in out:
                out[key] = float(np.mean(vals_k))
    first_err = next((e["error"] for e in extras_list if "error" in e), None)
    if first_err is not None:
        out["error"] = first_err
    return (float(vals.mean()) if vals.size else float("nan")), out


def _cap_index(n_rows: int, cap: int | None, seed: int) -> Tensor | None:
    """Sorted indices of a seeded random subsample of n_rows rows down to cap, or None if none is needed."""
    if cap is None or n_rows <= cap:
        return None
    return torch.randperm(n_rows, generator=torch.Generator().manual_seed(seed))[:cap].sort().values


def _cap(x: Tensor, cap: int | None, seed: int) -> Tensor:
    """Seeded random subsample of the rows of x down to cap (sorted indices), or x itself."""
    idx = _cap_index(x.shape[0], cap, seed)
    return x if idx is None else x[idx.to(x.device)]


def _neighbor_count(spec: MetricSpec, kw: Mapping[str, Any], role: int) -> int:
    """Neighbors an estimator reads: role 0 sizes its table, role 1 sets its value."""
    arg = _NEIGHBOR_ARGS[spec.name][role]
    if arg is None:
        return 2
    v = kw[arg] if arg in kw else inspect.signature(spec.fn).parameters[arg].default
    return int(v[1]) if isinstance(v, (tuple, list)) else int(v)


def _same_group_fraction(indices: Tensor, groups: Tensor, k: int) -> float:
    """Mean share of a point's k nearest neighbors (column 0 is the point) that belong to its own group."""
    g = groups.to(indices.device)
    k = min(k, indices.shape[1] - 1)
    return float((g[indices[:, 1 : k + 1]] == g[:, None]).double().mean())


def _points_sweep(
    layer_tensor: Tensor,
    specs: list[MetricSpec],
    params: Mapping[str, Mapping[str, Any]],
    seed: int = 0,
    limits: Mapping[str, int] | None = None,
    groups: Tensor | None = None,
) -> dict[str, tuple[float, dict]]:
    """Run point metrics on one (N, D) cloud, sharing Spectrum per preprocessing and one Neighbors table.

    Metrics with a registry max_items (or an entry in limits) see a seeded subsample
    of at most that many rows, with their own spectrum or neighbor table; the count
    used is written to extras["n_items_used"]. With groups (one id per row), the
    neighbor-table estimators also report extras["same_sample_fraction"], the mean share
    of a point's neighbors behind the value that share its group.
    """
    out: dict[str, tuple[float, dict]] = {}
    x = layer_tensor
    limits = dict(limits or {})
    subsets: dict[int, tuple[Tensor, Tensor | None]] = {x.shape[0]: (x, groups)}

    def subset(cap: int | None) -> tuple[Tensor, Tensor | None]:
        n = x.shape[0] if cap is None else min(int(cap), x.shape[0])
        if n not in subsets:
            idx = _cap_index(x.shape[0], n, seed)
            if idx is None:
                subsets[n] = (x, groups)
            else:
                subsets[n] = (x[idx.to(x.device)], None if groups is None else groups[idx])
        return subsets[n]

    spectra: dict[Any, Spectrum] = {}
    neighbors: dict[int, tuple[Tensor, Neighbors, Tensor | None]] = {}  # kNN estimators see the distinct points
    table = [s for s in specs if s.cache in ("neighbors", "neighbors_kw")]
    k_max = max((_neighbor_count(s, params.get(s.name, {}), 0) for s in table), default=0)
    for spec in specs:
        kw = dict(params.get(spec.name, {}))
        xs, gs = subset(limits.get(spec.name, spec.max_items))
        n_used = int(xs.shape[0])
        if spec.cache == "spectrum":
            pre, kw = _preprocess_args(spec, kw)
            key = (pre, n_used)
            if key not in spectra:
                spectra[key] = Spectrum.from_points(apply_preprocess(xs, pre), center=False)
            out[spec.name] = _run(spec, (spectra[key],), kw)
        elif spec.cache in ("neighbors", "neighbors_kw"):
            if n_used not in neighbors:
                try:
                    xu, inv = torch.unique(xs, dim=0, return_inverse=True)
                    gu = None
                    if gs is not None:  # a row duplicated across groups keeps one of them
                        gu = torch.empty(xu.shape[0], dtype=gs.dtype)
                        gu[inv.cpu()] = gs
                    neighbors[n_used] = (xu, Neighbors.from_points(xu, k_max), gu)
                except Exception as e:
                    out[spec.name] = (float("nan"), {"error": f"{type(e).__name__}: {e}"})
                    continue
            xu, nb, gu = neighbors[n_used]
            if spec.cache == "neighbors":
                out[spec.name] = _run(spec, (nb,), kw)
            else:
                out[spec.name] = _run(spec, (xu,), {**kw, "neighbors": nb})
            if xu.shape[0] < n_used:
                out[spec.name][1]["n_distinct"] = int(xu.shape[0])
            if gu is not None and "error" not in out[spec.name][1]:
                k_value = _neighbor_count(spec, kw, 1)
                out[spec.name][1]["same_sample_fraction"] = _same_group_fraction(nb.indices, gu, k_value)
        else:
            out[spec.name] = _run(spec, (xs,), kw)
        if n_used < x.shape[0]:
            out[spec.name][1]["n_items_used"] = n_used
    return out


def compute(
    layers: Mapping[int, Any],
    metrics: Sequence[str],
    *,
    level: str = "sequence",
    n: int | None = None,
    seed: int = 0,
    group_ids: Sequence | None = None,
    params: Mapping[str, Mapping[str, Any]] | None = None,
    model: str | None = None,
    pooling: str | None = None,
    corpus: str | None = None,
    views: ViewSpec | None = None,
    shifts: ShiftSpec | None = None,
    keep_per_sample: bool = False,
    limits: Mapping[str, int] | None = None,
    device: str | torch.device | None = None,
) -> Records:
    """Compute metrics for every layer and return one Record per (layer, metric).

    Args:
        layers: Layer index -> data. Point metrics: (N, D) at the sequence level, a sequence
            of (T_i, D) token tensors at the sample and population levels. View metrics:
            (q, N, D). PTE: (z, {k: z_k}). Jacobian effective rank: (B, k, M) Jacobian
            sketches (see jacobian_products).
        metrics: Registry names. All must share one input kind per call.
        level: "sequence", "sample" or "population" (point, trajectory and token-field
            metrics; the other input kinds are sequence level).
        n: Keep at most n samples chosen at random with seed, the same for every layer; at
            the population level, at most n tokens for the point metrics.
        seed: Subsampling seed, recorded.
        group_ids: One id per sample; one random sample per group is kept before subsampling.
        params: Metric name -> estimator keyword arguments; recorded in the records.
        model, pooling, corpus: Labels recorded verbatim.
        views: ViewSpec describing how view stacks were built (view metrics).
        shifts: ShiftSpec describing the pitch-shifted copies (PTE).
        keep_per_sample: At the sample level, keep every sample's value in extras["per_sample"].
        limits: Metric name -> cap on the items that estimator sees (overrides the registry
            max_items); a seeded subsample is drawn above the cap and recorded in
            extras["n_items_used"].
        device: Device the estimators run on; each layer, sample or population token cloud is
            moved there after subsetting, one at a time. Default: the device of the inputs.

    Returns:
        Records, one row per (layer, metric), failures recorded as nan with an error message.
    """
    dev = _device(device)
    params = {k: dict(v) for k, v in (params or {}).items()}
    specs = [get_metric(m) for m in metrics]
    kinds = {s.inputs for s in specs}
    token_kinds = {InputKind.POINTS, InputKind.TRAJECTORY, InputKind.TOKENS}
    if len(kinds) != 1 and not kinds <= token_kinds:
        raise ValueError(f"metrics of mixed input kinds in one call: {sorted(k.value for k in kinds)}")
    kind = kinds.pop() if len(kinds) == 1 else InputKind.TRAJECTORY  # mixed token kinds: handled per metric below
    layer_ids = sorted(layers)
    if not layer_ids:
        raise ValueError("layers is empty: no layer tensors were given")
    depths = _normalized_depths(layer_ids)
    records = Records()

    def record(
        spec: MetricSpec, layer: int, value: float, extras: dict, n_items: int, dim: int, n_views: int | None = None
    ) -> None:
        records.rows.append(
            Record(
                metric=spec.name,
                value=value,
                layer=layer,
                depth=depths[layer],
                model=model,
                level=level,
                pooling=pooling,
                corpus=corpus,
                n_items=n_items,
                dim=dim,
                n_views=n_views,
                preprocess=_preprocess_args(spec, params.get(spec.name, {}))[0].describe(),
                params=dict(params.get(spec.name, {})),
                views=views.describe() if views else None,
                shifts=shifts.describe() if shifts else None,
                seed=seed,
                tags=spec.tags,
                extras=extras,
                version=__version__,
            )
        )

    if kind == InputKind.VIEWS:
        first = layers[layer_ids[0]]
        idx = choose_indices(first.shape[1], n, seed, group_ids)
        caps = dict(limits or {})
        for l in layer_ids:
            v = _to(_tensor(layers[l])[:, idx], dev)
            for spec in specs:
                cap = caps.get(spec.name, spec.max_items)
                vs = v if cap is None or v.shape[1] <= cap else v[:, _cap(torch.arange(v.shape[1]), cap, seed)]
                value, extras = _run(spec, (vs,), params.get(spec.name, {}))
                if vs.shape[1] < v.shape[1]:
                    extras["n_items_used"] = int(vs.shape[1])
                record(spec, l, value, extras, len(idx), int(v.shape[-1]), int(v.shape[0]))
        return _warn_failed(records)

    if kind == InputKind.SHIFTED:
        z0, _ = layers[layer_ids[0]]
        idx = choose_indices(z0.shape[0], n, seed, group_ids)
        for l in layer_ids:
            z, shifted = layers[l]
            z = _to(_tensor(z)[idx], dev)
            sh = {k: _to(_tensor(v)[idx], dev) for k, v in shifted.items()}
            for spec in specs:
                value, extras = _run(spec, (z, sh), params.get(spec.name, {}))
                record(spec, l, value, extras, len(idx), int(z.shape[-1]), len(sh))
        return _warn_failed(records)

    if kind == InputKind.PAIR:
        raise ValueError("pair metrics compare two representations; use compute_pairs")

    if kind == InputKind.JACOBIAN:
        for l in layer_ids:
            j = _to(_tensor(layers[l]), dev)
            for spec in specs:
                value, extras = _run(spec, (j,), params.get(spec.name, {}))
                record(spec, l, value, extras, int(j.shape[0]), int(j.shape[-1]))
        return _warn_failed(records)

    if level == "sequence":
        if any(s.inputs in (InputKind.TRAJECTORY, InputKind.TOKENS) for s in specs):
            raise ValueError("trajectory and token-field metrics need level='sample' or 'population'")
        first = _tensor(layers[layer_ids[0]])
        counts = {l: int(_tensor(layers[l]).shape[0]) for l in layer_ids}
        if len(set(counts.values())) > 1:
            raise ValueError(f"layers must have the same number of rows, got {counts}")
        idx = choose_indices(first.shape[0], n, seed, group_ids)
        for l in layer_ids:
            x = _to(_tensor(layers[l])[idx], dev)
            for name, (value, extras) in _points_sweep(x, specs, params, seed, limits).items():
                record(get_metric(name), l, value, extras, len(idx), int(x.shape[-1]))
        return _warn_failed(records)

    def per_sample(samples: Sequence[Any], sample_specs: list[MetricSpec], idx: np.ndarray, l: int) -> None:
        """Run sample_specs on every selected sample's tokens and record their aggregates."""
        per_metric: dict[str, tuple[list[float], list[dict]]] = {s.name: ([], []) for s in sample_specs}
        point_specs = [s for s in sample_specs if s.inputs == InputKind.POINTS]
        other_specs = [s for s in sample_specs if s.inputs != InputKind.POINTS]
        for i in idx:
            c = _to(_tensor(samples[i]), dev)
            results = _points_sweep(c, point_specs, params, seed, limits) if point_specs else {}
            results.update({s.name: _run(s, (c,), params.get(s.name, {})) for s in other_specs})
            for name, (value, extras) in results.items():
                per_metric[name][0].append(value)
                per_metric[name][1].append(extras)
        dim = int(_tensor(samples[idx[0]]).shape[-1])
        for name, (values, extras_list) in per_metric.items():
            value, extras = _aggregate(values, extras_list, keep_per_sample)
            record(get_metric(name), l, value, extras, len(idx), dim)

    if level == "population":
        if any(s.inputs == InputKind.TRAJECTORY for s in specs):
            raise ValueError("trajectory metrics need level='sample'")
        pooled_specs = [s for s in specs if not s.per_sample]
        sample_specs = [s for s in specs if s.per_sample]  # their definition pairs tokens within a sample
        sample_idx = choose_indices(len(layers[layer_ids[0]]), None, seed, group_ids)
        for l in layer_ids:
            if pooled_specs:
                x, owner = _population_tokens(layers[l], sample_idx, n, seed)
                x = _to(x, dev)
                n_samples = int(torch.unique(owner).numel())
                for name, (value, extras) in _points_sweep(x, pooled_specs, params, seed, limits, owner).items():
                    extras["n_samples"] = n_samples
                    record(get_metric(name), l, value, extras, int(x.shape[0]), int(x.shape[-1]))
            if sample_specs:
                per_sample(layers[l], sample_specs, sample_idx, l)
        return _warn_failed(records)

    if level == "sample":
        idx = choose_indices(len(layers[layer_ids[0]]), n, seed, group_ids)
        for l in layer_ids:
            per_sample(layers[l], specs, idx, l)
        return _warn_failed(records)

    raise ValueError(f"unknown level {level!r}")


def _rank_table_rows(x_b: Tensor, rows: Tensor) -> Tensor:
    """Ranks (1 = nearest) of every item for the given query rows in space B; ties take the lower rank."""
    d = torch.cdist(x_b[rows], x_b, compute_mode=_cdist_mode(x_b.dtype))
    d[torch.arange(len(rows)), rows] = float("inf")
    order = d.argsort(dim=1)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(1, x_b.shape[0] + 1, device=d.device).expand(len(rows), -1))
    return ranks  # (c, N)


def compute_pairs(
    layers_a: Mapping[int, Tensor],
    layers_b: Mapping[int, Tensor] | None = None,
    *,
    k: int | None = None,
    n: int | None = None,
    seed: int = 0,
    group_ids: Sequence | None = None,
    model: str | None = None,
    model_b: str | None = None,
    pooling: str | None = None,
    corpus: str | None = None,
    chunk: int = 512,
    metric: str = "information_imbalance",
    params: Mapping[str, Any] | None = None,
    device: str | torch.device | None = None,
) -> Records:
    """A pair metric between every layer of A and every layer of B (B defaults to A).

    A and B are two representations of the same items with aligned rows: the layers of one model, two
    checkpoints, two models, or two modalities with paired items (image i and caption i); widths may
    differ. One Record per (layer_a, layer_b). metric is "information_imbalance" (default k = 1; value
    Delta(A -> B), extras["reverse"]), "neighborhood_overlap" (default k = 30; symmetric), "cycle_knn"
    (default k = 10; value A -> B, extras["reverse"]), "cka", "svcca" or "rsa" (symmetric). params passes
    the estimator's options, flat or keyed by the metric name as in compute(): debiased for CKA,
    threshold for SVCCA, distance and method for RSA, and l2 for the three neighbor metrics, which then
    rank cosine neighbors (unit-norm rows) as Huh et al. (2024) do.
    The preprocessing is recorded. The imbalance costs one chunked rank table per layer rather than one
    per pair: for each target layer the ranks of all items are computed once and gathered at the k
    nearest neighbors of every source layer, so L layers cost O(L N^2 D) instead of O(L^2 N^2 D). The
    overlap and the cycle consistency need only the k-nearest-neighbor tables, once per layer. CKA keeps
    the centered features
    and their norms, SVCCA the kept singular directions and RSA the ranked distance vector, each once
    per layer; RSA is capped at its registry max_items because the vectors are held for all layers.
    device is as in compute().
    """
    dev = _device(device)
    params = dict(params or {})
    if isinstance(params.get(metric), Mapping):  # the keyed form of compute()
        params = dict(params[metric])
    if metric in ("cka", "svcca", "rsa"):
        return _closed_form_pairs(
            metric, layers_a, layers_b, params, n, seed, group_ids, model, model_b, pooling, corpus, dev
        )
    if metric not in ("information_imbalance", "neighborhood_overlap", "cycle_knn"):
        raise ValueError(f"unknown layer-pair metric {metric!r}")
    if k is None:
        k = {"information_imbalance": 1, "neighborhood_overlap": 30, "cycle_knn": 10}[metric]
    l2 = bool(params.get("l2", False))
    spec = get_metric(metric)
    same = layers_b is None
    b_map: Mapping[int, Tensor] = layers_a if layers_b is None else layers_b
    ids_a, ids_b = sorted(layers_a), sorted(b_map)
    first = _tensor(layers_a[ids_a[0]])
    idx = choose_indices(first.shape[0], n, seed, group_ids)
    prep = l2_normalize if l2 else (lambda t: t)
    xa = {l: prep(_to(_tensor(layers_a[l])[idx], dev).double()) for l in ids_a}
    xb = xa if same else {l: prep(_to(_tensor(b_map[l])[idx], dev).double()) for l in ids_b}
    n_items = len(idx)
    nn_a = {l: Neighbors.from_points(xa[l], k).indices[:, 1 : k + 1] for l in ids_a}
    nn_b = nn_a if same else {l: Neighbors.from_points(xb[l], k).indices[:, 1 : k + 1] for l in ids_b}
    if metric in ("neighborhood_overlap", "cycle_knn"):
        from req_metrics.metrics.compare import _cycle_fraction, _shared_neighbor_fraction

        depths = _normalized_depths(ids_a)
        out = Records()
        for la in ids_a:
            for lb in ids_b:
                if metric == "neighborhood_overlap":
                    frac = _shared_neighbor_fraction(nn_a[la], nn_b[lb])
                    extras = {"std": float(frac.std()), "chance": k / (n_items - 1)}
                else:
                    frac = _cycle_fraction(nn_a[la], nn_b[lb])
                    reverse = float(_cycle_fraction(nn_b[lb], nn_a[la]).mean())
                    extras = {"reverse": reverse, "chance_bound": min(1.0, k * k / (n_items - 1))}
                out.rows.append(
                    Record(
                        metric=spec.name,
                        value=float(frac.mean()),
                        layer=la,
                        layer_b=lb,
                        depth=depths[la],
                        model=model if model_b is None else f"{model}->{model_b}",
                        level="sequence",
                        pooling=pooling,
                        corpus=corpus,
                        n_items=n_items,
                        dim=int(xa[la].shape[-1]),
                        preprocess="l2" if l2 else "none",
                        params={"k": k, "l2": l2},
                        seed=seed,
                        tags=spec.tags,
                        extras=extras,
                        version=__version__,
                    )
                )
        return out
    # mean rank in target space T of the k A-neighbors, accumulated over row chunks of T's rank table
    sums_ab = {(la, lb): 0.0 for la in ids_a for lb in ids_b}
    sums_ba = {(la, lb): 0.0 for la in ids_a for lb in ids_b}

    def accumulate(target_layers, target_x, source_layers, source_nn, sums, key):
        for lt in target_layers:
            for start in range(0, n_items, chunk):
                rows = torch.arange(start, min(start + chunk, n_items))
                ranks = _rank_table_rows(target_x[lt], rows)  # (c, N)
                for ls in source_layers:
                    gathered = torch.gather(ranks, 1, source_nn[ls][rows])  # (c, k)
                    sums[key(ls, lt)] += float(gathered.double().sum())

    accumulate(ids_b, xb, ids_a, nn_a, sums_ab, lambda ls, lt: (ls, lt))  # ranks in B of A's neighbors
    if same:  # Delta(B_lb -> A_la) is Delta(lb -> la), already accumulated
        sums_ba = {(la, lb): sums_ab[(lb, la)] for la in ids_a for lb in ids_b}
    else:
        accumulate(ids_a, xa, ids_b, nn_b, sums_ba, lambda ls, lt: (lt, ls))  # ranks in A of B's neighbors
    depths = _normalized_depths(ids_a)
    scale = (n_items * k) * (n_items / 2.0)
    out = Records()
    for la in ids_a:
        for lb in ids_b:
            out.rows.append(
                Record(
                    metric=spec.name,
                    value=sums_ab[(la, lb)] / scale,
                    layer=la,
                    layer_b=lb,
                    depth=depths[la],
                    model=model if model_b is None else f"{model}->{model_b}",
                    level="sequence",
                    pooling=pooling,
                    corpus=corpus,
                    n_items=n_items,
                    dim=int(xa[la].shape[-1]),
                    preprocess="l2" if l2 else "none",
                    params={"k": k, "l2": l2},
                    seed=seed,
                    tags=spec.tags,
                    extras={"reverse": sums_ba[(la, lb)] / scale},
                    version=__version__,
                )
            )
    return out


def _closed_form_pairs(
    metric: str,
    layers_a: Mapping[int, Tensor],
    layers_b: Mapping[int, Tensor] | None,
    params: dict[str, Any],
    n: int | None,
    seed: int,
    group_ids: Sequence | None,
    model: str | None,
    model_b: str | None,
    pooling: str | None,
    corpus: str | None,
    dev: torch.device | None = None,
) -> Records:
    """CKA, SVCCA or RSA for every layer pair, with per-layer summaries computed once."""
    from req_metrics.metrics.compare import _check_pair, _cka_from_stats, _cka_stats, _rsa_vector, _svd_directions

    spec = get_metric(metric)
    same = layers_b is None
    b_map: Mapping[int, Tensor] = layers_a if layers_b is None else layers_b
    ids_a, ids_b = sorted(layers_a), sorted(b_map)
    if spec.max_items is not None and (n is None or n > spec.max_items):
        n = spec.max_items
    idx = choose_indices(_tensor(layers_a[ids_a[0]]).shape[0], n, seed, group_ids)
    for l in ids_b:
        _check_pair(_tensor(layers_a[ids_a[0]])[idx], _tensor(b_map[l])[idx])
    if metric == "cka":
        debiased = bool(params.get("debiased", False))
        if debiased and len(idx) < 4:
            raise ValueError(f"debiased CKA needs N >= 4, got N = {len(idx)}")

        def summary(x: Tensor) -> Any:
            return _cka_stats(x)

        def pair(a: Any, b: Any) -> tuple[float, dict[str, Any]]:
            biased, unbiased = _cka_from_stats(a, b)
            return (unbiased if debiased else biased), {"biased": biased, "debiased": unbiased}

    elif metric == "rsa":
        distance, method = str(params.get("distance", "cosine")), str(params.get("method", "spearman"))

        def summary(x: Tensor) -> Any:
            return _rsa_vector(x, distance, method, int(params.get("chunk", 1024))).float()

        def pair(a: Any, b: Any) -> tuple[float, dict[str, Any]]:
            return float(torch.dot(a.double(), b.double())), {"n_pairs": float(a.numel())}

    else:
        threshold = float(params.get("threshold", 0.99))
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")

        def summary(x: Tensor) -> Any:
            return _svd_directions(x.double(), threshold)

        def pair(a: Any, b: Any) -> tuple[float, dict[str, Any]]:
            rho = torch.linalg.svdvals(a[0].T @ b[0]).clamp(0.0, 1.0)
            return float(rho.mean()), {"r2": float(rho.square().mean()), "k_a": float(a[1]), "k_b": float(b[1])}

    sa = {l: summary(_to(_tensor(layers_a[l])[idx], dev)) for l in ids_a}
    sb = sa if same else {l: summary(_to(_tensor(b_map[l])[idx], dev)) for l in ids_b}
    depths = _normalized_depths(ids_a)
    done: dict[tuple[int, int], tuple[float, dict[str, Any]]] = {}
    out = Records()
    for la in ids_a:
        for lb in ids_b:
            if same and (lb, la) in done:
                value, extras = done[(lb, la)]
                if metric == "svcca":
                    extras = {**extras, "k_a": extras["k_b"], "k_b": extras["k_a"]}
            else:
                value, extras = pair(sa[la], sb[lb])
                done[(la, lb)] = (value, extras)
            out.rows.append(
                Record(
                    metric=spec.name,
                    value=value,
                    layer=la,
                    layer_b=lb,
                    depth=depths[la],
                    model=model if model_b is None else f"{model}->{model_b}",
                    level="sequence",
                    pooling=pooling,
                    corpus=corpus,
                    n_items=len(idx),
                    dim=int(_tensor(layers_a[la]).shape[-1]),
                    preprocess="none",
                    params=dict(params),
                    seed=seed,
                    tags=spec.tags,
                    extras=dict(extras),
                    version=__version__,
                )
            )
    return out
