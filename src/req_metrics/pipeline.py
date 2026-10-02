"""Compute registered metrics over the layers of one model and return Records.

Inputs are plain tensors keyed by layer index; how they were produced is left
to the caller and is recorded through the labels and the ViewSpec and
ShiftSpec objects. Populations:

- "pooled": layers[l] is (N, D), one vector per clip.
- "frames": layers[l] is a sequence of (T_i, D) tensors, one per clip; point
  metrics run per clip and are aggregated, trajectory metrics run per clip.
- "tokens": layers[l] is a sequence of (T_i, D) tensors; all frames of all
  clips are pooled into one point cloud.
- Token-field metrics (norm outliers, token cosines) take the same per-clip
  (T_i, D) inputs under "frames" or "tokens".
View metrics take layers[l] of shape (q, N, D); PTE takes layers[l] = (z, shifted).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor

from req_metrics import __version__
from req_metrics._types import InputKind, MetricResult
from req_metrics.neighbors import Neighbors, _cdist_mode
from req_metrics.preprocess import apply_preprocess
from req_metrics.records import Record, Records
from req_metrics.registry import MetricSpec, get_metric
from req_metrics.spectrum import Spectrum
from req_metrics.view_construction import ShiftSpec, ViewSpec

_NEIGHBOR_K = {
    "intrinsic_dimension/gride": ("range_max", 64),
    "intrinsic_dimension/mle": ("k_range", (10, 20)),
    "mlid": ("k", 64),
    "neighborhood_curvature": ("k", 64),
}


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


def _normalized_depths(layer_ids: Sequence[int]) -> dict[int, float]:
    top = max(layer_ids) if layer_ids else 0
    return {l: (l / top if top > 0 else 0.0) for l in layer_ids}


def _run(spec: MetricSpec, args: tuple, kwargs: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    try:
        res: MetricResult = spec.fn(*args, **kwargs)
        return res.value, dict(res.extras)
    except Exception as e:  # the pipeline records failures instead of aborting a sweep
        return float("nan"), {"error": f"{type(e).__name__}: {e}"}


def _aggregate(
    values: list[float], extras_list: list[dict[str, Any]], keep_per_clip: bool
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
    if keep_per_clip:
        out["per_clip"] = [float(v) for v in values]
    numeric = [e for e in extras_list if e and "error" not in e]
    if numeric:  # per-clip extras shared by every clip are averaged under their own names
        for key in set.intersection(*(set(e) for e in numeric)):
            vals_k = [e[key] for e in numeric if isinstance(e[key], (int, float)) and not isinstance(e[key], bool)]
            if len(vals_k) == len(numeric) and key not in out:
                out[key] = float(np.mean(vals_k))
    first_err = next((e["error"] for e in extras_list if "error" in e), None)
    if first_err is not None:
        out["error"] = first_err
    return (float(vals.mean()) if vals.size else float("nan")), out


def _cap(x: Tensor, cap: int | None, seed: int) -> Tensor:
    """Seeded random subsample of the rows of x down to cap (sorted indices), or x itself."""
    if cap is None or x.shape[0] <= cap:
        return x
    idx = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(seed))[:cap].sort().values
    return x[idx.to(x.device)]


def _points_sweep(
    layer_tensor: Tensor,
    specs: list[MetricSpec],
    params: Mapping[str, Mapping[str, Any]],
    seed: int = 0,
    limits: Mapping[str, int] | None = None,
) -> dict[str, tuple[float, dict]]:
    """Run point metrics on one (N, D) cloud, sharing Spectrum per preprocessing and one Neighbors table.

    Metrics with a registry max_items (or an entry in limits) see a seeded subsample
    of at most that many rows, with their own spectrum or neighbor table; the count
    used is written to extras["n_items_used"].
    """
    out: dict[str, tuple[float, dict]] = {}
    x = layer_tensor
    limits = dict(limits or {})
    subsets: dict[int, Tensor] = {x.shape[0]: x}

    def subset(cap: int | None) -> Tensor:
        n = x.shape[0] if cap is None else min(int(cap), x.shape[0])
        if n not in subsets:
            subsets[n] = _cap(x, n, seed)
        return subsets[n]

    spectra: dict[Any, Spectrum] = {}
    neighbors: dict[int, Neighbors] = {}
    need_k: list[Any] = [
        (
            _NEIGHBOR_K[s.name][1]
            if _NEIGHBOR_K[s.name][0] not in params.get(s.name, {})
            else params[s.name][_NEIGHBOR_K[s.name][0]]
        )
        for s in specs
        if s.cache in ("neighbors", "neighbors_kw")
    ]
    k_max = 0
    for k in need_k:
        k_max = max(k_max, k[1] if isinstance(k, tuple) else int(k))
    for spec in specs:
        kw = dict(params.get(spec.name, {}))
        xs = subset(limits.get(spec.name, spec.max_items))
        n_used = int(xs.shape[0])
        if spec.cache == "spectrum":
            key = (spec.preprocess, n_used)
            if key not in spectra:
                spectra[key] = Spectrum.from_points(apply_preprocess(xs, spec.preprocess), center=False)
            out[spec.name] = _run(spec, (spectra[key],), kw)
        elif spec.cache in ("neighbors", "neighbors_kw"):
            if n_used not in neighbors:
                try:
                    neighbors[n_used] = Neighbors.from_points(xs, k_max)
                except Exception as e:
                    out[spec.name] = (float("nan"), {"error": f"{type(e).__name__}: {e}"})
                    continue
            if spec.cache == "neighbors":
                out[spec.name] = _run(spec, (neighbors[n_used],), kw)
            else:
                out[spec.name] = _run(spec, (xs,), {**kw, "neighbors": neighbors[n_used]})
        else:
            out[spec.name] = _run(spec, (xs,), kw)
        if n_used < x.shape[0]:
            out[spec.name][1]["n_items_used"] = n_used
    return out


def compute(
    layers: Mapping[int, Any],
    metrics: Sequence[str],
    *,
    population: str = "pooled",
    n: int | None = None,
    seed: int = 0,
    group_ids: Sequence | None = None,
    params: Mapping[str, Mapping[str, Any]] | None = None,
    model: str | None = None,
    pooling: str | None = None,
    corpus: str | None = None,
    views: ViewSpec | None = None,
    shifts: ShiftSpec | None = None,
    keep_per_clip: bool = False,
    limits: Mapping[str, int] | None = None,
) -> Records:
    """Compute metrics for every layer and return one Record per (layer, metric).

    Args:
        layers: Layer index -> data. Point metrics: (N, D) for "pooled", a sequence of
            (T_i, D) for "frames" and "tokens". View metrics: (q, N, D). PTE: (z, {k: z_k}).
        metrics: Registry names. All must share one input kind per call.
        population: "pooled", "frames" or "tokens" (point and trajectory metrics only).
        n: Keep at most n clips (or tokens, for "tokens") chosen at random with seed; the
            same clips are used for every layer.
        seed: Subsampling seed, recorded.
        group_ids: One id per clip; one random clip per group is kept before subsampling.
        params: Metric name -> estimator keyword arguments; recorded in the records.
        model, pooling, corpus: Labels recorded verbatim.
        views: ViewSpec describing how view stacks were built (view metrics).
        shifts: ShiftSpec describing the pitch-shifted copies (PTE).
        keep_per_clip: For "frames", keep every clip's value in extras["per_clip"].
        limits: Metric name -> cap on the items that estimator sees (overrides the registry
            max_items); a seeded subsample is drawn above the cap and recorded in
            extras["n_items_used"].

    Returns:
        Records, one row per (layer, metric), failures recorded as nan with an error message.
    """
    params = {k: dict(v) for k, v in (params or {}).items()}
    specs = [get_metric(m) for m in metrics]
    kinds = {s.inputs for s in specs}
    per_clip_kinds = {InputKind.POINTS, InputKind.TRAJECTORY, InputKind.TOKENS}
    if len(kinds) != 1 and not kinds <= per_clip_kinds:
        raise ValueError(f"metrics of mixed input kinds in one call: {sorted(k.value for k in kinds)}")
    kind = kinds.pop() if len(kinds) == 1 else InputKind.TRAJECTORY  # mixed per-clip kinds: handled per metric below
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
                population=population,
                pooling=pooling,
                corpus=corpus,
                n_items=n_items,
                dim=dim,
                n_views=n_views,
                preprocess=spec.preprocess.describe(),
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
            v = torch.as_tensor(layers[l])[:, idx]
            for spec in specs:
                cap = caps.get(spec.name, spec.max_items)
                vs = v if cap is None or v.shape[1] <= cap else v[:, _cap(torch.arange(v.shape[1]), cap, seed)]
                value, extras = _run(spec, (vs,), params.get(spec.name, {}))
                if vs.shape[1] < v.shape[1]:
                    extras["n_items_used"] = int(vs.shape[1])
                record(spec, l, value, extras, len(idx), int(v.shape[-1]), int(v.shape[0]))
        return records

    if kind == InputKind.SHIFTED:
        z0, _ = layers[layer_ids[0]]
        idx = choose_indices(z0.shape[0], n, seed, group_ids)
        for l in layer_ids:
            z, shifted = layers[l]
            z = torch.as_tensor(z)[idx]
            sh = {k: torch.as_tensor(v)[idx] for k, v in shifted.items()}
            for spec in specs:
                value, extras = _run(spec, (z, sh), params.get(spec.name, {}))
                record(spec, l, value, extras, len(idx), int(z.shape[-1]), len(sh))
        return records

    if kind == InputKind.PAIR:
        raise ValueError("pair metrics compare two representations; use compute_pairs")

    if population == "pooled":
        if any(s.inputs in (InputKind.TRAJECTORY, InputKind.TOKENS) for s in specs):
            raise ValueError("trajectory and token-field metrics need population='frames' or 'tokens'")
        first = torch.as_tensor(layers[layer_ids[0]])
        counts = {l: int(torch.as_tensor(layers[l]).shape[0]) for l in layer_ids}
        if len(set(counts.values())) > 1:
            raise ValueError(f"layers must have the same number of rows, got {counts}")
        idx = choose_indices(first.shape[0], n, seed, group_ids)
        for l in layer_ids:
            x = torch.as_tensor(layers[l])[idx]
            for name, (value, extras) in _points_sweep(x, specs, params, seed, limits).items():
                record(get_metric(name), l, value, extras, len(idx), int(x.shape[-1]))
        return records

    if population == "tokens":
        if any(s.inputs == InputKind.TRAJECTORY for s in specs):
            raise ValueError("trajectory metrics need population='frames'")
        for l in layer_ids:
            x = torch.cat([torch.as_tensor(c) for c in layers[l]], dim=0)
            idx = choose_indices(x.shape[0], n, seed)  # token subsample; group filtering is per clip, not per token
            x = x[idx]
            for name, (value, extras) in _points_sweep(x, specs, params, seed, limits).items():
                record(get_metric(name), l, value, extras, len(idx), int(x.shape[-1]))
        return records

    if population == "frames":
        clips0 = layers[layer_ids[0]]
        idx = choose_indices(len(clips0), n, seed, group_ids)
        for l in layer_ids:
            clips = layers[l]
            per_metric: dict[str, tuple[list[float], list[dict]]] = {s.name: ([], []) for s in specs}
            dim = int(torch.as_tensor(clips[idx[0]]).shape[-1])
            point_specs = [s for s in specs if s.inputs == InputKind.POINTS]
            other_specs = [s for s in specs if s.inputs != InputKind.POINTS]
            for i in idx:
                c = torch.as_tensor(clips[i])
                results = _points_sweep(c, point_specs, params, seed, limits) if point_specs else {}
                results.update({s.name: _run(s, (c,), params.get(s.name, {})) for s in other_specs})
                for name, (value, extras) in results.items():
                    per_metric[name][0].append(value)
                    per_metric[name][1].append(extras)
            for name, (values, extras_list) in per_metric.items():
                value, extras = _aggregate(values, extras_list, keep_per_clip)
                record(get_metric(name), l, value, extras, len(idx), dim)
        return records

    raise ValueError(f"unknown population {population!r}")


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
) -> Records:
    """A layer-pair metric between every layer of A and every layer of B (B defaults to A).

    Rows of all layers must describe the same clips in the same order. One
    Record per (layer_a, layer_b). metric is "information_imbalance" (default
    k = 1; value Delta(A -> B), extras["reverse"]) or "neighborhood_overlap"
    (default k = 30; symmetric). For the imbalance the cost is one chunked rank
    table per layer rather than one per pair: for each target layer the ranks
    of all items are computed once and gathered at the k nearest neighbors of
    every source layer, so L layers cost O(L N^2 D) instead of O(L^2 N^2 D).
    For the overlap only the k-nearest-neighbor tables are needed, once per layer.
    """
    if metric not in ("information_imbalance", "neighborhood_overlap"):
        raise ValueError(f"unknown layer-pair metric {metric!r}")
    if k is None:
        k = 1 if metric == "information_imbalance" else 30
    spec = get_metric(metric)
    same = layers_b is None
    b_map: Mapping[int, Tensor] = layers_a if layers_b is None else layers_b
    ids_a, ids_b = sorted(layers_a), sorted(b_map)
    first = torch.as_tensor(layers_a[ids_a[0]])
    idx = choose_indices(first.shape[0], n, seed, group_ids)
    xa = {l: torch.as_tensor(layers_a[l])[idx].double() for l in ids_a}
    xb = xa if same else {l: torch.as_tensor(b_map[l])[idx].double() for l in ids_b}
    n_items = len(idx)
    nn_a = {l: Neighbors.from_points(xa[l], k).indices[:, 1 : k + 1] for l in ids_a}
    nn_b = nn_a if same else {l: Neighbors.from_points(xb[l], k).indices[:, 1 : k + 1] for l in ids_b}
    if metric == "neighborhood_overlap":
        from req_metrics.metrics.compare import _shared_neighbor_fraction

        depths = _normalized_depths(ids_a)
        out = Records()
        for la in ids_a:
            for lb in ids_b:
                frac = _shared_neighbor_fraction(nn_a[la], nn_b[lb])
                out.rows.append(
                    Record(
                        metric=spec.name,
                        value=float(frac.mean()),
                        layer=la,
                        layer_b=lb,
                        depth=depths[la],
                        model=model if model_b is None else f"{model}->{model_b}",
                        population="pooled",
                        pooling=pooling,
                        corpus=corpus,
                        n_items=n_items,
                        dim=int(xa[la].shape[-1]),
                        preprocess="none",
                        params={"k": k},
                        seed=seed,
                        tags=spec.tags,
                        extras={"std": float(frac.std()), "chance": k / (n_items - 1)},
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
                    population="pooled",
                    pooling=pooling,
                    corpus=corpus,
                    n_items=n_items,
                    dim=int(xa[la].shape[-1]),
                    preprocess="none",
                    params={"k": k},
                    seed=seed,
                    tags=spec.tags,
                    extras={"reverse": sums_ba[(la, lb)] / scale},
                    version=__version__,
                )
            )
    return out
