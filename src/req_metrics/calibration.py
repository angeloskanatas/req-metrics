"""Permutation-null calibration of pairwise scores, and half-sample error bars.

Groger, Wen and Brbic (2026, ICML, arXiv:2602.14486, Sec. 5, Algorithms 1 and 2; reference code
github.com/mlbio-epfl/aristotelian, calibrated_similarity/calibration.py). The null distribution is the same score
with the rows of Y permuted, k times. tau is the ceil((1 - alpha)(k + 1))-th order statistic of the observed and null
scores together, the add-one p-value is (1 + #{null >= observed}) / (k + 1), and the calibrated score is
max((s - tau) / (s_best - tau), 0): zero at or below what chance reaches at level alpha, one at the metric's best
value. They report that raw CKA and Procrustes scores grow with width at a fixed number of items while the
calibrated ones do not, that neighborhood scores have a null of order k/N, and that a maximum over a grid of layer
pairs must be calibrated as a maximum: calibrate_layers applies one permutation to every layer of Y and calibrates
the aggregate. Their defaults: k = 200 (thresholds settle by about 50), alpha = 0.05.

Scores where lower means closer, such as the information imbalance (0 for identical neighborhoods, about 1 without
information), are calibrated on their negation (lower_is_better=True, best=0.0); tau and the null statistics are
reported in the metric's own units.

Frames of one clip are not exchangeable: permuting single frames keeps no clip together in the null, so a score
that only reflects which frames share a clip beats it. groups= permutes whole groups (clips) among groups of the
same size and keeps each group's internal order, the restricted permutation the paper names for dependent samples
(Appendix C.1).

jackknife: the spread of a score over random half-samples of the items, as Acevedo et al. (2026, TMLR,
arXiv:2505.17101) report for the information imbalance (five draws of 50%).

calibrate_grid calibrates every cell of a layer-pair grid from the same draws (Algorithm 1 applied to each cell).
Groger et al. report only the calibrated maximum over layer pairs (calibrate_layers), not a per-cell map.

For information_imbalance (with at most k and l2 set and N up to _FAST_IMBALANCE_MAX_N), calibrate, calibrate_layers,
calibrate_grid and jackknife compute each space's distances, ranks and neighbor weights once. A row permutation of Y only re-indexes
Y's rank table, so a null draw costs O(Nk) per layer pair instead of the full O(N^2 D) recomputation; a subset is
ranked from the stored distances in O(N^2). The values are information_imbalance's; on data with exactly tied
distances they can differ in the last digits, because tied distances may round differently here. Rank tables are
float32, which holds ranks (multiples of 1/2) exactly: 36 MB per layer of Y at N = 3,000. For neighborhood_overlap and cycle_knn (with at most k, l2 and jaccard
set) the k-nearest-neighbor table of every layer is built once: a row permutation of Y only relabels Y's table (row i
of the permuted Y is row perm[i] of Y, its neighbors the permuted positions of that row's neighbors), and the metric
receives the permuted rows with their table, so it computes exactly what it would from scratch without the O(N^2 D)
search per draw.
"""

from __future__ import annotations

import functools
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from req_metrics._types import MetricResult

PairScore = Callable[..., MetricResult | float]


@dataclass(frozen=True)
class Calibrated:
    """A score with its permutation null.

    score: the observed value. calibrated: max((s - tau) / (best - tau), 0) in the better-is-higher orientation, or
    the unnormalized excess when best is None. p_value: add-one right-tail permutation p-value in that orientation.
    tau, null_mean, null_std: the null threshold and moments in the metric's own units. k, alpha: the settings.
    """

    score: float
    calibrated: float
    p_value: float
    tau: float
    null_mean: float
    null_std: float
    k: int
    alpha: float


@dataclass(frozen=True)
class CalibratedGrid:
    """Every cell of a layer-pair grid with its permutation null.

    score, calibrated, p_value, tau, null_mean, null_std: (L_A, L_B) tensors with the meaning of the fields of
    Calibrated, cell by cell. The null of every cell comes from the same k draws, one permutation of the rows of B
    per draw applied to every layer of B. k, alpha: the settings.
    """

    score: Tensor
    calibrated: Tensor
    p_value: Tensor
    tau: Tensor
    null_mean: Tensor
    null_std: Tensor
    k: int
    alpha: float

    def cell(self, i: int, j: int) -> Calibrated:
        """The calibration of the pair (layers_a[i], layers_b[j])."""
        return Calibrated(
            score=float(self.score[i, j]),
            calibrated=float(self.calibrated[i, j]),
            p_value=float(self.p_value[i, j]),
            tau=float(self.tau[i, j]),
            null_mean=float(self.null_mean[i, j]),
            null_std=float(self.null_std[i, j]),
            k=self.k,
            alpha=self.alpha,
        )


def _value(result: MetricResult | float) -> float:
    return float(result.value) if isinstance(result, MetricResult) else float(result)


_FAST_IMBALANCE_MAX_N = 3000  # two N x N float64 distance tables per pair: 144 MB at this size


def _fast_imbalance(metric: PairScore, kwargs: dict, xs: Sequence[Tensor], n_min: int) -> tuple[int, bool] | None:
    """(k, l2) when stored tables can stand in for metric: information_imbalance, or a functools.partial of it (the
    way to set its k, since the calibration functions' own k counts permutations), with no keyword but k and l2,
    inputs of shape (N, D) with N <= _FAST_IMBALANCE_MAX_N, and more than k items in every draw of n_min items. None
    sends the caller to the general path."""
    from req_metrics.metrics.compare import information_imbalance

    if isinstance(metric, functools.partial) and metric.func is information_imbalance and not metric.args:
        kwargs = {**metric.keywords, **kwargs}
        metric = metric.func
    if metric is not information_imbalance or set(kwargs) - {"k", "l2"}:
        return None
    k = int(kwargs.get("k", 1))
    if any(x.ndim != 2 for x in xs) or xs[0].shape[0] > _FAST_IMBALANCE_MAX_N or not 1 <= k <= n_min - 1:
        return None
    return k, bool(kwargs.get("l2", False))


def _distances(x: Tensor, l2: bool) -> Tensor:
    """(N, N) Euclidean distances after information_imbalance's preprocessing, with an infinite diagonal."""
    from req_metrics.neighbors import _cdist_mode
    from req_metrics.preprocess import l2_normalize

    x = l2_normalize(x.double()) if l2 else x.double()
    return torch.cdist(x, x, compute_mode=_cdist_mode(x.dtype)).fill_diagonal_(float("inf"))


def _source_weights(d: Tensor, k: int) -> Tensor:
    """(N, N) weights of each row's k nearest columns; a tie at the k-th distance shares the remaining weight equally
    over the tied columns, information_imbalance's rule (an infinite diagonal never counts)."""
    kth = d.kthvalue(k, dim=1, keepdim=True).values
    closer, tied = d < kth, d == kth
    share = (k - closer.sum(dim=1, keepdim=True)).double() / tied.sum(dim=1, keepdim=True).double()
    return closer.double() + tied.double() * share


def _neighbor_entries(d: Tensor, k: int) -> tuple[Tensor, Tensor, Tensor]:
    """Rows, columns and weights of the nonzero entries of _source_weights(d, k)."""
    w = _source_weights(d, k)
    rows, cols = w.nonzero(as_tuple=True)
    return rows, cols, w[rows, cols]


def _rank_table(d: Tensor) -> Tensor:
    """Average ranks per row, as information_imbalance counts them, in float32."""
    from req_metrics.metrics.compare import _row_ranks

    return _row_ranks(d).float()


def _imbalance(entries: tuple[Tensor, Tensor, Tensor], ranks: Tensor, k: int, perm: Tensor | None = None) -> float:
    """Delta(A -> B) from A's neighbor entries and B's rank table, B's rows permuted by perm."""
    rows, cols, w = entries
    if perm is not None:
        rows, cols = perm[rows], perm[cols]
    n = ranks.shape[0]
    return float((w * ranks[rows, cols].double()).sum()) / (n * k * (n / 2.0))


class _ImbalanceTables:
    """Both spaces' distances, A's neighbor entries and B's ranks, computed once, for many values of Delta(A -> B)."""

    def __init__(self, x_a: Tensor, x_b: Tensor, k: int, l2: bool) -> None:
        self.k = k
        self.d_a, self.d_b = _distances(x_a, l2), _distances(x_b, l2)
        self.entries, self.r_b = _neighbor_entries(self.d_a, k), _rank_table(self.d_b)

    def value(self, perm: Tensor | None = None) -> float:
        """Delta(A -> B) with B's rows permuted by perm."""
        return _imbalance(self.entries, self.r_b, self.k, perm)

    def subset(self, rows: Tensor) -> float:
        """Delta(A -> B) on a subset of the items, ranked among the subset."""
        d_a, d_b = self.d_a[rows][:, rows], self.d_b[rows][:, rows]
        return _imbalance(_neighbor_entries(d_a, self.k), _rank_table(d_b), self.k)


class _LayerImbalanceTables:
    """A's neighbor entries and B's rank tables per layer, for the grid Delta(A_l -> B_m) under row permutations of B."""

    def __init__(self, layers_a: Sequence[Tensor], layers_b: Sequence[Tensor], k: int, l2: bool) -> None:
        self.k = k
        self.entries = [_neighbor_entries(_distances(x, l2), k) for x in layers_a]
        self.ranks = [_rank_table(_distances(y, l2)) for y in layers_b]

    def grid(self, perm: Tensor | None) -> Tensor:
        """Imbalance of every A layer toward every B layer, with B's items permuted by `perm` when given."""
        out = torch.empty(len(self.entries), len(self.ranks), dtype=torch.float64)
        for j, r in enumerate(self.ranks):
            p = None if perm is None else perm.to(r.device)
            for i, e in enumerate(self.entries):
                out[i, j] = _imbalance(e, r, self.k, p)
        return out


def _fast_neighbors(metric: PairScore, kwargs: dict, xs: Sequence[Tensor], n_min: int) -> tuple[int, bool] | None:
    """(neighbors per table, l2) when stored k-nearest-neighbor tables can stand in for metric: neighborhood_overlap or
    cycle_knn, or a functools.partial of either, with no keyword but k, l2 and (for the overlap) jaccard, inputs of
    shape (N, D) and more than k items in every draw of n_min items. The tables hold k + 1 neighbors, the one more
    the overlap reads to see a tie at the k-th. None sends the caller to the general path."""
    from req_metrics.metrics.compare import cycle_knn, neighborhood_overlap

    if isinstance(metric, functools.partial) and not metric.args:
        kwargs = {**metric.keywords, **kwargs}
        metric = metric.func
    allowed = {neighborhood_overlap: ({"k", "l2", "jaccard"}, 30), cycle_knn: ({"k", "l2"}, 10)}
    if metric not in allowed or set(kwargs) - allowed[metric][0]:
        return None
    k = int(kwargs.get("k", allowed[metric][1]))
    if any(x.ndim != 2 for x in xs) or not 1 <= k <= n_min - 1:
        return None
    return k + 1, bool(kwargs.get("l2", False))


class _NeighborTables:
    """The k-nearest-neighbor table of every layer of A and of B, for a neighbor metric under row permutations of B."""

    def __init__(self, layers_a: Sequence[Tensor], layers_b: Sequence[Tensor], need: int, l2: bool) -> None:
        from req_metrics.neighbors import Neighbors
        from req_metrics.preprocess import l2_normalize

        n = layers_b[0].shape[0]
        prep = (lambda x: l2_normalize(x.double())) if l2 else (lambda x: x.double())
        self.a = [Neighbors.from_points(prep(x), min(need, n - 1)) for x in layers_a]
        self.b = [Neighbors.from_points(prep(y), min(need, n - 1)) for y in layers_b]

    @staticmethod
    def permuted(table, perm: Tensor):
        """The table of y[perm] from the table of y: rows reordered, neighbor indices relabeled."""
        perm = perm.to(table.indices.device)
        inverse = torch.empty_like(perm)
        inverse[perm] = torch.arange(perm.numel(), device=perm.device)
        return type(table)(table.distances[perm], inverse[table.indices[perm]])


def _grid_function(
    metric: PairScore, layers_a: Sequence[Tensor], layers_b: Sequence[Tensor], metric_kwargs: dict, n: int
) -> Callable[[Tensor | None], Tensor]:
    """grid(perm): the (L_A, L_B) float64 grid of metric(layers_a[l], layers_b[m][perm]), through the imbalance or the
    neighbor tables when the metric allows and through the metric itself otherwise; perm None gives the observed grid."""
    fast = _fast_imbalance(metric, metric_kwargs, [*layers_a, *layers_b], n)
    if fast is not None:
        return _LayerImbalanceTables(layers_a, layers_b, *fast).grid
    neighbors = _fast_neighbors(metric, metric_kwargs, [*layers_a, *layers_b], n)
    tables = _NeighborTables(layers_a, layers_b, *neighbors) if neighbors is not None else None

    def grid(perm: Tensor | None) -> Tensor:
        out = torch.empty(len(layers_a), len(layers_b), dtype=torch.float64)
        for j, y in enumerate(layers_b):
            y_perm = y if perm is None else y[perm.to(y.device)]
            table_b = None if tables is None else (tables.b[j] if perm is None else tables.permuted(tables.b[j], perm))
            for i, x in enumerate(layers_a):
                kwargs = (
                    metric_kwargs
                    if tables is None
                    else {**metric_kwargs, "neighbors_a": tables.a[i], "neighbors_b": table_b}
                )
                out[i, j] = _value(metric(x, y_perm, **kwargs))
        return out

    return grid


def group_permutation(groups: Tensor | np.ndarray, generator: torch.Generator) -> Tensor:
    """Row indices that move whole groups onto other groups of the same size, keeping each group's row order.

    groups: (N,) group label of every row. Rows of a group need not be contiguous; within a group, rows keep their
    order of appearance, so frame t of one clip is paired with frame t of another.
    """
    labels = torch.as_tensor(groups)
    if labels.ndim != 1:
        raise ValueError(f"groups must be one label per row, got shape {tuple(labels.shape)}")
    unique, inverse = torch.unique(labels, return_inverse=True)
    members = [torch.nonzero(inverse == g).flatten() for g in range(len(unique))]
    perm = torch.arange(labels.numel())
    by_size: dict[int, list[int]] = {}
    for g, rows in enumerate(members):
        by_size.setdefault(len(rows), []).append(g)
    movable = sum(size * len(same) for size, same in by_size.items() if len(same) > 1)
    if movable == 0:
        raise ValueError("no two groups have the same size, so no group can be permuted; give groups of equal size")
    if movable < labels.numel() / 2:
        warnings.warn(
            f"only {movable} of {labels.numel()} rows are in groups that share their size with another group; the "
            "other groups keep their pairing in every draw, which inflates the null",
            RuntimeWarning,
            stacklevel=2,
        )
    for same in by_size.values():
        order = torch.randperm(len(same), generator=generator)
        for src, dst in zip(same, [same[i] for i in order.tolist()], strict=True):
            perm[members[src]] = members[dst]
    return perm


def _null_threshold(observed: float, null: Tensor, alpha: float) -> tuple[float, float]:
    """tau as the ceil((1 - alpha)(k + 1))-th order statistic of observed and null together, and the p-value."""
    k = null.numel()
    combined = torch.sort(torch.cat([torch.tensor([observed], dtype=null.dtype), null])).values
    idx = min(max(math.ceil((1.0 - alpha) * (k + 1)) - 1, 0), k)
    p_value = (1.0 + float((null >= observed).sum())) / (k + 1)
    return float(combined[idx]), p_value


def _calibrated(
    observed: float, null: Tensor, alpha: float, best: float | None, lower_is_better: bool, k: int
) -> Calibrated:
    sign = -1.0 if lower_is_better else 1.0
    t_obs, t_null = sign * observed, sign * null
    tau_t, p_value = _null_threshold(t_obs, t_null, alpha)
    if best is None:
        cal = max(t_obs - tau_t, 0.0)
    else:
        t_best = sign * best
        cal = min(max((t_obs - tau_t) / max(t_best - tau_t, 1e-12), 0.0), 1.0)
    return Calibrated(
        score=observed,
        calibrated=cal,
        p_value=p_value,
        tau=sign * tau_t,
        null_mean=float(null.mean()),
        null_std=float(null.std(unbiased=True)) if k > 1 else 0.0,
        k=k,
        alpha=alpha,
    )


def _check(k: int, alpha: float) -> None:
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def calibrate(
    metric: PairScore,
    x_a: Tensor,
    x_b: Tensor,
    *,
    k: int = 200,
    alpha: float = 0.05,
    best: float | None = 1.0,
    lower_is_better: bool = False,
    groups: Tensor | np.ndarray | None = None,
    seed: int = 0,
    **metric_kwargs,
) -> Calibrated:
    """Calibrate metric(x_a, x_b) against k permutations of x_b's rows (Algorithm 1 of Groger et al.).

    Args:
        metric: A pairwise score, e.g. req_metrics.cka or information_imbalance; extra keywords are passed on.
        x_a, x_b: (N, D_a) and (N, D_b), rows of the same items.
        k: Permutations.
        alpha: Level of the null threshold.
        best: The metric's best attainable value (1 for CKA, RSA and neighborhood overlap; 0 for the information
            imbalance), or None for the unnormalized excess over tau.
        lower_is_better: The score falls as the spaces agree (information imbalance).
        groups: (N,) labels; whole groups are permuted instead of single rows (frames of one clip).
        seed: Seed of the permutations.
    """
    _check(k, alpha)
    if x_a.shape[0] != x_b.shape[0]:
        raise ValueError(f"x_a and x_b need the same rows, got {x_a.shape[0]} and {x_b.shape[0]}")
    gen = torch.Generator().manual_seed(seed)
    n = x_b.shape[0]
    grid = _grid_function(metric, [x_a], [x_b], metric_kwargs, n)
    observed = float(grid(None)[0, 0])
    null = torch.empty(k, dtype=torch.float64)
    for i in range(k):
        perm = group_permutation(groups, gen) if groups is not None else torch.randperm(n, generator=gen)
        null[i] = float(grid(perm)[0, 0])
    return _calibrated(observed, null, alpha, best, lower_is_better, k)


_AGGREGATES: dict[str, Callable[[Tensor], Tensor]] = {
    "max": lambda m: m.max(),
    "min": lambda m: m.min(),
    "mean": lambda m: m.mean(),
    "row_max_mean": lambda m: m.max(dim=1).values.mean(),
    "col_max_mean": lambda m: m.max(dim=0).values.mean(),
}


def calibrate_layers(
    metric: PairScore,
    layers_a: Sequence[Tensor],
    layers_b: Sequence[Tensor],
    *,
    aggregate: str | Callable[[Tensor], Tensor] = "max",
    k: int = 200,
    alpha: float = 0.05,
    best: float | None = 1.0,
    lower_is_better: bool = False,
    groups: Tensor | np.ndarray | None = None,
    seed: int = 0,
    **metric_kwargs,
) -> tuple[Calibrated, Tensor]:
    """Calibrate an aggregate of the layer-pair grid S[l, m] = metric(layers_a[l], layers_b[m]) (Algorithm 2).

    Each null draw applies one permutation to the rows of every layer of B, so the null of the aggregate keeps the
    correlation between layers. For a lower-is-better score pass aggregate="min" to calibrate the best pair.

    Returns:
        The calibrated aggregate, and the observed (L_A, L_B) grid.
    """
    _check(k, alpha)
    agg = _AGGREGATES[aggregate] if isinstance(aggregate, str) else aggregate
    n = layers_b[0].shape[0]
    if any(x.shape[0] != n for x in [*layers_a, *layers_b]):
        raise ValueError("every layer needs the same rows")
    grid = _grid_function(metric, layers_a, layers_b, metric_kwargs, n)
    gen = torch.Generator().manual_seed(seed)
    observed_grid = grid(None)
    observed = float(agg(observed_grid))
    null = torch.empty(k, dtype=torch.float64)
    for i in range(k):
        perm = group_permutation(groups, gen) if groups is not None else torch.randperm(n, generator=gen)
        null[i] = float(agg(grid(perm)))
    return _calibrated(observed, null, alpha, best, lower_is_better, k), observed_grid


def calibrate_grid(
    metric: PairScore,
    layers_a: Sequence[Tensor],
    layers_b: Sequence[Tensor],
    *,
    k: int = 200,
    alpha: float = 0.05,
    best: float | None = 1.0,
    lower_is_better: bool = False,
    groups: Tensor | np.ndarray | None = None,
    seed: int = 0,
    **metric_kwargs,
) -> CalibratedGrid:
    """Calibrate every cell of the layer-pair grid S[l, m] = metric(layers_a[l], layers_b[m]) (Algorithm 1 per cell).

    Groger et al. report only the calibrated maximum over layer pairs (calibrate_layers here), not a per-cell map;
    the map serves to relate cells to other per-layer quantities. The k draws are shared by the cells, one permutation
    of the rows of every layer of B per draw, so with the same seed a cell equals calibrate() on that pair and the
    cells' nulls are correlated as the layers are. The maximum over the calibrated cells is not the calibrated maximum.

    Returns:
        CalibratedGrid, one (L_A, L_B) tensor per field of Calibrated.
    """
    _check(k, alpha)
    n = layers_b[0].shape[0]
    if any(x.shape[0] != n for x in [*layers_a, *layers_b]):
        raise ValueError("every layer needs the same rows")
    grid = _grid_function(metric, layers_a, layers_b, metric_kwargs, n)
    gen = torch.Generator().manual_seed(seed)
    observed = grid(None)
    null = torch.empty(k, *observed.shape, dtype=torch.float64)
    for i in range(k):
        perm = group_permutation(groups, gen) if groups is not None else torch.randperm(n, generator=gen)
        null[i] = grid(perm)
    fields = {name: torch.empty_like(observed) for name in ("calibrated", "p_value", "tau", "null_mean", "null_std")}
    for i in range(observed.shape[0]):
        for j in range(observed.shape[1]):
            cell = _calibrated(float(observed[i, j]), null[:, i, j], alpha, best, lower_is_better, k)
            for name, grid_field in fields.items():
                grid_field[i, j] = getattr(cell, name)
    return CalibratedGrid(score=observed, k=k, alpha=alpha, **fields)


def jackknife(
    metric: PairScore,
    x_a: Tensor,
    x_b: Tensor,
    *,
    n_draws: int = 5,
    fraction: float = 0.5,
    seed: int = 0,
    **metric_kwargs,
) -> tuple[float, float, list[float]]:
    """Mean and standard deviation of metric over n_draws random subsets of fraction of the items.

    Acevedo et al. (2026, TMLR, arXiv:2505.17101) report the information imbalance with the standard deviation over
    five half-samples; their code reports the half-sample mean as the value. Scores that depend on N, such as the
    information imbalance, differ between N and fraction * N, so report the full-sample value with this spread, or
    state which is reported.

    Returns:
        (mean, std, values).
    """
    if n_draws < 1 or not 0.0 < fraction <= 1.0:
        raise ValueError("need n_draws >= 1 and 0 < fraction <= 1")
    gen = torch.Generator().manual_seed(seed)
    n = x_a.shape[0]
    m = max(2, int(round(fraction * n)))
    fast = _fast_imbalance(metric, metric_kwargs, [x_a, x_b], m)
    tables = _ImbalanceTables(x_a, x_b, *fast) if fast is not None else None
    values = []
    for _ in range(n_draws):
        rows = torch.randperm(n, generator=gen)[:m]
        if tables is not None:
            values.append(tables.subset(rows.to(tables.d_a.device)))
        else:
            values.append(_value(metric(x_a[rows.to(x_a.device)], x_b[rows.to(x_b.device)], **metric_kwargs)))
    t = torch.tensor(values, dtype=torch.float64)
    return float(t.mean()), float(t.std(unbiased=True)) if n_draws > 1 else 0.0, values
