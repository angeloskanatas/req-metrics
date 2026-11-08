"""Metrics that compare two representations of the same items.

Rows of the two inputs describe the same items in the same order; widths may differ.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from req_metrics._types import InputKind, MetricResult, Preprocess
from req_metrics.neighbors import Neighbors, _cdist_mode
from req_metrics.preprocess import l2_normalize
from req_metrics.registry import register_metric


def _row_ranks(d: Tensor) -> Tensor:
    """Ranks (1 = nearest) of every column in each row of d, tied distances sharing their average rank.

    This is scipy.stats.rankdata(method="average") per row, the convention of the code of Del Tatto et al. (2024,
    PNAS, arXiv:2305.10817); without ties it is one plus the number of items strictly closer. An infinite diagonal
    (the point itself) ranks last.
    """
    s = d.sort(dim=1).values
    lo = torch.searchsorted(s, d)  # items strictly closer
    hi = torch.searchsorted(s, d, right=True)  # items closer or tied, the item itself included
    return lo.double() + (hi - lo + 1).double() / 2.0


def _boundary_ties(x: Tensor, table: Neighbors, k: int, chunk: int = 512) -> dict[int, tuple[Tensor, Tensor]]:
    """The rows whose k-th neighbor is tied with an item outside the first k: {row: (columns, weights)}.

    The items strictly closer than the k-th distance get weight 1 and the m items at that distance share the
    remaining k - c equally, which is the expected rank sum under a uniformly random choice among the tied items.
    Needs a table with at least k + 1 neighbors (a tie with the next item shows as equal distances); with k = N - 1
    every other item is a neighbor and nothing can tie at the boundary.
    """
    if table.k <= k:
        return {}
    rows_all = torch.nonzero(table.distances[:, k] == table.distances[:, k + 1]).flatten()
    out: dict[int, tuple[Tensor, Tensor]] = {}
    for start in range(0, len(rows_all), chunk):
        rows = rows_all[start : start + chunk].to(x.device)
        d = torch.cdist(x[rows], x, compute_mode=_cdist_mode(x.dtype))
        d[torch.arange(len(rows), device=x.device), rows] = float("inf")
        kth = d.kthvalue(k, dim=1, keepdim=True).values
        lt, eq = d < kth, d == kth
        for r, row in enumerate(rows.tolist()):
            closer, tied = torch.nonzero(lt[r]).flatten(), torch.nonzero(eq[r]).flatten()
            share = (k - len(closer)) / len(tied)
            ones = torch.ones(len(closer), dtype=torch.float64, device=x.device)
            shared = torch.full((len(tied),), share, dtype=torch.float64, device=x.device)
            out[row] = (torch.cat([closer, tied]), torch.cat([ones, shared]))
    return out


def _rank_sums(ranks: Tensor, nn: Tensor, ties: dict[int, tuple[Tensor, Tensor]], rows: Tensor) -> Tensor:
    """Per query row: the sum of its target ranks over its k source neighbors, tied boundaries taken from `ties`."""
    sums = torch.gather(ranks, 1, nn).sum(dim=1)
    for r, row in enumerate(rows.tolist()):
        if row in ties:
            cols, weights = ties[row]
            sums[r] = (ranks[r, cols] * weights).sum()
    return sums


def _mean_rank_in_b(x_b: Tensor, x_a: Tensor, table_a: Neighbors, k: int, chunk: int = 512) -> float:
    """Mean over points of the rank, in space B, of each point's k neighbors chosen in space A.

    Ranks in B share their average over tied distances (_row_ranks), and a tie at A's k-th neighbor is averaged over
    the tied items (_boundary_ties). B's distances are computed per chunk of rows and never stored as an N x N table.
    """
    n = x_b.shape[0]
    nn = table_a.indices[:, 1 : k + 1]
    ties = _boundary_ties(x_a, table_a, k)
    total = 0.0
    for start in range(0, n, chunk):
        rows = torch.arange(start, min(start + chunk, n), device=x_b.device)
        d = torch.cdist(x_b[rows], x_b, compute_mode=_cdist_mode(x_b.dtype))  # (c, N)
        d[torch.arange(len(rows), device=x_b.device), rows] = float("inf")  # exclude the point itself
        total += float(_rank_sums(_row_ranks(d), nn[rows], ties, rows).sum())
    return total / (n * k)


def information_imbalance(
    x_a: Tensor,
    x_b: Tensor,
    *,
    k: int = 1,
    l2: bool = False,
    neighbors_a: Neighbors | None = None,
    neighbors_b: Neighbors | None = None,
) -> MetricResult:
    """Information imbalance from representation A to representation B.

    Glielmo et al. (2022, PNAS Nexus, arXiv:2104.15079, Eq. 2): Delta(A -> B) = 2 <r_B | r_A = 1>
    / N, the mean rank in B of each point's nearest neighbor in A; 2/N for identical spaces without
    ties and 1 in expectation for independent ones. A small Delta(A -> B) with a large
    Delta(B -> A) means A contains the information in B. k > 1 averages the ranks of the k nearest
    A-neighbors, as in DADApy (Glielmo et al., 2022, Patterns, arXiv:2205.03373). Ranks are counted
    exactly, without storing an N x N table. Cheng et al. (2025, ICLR, arXiv:2405.15471) compare the
    layers of one model and different models with it. Acevedo et al. (2026, TMLR, arXiv:2505.17101
    v7) compare translations, same-class images and image-caption pairs, on token-averaged
    representations with unit-length (l2=True) or plain Euclidean distances. Their definition
    divides by N - 1 instead of N (N / (N - 1) in expectation for independent spaces), and their
    code ranks each query against a disjoint half of the items; both change the value by O(1/N).
    They report the standard deviation over five half-samples (req_metrics.calibration.jackknife)
    and a shuffled-pair null near 1 (req_metrics.calibration.calibrate with lower_is_better=True
    and best=0.0).

    Ties (discrete data such as binary rolls, labels or repeated items) are handled without an
    arbitrary order: tied B-distances share their average rank, as in the code of Del Tatto et al.
    (2024, PNAS, arXiv:2305.10817), which keeps 1 as the expectation for independent spaces, and
    items tied at A's k-th distance share the remaining weight equally, the expectation under a
    random choice. Without ties the value is Glielmo et al.'s count exactly.

    Args:
        x_a: Representation A, (N, D_a).
        x_b: Representation B, (N, D_b), rows of the same items as x_a.
        k: Nearest A-neighbors whose B-ranks are averaged.
        l2: Scale rows to unit norm first, so neighbors and ranks are cosine ones.
        neighbors_a, neighbors_b: Precomputed Neighbors tables of the (preprocessed) inputs; reused when they hold
            at least k + 1 neighbors (the extra one detects a tie at the k-th), rebuilt otherwise.

    Returns:
        value: Delta(A -> B).
        extras: reverse, Delta(B -> A); asymmetry, Delta(A -> B) - Delta(B -> A) (Acevedo et al.'s A).
    """
    if x_a.ndim != 2 or x_b.ndim != 2 or x_a.shape[0] != x_b.shape[0]:
        raise ValueError(f"expected two (N, D) tensors with equal N, got {tuple(x_a.shape)} and {tuple(x_b.shape)}")
    xa, xb = (l2_normalize(x_a.double()), l2_normalize(x_b.double())) if l2 else (x_a.double(), x_b.double())
    n = xa.shape[0]
    if not 1 <= k <= n - 1:
        raise ValueError(f"k must be in [1, N - 1], got {k} for N = {n}")
    need = min(k + 1, n - 1)  # one neighbor more than k, so a tie at the k-th is visible
    nb_a = neighbors_a if neighbors_a is not None and neighbors_a.k >= need else Neighbors.from_points(xa, need)
    nb_b = neighbors_b if neighbors_b is not None and neighbors_b.k >= need else Neighbors.from_points(xb, need)
    forward = _mean_rank_in_b(xb, xa, nb_a, k) / (n / 2.0)
    reverse = _mean_rank_in_b(xa, xb, nb_b, k) / (n / 2.0)
    return MetricResult(forward, {"reverse": reverse, "asymmetry": forward - reverse})


def _shared_neighbor_fraction(
    nn_a: Tensor,
    nn_b: Tensor,
    jaccard: bool = False,
    ties_a: dict[int, tuple[Tensor, Tensor]] | None = None,
    ties_b: dict[int, tuple[Tensor, Tensor]] | None = None,
) -> Tensor:
    """Per-point overlap of two (N, k) index sets with distinct entries per row: shared / k, or shared / union.

    For the rows in ties_a or ties_b (_boundary_ties), the shared count is its expectation when the items tied at
    the k-th distance are drawn at random, independently in the two spaces: the sum over items of the product of
    their inclusion probabilities.
    """
    both = torch.sort(torch.cat([nn_a, nn_b], dim=1), dim=1).values
    shared = (both[:, 1:] == both[:, :-1]).sum(dim=1).double()
    n, k = nn_a.shape
    tied_rows = set(ties_a or {}) | set(ties_b or {})
    for row in tied_rows:
        p = []
        for nn, ties in ((nn_a, ties_a or {}), (nn_b, ties_b or {})):
            v = torch.zeros(n, dtype=torch.float64, device=nn.device)
            if row in ties:
                cols, weights = ties[row]
                v[cols] = weights
            else:
                v[nn[row]] = 1.0
            p.append(v)
        shared[row] = (p[0] * p[1]).sum()
    return shared / ((2 * k - shared) if jaccard else k)


def _overlap_chance(n: int, k: int, jaccard: bool = False) -> float:
    """Expected overlap of two independent uniform k-subsets of N - 1 items: k/(N - 1), or for the Jaccard
    normalization the expectation of m/(2k - m) over the hypergeometric shared count m."""
    if not jaccard:
        return k / (n - 1)

    def log_binom(a: int, b: int) -> float:
        return math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)

    total = 0.0
    for m in range(max(0, 2 * k - (n - 1)), k + 1):
        log_p = log_binom(k, m) + log_binom(n - 1 - k, k - m) - log_binom(n - 1, k)
        total += math.exp(log_p) * m / (2 * k - m)
    return total


def neighborhood_overlap(
    x_a: Tensor,
    x_b: Tensor,
    *,
    k: int = 30,
    l2: bool = False,
    jaccard: bool = False,
    neighbors_a: Neighbors | None = None,
    neighbors_b: Neighbors | None = None,
) -> MetricResult:
    """Neighborhood overlap: mean fraction of the k nearest neighbors shared by two representations.

    Doimo, Glielmo, Ansuini and Laio (2020, NeurIPS, arXiv:2007.03506, Eq. 1): 1 when every point
    keeps its neighbors, k/(N-1) in expectation for unrelated spaces (Groger, Wen and Brbic, 2026,
    Prop. 4.2). Neighbors are Euclidean, the point itself excluded, and k = 30 as in Doimo et al.
    and Valeriani et al. (2023). The mutual k-nearest-neighbor alignment of Huh et al. (2024, ICML,
    App. A) is the same quantity on cosine neighbors with k = 10 on 1,024 image-caption pairs;
    l2=True ranks cosine neighbors. At fixed k the value falls as N grows and when an item has
    several valid partners (Koepke et al., 2026), so compare at equal N and k on one-to-one pairs.
    jaccard=True divides each point's shared count by the size of the union of its two neighbor
    sets instead of by k. This is the k-NN Jaccard similarity of the ReSi benchmark (Klabunde et
    al., 2025, ICLR, Eq. 24 and code), attributed there to Wang et al. (2022) and used with k = 10
    on cosine neighbors. Its chance level is the expectation of m/(2k - m) over the hypergeometric
    shared count m.

    When the k-th distance of a point is tied with further items (discrete data), its neighbor set
    is not unique; the shared count is then its expectation with the tied items drawn at random,
    independently in each space, which keeps the chance level of unrelated spaces (a fixed order
    of tied items, shared by both spaces, inflates it). This is exact for the overlap and a plug-in
    for jaccard=True. Without ties the value is the plain count.

    Args:
        x_a, x_b: (N, D_a) and (N, D_b), rows of the same items: two layers, checkpoints, models
            or modalities.
        k: Neighborhood size.
        l2: Scale rows to unit norm first, so Euclidean neighbors are cosine neighbors.
        jaccard: Normalize each point's shared count by the union size instead of k.
        neighbors_a, neighbors_b: Precomputed Neighbors tables of the (preprocessed) inputs; reused when they hold
            at least k + 1 neighbors (the extra one detects a tie at the k-th), rebuilt otherwise.

    Returns:
        value: mean overlap in [0, 1].
        extras: std of the per-point overlaps, chance level under independent neighbor sets.
    """
    if x_a.ndim != 2 or x_b.ndim != 2 or x_a.shape[0] != x_b.shape[0]:
        raise ValueError(f"expected two (N, D) tensors with equal N, got {tuple(x_a.shape)} and {tuple(x_b.shape)}")
    n = x_a.shape[0]
    if k < 1 or k > n - 1:
        raise ValueError(f"k must be in [1, N - 1], got {k} for N = {n}")
    xa, xb = (l2_normalize(x_a.double()), l2_normalize(x_b.double())) if l2 else (x_a.double(), x_b.double())
    need = min(k + 1, n - 1)  # one neighbor more than k, so a tie at the k-th is visible
    nb_a = neighbors_a if neighbors_a is not None and neighbors_a.k >= need else Neighbors.from_points(xa, need)
    nb_b = neighbors_b if neighbors_b is not None and neighbors_b.k >= need else Neighbors.from_points(xb, need)
    per_point = _shared_neighbor_fraction(
        nb_a.indices[:, 1 : k + 1],
        nb_b.indices[:, 1 : k + 1],
        jaccard,
        _boundary_ties(xa, nb_a, k),
        _boundary_ties(xb, nb_b, k),
    )
    return MetricResult(
        float(per_point.mean()),
        {"std": float(per_point.std()) if n > 1 else 0.0, "chance": _overlap_chance(n, k, jaccard)},
    )


def _cycle_fraction(nn_a: Tensor, nn_b: Tensor) -> Tensor:
    """Per-point indicator that i is among the A-neighbors of one of its B-neighbors, from (N, k) index sets."""
    back = nn_a[nn_b]  # (N, k, k): the A-neighbors of each B-neighbor of i
    hit = back == torch.arange(nn_a.shape[0], device=nn_a.device)[:, None, None]
    return hit.flatten(1).any(dim=1).double()


def cycle_knn(
    x_a: Tensor,
    x_b: Tensor,
    *,
    k: int = 10,
    l2: bool = False,
    neighbors_a: Neighbors | None = None,
    neighbors_b: Neighbors | None = None,
) -> MetricResult:
    """Cycle k-nearest-neighbor consistency from representation A to representation B.

    Huh et al. (2024, ICML, App. B, Fig. 11, and their code): the fraction of items whose k nearest
    neighbors in B have the item among their own k nearest neighbors in A, a first hop in B and a
    return hop in A. Zhang et al. (2026, Eq. 1) and Groger, Wen and Brbic (2026, Eq. 35) use the
    same definition. The score is not symmetric for k >= 2 (Zhang et al., App. A), so both
    orderings are returned. Identical representations score 1 only when every item is a nearest
    neighbor of one of its own nearest neighbors; items with no reciprocal neighbor lower the
    value. For independent representations each return hop succeeds with probability k/(N - 1)
    (Groger et al., Prop. C.9), so the chance level is at most k^2/(N - 1). Neighbors are
    Euclidean, the point itself excluded; these papers use k = 10 and cosine neighbors (l2=True).
    A tie at the k-th distance (discrete data) is resolved by the order of the neighbor search, as
    in those papers; information_imbalance and neighborhood_overlap average over the tied items
    instead.

    Args:
        x_a, x_b: (N, D_a) and (N, D_b), rows of the same items.
        k: Neighborhood size.
        l2: Scale rows to unit norm first, so Euclidean neighbors are cosine neighbors.
        neighbors_a, neighbors_b: Precomputed Neighbors tables with at least k neighbors.

    Returns:
        value: the fraction in [0, 1], A -> B.
        extras: reverse (B -> A), chance_bound k^2/(N - 1).
    """
    if x_a.ndim != 2 or x_b.ndim != 2 or x_a.shape[0] != x_b.shape[0]:
        raise ValueError(f"expected two (N, D) tensors with equal N, got {tuple(x_a.shape)} and {tuple(x_b.shape)}")
    n = x_a.shape[0]
    if k < 1 or k > n - 1:
        raise ValueError(f"k must be in [1, N - 1], got {k} for N = {n}")
    xa, xb = (l2_normalize(x_a.double()), l2_normalize(x_b.double())) if l2 else (x_a.double(), x_b.double())
    nb_a = neighbors_a if neighbors_a is not None else Neighbors.from_points(xa, k)
    nb_b = neighbors_b if neighbors_b is not None else Neighbors.from_points(xb, k)
    if nb_a.k < k or nb_b.k < k:
        raise ValueError("Neighbors tables need at least k neighbors")
    nn_a, nn_b = nb_a.indices[:, 1 : k + 1], nb_b.indices[:, 1 : k + 1]
    forward = float(_cycle_fraction(nn_a, nn_b).mean())
    reverse = float(_cycle_fraction(nn_b, nn_a).mean())
    return MetricResult(forward, {"reverse": reverse, "chance_bound": min(1.0, k * k / (n - 1))})


def _check_pair(x_a: Tensor, x_b: Tensor) -> None:
    if x_a.ndim != 2 or x_b.ndim != 2 or x_a.shape[0] != x_b.shape[0]:
        raise ValueError(f"expected two (N, D) tensors with equal N, got {tuple(x_a.shape)} and {tuple(x_b.shape)}")
    if not (torch.isfinite(x_a).all() and torch.isfinite(x_b).all()):
        raise ValueError("input contains non-finite values")


def _debiased_hsic(xty: Tensor, rows_a: Tensor, rows_b: Tensor, sq_a: Tensor, sq_b: Tensor, n: int) -> Tensor:
    """Unbiased linear HSIC up to the factor 1/(n(n-3)), from centered-feature summaries."""
    return xty - n / (n - 2.0) * (rows_a @ rows_b) + sq_a * sq_b / ((n - 1) * (n - 2))


def _cka_stats(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Column-centered float64 features, ||X^T X||_F^2 and the squared row norms."""
    xc = x.double() - x.double().mean(0, keepdim=True)
    return xc, (xc.T @ xc).square().sum(), xc.square().sum(1)


def _cka_from_stats(a: tuple[Tensor, Tensor, Tensor], b: tuple[Tensor, Tensor, Tensor]) -> tuple[float, float]:
    """(biased, debiased) linear CKA from two _cka_stats; debiased is nan for N < 4."""
    xa, xtx, rows_a = a
    xb, yty, rows_b = b
    xty = (xb.T @ xa).square().sum()
    biased = float(xty / (xtx.sqrt() * yty.sqrt()))
    n = xa.shape[0]
    if n < 4:
        return biased, float("nan")
    sq_a, sq_b = rows_a.sum(), rows_b.sum()
    num = _debiased_hsic(xty, rows_a, rows_b, sq_a, sq_b, n)
    den_a = _debiased_hsic(xtx, rows_a, rows_a, sq_a, sq_a, n)
    den_b = _debiased_hsic(yty, rows_b, rows_b, sq_b, sq_b, n)
    return biased, float(num / (den_a.sqrt() * den_b.sqrt()))


def _regress_out(k: Tensor, k0: Tensor) -> tuple[Tensor, float, float]:
    """K - alpha K0, alpha by least squares over all entries without intercept (Cui et al., Eq. 3-5); alpha and the
    share of K's variance explained, 1 - var(residual) / var(K)."""
    alpha = float((k0 * k).sum() / (k0 * k0).sum().clamp_min(1e-300))
    res = k - alpha * k0
    return res, alpha, float(1.0 - res.var() / k.var().clamp_min(1e-300))


def _deconfounded_kernel(x: Tensor, confound: Tensor) -> tuple[Tensor, dict[str, float]]:
    """Centered PSD linear kernel of x with the confound's linear kernel regressed out (Cui et al., Eq. 3-8).

    Columns are centered; the Frobenius scaling of the paper changes neither the residual's direction nor CKA,
    and neither does Eq. 8's factor rho^2, so the PSD part is Q Lambda_+ Q^T. Returns the kernel and alpha, the
    share of the kernel's variance the confound explains, and the share of eigenvalue mass removed.
    """
    xc = x.double() - x.double().mean(0, keepdim=True)
    zc = confound.double() - confound.double().mean(0, keepdim=True)
    res, alpha, explained = _regress_out(xc @ xc.T, zc @ zc.T)
    lam, q = torch.linalg.eigh((res + res.T) / 2.0)
    k = (q * lam.clamp_min(0.0)) @ q.T
    k = k - k.mean(0, keepdim=True) - k.mean(1, keepdim=True) + k.mean()
    dropped = float(lam.clamp_max(0.0).abs().sum() / lam.abs().sum().clamp_min(1e-300))
    return k, {"alpha": alpha, "explained": explained, "psd_dropped": dropped}


def _kernel_cka(k_a: Tensor, k_b: Tensor) -> float:
    """CKA of two centered kernels with the biased HSIC: <K_a, K_b>_F / (||K_a||_F ||K_b||_F)."""
    return float((k_a * k_b).sum() / ((k_a * k_a).sum() * (k_b * k_b).sum()).sqrt().clamp_min(1e-300))


def cka(
    x_a: Tensor,
    x_b: Tensor,
    *,
    debiased: bool = False,
    confound: Tensor | None = None,
    confound_b: Tensor | None = None,
) -> MetricResult:
    """Linear centered kernel alignment, optionally deconfounded.

    Kornblith, Norouzi, Lee and Hinton (2019, ICML, arXiv:1905.00414, Table 1): ||Y^T X||_F^2 /
    (||X^T X||_F ||Y^T Y||_F) for column-centered X and Y. It is invariant to orthogonal maps and
    isotropic scaling. The plug-in estimate is biased upward unless N is large relative to the
    widths; debiased=True uses the unbiased HSIC estimator, as in the authors' notebook, which
    can be negative and needs N >= 4. The value is computed from D x D cross-products in float64.

    With confound, the deconfounded CKA of Cui, Kumar, Marttinen and Kaski (2022, NeurIPS,
    arXiv:2202.00095, Eq. 3-11): the linear kernel of a representation of the same items (their
    input space; the previous layer in their recursive variant, App. D.2) is regressed out of each
    kernel by least squares over all N^2 entries, the residuals are projected onto the PSD cone and
    compared with the biased HSIC. It removes the similarity both sides inherit from the confound,
    which can make two random networks look alike (their Fig. 1: CKA 0.99, dCKA 0.43), and keeps
    the invariances of CKA (their Props. 3.1-3.2). The authors expect the same conclusions as CKA
    within one domain (App. G), the linear fit holds best for shallow layers (App. A), and where
    the input structure is itself of interest it should not be removed (Sec. 5). Costs O(N^2)
    memory and an O(N^3) eigendecomposition per representation.

    Args:
        x_a, x_b: (N, D_a) and (N, D_b).
        debiased: Return the debiased estimate (not with confound).
        confound: (N, D_0) representation whose similarity structure is removed from both kernels.
        confound_b: The confound of x_b when it differs from x_a's (defaults to confound).

    Returns:
        value: CKA, or the deconfounded CKA when confound is given.
        extras: biased, debiased (plain CKA; nan for N < 4); with confound also alpha_a, alpha_b (regression
            coefficients), explained_a, explained_b (share of each kernel's variance the confound explains) and
            psd_dropped_a, psd_dropped_b (share of eigenvalue mass removed by the projection).
    """
    _check_pair(x_a, x_b)
    if debiased and x_a.shape[0] < 4:
        raise ValueError(f"debiased CKA needs N >= 4, got N = {x_a.shape[0]}")
    biased, unbiased = _cka_from_stats(_cka_stats(x_a), _cka_stats(x_b))
    extras: dict[str, float] = {"biased": biased, "debiased": unbiased}
    if confound is None:
        if confound_b is not None:
            raise ValueError("confound_b needs confound")
        return MetricResult(unbiased if debiased else biased, extras)
    if debiased:
        raise ValueError("the deconfounded CKA uses the biased HSIC (Cui et al.); debiased is not defined for it")
    conf_b = confound if confound_b is None else confound_b
    _check_pair(x_a, confound)
    _check_pair(x_b, conf_b)
    k_a, info_a = _deconfounded_kernel(x_a, confound)
    k_b, info_b = _deconfounded_kernel(x_b, conf_b)
    extras.update({f"{key}_a": v for key, v in info_a.items()})
    extras.update({f"{key}_b": v for key, v in info_b.items()})
    return MetricResult(_kernel_cka(k_a, k_b), extras)


def _svd_directions(x: Tensor, threshold: float) -> tuple[Tensor, int]:
    """Left singular vectors of the centered x whose singular values sum to a fraction threshold of the total."""
    u, s, _ = torch.linalg.svd(x - x.mean(0, keepdim=True), full_matrices=False)
    cum = torch.cumsum(s, 0)
    if float(cum[-1]) == 0.0:
        raise ValueError("zero matrix after centering: no singular directions")
    k = min(int(torch.searchsorted(cum, threshold * cum[-1]).item()) + 1, s.numel())
    return u[:, :k], k


def svcca(x_a: Tensor, x_b: Tensor, *, threshold: float = 0.99) -> MetricResult:
    """SVCCA: mean canonical correlation between the leading SVD directions of two representations.

    Raghu, Gilmer, Yosinski and Sohl-Dickstein (2017, NeurIPS, arXiv:1706.05806, Eq. 1, App. A):
    each centered representation keeps the fewest directions whose singular values sum to
    threshold of the total, and the value is the mean of the min(k_a, k_b) canonical
    correlations between them, computed as the singular values of U_a^T U_b. Invariant to
    invertible linear maps of the kept subspaces, so N must be well above the kept widths; as
    they approach N, any two representations score near 1.

    Args:
        x_a, x_b: (N, D_a) and (N, D_b).
        threshold: Fraction of the summed singular values kept.

    Returns:
        value: mean canonical correlation in [0, 1].
        extras: r2 (mean squared correlation), k_a, k_b.
    """
    _check_pair(x_a, x_b)
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    u_a, k_a = _svd_directions(x_a.double(), threshold)
    u_b, k_b = _svd_directions(x_b.double(), threshold)
    rho = torch.linalg.svdvals(u_a.T @ u_b).clamp(0.0, 1.0)
    return MetricResult(float(rho.mean()), {"r2": float(rho.square().mean()), "k_a": float(k_a), "k_b": float(k_b)})


def _folds(n: int, n_folds: int, seed: int) -> list[tuple[Tensor, Tensor]]:
    """Seeded K-fold split of n rows: (training rows, held-out rows) per fold."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    parts = list(perm.tensor_split(n_folds))  # exactly n_folds parts, sizes differing by at most one
    return [(torch.cat([q for j, q in enumerate(parts) if j != i]), parts[i]) for i in range(len(parts))]


def _ridge_fit(x_train: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Column mean and thin SVD (U, S, V) of the centered training rows of a predictor, in float64."""
    mean = x_train.double().mean(0, keepdim=True)
    u, s, vh = torch.linalg.svd(x_train.double() - mean, full_matrices=False)
    return mean, u, s, vh.T


def _ridge_loo_error(u: Tensor, s: Tensor, y: Tensor, z: Tensor, lam: float) -> Tensor:
    """Mean squared leave-one-out error of ridge regression with an intercept, per target column, in closed form.

    The predictor and the target y are centered on the n training rows and z = U^T y. With the hat matrix
    H = 11^T / n + U diag(s^2 / (s^2 + lambda)) U^T, the left-out residual of row i is (y_i - (H y)_i) / (1 - H_ii);
    the intercept adds nothing to H y because y is centered, and 1 / n to every H_ii.
    """
    d = s.square() / (s.square() + lam)
    fitted = u @ (d[:, None] * z)
    leverage = (u.square() * d).sum(1) + 1.0 / u.shape[0]
    return ((y - fitted) / (1.0 - leverage).clamp_min(1e-12)[:, None]).square().mean(0)


def _ridge_direction(
    fits: list[tuple[Tensor, Tensor, Tensor, Tensor]],
    x_pred: Tensor,
    x_target: Tensor,
    folds: list[tuple[Tensor, Tensor]],
    grid: Tensor,
) -> tuple[Tensor, Tensor]:
    """Held-out r^2 per target column, averaged over folds, of ridge fits from the predictor (one fit per fold);
    lambda per column minimizes the leave-one-out error on the training rows over grid x the largest squared
    singular value. Returns (r^2 per column, chosen lambdas per fold and column)."""
    scores, lambdas = [], []
    for (mean_a, u, s, v), (tr, te) in zip(fits, folds, strict=True):
        y_mean = x_target[tr].double().mean(0, keepdim=True)
        yt = x_target[tr].double() - y_mean
        z = u.T @ yt
        candidates = grid * float(s.square().max())
        errors = torch.stack([_ridge_loo_error(u, s, yt, z, float(lam)) for lam in candidates])  # (grid, D_b)
        best = candidates[errors.argmin(0)]
        proj = (x_pred[te].double() - mean_a) @ v
        pred = torch.empty((te.numel(), yt.shape[1]), dtype=torch.float64, device=yt.device)
        for lam in best.unique():
            cols = torch.nonzero(best == lam).flatten()
            pred[:, cols] = proj @ ((s / (s.square() + lam))[:, None] * z[:, cols])
        obs = x_target[te].double() - y_mean
        pc, pp = obs - obs.mean(0, keepdim=True), pred - pred.mean(0, keepdim=True)
        denom = pc.norm(dim=0) * pp.norm(dim=0)
        r = torch.where(denom > 0, (pc * pp).sum(0) / denom.clamp_min(1e-300), torch.zeros_like(denom))
        scores.append(r.clamp_min(0.0).square())
        lambdas.append(best)
    return torch.stack(scores).mean(0), torch.stack(lambdas)


def ridge_predictivity(
    x_a: Tensor,
    x_b: Tensor,
    *,
    n_folds: int = 5,
    seed: int = 0,
    lambda_grid: tuple[float, float, int] = (1e-8, 1e4, 25),
) -> MetricResult:
    """Held-out variance of x_b's columns predicted by ridge regression from x_a.

    The encoding-model protocol of Tuckute, Feather, Boebinger and McDermott (2023, PLOS Biology,
    doi:10.1371/journal.pbio.3002366, Methods) and Pepino, Riera, Kamienkowski and Ferrer (2025,
    arXiv:2511.16849, Sec. 4.3). The items are split into folds. On each training fold the
    predictor's columns are demeaned but not scaled (their non-isotropic prior), and the ridge
    penalty of each target column is chosen by leave-one-out cross-validation within the training
    rows. The held-out predictions are scored by the squared Pearson correlation with the target
    column, with negative correlations set to 0, and the folds are averaged; Tuckute et al. take the
    median over ten random 83/82 splits instead. Regression finds the subspace of x_a that predicts
    x_b and ignores the rest, where RSA and CKA weigh every direction (Tuckute et al., Results;
    Pepino et al., Sec. 2.1.2). The reliability (noise-ceiling) correction of Tuckute et al. is not
    applied. The penalty grid is relative to the data: lambda_grid gives (low, high, count)
    multiples of the largest squared singular value of the training predictor, log-spaced. Each fold
    needs one SVD of the training predictor, in float64; the reverse direction is computed the same
    way.

    Args:
        x_a: (N, D_a), the predictor.
        x_b: (N, D_b), the representation predicted, scored column by column.
        n_folds: Seeded K-fold split of the items.
        seed: Seed of the split.
        lambda_grid: (low, high, count) of the relative penalty grid.

    Returns:
        value: mean over x_b's columns of the held-out r^2.
        extras: median (over columns), reverse and reverse_median (x_a predicted from x_b), n_folds,
            log10_lambda_median and reverse_log10_lambda_median (median log10 of the chosen penalties, per
            direction).
    """
    _check_pair(x_a, x_b)
    n = x_a.shape[0]
    if not 2 <= n_folds <= n // 2:
        raise ValueError(f"n_folds must be between 2 and N/2, got {n_folds} for N = {n}")
    lo, hi, count = lambda_grid
    if not (0 < lo < hi and count >= 1):
        raise ValueError(f"lambda_grid must be (low, high, count) with 0 < low < high, got {lambda_grid}")
    folds = _folds(n, n_folds, seed)
    grid = torch.logspace(math.log10(lo), math.log10(hi), int(count), dtype=torch.float64, device=x_a.device)
    fits_a = [_ridge_fit(x_a[tr]) for tr, _ in folds]
    fits_b = [_ridge_fit(x_b[tr]) for tr, _ in folds]
    forward, lambdas = _ridge_direction(fits_a, x_a, x_b, folds, grid)
    reverse, reverse_lambdas = _ridge_direction(fits_b, x_b, x_a, folds, grid)
    extras = {
        "median": float(forward.median()),
        "reverse": float(reverse.mean()),
        "reverse_median": float(reverse.median()),
        "n_folds": float(len(folds)),
        "log10_lambda_median": float(torch.log10(lambdas).median()),
        "reverse_log10_lambda_median": float(torch.log10(reverse_lambdas).median()),
    }
    return MetricResult(float(forward.mean()), extras)


def _procrustes_factors(x: Tensor, rank_tol: float) -> tuple[Tensor, int]:
    """U S of the thin SVD of the column-centered, Frobenius-normalized x in float64, singular values below rank_tol
    times the largest dropped, and the number kept."""
    centered = (x - x.mean(0, keepdim=True)).double()
    norm = float(centered.norm())
    if norm == 0.0:
        raise ValueError("zero matrix after centering: no Procrustes distance")
    u, s, _ = torch.linalg.svd(centered / norm, full_matrices=False)
    keep = s > rank_tol * s[0]
    return u[:, keep] * s[keep], int(keep.sum())


def _procrustes_from_factors(f_a: Tensor, f_b: Tensor) -> tuple[float, float]:
    """(squared distance, nuclear norm): the singular values of x_a^T x_b are those of (U_a S_a)^T (U_b S_b)."""
    nuclear = float(torch.linalg.svdvals(f_a.T @ f_b).sum())
    return max(2.0 - 2.0 * nuclear, 0.0), nuclear


def procrustes(x_a: Tensor, x_b: Tensor, *, rank_tol: float = 1e-10) -> MetricResult:
    """Squared orthogonal Procrustes distance between two representations, in Ding et al.'s normalization.

    Ding, Denain and Steinhardt (2021, NeurIPS, arXiv:2108.01661, Sec. 2, Eqs. 6-7): the columns of each
    representation are centered and the matrix divided by its Frobenius norm; the squared distance
    min_R ||x_b - x_a R||_F^2 over orthogonal R is ||x_a||_F^2 + ||x_b||_F^2 - 2 ||x_a^T x_b||_*, here
    2 - 2 ||x_a^T x_b||_* in [0, 2] with the nuclear norm, which unequal widths do not change (ReSi zero-pads
    the narrower side to the same effect). With shared singular vectors it is the squared distance between
    the singular values, where CKA is a quartic dominated by the largest ones (their Discussion). The
    distance is symmetric and invariant to orthogonal maps and isotropic scaling of either side, but not to
    invertible maps (CCA is) or to anisotropic scaling. Across models of different widths, the convergence it
    reports is a width effect that permutation calibration removes (Groger et al., 2026; `calibrate`). It
    costs one thin SVD per side, in float64.

    Args:
        x_a: (N, D_a).
        x_b: (N, D_b).
        rank_tol: Singular values below rank_tol times the largest are dropped on each side.

    Returns:
        value: the squared distance, 0 for representations equal up to an orthogonal map and a scale, at most 2;
            lower is closer.
        extras: alignment (||x_a^T x_b||_* in [0, 1], 1 minus half the value), distance (the square root),
            rank_a, rank_b.
    """
    _check_pair(x_a, x_b)
    f_a, r_a = _procrustes_factors(x_a, rank_tol)
    f_b, r_b = _procrustes_factors(x_b, rank_tol)
    value, nuclear = _procrustes_from_factors(f_a, f_b)
    extras = {"alignment": nuclear, "distance": math.sqrt(value), "rank_a": float(r_a), "rank_b": float(r_b)}
    return MetricResult(value, extras)


def _condensed_distances(x: Tensor, distance: str, chunk: int) -> Tensor:
    """Pairwise distances of the rows for i < j, row-major, built in row chunks."""
    n = x.shape[0]
    if distance == "correlation":
        x = x - x.mean(dim=1, keepdim=True)
    if distance in ("cosine", "correlation"):
        x = l2_normalize(x)
    elif distance != "euclidean":
        raise ValueError(f"distance must be cosine, euclidean or correlation, got {distance!r}")
    cols = torch.arange(n, device=x.device)
    parts = []
    for start in range(0, n - 1, chunk):
        rows = torch.arange(start, min(start + chunk, n - 1), device=x.device)
        d = (
            torch.cdist(x[rows], x, compute_mode=_cdist_mode(x.dtype))
            if distance == "euclidean"
            else 1.0 - x[rows] @ x.T
        )
        parts.append(d[cols[None, :] > rows[:, None]])
    return torch.cat(parts)


def _average_ranks(v: Tensor) -> Tensor:
    """1-based ranks of a vector; tied values share their mean rank."""
    order = torch.argsort(v)
    _, inverse, counts = torch.unique_consecutive(v[order], return_inverse=True, return_counts=True)
    ends = torch.cumsum(counts, 0).double()
    ranks = torch.empty_like(v, dtype=torch.float64)
    ranks[order] = (ends - (counts.double() - 1) / 2)[inverse]
    return ranks


def _standardized(v: Tensor, method: str) -> Tensor:
    """Ranked for Spearman, then zero-mean and unit-norm, so the correlation of two is their dot product."""
    if method == "spearman":
        v = _average_ranks(v)
    elif method != "pearson":
        raise ValueError(f"method must be spearman or pearson, got {method!r}")
    v = v - v.mean()
    return v / v.norm().clamp_min(1e-300)


def _rsa_vector(x: Tensor, distance: str, method: str, chunk: int) -> Tensor:
    """Standardized distance vector; RSA is the dot product of two."""
    return _standardized(_condensed_distances(x.double(), distance, chunk), method)


def _drsa_vector(x: Tensor, confound: Tensor, method: str, chunk: int) -> tuple[Tensor, float]:
    """Standardized residual of x's squared Euclidean distances after regressing out the confound's (Cui et al.,
    Eq. 3-5 and 12), and alpha. The regression over the pairs i < j equals theirs over all N^2 entries (the diagonal
    is zero and the matrix symmetric), and their Frobenius scaling changes neither the residual's direction nor a
    correlation."""
    d = _condensed_distances(x.double(), "euclidean", chunk).square()
    d0 = _condensed_distances(confound.double(), "euclidean", chunk).square()
    res, alpha, _ = _regress_out(d, d0)
    return _standardized(res, method), alpha


def rsa(
    x_a: Tensor,
    x_b: Tensor,
    *,
    distance: str = "cosine",
    method: str = "spearman",
    chunk: int = 1024,
    confound: Tensor | None = None,
    confound_b: Tensor | None = None,
) -> MetricResult:
    """Representational similarity analysis: correlation of the pairwise-distance vectors of two representations.

    Kriegeskorte, Mur and Bandettini (2008, Frontiers in Systems Neuroscience): each representation
    gives the distances between all pairs of items, and the two vectors (i < j) are compared by
    Spearman rank correlation; 1 for the same geometry, 0 in expectation for unrelated ones. The
    default distance is "cosine", as in Koepke et al. (2026, arXiv:2604.18572); "euclidean" and
    "correlation" (one minus the Pearson correlation across features, the choice of Kriegeskorte et
    al.) are the alternatives. RSA reads the full ordering of pairs, whereas CKA weights the leading
    directions. Each representation gives N(N - 1)/2 distances, built in row chunks.

    With confound, the deconfounded RSA of Cui, Kumar, Marttinen and Kaski (2022, NeurIPS,
    arXiv:2202.00095, Eq. 12): the squared Euclidean distances of a representation of the same items
    are regressed out of each representation's squared Euclidean distances (least squares, no
    intercept) and the residuals are correlated. As in the paper, this needs distance="euclidean".
    See cka for when deconfounding applies.

    Args:
        x_a, x_b: (N, D_a) and (N, D_b), rows of the same items.
        distance: "cosine", "euclidean" or "correlation".
        method: "spearman" (ranks) or "pearson" (raw distances).
        chunk: Rows per distance block.
        confound: (N, D_0) representation whose distance structure is removed from both.
        confound_b: The confound of x_b when it differs from x_a's (defaults to confound).

    Returns:
        value: the correlation, or the deconfounded correlation when confound is given.
        extras: n_pairs; with confound also rsa (the plain correlation), alpha_a and alpha_b.
    """
    _check_pair(x_a, x_b)
    if confound is None:
        if confound_b is not None:
            raise ValueError("confound_b needs confound")
        a, b = _rsa_vector(x_a, distance, method, chunk), _rsa_vector(x_b, distance, method, chunk)
        return MetricResult(float(torch.dot(a, b)), {"n_pairs": float(a.numel())})
    if distance != "euclidean":
        raise ValueError(
            "the deconfounded RSA regresses squared Euclidean distances (Cui et al.); pass distance='euclidean'"
        )
    conf_b = confound if confound_b is None else confound_b
    _check_pair(x_a, confound)
    _check_pair(x_b, conf_b)
    a, alpha_a = _drsa_vector(x_a, confound, method, chunk)
    b, alpha_b = _drsa_vector(x_b, conf_b, method, chunk)
    plain = float(torch.dot(_rsa_vector(x_a, distance, method, chunk), _rsa_vector(x_b, distance, method, chunk)))
    extras = {"n_pairs": float(a.numel()), "rsa": plain, "alpha_a": alpha_a, "alpha_b": alpha_b}
    return MetricResult(float(torch.dot(a, b)), extras)


register_metric(
    "rsa",
    inputs=InputKind.PAIR,
    preprocess=Preprocess(),
    citation=("kriegeskorte2008rsa", "koepke2026cave", "cui2022deconfounded"),
    max_items=4000,
    description="Spearman correlation of the pairwise-distance vectors of two representations (Kriegeskorte et al., 2008).",
)(rsa)
register_metric(
    "cka",
    inputs=InputKind.PAIR,
    preprocess=Preprocess(),
    citation=("kornblith2019similarity", "cui2022deconfounded"),
    arxiv="1905.00414",
    description="Linear centered kernel alignment between two representations of the same items (Kornblith et al., 2019).",
)(cka)
register_metric(
    "svcca",
    inputs=InputKind.PAIR,
    preprocess=Preprocess(),
    citation=("raghu2017svcca",),
    arxiv="1706.05806",
    description="Mean canonical correlation of the leading SVD directions of two representations (Raghu et al., 2017).",
)(svcca)
register_metric(
    "ridge_predictivity",
    inputs=InputKind.PAIR,
    preprocess=Preprocess(),
    citation=("tuckute2023many", "pepino2025brainlike"),
    max_items=4000,
    description=(
        "Held-out variance of one representation's columns predicted by ridge regression from the other "
        "(Tuckute et al., 2023); the ordering matters."
    ),
)(ridge_predictivity)
register_metric(
    "procrustes",
    inputs=InputKind.PAIR,
    preprocess=Preprocess(),
    citation=("ding2021grounding", "klabunde2025resi"),
    arxiv="2108.01661",
    max_items=4000,
    description=(
        "Squared orthogonal Procrustes distance between centered, Frobenius-normalized representations "
        "(Ding et al., 2021); lower is closer."
    ),
)(procrustes)
register_metric(
    "neighborhood_overlap",
    inputs=InputKind.PAIR,
    preprocess=Preprocess(),
    citation=(
        "doimo2020nucleation",
        "valeriani2023geometry",
        "huh2024platonic",
        "glielmo2022dadapy",
        "wang2022instability",
        "klabunde2025resi",
    ),
    arxiv="2007.03506",
    tags=("relational",),
)(neighborhood_overlap)
register_metric(
    "cycle_knn",
    inputs=InputKind.PAIR,
    preprocess=Preprocess(),
    citation=("huh2024platonic", "zhang2026wittgensteinian", "groger2026aristotelian"),
    arxiv="2405.07987",
    tags=("relational",),
    description=(
        "Fraction of items that are a nearest neighbor in A of one of their nearest neighbors in B "
        "(Huh et al., 2024); the ordering matters."
    ),
)(cycle_knn)
register_metric(
    "information_imbalance",
    inputs=InputKind.PAIR,
    preprocess=Preprocess(),
    citation=(
        "glielmo2022imbalance",
        "cheng2025emergence",
        "acevedo2025semantic",
        "glielmo2022dadapy",
        "deltatto2024robust",
    ),
    arxiv="2104.15079",
)(information_imbalance)
