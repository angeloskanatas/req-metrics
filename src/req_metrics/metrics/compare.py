"""Metrics that compare two representations of the same items.

Rows of the two inputs must describe the same items in the same order; the
representations may have different widths (different layers, models or
feature subsets).
"""

from __future__ import annotations

import torch
from torch import Tensor

from req_metrics._types import InputKind, MetricResult, Preprocess
from req_metrics.neighbors import Neighbors, _cdist_mode
from req_metrics.registry import register_metric


def _mean_rank_in_b(x_b: Tensor, nn_a: Tensor, chunk: int = 512) -> float:
    """Mean over points of the rank, in space B, of each point's neighbors chosen in space A.

    The rank of item j for point i is one plus the number of other items strictly
    closer to i than j in space B, so exact ties count as the lower rank. Distances
    are computed per chunk of rows and never stored as an N x N table.
    """
    n = x_b.shape[0]
    total = 0.0
    for start in range(0, n, chunk):
        rows = torch.arange(start, min(start + chunk, n), device=x_b.device)
        d = torch.cdist(x_b[rows], x_b, compute_mode=_cdist_mode(x_b.dtype))  # (c, N)
        d[torch.arange(len(rows)), rows] = float("inf")  # exclude the point itself
        target = torch.gather(d, 1, nn_a[rows])  # (c, k) distances to A's neighbors
        ranks = (d.unsqueeze(1) < target.unsqueeze(2)).sum(dim=2) + 1  # (c, k)
        total += float(ranks.double().sum())
    return total / (n * nn_a.shape[1])


def information_imbalance(
    x_a: Tensor, x_b: Tensor, *, k: int = 1, neighbors_a: Neighbors | None = None, neighbors_b: Neighbors | None = None
) -> MetricResult:
    """Information imbalance from representation A to representation B.

    Glielmo et al. (2022, PNAS Nexus, arXiv:2104.15079), Eq. 2:
    Delta(A -> B) = 2 <r_B | r_A = 1> / N, the mean rank in space B of each
    point's nearest neighbor in space A, scaled so that identical spaces give
    about 2/N and independent spaces about 1. Asymmetric: a small Delta(A -> B)
    with a large Delta(B -> A) means A contains the information in B and more.
    The k-neighbor generalization averages the ranks of the k nearest
    A-neighbors, as in DADApy's implementation (Glielmo et al., 2022,
    Patterns); k = 1 is the original definition. Ranks are exact and computed by
    counting, so no N x N index table is stored; DADApy instead looks neighbors
    up in a truncated table and draws a random rank for items beyond it.

    Args:
        x_a: Representation A, shape (N, D_a).
        x_b: Representation B, shape (N, D_b), same items in the same order.
        k: Number of nearest A-neighbors whose B-ranks are averaged.
        neighbors_a: Precomputed Neighbors of A with at least k neighbors.
        neighbors_b: Precomputed Neighbors of B with at least k neighbors (for the reverse direction).

    Returns:
        value: Delta(A -> B).
        extras: reverse (Delta(B -> A)).
    """
    if x_a.ndim != 2 or x_b.ndim != 2 or x_a.shape[0] != x_b.shape[0]:
        raise ValueError(f"expected two (N, D) tensors with equal N, got {tuple(x_a.shape)} and {tuple(x_b.shape)}")
    xa, xb = x_a.double(), x_b.double()
    nb_a = neighbors_a if neighbors_a is not None else Neighbors.from_points(xa, k)
    nb_b = neighbors_b if neighbors_b is not None else Neighbors.from_points(xb, k)
    if nb_a.k < k or nb_b.k < k:
        raise ValueError("Neighbors tables need at least k neighbors")
    n = xa.shape[0]
    forward = _mean_rank_in_b(xb, nb_a.indices[:, 1 : k + 1]) / (n / 2.0)
    reverse = _mean_rank_in_b(xa, nb_b.indices[:, 1 : k + 1]) / (n / 2.0)
    return MetricResult(forward, {"reverse": reverse})


def _shared_neighbor_fraction(nn_a: Tensor, nn_b: Tensor) -> Tensor:
    """Per-point fraction of shared entries between two (N, k) index sets with distinct entries per row."""
    both = torch.sort(torch.cat([nn_a, nn_b], dim=1), dim=1).values
    shared = (both[:, 1:] == both[:, :-1]).sum(dim=1)
    return shared.double() / nn_a.shape[1]


def neighborhood_overlap(
    x_a: Tensor, x_b: Tensor, *, k: int = 30, neighbors_a: Neighbors | None = None, neighbors_b: Neighbors | None = None
) -> MetricResult:
    """Neighborhood overlap: mean fraction of k nearest neighbors shared by two representations of the same items.

    Doimo, Glielmo, Ansuini and Laio (2020, NeurIPS, arXiv:2007.03506, Eq. 1):
    chi_k(A, B) = (1/N) sum_i (1/k) sum_j A_ij B_ij for the k-nearest-neighbor
    adjacency matrices of the two spaces, 1 when every point keeps its
    neighbors and k/(N-1) in expectation for unrelated spaces. Between
    consecutive layers it measures how much of the local neighbor structure a
    layer rewires; Valeriani et al. (2023, NeurIPS, arXiv:2302.00294) use it to
    locate the layers where transformers reorganize representations. Both
    works use k = 30 for networks on ImageNet-scale data and report the trend
    robust to k (Doimo App. A.2, Valeriani Fig. S5). Symmetric, label-free,
    Euclidean neighbors, the point itself excluded; same convention as
    DADApy's return_data_overlap. Complements the information imbalance, which
    asks the directional question whether A's neighbors are B's near ranks.

    Args:
        x_a, x_b: (N, D_a) and (N, D_b), rows describing the same items.
        k: Neighborhood size.
        neighbors_a, neighbors_b: Precomputed Neighbors tables with at least k neighbors.

    Returns:
        value: mean overlap in [0, 1].
        extras: std of the per-point overlaps, chance level k / (N - 1).
    """
    if x_a.ndim != 2 or x_b.ndim != 2 or x_a.shape[0] != x_b.shape[0]:
        raise ValueError(f"expected two (N, D) tensors with equal N, got {tuple(x_a.shape)} and {tuple(x_b.shape)}")
    n = x_a.shape[0]
    if k < 1 or k > n - 1:
        raise ValueError(f"k must be in [1, N - 1], got {k} for N = {n}")
    nb_a = neighbors_a if neighbors_a is not None else Neighbors.from_points(x_a.double(), k)
    nb_b = neighbors_b if neighbors_b is not None else Neighbors.from_points(x_b.double(), k)
    if nb_a.k < k or nb_b.k < k:
        raise ValueError("Neighbors tables need at least k neighbors")
    per_point = _shared_neighbor_fraction(nb_a.indices[:, 1 : k + 1], nb_b.indices[:, 1 : k + 1])
    return MetricResult(
        float(per_point.mean()), {"std": float(per_point.std()) if n > 1 else 0.0, "chance": k / (n - 1)}
    )


def _check_pair(x_a: Tensor, x_b: Tensor) -> None:
    if x_a.ndim != 2 or x_b.ndim != 2 or x_a.shape[0] != x_b.shape[0]:
        raise ValueError(f"expected two (N, D) tensors with equal N, got {tuple(x_a.shape)} and {tuple(x_b.shape)}")


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


def cka(x_a: Tensor, x_b: Tensor, *, debiased: bool = False) -> MetricResult:
    """Linear centered kernel alignment between two representations of the same items.

    Kornblith, Norouzi, Lee and Hinton (2019, ICML, arXiv:1905.00414), Table 1:
    CKA = ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F) for column-centered X (N, D_a)
    and Y (N, D_b), the normalized HSIC of Eq. 4 with linear kernels. 1 for
    representations equal up to an orthogonal map and an isotropic scaling; it is
    not invariant to arbitrary invertible linear maps, which is what lets it
    distinguish layers wider than N. The plug-in estimate is biased upward when N
    is not large relative to the widths. debiased=True uses the unbiased HSIC
    estimator of Song et al. (2007) in the feature-space form of the authors'
    reference notebook; it reduces the bias, can be negative and needs N >= 4. Both
    estimates are in the extras. Computed in float64 from the D x D cross-products
    in O(N D_a D_b), without N x N Gram matrices.

    Args:
        x_a, x_b: (N, D_a) and (N, D_b), rows describing the same items.
        debiased: Return the debiased estimate.

    Returns:
        value: CKA in [0, 1] (debiased: can fall slightly below 0).
        extras: biased, debiased (nan for N < 4).
    """
    _check_pair(x_a, x_b)
    if debiased and x_a.shape[0] < 4:
        raise ValueError(f"debiased CKA needs N >= 4, got N = {x_a.shape[0]}")
    biased, unbiased = _cka_from_stats(_cka_stats(x_a), _cka_stats(x_b))
    return MetricResult(unbiased if debiased else biased, {"biased": biased, "debiased": unbiased})


def _svd_directions(x: Tensor, threshold: float) -> tuple[Tensor, int]:
    """Left singular vectors of the centered x whose singular values sum to a fraction threshold of the total."""
    u, s, _ = torch.linalg.svd(x - x.mean(0, keepdim=True), full_matrices=False)
    cum = torch.cumsum(s, 0)
    k = min(int(torch.searchsorted(cum, threshold * cum[-1]).item()) + 1, s.numel())
    return u[:, :k], k


def svcca(x_a: Tensor, x_b: Tensor, *, threshold: float = 0.99) -> MetricResult:
    """SVCCA similarity: mean canonical correlation between the leading SVD directions of two representations.

    Raghu, Gilmer, Yosinski and Sohl-Dickstein (2017, NeurIPS, arXiv:1706.05806):
    each representation is centered and reduced by SVD to the fewest directions
    whose singular values sum to at least threshold of their total (App. A, the
    99% rule on singular values), canonical correlation analysis between the two
    reduced representations gives min(k_a, k_b) correlations, and their mean is the
    similarity (Eq. 1, averaged over the aligned directions as in the reference
    tutorial and in Kornblith et al., 2019, Table 1). The correlations are the
    singular values of U_a^T U_b for the orthonormal bases of the kept directions,
    which equals the covariance-based CCA of the reference code at epsilon = 0
    without inverting covariance matrices. Invariant to invertible linear maps of
    the kept subspaces, so it needs N well above the kept widths: the reference
    tutorial asks for 5 to 10 times as many items as neurons, and as a kept width
    approaches N any two representations score near 1 (Kornblith et al., 2019,
    Theorem 1).

    Args:
        x_a, x_b: (N, D_a) and (N, D_b), rows describing the same items.
        threshold: Fraction of the summed singular values kept per representation.

    Returns:
        value: mean canonical correlation in [0, 1].
        extras: r2 (mean squared correlation), k_a, k_b (directions kept).
    """
    _check_pair(x_a, x_b)
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    u_a, k_a = _svd_directions(x_a.double(), threshold)
    u_b, k_b = _svd_directions(x_b.double(), threshold)
    rho = torch.linalg.svdvals(u_a.T @ u_b).clamp(0.0, 1.0)
    return MetricResult(float(rho.mean()), {"r2": float(rho.square().mean()), "k_a": float(k_a), "k_b": float(k_b)})


register_metric(
    "cka",
    inputs=InputKind.PAIR,
    preprocess=Preprocess(),
    citation=("kornblith2019similarity",),
    arxiv="1905.00414",
    description="Linear centered kernel alignment between two layers (Kornblith et al., 2019).",
)(cka)
register_metric(
    "svcca",
    inputs=InputKind.PAIR,
    preprocess=Preprocess(),
    citation=("raghu2017svcca",),
    arxiv="1706.05806",
    description="Mean canonical correlation of the leading SVD directions of two layers (Raghu et al., 2017).",
)(svcca)
register_metric(
    "neighborhood_overlap",
    inputs=InputKind.PAIR,
    preprocess=Preprocess(),
    citation=("doimo2020nucleation", "valeriani2023geometry", "glielmo2022dadapy"),
    arxiv="2007.03506",
    tags=("relational",),
)(neighborhood_overlap)
register_metric(
    "information_imbalance",
    inputs=InputKind.PAIR,
    preprocess=Preprocess(),
    citation=("glielmo2022imbalance", "glielmo2022dadapy"),
    arxiv="2104.15079",
)(information_imbalance)
