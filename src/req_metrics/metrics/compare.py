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
    / N, the mean rank in B of each point's nearest neighbor in A; about 2/N for identical spaces
    and 1 for independent ones. A small Delta(A -> B) with a large Delta(B -> A) means A contains
    the information in B. k > 1 averages the ranks of the k nearest A-neighbors, as in DADApy.
    Ranks are counted exactly, without an N x N table. Used between the layers of one model and
    between models by Cheng et al. (2025, ICLR), and between languages, images and image-caption
    pairs by Acevedo et al. (2025), who binarize activations and rank Hamming distances: pass
    torch.sign(x), whose Euclidean ranks are the Hamming ranks.

    Args:
        x_a: Representation A, (N, D_a).
        x_b: Representation B, (N, D_b), rows of the same items as x_a.
        k: Nearest A-neighbors whose B-ranks are averaged.
        l2: Scale rows to unit norm first, so neighbors and ranks are cosine ones.
        neighbors_a, neighbors_b: Precomputed Neighbors tables with at least k neighbors.

    Returns:
        value: Delta(A -> B).
        extras: reverse, Delta(B -> A).
    """
    if x_a.ndim != 2 or x_b.ndim != 2 or x_a.shape[0] != x_b.shape[0]:
        raise ValueError(f"expected two (N, D) tensors with equal N, got {tuple(x_a.shape)} and {tuple(x_b.shape)}")
    xa, xb = (l2_normalize(x_a.double()), l2_normalize(x_b.double())) if l2 else (x_a.double(), x_b.double())
    nb_a = neighbors_a if neighbors_a is not None else Neighbors.from_points(xa, k)
    nb_b = neighbors_b if neighbors_b is not None else Neighbors.from_points(xb, k)
    if nb_a.k < k or nb_b.k < k:
        raise ValueError("Neighbors tables need at least k neighbors")
    n = xa.shape[0]
    forward = _mean_rank_in_b(xb, nb_a.indices[:, 1 : k + 1]) / (n / 2.0)
    reverse = _mean_rank_in_b(xa, nb_b.indices[:, 1 : k + 1]) / (n / 2.0)
    return MetricResult(forward, {"reverse": reverse})


def _shared_neighbor_fraction(nn_a: Tensor, nn_b: Tensor, jaccard: bool = False) -> Tensor:
    """Per-point overlap of two (N, k) index sets with distinct entries per row: shared / k, or shared / union."""
    both = torch.sort(torch.cat([nn_a, nn_b], dim=1), dim=1).values
    shared = (both[:, 1:] == both[:, :-1]).sum(dim=1)
    k = nn_a.shape[1]
    return shared.double() / ((2 * k - shared).double() if jaccard else k)


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
    Prop. 4.2). Euclidean neighbors, the point itself excluded; k = 30 as in Doimo et al. and
    Valeriani et al. (2023). The mutual k-nearest-neighbor alignment of Huh et al. (2024, ICML,
    App. A) is the same quantity on cosine neighbors with k = 10 on 1,024 image-caption pairs;
    l2=True ranks cosine neighbors. At fixed k the value falls as N grows and when an item has
    several valid partners (Koepke et al., 2026), so compare at equal N and k on one-to-one pairs.
    jaccard=True divides each point's shared count by the size of the union of its two neighbor
    sets instead of by k: the k-NN Jaccard similarity of the ReSi benchmark (Klabunde et al.,
    2025, ICLR, Eq. 24 and code), attributed there to Wang et al. (2022) and ranked first in
    vision with k = 10 on cosine neighbors. Its chance level is the expectation of m/(2k - m)
    over the hypergeometric shared count m.

    Args:
        x_a, x_b: (N, D_a) and (N, D_b), rows of the same items: two layers, checkpoints, models
            or modalities.
        k: Neighborhood size.
        l2: Scale rows to unit norm first, so Euclidean neighbors are cosine neighbors.
        jaccard: Normalize each point's shared count by the union size instead of k.
        neighbors_a, neighbors_b: Precomputed Neighbors tables with at least k neighbors.

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
    nb_a = neighbors_a if neighbors_a is not None else Neighbors.from_points(xa, k)
    nb_b = neighbors_b if neighbors_b is not None else Neighbors.from_points(xb, k)
    if nb_a.k < k or nb_b.k < k:
        raise ValueError("Neighbors tables need at least k neighbors")
    per_point = _shared_neighbor_fraction(nb_a.indices[:, 1 : k + 1], nb_b.indices[:, 1 : k + 1], jaccard)
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

    Huh et al. (2024, ICML, App. A, Table 11, and their code): the fraction of items whose k nearest
    neighbors in B have the item among their own k nearest neighbors in A, a first hop in B and a
    return hop in A. Written cycle-kNN(A -> B) by Zhang et al. (2026, Eq. 1) and cycle-kNN_k(A, B)
    by Groger, Wen and Brbic (2026, Eq. 35). The ordering matters for k >= 2 (Zhang et al., App. A),
    and they report both orderings and their gap. Identical representations score 1 only when every
    item is a nearest neighbor of one of its own nearest neighbors; items with no reciprocal
    neighbor lower the value. For independent representations each return hop succeeds with
    probability k/(N - 1) (Groger et al., Prop. C.9), so the chance level is at most k^2/(N - 1).
    Euclidean neighbors, the point itself excluded; k = 10 and cosine neighbors (l2=True) in Huh
    et al., Zhang et al. and Groger et al.

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
    """Linear centered kernel alignment.

    Kornblith, Norouzi, Lee and Hinton (2019, ICML, arXiv:1905.00414, Table 1): ||Y^T X||_F^2 /
    (||X^T X||_F ||Y^T Y||_F) for column-centered X and Y. Invariant to orthogonal maps and
    isotropic scaling. The plug-in estimate is biased upward unless N is large relative to the
    widths; debiased=True uses the unbiased HSIC estimator, as in the authors' notebook, which
    can be negative and needs N >= 4. Computed from D x D cross-products in float64.

    Args:
        x_a, x_b: (N, D_a) and (N, D_b).
        debiased: Return the debiased estimate.

    Returns:
        value: CKA.
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


def _rsa_vector(x: Tensor, distance: str, method: str, chunk: int) -> Tensor:
    """Standardized (zero-mean, unit-norm) distance vector, ranked for Spearman; RSA is the dot product of two."""
    v = _condensed_distances(x.double(), distance, chunk)
    if method == "spearman":
        v = _average_ranks(v)
    elif method != "pearson":
        raise ValueError(f"method must be spearman or pearson, got {method!r}")
    v = v - v.mean()
    return v / v.norm().clamp_min(1e-300)


def rsa(
    x_a: Tensor, x_b: Tensor, *, distance: str = "cosine", method: str = "spearman", chunk: int = 1024
) -> MetricResult:
    """Representational similarity analysis: correlation of the pairwise-distance vectors of two representations.

    Kriegeskorte, Mur and Bandettini (2008, Frontiers in Systems Neuroscience): each representation
    gives the distances between all pairs of items, and the two vectors (i < j) are compared by
    Spearman rank correlation; 1 for the same geometry, 0 in expectation for unrelated ones. The
    distance is cosine by default, as in Koepke et al. (2026), "euclidean", or "correlation" (one
    minus the Pearson correlation across features, the paper's choice). Reads the full ordering of
    pairs, where CKA weights the leading directions. N(N - 1)/2 distances per representation, built
    in row chunks.

    Args:
        x_a, x_b: (N, D_a) and (N, D_b), rows of the same items.
        distance: "cosine", "euclidean" or "correlation".
        method: "spearman" (ranks) or "pearson" (raw distances).
        chunk: Rows per distance block.

    Returns:
        value: the correlation.
        extras: n_pairs.
    """
    _check_pair(x_a, x_b)
    a, b = _rsa_vector(x_a, distance, method, chunk), _rsa_vector(x_b, distance, method, chunk)
    return MetricResult(float(torch.dot(a, b)), {"n_pairs": float(a.numel())})


register_metric(
    "rsa",
    inputs=InputKind.PAIR,
    preprocess=Preprocess(),
    citation=("kriegeskorte2008rsa", "koepke2026cave"),
    max_items=4000,
    description="Spearman correlation of the pairwise-distance vectors of two representations (Kriegeskorte et al., 2008).",
)(rsa)
register_metric(
    "cka",
    inputs=InputKind.PAIR,
    preprocess=Preprocess(),
    citation=("kornblith2019similarity",),
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
    citation=("glielmo2022imbalance", "cheng2025emergence", "acevedo2025semantic", "glielmo2022dadapy"),
    arxiv="2104.15079",
)(information_imbalance)
