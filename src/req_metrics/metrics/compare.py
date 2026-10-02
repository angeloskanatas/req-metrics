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
    Patterns), which the cross-layer and cross-model analysis of Kanatas et al. (2026)
    used with k = 1 and full neighbor tables. Ranks are exact and computed by
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
    tags=("paper-canonical",),
)(information_imbalance)
