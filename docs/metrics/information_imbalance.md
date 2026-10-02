# `information_imbalance`

Information imbalance from representation A to representation B.

- Input: `pair`
- Canonical preprocessing: `none`
- Tags: paper-canonical
- Shared cache: none
- Origin: https://arxiv.org/abs/2104.15079
- Cite: `glielmo2022imbalance` (Ranking the information content of distance measures (2022)); `glielmo2022dadapy` (DADApy: Distance-based Analysis of DAta-manifolds in Python (2022))

## Definition, protocol and pitfalls

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
