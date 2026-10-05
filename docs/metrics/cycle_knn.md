# `cycle_knn`

Fraction of items that are a nearest neighbor in A of one of their nearest neighbors in B (Huh et al., 2024); the ordering matters.

- Input: `pair`
- Canonical preprocessing: `none`
- Tags: relational
- Shared cache: none
- Origin: https://arxiv.org/abs/2405.07987
- Cite: `huh2024platonic` (Position: The Platonic Representation Hypothesis (2024)); `zhang2026wittgensteinian` (The Wittgensteinian Representation Hypothesis: Is Language the Attractor of Multimodal Convergence? (2026)); `groger2026aristotelian` (Revisiting the Platonic Representation Hypothesis: An Aristotelian View (2026))

## Definition, protocol and pitfalls

Cycle k-nearest-neighbor consistency from representation A to representation B.

Huh et al. (2024, ICML, App. A, Table 11, and their code): the fraction of items whose k nearest
neighbors in B have the item among their own k nearest neighbors in A, a first hop in B and a
return hop in A; Eq. 1 of Zhang et al. (2026) and Eq. 35 of Groger, Wen and Brbic (2026). Not
symmetric for k >= 2 (Zhang et al., App. A), so both orderings are returned. Identical
representations score 1 only when every item is a nearest neighbor of one of its own nearest
neighbors; items with no reciprocal neighbor lower the value. For independent representations
each return hop succeeds with probability k/(N - 1) (Groger et al., Prop. C.9), so the chance
level is at most k^2/(N - 1). Euclidean neighbors, the point itself excluded; k = 10 and cosine
neighbors (l2=True) in all three sources.

Args:
    x_a, x_b: (N, D_a) and (N, D_b), rows of the same items.
    k: Neighborhood size.
    l2: Scale rows to unit norm first, so Euclidean neighbors are cosine neighbors.
    neighbors_a, neighbors_b: Precomputed Neighbors tables with at least k neighbors.

Returns:
    value: the fraction in [0, 1], A -> B.
    extras: reverse (B -> A), chance_bound k^2/(N - 1).
