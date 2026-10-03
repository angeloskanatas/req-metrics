# `neighborhood_overlap`

Neighborhood overlap: mean fraction of the k nearest neighbors shared by two representations.

- Input: `pair`
- Canonical preprocessing: `none`
- Tags: relational
- Shared cache: none
- Origin: https://arxiv.org/abs/2007.03506
- Cite: `doimo2020nucleation` (Hierarchical nucleation in deep neural networks (2020)); `valeriani2023geometry` (The geometry of hidden representations of large transformer models (2023)); `glielmo2022dadapy` (DADApy: Distance-based Analysis of DAta-manifolds in Python (2022))

## Definition, protocol and pitfalls

Doimo, Glielmo, Ansuini and Laio (2020, NeurIPS, arXiv:2007.03506, Eq. 1): 1 when every point
keeps its neighbors, k/(N-1) in expectation for unrelated spaces. Euclidean neighbors, the
point itself excluded; k = 30 as in Doimo et al. and Valeriani et al. (2023).

Args:
    x_a, x_b: (N, D_a) and (N, D_b).
    k: Neighborhood size.
    neighbors_a, neighbors_b: Precomputed Neighbors tables with at least k neighbors.

Returns:
    value: mean overlap in [0, 1].
    extras: std of the per-point overlaps, chance level k / (N - 1).
