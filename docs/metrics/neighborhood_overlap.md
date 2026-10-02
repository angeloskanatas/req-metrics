# `neighborhood_overlap`

Neighborhood overlap: mean fraction of k nearest neighbors shared by two representations of the same items.

- Input: `pair`
- Canonical preprocessing: `none`
- Tags: relational
- Shared cache: none
- Origin: https://arxiv.org/abs/2007.03506
- Cite: `doimo2020nucleation` (Hierarchical nucleation in deep neural networks (2020)); `valeriani2023geometry` (The geometry of hidden representations of large transformer models (2023)); `glielmo2022dadapy` (DADApy: Distance-based Analysis of DAta-manifolds in Python (2022))

## Definition, protocol and pitfalls

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
