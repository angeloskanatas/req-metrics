# `neighborhood_overlap`

Neighborhood overlap: mean fraction of the k nearest neighbors shared by two representations.

- Input: `pair`
- Canonical preprocessing: `none`
- Tags: relational
- Shared cache: none
- Origin: https://arxiv.org/abs/2007.03506
- Cite: `doimo2020nucleation` (Hierarchical nucleation in deep neural networks (2020)); `valeriani2023geometry` (The geometry of hidden representations of large transformer models (2023)); `huh2024platonic` (Position: The Platonic Representation Hypothesis (2024)); `glielmo2022dadapy` (DADApy: Distance-based Analysis of DAta-manifolds in Python (2022))

## Definition, protocol and pitfalls

Doimo, Glielmo, Ansuini and Laio (2020, NeurIPS, arXiv:2007.03506, Eq. 1): 1 when every point
keeps its neighbors, k/(N-1) in expectation for unrelated spaces (Groger, Wen and Brbic, 2026,
Prop. 4.2). Euclidean neighbors, the point itself excluded; k = 30 as in Doimo et al. and
Valeriani et al. (2023). The mutual k-nearest-neighbor alignment of Huh et al. (2024, ICML,
App. A) is the same quantity on cosine neighbors with k = 10 on 1,024 image-caption pairs;
l2=True ranks cosine neighbors. At fixed k the value falls as N grows and when an item has
several valid partners (Koepke et al., 2026), so compare at equal N and k on one-to-one pairs.

Args:
    x_a, x_b: (N, D_a) and (N, D_b), rows of the same items: two layers, checkpoints, models
        or modalities.
    k: Neighborhood size.
    l2: Scale rows to unit norm first, so Euclidean neighbors are cosine neighbors.
    neighbors_a, neighbors_b: Precomputed Neighbors tables with at least k neighbors.

Returns:
    value: mean overlap in [0, 1].
    extras: std of the per-point overlaps, chance level k / (N - 1).
