# `self_clustering`

Self-clustering score: excess squared cosine over a uniform spherical cloud.

- Input: `points`
- Canonical preprocessing: `l2`
- Tags: relational
- Shared cache: none
- Origin: https://arxiv.org/abs/2305.16562
- Cite: `tsitsulin2023unsupervised` (Unsupervised Embedding Quality Evaluation (2023)); `arputharaj2026comparative` (A Comparative Study of Label-free Representation Quality Metrics in Deep Learning (2026))

## Definition, protocol and pitfalls

Tsitsulin, Munkhoeva and Perozzi (2023, TAG-ML at ICML, arXiv:2305.16562, Def. 3.5): with Q
the sum of squared cosines over all pairs of L2-normalized rows,
(Q - N - N(N-1)/D) / (N^2 - N - N(N-1)/D), 0 for a uniform cloud and 1 for a single point.
The paper writes Q as a Frobenius norm but states the expectation and maximum of its
square, which is used here. Computed from the D x D second-moment matrix in O(N D^2).

Args:
    x: Points (N, D).
    center: Mean-center before normalizing.

Returns:
    value: score, 0 (uniform) to 1 (collapsed).
    extras: mean_squared_cosine over distinct pairs, and its uniform value 1/D.
