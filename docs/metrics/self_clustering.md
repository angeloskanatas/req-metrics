# `self_clustering`

Self-clustering score: excess squared cosine over a uniform spherical cloud.

- Input: `points`
- Canonical preprocessing: `l2`
- Tags: relational
- Shared cache: none
- Origin: https://arxiv.org/abs/2305.16562
- Cite: `tsitsulin2023unsupervised` (Unsupervised Embedding Quality Evaluation (2023)); `arputharaj2026comparative` (A Comparative Study of Label-free Representation Quality Metrics in Deep Learning (2026))

## Definition, protocol and pitfalls

Tsitsulin, Munkhoeva and Perozzi (2023, TAG-ML at ICML, arXiv:2305.16562,
Def. 3.5): for L2-normalized rows W, the pairwise dot-product mass
Q = sum_ij (w_i . w_j)^2 is compared with its expectation for N points
uniform on the sphere, N + N(N-1)/D, and with its maximum N^2 at complete
collapse: SelfCluster = (Q - N - N(N-1)/D) / (N^2 - N - N(N-1)/D), 0 for a
uniform cloud and 1 for a single point. The paper writes Q as the Frobenius
norm of W W^T; the expectation and maximum it states are those of the squared
norm, which is what is used here so that collapse gives exactly 1. Computed
through the D x D second-moment matrix, ||W^T W||_F^2 = ||W W^T||_F^2, in
O(N D^2) (the reformulation of Arputharaj et al., 2026, TMLR). Arputharaj et
al. report it as a reliable negative predictor of accuracy for
self-supervised vision models, uninformative for supervised ones, and
anti-correlated at -0.999 with diffusion spectral entropy under
L2-normalization. The paper found its sign on graph embeddings consistent
where spectral metrics flipped.

Args:
    x: Points (N, D).
    center: Mean-center before normalizing (off in the source paper).

Returns:
    value: SelfCluster, 0 (uniform) to 1 (collapsed).
    extras: mean_squared_cosine over distinct pairs, and its uniform value 1/D.
