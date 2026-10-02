# `lidar`

LiDAR: effective rank of the LDA matrix with clips as classes and views as samples.

- Input: `views`
- Canonical preprocessing: `none`
- Tags: paper-canonical
- Shared cache: none
- Origin: https://arxiv.org/abs/2312.04000
- Cite: `DBLP:conf/iclr/Thilak0SDGNSL24` (LiDAR: Sensing Linear Probing Performance in Joint Embedding SSL (2024))

## Definition, protocol and pitfalls

Thilak et al. (2024, ICLR, arXiv:2312.04000), Eq. 1-4. Each clean clip
is a surrogate class and its q views the within-class samples:
S_b is the scatter of the class means around the grand mean, S_w the
scatter of the views around their class mean plus delta I, and
LiDAR = exp(-sum p_i log p_i) over the eigenvalues of
S_w^{-1/2} S_b S_w^{-1/2}, normalized by their sum. It counts the
directions that separate clips after whitening the variability the
augmentations induce, so it tracks the training objective's own
invariances rather than raw covariance rank. The paper uses unbiased
estimates (n-1 and n(q-1) denominators), recommends n above the feature
width because rank(S_b) <= n, and finds q = 10 within one percent of
q = 50. Computed in float64 with symmetric eigendecompositions; the
epsilon the paper adds inside the logarithm is omitted, zero eigenvalues
contributing nothing. The paper does not print delta; 1e-4 is the value
of the reference implementation of Skean et al. (2025). Protocol of Kanatas et al. (2026):
10,000 clips, 10 views, biased denominators, delta 1e-6, a shared
augmentation chain across models with task-defining augmentations
removed per task family, whereas the original work uses each method's
own training augmentations. On autoregressive decoders the layer-wise
correlation with downstream accuracy reverses sign in Kanatas et al. (2026).

Args:
    views: Augmented representations, shape (q, N, D), q >= 2.
    delta: Ridge added to the within-class scatter.
    unbiased: Use n-1 and n(q-1) denominators (paper) or n and nq.
    max_eigenvalues: Keep only the largest eigenvalues, if set.

Returns:
    value: LiDAR effective rank.
    extras: entropy, n_positive_eigenvalues.
