# `dime`

DiME: permuted minus paired matrix-based joint entropy of two views.

- Input: `views`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- arXiv: https://arxiv.org/abs/2301.08164
- Cite: `skean2023dime` (DiME: Maximizing Mutual Information by a Difference of Matrix-Based Entropies (2023))

## Definition, protocol and pitfalls

Skean et al. (2023, arXiv:2301.08164, Sec. 2.2): with K_X and K_Y the normalized N x N Gram
matrices of the two views, DiME is the mean joint entropy S_alpha(K_X o K_Y) under random
re-pairings minus the paired joint entropy; it is zero for unrelated views. The paper uses
Gaussian kernels and alpha = 1.01. The Hadamard product has no D x D covariance form, so the
N x N matrices are needed and the cost grows as N^3 per permutation.

Args:
    views: Two views (2, N, D).
    alpha: Renyi order.
    n_perm: Random re-pairings averaged.
    kernel: "linear" (cosine Gram) or "rbf" (median bandwidth).
    seed: Permutation seed.
    normalization: "raw" or "max" (divide by log N).
    center: Mean-center each view before the kernel, as the code of Skean et al. (2025) does (their
        paper does not state it).

Returns:
    value: DiME.
    extras: joint_entropy, permuted_joint_entropy.
