# `dime`

DiME: permuted minus paired matrix-based joint entropy of two views.

- Input: `views`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- Origin: https://arxiv.org/abs/2301.08164
- Cite: `skean2023dime` (DiME: Maximizing Mutual Information by a Difference of Matrix-Based Entropies (2023))

## Definition, protocol and pitfalls

Skean et al. (2023, arXiv:2301.08164, Sec. 2.2): with K_X and K_Y the normalized N x N Gram
matrices of the two views, DiME is the mean joint entropy S_alpha(K_X o K_Y) under random
re-pairings minus the paired joint entropy; zero for unrelated views. The paper uses Gaussian
kernels and alpha = 1.01. The N x N definition is kept, since the Hadamard product does not
survive a swap to D x D covariances, so cost grows as N^3 per permutation.

Args:
    views: Two views (2, N, D).
    alpha: Renyi order.
    n_perm: Random re-pairings averaged.
    kernel: "linear" (cosine Gram) or "rbf" (median bandwidth).
    seed: Permutation seed.
    normalization: "raw" or "max" (divide by log N).
    center: Mean-center each view before the kernel, as Skean et al. (2025) do.

Returns:
    value: DiME.
    extras: joint_entropy, permuted_joint_entropy.
