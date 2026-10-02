# `dime`

DiME: permuted minus paired matrix-based joint entropy of two views.

- Input: `views`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- Origin: https://arxiv.org/abs/2301.08164
- Cite: `skean2023dime` (DiME: Maximizing Mutual Information by a Difference of Matrix-Based Entropies (2023))

## Definition, protocol and pitfalls

Skean et al. (2023, arXiv:2301.08164): with K_X and K_Y the N x N
normalized Gram matrices of the two views (unit diagonal), the joint
entropy is S_alpha(K_X o K_Y) over the Hadamard product (their Sec. 2.2),
and DiME is the expected joint entropy under random re-pairings of one
view minus the paired joint entropy. It behaves like a mutual
information between the views and is zero when they are unrelated. The
paper uses Gaussian kernels with alpha = 1.01; the Skean et al. (2025)
layer-wise reference implementation uses the linear Gram of the states. Both
that implementation replaces the N x N Gram matrices by D x D covariances
whenever N > D; the Hadamard product does not commute with
that swap, so those values are not the published quantity. This
implementation follows the definition and therefore costs an N x N
eigendecomposition per permutation; subsample to a few thousand clips.

Args:
    views: Two views, shape (2, N, D).
    alpha: Renyi order; 1.0 gives von Neumann entropy.
    n_perm: Random re-pairings averaged for the baseline.
    kernel: "linear" (cosine Gram of unit rows) or "rbf" (median bandwidth).
    seed: Permutation seed.
    normalization: "raw" or "max" (divide by log N).

Returns:
    value: DiME.
    extras: joint_entropy, permuted_joint_entropy.
