# `cka`

Linear centered kernel alignment between two layers (Kornblith et al., 2019).

- Input: `pair`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- Origin: https://arxiv.org/abs/1905.00414
- Cite: `kornblith2019similarity` (Similarity of Neural Network Representations Revisited (2019))

## Definition, protocol and pitfalls

Linear centered kernel alignment between two representations of the same items.

Kornblith, Norouzi, Lee and Hinton (2019, ICML, arXiv:1905.00414), Table 1:
CKA = ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F) for column-centered X (N, D_a)
and Y (N, D_b), the normalized HSIC of Eq. 4 with linear kernels. 1 for
representations equal up to an orthogonal map and an isotropic scaling; it is
not invariant to arbitrary invertible linear maps, which is what lets it
distinguish layers wider than N. The plug-in estimate is biased upward when N
is not large relative to the widths. debiased=True uses the unbiased HSIC
estimator of Song et al. (2007) in the feature-space form of the authors'
reference notebook; it reduces the bias, can be negative and needs N >= 4. Both
estimates are in the extras. Computed in float64 from the D x D cross-products
in O(N D_a D_b), without N x N Gram matrices.

Args:
    x_a, x_b: (N, D_a) and (N, D_b), rows describing the same items.
    debiased: Return the debiased estimate.

Returns:
    value: CKA in [0, 1] (debiased: can fall slightly below 0).
    extras: biased, debiased (nan for N < 4).
