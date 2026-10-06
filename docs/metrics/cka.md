# `cka`

Linear centered kernel alignment between two representations of the same items (Kornblith et al., 2019).

- Input: `pair`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- arXiv: https://arxiv.org/abs/1905.00414
- Cite: `kornblith2019similarity` (Similarity of Neural Network Representations Revisited (2019))

## Definition, protocol and pitfalls

Linear centered kernel alignment.

Kornblith, Norouzi, Lee and Hinton (2019, ICML, arXiv:1905.00414, Table 1): ||Y^T X||_F^2 /
(||X^T X||_F ||Y^T Y||_F) for column-centered X and Y. Invariant to orthogonal maps and
isotropic scaling. The plug-in estimate is biased upward unless N is large relative to the
widths; debiased=True uses the unbiased HSIC estimator, as in the authors' notebook, which
can be negative and needs N >= 4. Computed from D x D cross-products in float64.

Args:
    x_a, x_b: (N, D_a) and (N, D_b).
    debiased: Return the debiased estimate.

Returns:
    value: CKA.
    extras: biased, debiased (nan for N < 4).
