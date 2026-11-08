# `cka`

Linear centered kernel alignment between two representations of the same items (Kornblith et al., 2019).

- Input: `pair`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- arXiv: https://arxiv.org/abs/1905.00414
- Cite: `kornblith2019similarity` (Similarity of Neural Network Representations Revisited (2019)); `cui2022deconfounded` (Deconfounded Representation Similarity for Comparison of Neural Networks (2022))

## Definition, protocol and pitfalls

Linear centered kernel alignment, optionally deconfounded.

Kornblith, Norouzi, Lee and Hinton (2019, ICML, arXiv:1905.00414, Table 1): ||Y^T X||_F^2 /
(||X^T X||_F ||Y^T Y||_F) for column-centered X and Y. It is invariant to orthogonal maps and
isotropic scaling. The plug-in estimate is biased upward unless N is large relative to the
widths; debiased=True uses the unbiased HSIC estimator, as in the authors' notebook, which
can be negative and needs N >= 4. The value is computed from D x D cross-products in float64.

With confound, the deconfounded CKA of Cui, Kumar, Marttinen and Kaski (2022, NeurIPS,
arXiv:2202.00095, Eq. 3-11): the linear kernel of a representation of the same items (their
input space; the previous layer in their recursive variant, App. D.2) is regressed out of each
kernel by least squares over all N^2 entries, the residuals are projected onto the PSD cone and
compared with the biased HSIC. It removes the similarity both sides inherit from the confound,
which can make two random networks look alike (their Fig. 1: CKA 0.99, dCKA 0.43), and keeps
the invariances of CKA (their Props. 3.1-3.2). The authors expect the same conclusions as CKA
within one domain (App. G), the linear fit holds best for shallow layers (App. A), and where
the input structure is itself of interest it should not be removed (Sec. 5). Costs O(N^2)
memory and an O(N^3) eigendecomposition per representation.

Args:
    x_a, x_b: (N, D_a) and (N, D_b).
    debiased: Return the debiased estimate (not with confound).
    confound: (N, D_0) representation whose similarity structure is removed from both kernels.
    confound_b: The confound of x_b when it differs from x_a's (defaults to confound).

Returns:
    value: CKA, or the deconfounded CKA when confound is given.
    extras: biased, debiased (plain CKA; nan for N < 4); with confound also alpha_a, alpha_b (regression
        coefficients), explained_a, explained_b (share of each kernel's variance the confound explains) and
        psd_dropped_a, psd_dropped_b (share of eigenvalue mass removed by the projection).
