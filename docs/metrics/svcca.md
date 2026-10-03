# `svcca`

Mean canonical correlation of the leading SVD directions of two layers (Raghu et al., 2017).

- Input: `pair`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- Origin: https://arxiv.org/abs/1706.05806
- Cite: `raghu2017svcca` (SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability (2017))

## Definition, protocol and pitfalls

SVCCA similarity: mean canonical correlation between the leading SVD directions of two representations.

Raghu, Gilmer, Yosinski and Sohl-Dickstein (2017, NeurIPS, arXiv:1706.05806):
each representation is centered and reduced by SVD to the fewest directions
whose singular values sum to at least threshold of their total (App. A, the
99% rule on singular values), canonical correlation analysis between the two
reduced representations gives min(k_a, k_b) correlations, and their mean is the
similarity (Eq. 1, averaged over the aligned directions as in the reference
tutorial and in Kornblith et al., 2019, Table 1). The correlations are the
singular values of U_a^T U_b for the orthonormal bases of the kept directions,
which equals the covariance-based CCA of the reference code at epsilon = 0
without inverting covariance matrices. Invariant to invertible linear maps of
the kept subspaces, so it needs N well above the kept widths: the reference
tutorial asks for 5 to 10 times as many items as neurons, and as a kept width
approaches N any two representations score near 1 (Kornblith et al., 2019,
Theorem 1).

Args:
    x_a, x_b: (N, D_a) and (N, D_b), rows describing the same items.
    threshold: Fraction of the summed singular values kept per representation.

Returns:
    value: mean canonical correlation in [0, 1].
    extras: r2 (mean squared correlation), k_a, k_b (directions kept).
