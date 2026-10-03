# `svcca`

Mean canonical correlation of the leading SVD directions of two layers (Raghu et al., 2017).

- Input: `pair`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- Origin: https://arxiv.org/abs/1706.05806
- Cite: `raghu2017svcca` (SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability (2017))

## Definition, protocol and pitfalls

SVCCA: mean canonical correlation between the leading SVD directions of two representations.

Raghu, Gilmer, Yosinski and Sohl-Dickstein (2017, NeurIPS, arXiv:1706.05806, Eq. 1, App. A):
each centered representation keeps the fewest directions whose singular values sum to
threshold of the total, and the value is the mean of the min(k_a, k_b) canonical
correlations between them, computed as the singular values of U_a^T U_b. Invariant to
invertible linear maps of the kept subspaces, so N must be well above the kept widths; as
they approach N, any two representations score near 1.

Args:
    x_a, x_b: (N, D_a) and (N, D_b).
    threshold: Fraction of the summed singular values kept.

Returns:
    value: mean canonical correlation in [0, 1].
    extras: r2 (mean squared correlation), k_a, k_b.
