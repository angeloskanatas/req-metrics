# `procrustes`

Squared orthogonal Procrustes distance between centered, Frobenius-normalized representations (Ding et al., 2021); lower is closer.

- Input: `pair`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- arXiv: https://arxiv.org/abs/2108.01661
- Cite: `ding2021grounding` (Grounding representation similarity through statistical testing (2021)); `klabunde2025resi` (ReSi: A Comprehensive Benchmark for Representational Similarity Measures (2025))

## Definition, protocol and pitfalls

Squared orthogonal Procrustes distance between two representations, in Ding et al.'s normalization.

Ding, Denain and Steinhardt (2021, NeurIPS, arXiv:2108.01661, Sec. 2, Eqs. 6-7): the columns of each
representation are centered and the matrix divided by its Frobenius norm; the squared distance
min_R ||x_b - x_a R||_F^2 over orthogonal R is ||x_a||_F^2 + ||x_b||_F^2 - 2 ||x_a^T x_b||_*, here
2 - 2 ||x_a^T x_b||_* in [0, 2] with the nuclear norm, which unequal widths do not change (ReSi zero-pads
the narrower side to the same effect). With shared singular vectors it is the squared distance between
the singular values, where CKA is a quartic dominated by the largest ones (their Discussion). The
distance is symmetric and invariant to orthogonal maps and isotropic scaling of either side, but not to
invertible maps (CCA is) or to anisotropic scaling. Across models of different widths, the convergence it
reports is a width effect that permutation calibration removes (Groger et al., 2026; `calibrate`). It
costs one thin SVD per side, in float64.

Args:
    x_a: (N, D_a).
    x_b: (N, D_b).
    rank_tol: Singular values below rank_tol times the largest are dropped on each side.

Returns:
    value: the squared distance, 0 for representations equal up to an orthogonal map and a scale, at most 2;
        lower is closer.
    extras: alignment (||x_a^T x_b||_* in [0, 1], 1 minus half the value), distance (the square root),
        rank_a, rank_b.
