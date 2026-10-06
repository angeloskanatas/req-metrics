# `intrinsic_dimension/mlid`

Geometric mean of per-point local intrinsic dimension (mLID).

- Input: `points`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: neighbors
- arXiv: https://arxiv.org/abs/2401.10474
- Cite: `huang2024ldreg` (LDReg: Local Dimensionality Regularized Self-Supervised Learning (2024)); `amsaleg2018lid` (Extreme-value-theoretic estimation of local intrinsic dimensionality (2018))

## Definition, protocol and pitfalls

LID_i = mu_k / (w_k - mu_k) by the method of moments (Amsaleg et al., 2018), aggregated as
the geometric mean, the Frechet mean of LDReg (Huang et al., 2024, ICLR, arXiv:2401.10474).
w_k is the k-th neighbor distance and mu_k the mean of the k - 1 nearer ones, as in the
LDReg code (lid_mom_est); the paper's text averages all k.

Args:
    x: Points (N, D), duplicates removed first, or a Neighbors table with at least k neighbors.
    k: Neighborhood size.

Returns:
    value: mLID.
    extras: frechet_var (variance of log LID), n_valid.
