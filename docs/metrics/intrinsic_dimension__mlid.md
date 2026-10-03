# `intrinsic_dimension/mlid`

Geometric mean of per-point local intrinsic dimension (mLID).

- Input: `points`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: neighbors
- Origin: https://arxiv.org/abs/2401.10474
- Cite: `huang2024ldreg` (LDReg: Local Dimensionality Regularized Self-Supervised Learning (2024)); `amsaleg2018lid` (Extreme-value-theoretic estimation of local intrinsic dimensionality (2018))

## Definition, protocol and pitfalls

Per-point LID by the method of moments (Amsaleg et al., 2018, Data
Mining and Knowledge Discovery): LID_i = mu_k / (w_k - mu_k), with mu_k
the mean of the k-1 nearest distances and w_k the k-th. Aggregated as
the geometric mean, which LDReg (Huang et al., 2024, ICLR,
arXiv:2401.10474) motivates as the Frechet mean under the Fisher-Rao
metric; the variance of log LID is reported as the spread of local
dimensionality across points.

Args:
    x: Points (N, D), or a Neighbors table with at least k neighbors.
    k: Neighborhood size.

Returns:
    value: mLID.
    extras: frechet_var (variance of log LID), n_valid.
