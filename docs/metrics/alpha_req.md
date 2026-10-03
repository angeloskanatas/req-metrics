# `alpha_req`

Power-law decay exponent of the covariance eigenspectrum (alpha-ReQ).

- Input: `points`
- Canonical preprocessing: `center`
- Tags: none
- Shared cache: spectrum
- Cite: `DBLP:conf/nips/AgrawalMGR22` ($\alpha$-ReQ: Assessing Representation
                  Quality in Self-Supervised Learning by measuring eigenspectrum decay (2022)); `stringer2019highdim` (High-dimensional geometry of population responses in visual cortex (2019))

## Definition, protocol and pitfalls

Agrawal et al. (2022, NeurIPS): lambda_j ~ j^(-alpha), fitted by weighted least squares in
log-log space with weights 1/j over fit_range (Stringer et al., 2019). Values near 1 go with
good representations. The fit range can flip the sign of the correlation with accuracy
(Arputharaj et al., 2026, App. B.1), so compare alphas only at equal fit_range.

Args:
    x: Points (N, D), or a Spectrum of preprocessed points.
    fit_range: Half-open range of 0-based eigenvalue indices.
    center: Mean-center before the SVD.

Returns:
    value: alpha.
    extras: r2 of the log-log fit.
