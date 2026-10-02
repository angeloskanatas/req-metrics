# `alpha_req`

Power-law decay exponent of the covariance eigenspectrum (alpha-ReQ).

- Input: `points`
- Canonical preprocessing: `center`
- Tags: computed-not-in-paper
- Shared cache: spectrum
- Cite: `DBLP:conf/nips/AgrawalMGR22` ($\alpha$-ReQ: Assessing Representation
                  Quality in Self-Supervised Learning by measuring eigenspectrum decay (2022)); `stringer2019highdim` (High-dimensional geometry of population responses in visual cortex (2019))

## Definition, protocol and pitfalls

Agrawal et al. (2022, NeurIPS) fit lambda_j ~ j^(-alpha) to the sorted
eigenvalues of the empirical covariance. Small alpha (at most about 1)
indicates a dense encoding, large alpha a rapidly decaying, sparse one;
both too high and too low a value imply poor generalization, with good
representations in a range close to 1, matching the infinite-width
linear-regression result that min-norm solutions generalize iff alpha = 1.
The fit is the Stringer et al. (2019, Nature) recipe used by the
reference implementation: weighted least squares in log-log space over
the eigenvalue indices in fit_range, with weights 1/j to down-weight the
tail. The exponent is independent of how the spectrum is normalized.
The fit range changes conclusions: Arputharaj et al. (2026, TMLR,
Appendix B.1) find the correlation of alpha with accuracy flipping sign
between this range and indices [10, 0.9 D], so record fit_range with
every value and do not compare alphas fitted over different ranges.

Args:
    x: Points (N, D), or a Spectrum of already preprocessed points.
    fit_range: Half-open range of 0-based eigenvalue indices used in the fit.
    center: Mean-center before the SVD (the covariance is centered by definition).

Returns:
    value: alpha.
    extras: r2 of the log-log fit over fit_range.
