# `participation_ratio`

Participation ratio (sum lambda)^2 / sum lambda^2 of the covariance eigenvalues.

- Input: `points`
- Canonical preprocessing: `center`
- Tags: none
- Shared cache: none
- arXiv: https://arxiv.org/abs/2602.03282
- Cite: `chung2026globalgeometry` (Global Geometry Is Not Enough for Vision Representations (2026)); `jha2026nerve` (NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks (2026)); `chun2026dimensionality` (Estimating Dimensionality of Neural Representations from Finite Samples (2026))

## Definition, protocol and pitfalls

Jha et al. (2026, Eq. 2); divided by D, the G.PR of Chung and Kim (2026). The plug-in
estimate is biased low by about PR/N. Chun, Canatar, Chung and Lee (2026, ICLR,
arXiv:2509.26560, Sec. 4) give unbiased estimators: "row" corrects the sample size (all
units observed), "col" the unit subsampling, "both" the two. All four come from one
closed-form pass over the (N, D) matrix; correction chooses the value. A Spectrum, or
center=False, gives the plug-in estimate only.

Args:
    x: Points (N, D), or a Spectrum of preprocessed points (plug-in only).
    normalized: Divide by D.
    center: Mean-center; False gives the uncentered plug-in estimate only.
    correction: "none", "row", "col" or "both".

Returns:
    value: the chosen estimate, divided by D if normalized.
    extras: participation_ratio (the chosen estimate) and the naive, row, col and both
        estimates, none divided by D.
