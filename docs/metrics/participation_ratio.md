# `participation_ratio`

Participation ratio of the covariance eigenvalues, (sum lambda)^2 / sum lambda^2.

- Input: `points`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- Origin: https://arxiv.org/abs/2602.03282
- Cite: `chung2026globalgeometry` (Global Geometry Is Not Enough for Vision Representations (2026)); `jha2026nerve` (NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks (2026)); `chun2026dimensionality` (Estimating Dimensionality of Neural Representations from Finite Samples (2026))

## Definition, protocol and pitfalls

A classical count of directions that carry variance, between 1 and D
(NerVE, Jha et al., 2026, Eq. 2). Divided by D it is the global
participation ratio G.PR of Chung and Kim (2026, arXiv:2602.03282),
in (0, 1] with 1 for an isotropic cloud; that paper finds it, like other
global geometry statistics, uncorrelated with compositional binding,
which is the caveat to carry.

The plug-in estimate is biased downward at finite N: Chun, Canatar, Chung
and Lee (2026, ICLR, arXiv:2509.26560, Sec. 3) show 1/PR_naive is about
1/N + 1/D + 1/PR, so the relative bias is about PR/N. Their unbiased
estimators average the quartic index sums over distinct indices (Sec. 4):
"row" removes the sample-size bias, the case of network activations where
all D units are observed (their Sec. 4.5), "col" removes the
unit-subsampling term and "both" removes the two. The plug-in and the three
corrected estimates come from one pass over the (N, D) matrix, in closed
form from the D x D second-moment matrix and column moments, checked
against the reference implementation; all four are in the extras and
correction chooses the value. Use "row" when N is below about 100 times the
expected ratio, which includes small monitoring buffers. A Spectrum, or
center=False, gives the plug-in estimate only.

Args:
    x: Points (N, D), or a Spectrum of already preprocessed points (plug-in only).
    normalized: Divide by D.
    center: Mean-center; False gives the uncentered plug-in estimate only.
    correction: "none" (plug-in), "row", "col" or "both" (Chun et al., 2026).

Returns:
    value: the chosen estimate, divided by D if normalized.
    extras: participation_ratio (the chosen estimate, not divided by D) and the
        naive, row, col and both estimates.
