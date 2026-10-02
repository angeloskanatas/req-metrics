# `participation_ratio/corrected`

Row-bias-corrected participation ratio of Chun et al. (2026) on the raw matrix.

- Input: `points`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- Origin: https://arxiv.org/abs/2509.26560
- Cite: `chun2026dimensionality` (Estimating Dimensionality of Neural Representations from Finite Samples (2026)); `chung2026globalgeometry` (Global Geometry Is Not Enough for Vision Representations (2026))

## Definition, protocol and pitfalls

Participation ratio of the covariance eigenvalues, (sum lambda)^2 / sum lambda^2.

A classical count of directions that carry variance, between 1 and D
(NerVE, Jha et al., 2026, Eq. 2). Divided by D it is the global
participation ratio G.PR of Chung and Kim (2026, arXiv:2602.03282),
in (0, 1] with 1 for an isotropic cloud; that paper finds it, like other
global geometry statistics, uncorrelated with compositional binding,
which is the caveat to carry.

The plug-in estimate is biased downward at finite N: Chun, Canatar, Chung
and Lee (2026, ICLR, arXiv:2509.26560, Sec. 3) show 1/PR_naive is about
1/N + 1/D + 1/PR, so the relative bias is about PR/N. Their unbiased
estimators average the quartic index sums over distinct indices (Sec. 4);
correction="row" removes the sample-size bias, the case of network
activations where all D units are observed (their Sec. 4.5), and "both"
also removes the unit-subsampling term. The corrected estimators center
algebraically and therefore need the raw (N, D) matrix, not a Spectrum or
pre-centered data; the quartic sums are computed in closed form from the
D x D second-moment matrix and column moments, checked against the
reference implementation. Use "row" when N is below about 100 times the
expected ratio, which includes small monitoring buffers.

Args:
    x: Points (N, D), or a Spectrum of already preprocessed points (correction="none" only).
    normalized: Divide by D.
    center: Mean-center before the SVD (correction="none").
    correction: "none" (plug-in), "row", "col" or "both" (Chun et al., 2026).

Returns:
    value: participation ratio, normalized or raw.
    extras: the raw ratio; with a correction also the plug-in ratio.
