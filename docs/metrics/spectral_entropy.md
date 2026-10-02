# `spectral_entropy`

Shannon entropy of the normalized spectrum of the centered covariance.

- Input: `points`
- Canonical preprocessing: `center`
- Tags: none
- Shared cache: spectrum
- Origin: https://arxiv.org/abs/2603.06922
- Cite: `jha2026nerve` (NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks (2026))

## Definition, protocol and pitfalls

NerVE (Jha et al., 2026, arXiv:2603.06922, Eq. 1) reads the eigenvalues
of the activation covariance as a distribution and reports its entropy:
near 0 for a collapsed spectrum, log D for a uniform one. It is the
logarithm of the effective rank under the same spectrum convention, so
the two agree in rank; it is kept because the normalized form in [0, 1]
is the quantity usually plotted during training.

Args:
    x: Points (N, D), or a Spectrum of already preprocessed points.
    spectrum: "variance" (covariance eigenvalues, NerVE) or "singular".
    normalization: "max" divides by log min(N, D); "logN", "logD", "raw".
    center: Mean-center before the SVD.

Returns:
    value: normalized entropy.
    extras: raw entropy in nats.
