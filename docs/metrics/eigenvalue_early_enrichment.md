# `eigenvalue_early_enrichment`

Top-heaviness of the covariance spectrum over the ambient dimension (EEE).

- Input: `points`
- Canonical preprocessing: `center`
- Tags: none
- Shared cache: spectrum
- Origin: https://arxiv.org/abs/2603.06922
- Cite: `jha2026nerve` (NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks (2026))

## Definition, protocol and pitfalls

Jha et al. (2026, arXiv:2603.06922, Eq. 3): EEE = (2/D) sum_k (S_k - k/D), with S_k the
cumulative variance fraction of the k largest eigenvalues over all D directions. 0 for a flat
spectrum, approaching 1 when one direction carries all the variance; scale-invariant.

Args:
    x: Points (N, D), or a Spectrum of preprocessed points.
    center: Mean-center before the SVD.

Returns:
    value: EEE in [0, 1).
    extras: none.
