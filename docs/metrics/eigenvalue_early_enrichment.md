# `eigenvalue_early_enrichment`

Top-heaviness of the covariance spectrum over the ambient dimension (EEE).

- Input: `points`
- Canonical preprocessing: `center`
- Tags: none
- Shared cache: spectrum
- Origin: https://arxiv.org/abs/2603.06922
- Cite: `jha2026nerve` (NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks (2026))

## Definition, protocol and pitfalls

NerVE (Jha et al., 2026, arXiv:2603.06922, Eq. 3): the mean gap between
the cumulative variance fraction of the k largest eigenvalues and the
uniform reference k/D, normalized to [0, 1): EEE = (2/D) sum_k (S_k - k/D)
over all D ambient directions, unused ones counted as zero variance.
0 for a flat spectrum, approaching 1 when one direction carries all the
variance. Scale-invariant.

Args:
    x: Points (N, D), or a Spectrum of already preprocessed points.
    center: Mean-center before the SVD.

Returns:
    value: EEE in [0, 1).
    extras: none.
