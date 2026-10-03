# `effective_rank`

Effective rank: exponential of the Shannon entropy of the normalized spectrum.

- Input: `points`
- Canonical preprocessing: `center`
- Tags: paper-canonical
- Shared cache: spectrum
- Origin: https://arxiv.org/abs/2210.02885
- Cite: `DBLP:conf/eusipco/RoyV07` (The effective rank: A measure of effective dimensionality (2007)); `DBLP:conf/icml/GarridoBNL23` (RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised
                  Representations by Their Rank (2023)); `jha2026nerve` (NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks (2026))

## Definition, protocol and pitfalls

Roy and Vetterli (2007, EUSIPCO); RankMe (Garrido et al., 2023, ICML, arXiv:2210.02885).
p_k = s_k / sum(s) over the singular values, or s_k^2 / sum(s^2) with spectrum="variance"
(Skean et al., 2025). Centered by default: on autoregressive decoders the uncentered
spectrum is dominated by the mean direction. RankMe uses 25,600 samples; N should be well
above D.

Args:
    x: Points (N, D), or a Spectrum of preprocessed points.
    spectrum: "singular" or "variance".
    center: Mean-center before the SVD.

Returns:
    value: effective rank in [1, min(N, D)].
    extras: entropy; normalized_entropy, over log min(N, D) (with spectrum="variance", the
        spectral entropy of Jha et al., 2026, Eq. 1); normalized_rank, value / D; and the
        effective rank under the other convention.
