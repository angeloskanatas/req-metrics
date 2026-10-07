# `effective_rank`

Effective rank: exponential of the Shannon entropy of the normalized spectrum.

- Input: `points`
- Canonical preprocessing: `center`
- Tags: paper-canonical
- Shared cache: spectrum
- arXiv: https://arxiv.org/abs/2210.02885
- Cite: `DBLP:conf/eusipco/RoyV07` (The effective rank: A measure of effective dimensionality (2007)); `DBLP:conf/icml/GarridoBNL23` (RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised
                  Representations by Their Rank (2023)); `jha2026nerve` (NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks (2026))

## Definition, protocol and pitfalls

Roy and Vetterli (2007, EUSIPCO); RankMe (Garrido et al., 2023, ICML, arXiv:2210.02885).
p_k = s_k / sum(s) over the singular values, or s_k^2 / sum(s^2) with spectrum="variance"
(Skean et al., 2025). RankMe takes the uncentered embeddings and adds a small epsilon to
p_k; here the default centers, because on autoregressive decoders the uncentered spectrum is
dominated by the mean direction, and there is no epsilon. RankMe uses 25,600 samples; N
should be well above D.

Depth lowers the converged rank by design in joint-embedding predictive training: in a deep

linear JEPA every mode converges to rho_i^L, its across-view regression coefficient to the

power of the encoder depth (Littwin et al., 2024, Cor. 4.3), so a deeper encoder reads lower

without being collapsed, and in BYOL the predictor input keeps about a quarter of its modes

alive while accuracy rises (Tian et al., 2021, Fig. 2). Compare runs at one depth, and read a

low rank as necessary-but-not-sufficient evidence.


Args:
    x: Points (N, D), or a Spectrum of preprocessed points.
    spectrum: "singular" or "variance".
    center: Mean-center before the SVD.
    max_eigenvalues: Keep only the largest singular values, as a truncated PCA does.

Returns:
    value: effective rank in [1, min(N, D)]; 0 for a zero matrix (every row equal after
        centering), which has rank 0.
    extras: entropy; normalized_entropy, over log min(N, D) (with spectrum="variance", the
        spectral entropy of Jha et al., 2026, Eq. 1); normalized_rank, value / D; and the
        effective rank under the other convention.
