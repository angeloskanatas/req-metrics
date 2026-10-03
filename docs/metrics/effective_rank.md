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

Roy and Vetterli (2007, EUSIPCO) define it on the singular values,
p_k = s_k / sum(s), erank = exp(-sum p_k log p_k), bounded by 1 and the
rank. RankMe (Garrido et al., 2023, ICML) uses the same quantity as a
label-free predictor of linear-probe accuracy and for hyperparameter
selection, computed on 25,600 samples; their appendix shows convergence
in the number of samples for 2048-dimensional outputs, so N should be an
order of magnitude above D. RankMe-t (Aldeneh et al., 2024) is the same
quantity on frame sequences summed over time, one vector per utterance,
which is the pooled population with mean pooling up to a per-clip scale
that the effective rank ignores. Skean et al. (2025, ICML) and the reptrix
library normalize the eigenvalues of the covariance instead (s_k^2),
which weights the leading directions more heavily; pass
spectrum="variance" for that convention. Both conventions come from the
same spectrum, so the other one is always in the extras. With
spectrum="variance", normalized_entropy is the spectral entropy of NerVE
(Jha et al., 2026, arXiv:2603.06922, Eq. 1).

The matrix is mean-centered before the SVD. RankMe as published does
not center; the reptrix reference implementation does, through PCA.
Centering changes conclusions: on autoregressive decoders the
uncentered spectrum is dominated by the mean direction and the
layer-wise correlation with downstream accuracy changes sign. RankMe's
epsilon inside the logarithm is omitted; zero singular values contribute
zero entropy exactly.

Args:
    x: Points (N, D), or a Spectrum of already preprocessed points.
    spectrum: "singular" (Roy-Vetterli, RankMe) or "variance" (Skean, reptrix).
    center: Mean-center before the SVD.

Returns:
    value: effective rank in [1, min(N, D)].
    extras: entropy, normalized_entropy (over log min(N, D)), normalized_rank = value / D
        (RankMe* of Tsitsulin et al., 2023, the fraction of the width in use), and the
        effective rank under the other spectrum convention.
