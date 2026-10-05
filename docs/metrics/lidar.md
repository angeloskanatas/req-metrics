# `lidar`

LiDAR: effective rank of the LDA matrix, with samples as classes and their views as within-class points.

- Input: `views`
- Canonical preprocessing: `none`
- Tags: paper-canonical
- Shared cache: none
- Origin: https://arxiv.org/abs/2312.04000
- Cite: `DBLP:conf/iclr/Thilak0SDGNSL24` (LiDAR: Sensing Linear Probing Performance in Joint Embedding SSL (2024))

## Definition, protocol and pitfalls

Thilak et al. (2024, ICLR, arXiv:2312.04000, Eqs. 1-2; Eqs. 1-4 in arXiv v1): S_b is the
scatter of the samples' view means and S_w the scatter of the views around their sample mean plus
delta I; LiDAR is the exponential of the entropy of the normalized eigenvalues of
S_w^{-1/2} S_b S_w^{-1/2}. The clean sample names the class and is not one of the q views. Use
the training objective's own positives when monitoring one model (their Sec. 4.2) and one
shared chain when comparing models. The denominators rescale S_b and S_w by constants, which
leaves the value unchanged at delta = 0; an absolute delta makes it scale-dependent when
within-sample variance approaches delta. Directions without sample signal keep eigenvalues of
order 1/q, so compare at equal q and width, with n above the width (App. 11). The paper's
epsilon is omitted.

Args:
    views: Augmented representations (q, N, D), q >= 2.
    delta: Ridge added to S_w; the paper gives no value, 1e-4 is that of Skean et al. (2025).
    unbiased: Denominators n - 1 and n(q - 1), or n and nq.
    max_eigenvalues: Keep only the largest eigenvalues.

Returns:
    value: LiDAR; 0 when the LDA matrix has no positive eigenvalue (no sample separates).
    extras: entropy, n_positive_eigenvalues.
