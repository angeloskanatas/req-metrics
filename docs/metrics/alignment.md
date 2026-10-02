# `alignment`

Alignment: mean distance between L2-normalized views of the same clip, to the power alpha.

- Input: `views`
- Canonical preprocessing: `l2`
- Tags: relational
- Shared cache: none
- Origin: https://arxiv.org/abs/2005.10242
- Cite: `wang2020uniformity` (Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere (2020))

## Definition, protocol and pitfalls

Wang and Isola (2020, ICML, arXiv:2005.10242, Sec. 4.1.1):
L_align = E ||f(x) - f(y)||^alpha over positive pairs, alpha = 2 in Wang and
Isola (2020) and their reference code; 0 for a perfectly aligned encoder, 2 for unrelated
unit vectors in high dimension. Averaged over all pairs of views and all
clips. The companion of uniformity: the two together are the asymptotic
form of the contrastive loss (Theorem 1), and encoders with low values of
both perform best in that study. With two views that are a sample and its
transformed copy, 1 - value / 2 is the cosine similarity that Plachouras et
al. (2025, IJCNN) report as an invariance score over a transformation's
parameter range.

Args:
    views: Augmented representations, shape (q, N, D), q >= 2; rows are L2-normalized here.
    alpha: Distance exponent.

Returns:
    value: L_align.
    extras: n_pairs of views averaged.
