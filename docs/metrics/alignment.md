# `alignment`

Alignment: mean distance between L2-normalized views of the same clip, to the power alpha.

- Input: `views`
- Canonical preprocessing: `l2`
- Tags: relational
- Shared cache: none
- Origin: https://arxiv.org/abs/2005.10242
- Cite: `wang2020uniformity` (Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere (2020))

## Definition, protocol and pitfalls

Wang and Isola (2020, ICML, arXiv:2005.10242, Sec. 4.1.1), alpha = 2: 0 for perfectly aligned
views, 2 for unrelated unit vectors in high dimension; averaged over all view pairs and clips.

Args:
    views: Augmented representations (q, N, D), q >= 2.
    alpha: Distance exponent.

Returns:
    value: alignment.
    extras: n_pairs.
