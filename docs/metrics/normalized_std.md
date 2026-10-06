# `normalized_std`

Mean per-channel standard deviation of the L2-normalized output.

- Input: `points`
- Canonical preprocessing: `l2`
- Tags: relational, collapse-indicator
- Shared cache: none
- arXiv: https://arxiv.org/abs/2011.10566
- Cite: `chen2021simsiam` (Exploring Simple Siamese Representation Learning (2021))

## Definition, protocol and pitfalls

Chen and He (2021, CVPR, arXiv:2011.10566, Sec. 4.1): 0 under complete collapse and about
1 / sqrt(D) for an isotropic Gaussian. It does not detect dimensional collapse.

Args:
    x: Points (N, D).

Returns:
    value: mean channel std of the unit-norm rows (N - 1 denominator).
    extras: reference 1 / sqrt(D) and the ratio value * sqrt(D).
