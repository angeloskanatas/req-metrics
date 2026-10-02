# `normalized_std`

Mean per-channel standard deviation of the L2-normalized output.

- Input: `points`
- Canonical preprocessing: `l2`
- Tags: collapse-indicator
- Shared cache: none
- Origin: https://arxiv.org/abs/2011.10566
- Cite: `chen2021simsiam` (Exploring Simple Siamese Representation Learning (2021))

## Definition, protocol and pitfalls

Chen and He (2021, CVPR, arXiv:2011.10566, Sec. 4.1): the std over samples of
z / ||z||_2, averaged over channels, is 0 when all outputs collapse to one
vector and about 1 / sqrt(D) when z is a zero-mean isotropic Gaussian, so it
is the standard complete-collapse monitor of Siamese self-supervised
learning (lightly's std_of_l2_normalized). It does not see dimensional
collapse; pair it with a spectral metric. Sample std uses the N - 1
denominator, as in that implementation.

Args:
    x: Points (N, D).

Returns:
    value: mean channel std of the unit-norm rows.
    extras: reference 1 / sqrt(D) and the ratio value * sqrt(D) (about 1 for an isotropic cloud).
