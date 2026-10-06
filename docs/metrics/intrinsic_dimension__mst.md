# `intrinsic_dimension/mst`

Intrinsic dimension from the growth of minimum-spanning-tree length.

- Input: `points`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- arXiv: https://arxiv.org/abs/2606.03338
- Cite: `mordacq2026idest` (IdEst: Assessing Self-Supervised Learning Representations via Intrinsic Dimension (2026)); `costa2006mst` (Determining Intrinsic Dimension and Entropy of High-Dimensional Shape Spaces (2006))

## Definition, protocol and pitfalls

Costa and Hero (2006); IdEst (Mordacq et al., 2026, arXiv:2606.03338, Alg. 1). The MST
length of n points grows as n^((d-1)/d): log L is regressed on log n over subsamples of size
n_min, n_min + step, ... and d = 1 / (1 - slope). Each MST is O(n^2) in memory; use a few
thousand points.

Args:
    x: Points (N, D).
    n_min: Smallest subsample size (default N/16, at least 8).
    step: Increment between subsample sizes (default N/16).
    seed: Subsampling seed.

Returns:
    value: estimated dimension; nan if the slope leaves (0, 1).
    extras: slope, n_sizes.
