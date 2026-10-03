# `intrinsic_dimension/mst`

Intrinsic dimension from the scaling of minimum-spanning-tree length.

- Input: `points`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- Origin: https://arxiv.org/abs/2606.03338
- Cite: `mordacq2026idest` (IdEst: Assessing Self-Supervised Learning Representations via Intrinsic Dimension (2026)); `costa2006mst` (Determining Intrinsic Dimension and Entropy of High-Dimensional Shape Spaces (2006))

## Definition, protocol and pitfalls

The MST dimension of Costa and Hero (2006), used by IdEst (Mordacq et
al., 2026, arXiv:2606.03338) as a probe-free proxy for linear-probe
accuracy. The total MST length of an n-point sample grows as
n^((d-1)/d); IdEst's Algorithm 1 draws subsamples of size n_min,
n_min + step, ... below N, computes each MST length, regresses log L on
log n, and returns d = 1 / (1 - slope). The MST length equals the total
zero-dimensional persistence, so this is also the PH_0 dimension. Each
MST is O(n^2) in memory, so IdEst caps N at 50,000 and this
implementation expects a few thousand points at most.

Args:
    x: Points (N, D).
    n_min: Smallest subsample size (default N/16, at least 8).
    step: Increment between subsample sizes (default N/16).
    seed: Subsampling seed.

Returns:
    value: estimated dimension; nan if the slope leaves (0, 1).
    extras: slope, n_sizes.
