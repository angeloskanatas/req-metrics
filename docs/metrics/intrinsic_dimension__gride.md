# `intrinsic_dimension/gride`

GRIDE intrinsic dimension at the 8th-neighbor scale.

- Input: `points`
- Canonical preprocessing: `none`
- Tags: paper-canonical
- Shared cache: neighbors
- Origin: https://arxiv.org/abs/2104.13832
- Cite: `denti2022gride` (The generalized ratios intrinsic dimension estimator (2022)); `glielmo2022dadapy` (DADApy: Distance-based Analysis of DAta-manifolds in Python (2022)); `DBLP:journals/corr/abs-1803-06992` (Estimating the intrinsic dimension of datasets by a minimal neighborhood
                  information (2017))

## Definition, protocol and pitfalls

GRIDE intrinsic dimension at doubling neighbor scales.

Denti et al. (2022, Scientific Reports; arXiv:2104.13832) generalize
TwoNN to the ratio mu = r_n2 / r_n1 of the n2-th to the n1-th neighbor
distance, whose density is d (mu^d - 1)^(n2-n1-1) / (mu^((n2-1)d+1)
B(n2-n1, n1)) (their Eq. 13). The estimator maximizes the likelihood
over all points at each scale (n1, n2) = (k, 2k) for k = 1, 2, 4, ... up
to range_max, which traces the dimension as a function of the
neighborhood size. This follows dadapy's reference implementation: the
same ratio filter, bisection on the likelihood derivative, and
Fisher-information standard error. The returned value is the estimate
whose outer rank n2 equals scale. Kanatas et al. (2026) report GRIDE profiles
qualitatively consistent with TwoNN; their protocol entry uses scale 8, the
ratio of the 8th to the 4th neighbor distance. Larger range_max
needs a Neighbors table with that many neighbors per point.

Args:
    x: Points (N, D), or a Neighbors table with k >= range_max.
    scale: Outer neighbor rank n2 whose estimate is the value; a power of two.
    range_max: Largest outer rank; log2(range_max) scales are computed.
    d0, d1: Bisection bounds on the dimension.
    eps: Bisection precision.

Returns:
    value: dimension at the requested scale.
    extras: id_rank{n2}, err_rank{n2}, r_rank{n2} for every computed scale.
