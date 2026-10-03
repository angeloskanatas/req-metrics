# `intrinsic_dimension/gride`

GRIDE intrinsic dimension at the 8th-neighbor scale.

- Input: `points`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: neighbors
- Origin: https://arxiv.org/abs/2104.13832
- Cite: `denti2022gride` (The generalized ratios intrinsic dimension estimator (2022)); `glielmo2022dadapy` (DADApy: Distance-based Analysis of DAta-manifolds in Python (2022)); `DBLP:journals/corr/abs-1803-06992` (Estimating the intrinsic dimension of datasets by a minimal neighborhood
                  information (2017))

## Definition, protocol and pitfalls

GRIDE intrinsic dimension at doubling neighbor scales.

Denti et al. (2022, Scientific Reports, arXiv:2104.13832, Eq. 13): the maximum-likelihood
dimension from the ratio of the n2-th to the n1-th neighbor distance, at (n1, n2) = (k, 2k)
for k = 1, 2, 4, ... up to range_max. The value is the estimate at n2 = scale; scale 8 uses
the 8th and 4th neighbors. Bisection on the likelihood derivative and Fisher-information
errors follow DADApy.

Args:
    x: Points (N, D), or a Neighbors table with k >= range_max.
    scale: Outer neighbor rank of the returned estimate; a power of two.
    range_max: Largest outer rank.
    d0, d1: Bisection bounds on the dimension.
    eps: Bisection precision.

Returns:
    value: dimension at the requested scale.
    extras: id_rank{n2}, err_rank{n2}, r_rank{n2} for every computed scale.
