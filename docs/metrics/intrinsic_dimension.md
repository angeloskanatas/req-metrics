# `intrinsic_dimension`

TwoNN intrinsic dimension (Facco et al.).

- Input: `points`
- Canonical preprocessing: `none`
- Tags: paper-canonical
- Shared cache: none
- Origin: https://arxiv.org/abs/1803.06992
- Cite: `DBLP:journals/corr/abs-1803-06992` (Estimating the intrinsic dimension of datasets by a minimal neighborhood
                  information (2017)); `glielmo2022dadapy` (DADApy: Distance-based Analysis of DAta-manifolds in Python (2022))

## Definition, protocol and pitfalls

TwoNN intrinsic dimension from the ratio of second- to first-neighbor distance.

Facco et al. (2017, Scientific Reports): under local uniformity the ratio
mu = r_2 / r_1 is Pareto with shape d, so -log(1 - F(mu)) = d log mu. The
"base" algorithm sorts the ratios, keeps the lowest mu_fraction (the
paper discards the top 10 percent as unstable), sets the empirical CDF
to i/N, and fits a line through the origin by least squares; "ml" is the
closed-form maximum likelihood d = (N - 1) / sum(log mu). Exact
duplicate rows are removed first (DADApy keeps them unless
remove_identical_points is called; a duplicate gives r_1 = 0 and an
infinite ratio, and its neighbors a ratio of 1). The choice matters: on a
corpus with 13 duplicate clips among 990 the estimate differs by about 20
percent between the two conventions. extras["n_used"] records the count
after removal.
The intrinsic-dimension estimator of Kanatas et al. (2026).

Args:
    x: Points (N, D).
    mu_fraction: Fraction of the smallest ratios kept in the fit.
    algorithm: "base" (linear fit) or "ml" (maximum likelihood).

Returns:
    value: estimated dimension.
    extras: r (mean distance to the first two neighbors), n_used.
