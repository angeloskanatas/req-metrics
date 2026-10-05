# `intrinsic_dimension/twonn`

TwoNN intrinsic dimension (Facco et al.).

- Input: `points`
- Canonical preprocessing: `none`
- Tags: paper-canonical
- Shared cache: neighbors
- Origin: https://arxiv.org/abs/1803.06992
- Cite: `DBLP:journals/corr/abs-1803-06992` (Estimating the intrinsic dimension of datasets by a minimal neighborhood
                  information (2017)); `glielmo2022dadapy` (DADApy: Distance-based Analysis of DAta-manifolds in Python (2022))

## Definition, protocol and pitfalls

TwoNN intrinsic dimension from the ratio of second- to first-neighbor distance.

Facco et al. (2017, Scientific Reports): under local uniformity mu = r_2 / r_1 is Pareto with
shape d. "base" fits -log(1 - F(mu)) = d log mu through the origin on the lowest mu_fraction
of the ratios (the paper discards the top 10 percent); "ml" is d = (N - 1) / sum(log mu).
Exact duplicate rows are removed first, which DADApy does not do by default; see
extras["n_used"].

Args:
    x: Points (N, D), or a Neighbors table of the distinct points with k >= 2.
    mu_fraction: Fraction of the smallest ratios kept in the fit.
    algorithm: "base" or "ml".

Returns:
    value: estimated dimension.
    extras: r (mean distance to the first two neighbors), n_used.
