# `neighborhood_curvature`

Neighborhood curvature: mean cosine between the unit edges to the k nearest neighbors.

- Input: `points`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: neighbors_kw
- Origin: https://arxiv.org/abs/2511.17426
- Cite: `ghojogh2025curvssl` (Self-Supervised Learning by Curvature Alignment (2025))

## Definition, protocol and pitfalls

CurvSSL (Ghojogh et al., 2025, arXiv:2511.17426): per point, the mean pairwise cosine of the
unit vectors to its k neighbors, averaged over points; near 0 for isotropic neighborhoods,
toward 1 at boundaries. A Gaussian cloud scores about 0.2 at k = 32 in eight dimensions, so
compare layers or runs rather than reading the value against zero.

Args:
    x: Points (N, D).
    k: Neighborhood size.
    neighbors: Precomputed Neighbors table with at least k neighbors.

Returns:
    value: mean pairwise neighbor cosine.
    extras: none.
