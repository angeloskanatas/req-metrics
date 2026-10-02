# `neighborhood_curvature`

Local bending of the cloud: mean cosine between unit edges to the k nearest neighbors.

- Input: `points`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: neighbors_kw
- Origin: https://arxiv.org/abs/2511.17426
- Cite: `ghojogh2025curvssl` (Self-Supervised Learning by Curvature Alignment (2025))

## Definition, protocol and pitfalls

The discrete curvature score of CurvSSL (Ghojogh et al., 2025,
arXiv:2511.17426): for each point, the unit vectors to its k nearest
neighbors and the mean of their pairwise cosines, averaged over points.
Near 0 when neighbors surround the point isotropically, toward 1 when
they lie to one side (a boundary or a sharp bend), negative when they
lie on opposite sides, as on a curve. A Gaussian cloud is not a null
case: its outer points see neighbors biased toward the center and score
around 0.2 at k = 32 in eight dimensions, so compare layers or runs
rather than reading the value against zero. A point-cloud quantity,
unrelated to the trajectory curvature of a frame sequence.

Args:
    x: Points (N, D).
    k: Neighborhood size.
    neighbors: Precomputed Neighbors table with at least k neighbors.

Returns:
    value: mean pairwise neighbor cosine.
    extras: none.
