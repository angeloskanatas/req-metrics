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
compare layers or runs rather than reading the value against zero. Points are processed in
chunks, so memory is bounded by chunk * k * D float64 values.

Args:
    x: Points (N, D); without neighbors, duplicates are removed first.
    k: Neighborhood size.
    neighbors: Precomputed Neighbors table of x with at least k neighbors.
    chunk: Points per batch; default keeps a batch near 2^25 values.

Returns:
    value: mean pairwise neighbor cosine.
    extras: none.
