# `neighborhood_curvature`

Neighborhood curvature: mean cosine between the unit edges to the k nearest neighbors.

- Input: `points`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: neighbors_kw
- arXiv: https://arxiv.org/abs/2511.17426
- Cite: `ghojogh2025curvssl` (Self-Supervised Learning by Curvature Alignment (2025))

## Definition, protocol and pitfalls

CurvSSL (Ghojogh et al., 2025, arXiv:2511.17426, Eq. 6): per point, the pairwise cosines of
the unit vectors to its k neighbors, summed in the paper and averaged over the k(k - 1)/2 pairs
here (the same ranking at fixed k), then averaged over points; near 0 for isotropic neighborhoods,
toward 1 at boundaries. A Gaussian cloud scores about 0.2 at k = 32 in eight dimensions, so
compare layers or runs rather than reading the value against zero. Points are processed in
chunks, so memory is bounded by chunk * k * D float64 values.

Args:
    x: Points (N, D); without neighbors, duplicates are removed first.
    k: Neighborhood size.
    neighbors: Precomputed Neighbors table of x with at least k neighbors, built on the distinct
        rows of x (duplicate rows give zero-length edges and bias the value).
    chunk: Points per batch; default keeps a batch near 2^25 values.

Returns:
    value: mean pairwise neighbor cosine.
    extras: none.
