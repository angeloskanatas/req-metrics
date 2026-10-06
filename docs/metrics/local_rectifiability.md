# `local_rectifiability`

UR-JEPA beta-number flatness around an n-plane across dyadic scales (Eq. 24).

- Input: `points`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- arXiv: https://arxiv.org/abs/2606.01443
- Cite: `le2026urjepa` (UR-JEPA: Uniform Rectifiability as a Regularizer for Joint-Embedding Predictive Architectures (2026))

## Definition, protocol and pitfalls

Multi-scale flatness of the cloud around an n-dimensional tangent plane.

The empirical beta-number of UR-JEPA (Le et al., 2026, arXiv:2606.01443, Eqs. 23-24): at
anchors x and dyadic scales r_k = 2^-k r_max, the Gaussian-weighted variance orthogonal to the
best-fit affine n-plane, divided by r^2 and the neighborhood mass. Small and decaying with r
means locally flat; near zero together with the trace means collapse. r_max is the largest
anchor-to-point distance. The paper fixes n (Sec. 6.2); the default D / 8, at least 4,
depends on the width only, so layers of equal width are compared at the same n. One local
PCA per anchor and scale, in float32 with TF32 matmuls disabled; use a few thousand points.

Args:
    x: Points (N, D).
    n: Tangent dimension, 1 <= n < D.
    n_anchors: Anchor points per scale.
    n_scales: Dyadic scales, the largest r_max.
    chunk: Anchors per batch.
    seed: Anchor sampling seed.

Returns:
    value: beta_2 squared (Eq. 24) at scale index (n_scales - 1) // 2, r_max / 4 with the defaults.
    extras: n, and beta2_scale{i}, trace_scale{i}, local_id_scale{i} (eigenvalues of the
        local scatter above their mean), r_scale{i}, with scale 0 the coarsest.
