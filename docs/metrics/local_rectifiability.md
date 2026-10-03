# `local_rectifiability`

UR-JEPA beta-number flatness around an n-plane across dyadic scales (Eq. 24).

- Input: `points`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- Origin: https://arxiv.org/abs/2606.01443
- Cite: `le2026urjepa` (UR-JEPA: Uniform Rectifiability as a Regularizer for Joint-Embedding Predictive Architectures (2026))

## Definition, protocol and pitfalls

Multi-scale flatness of the cloud around an n-dimensional tangent plane.

The empirical beta-number of UR-JEPA (Le et al., 2026, arXiv:2606.01443, Eqs. 23-24): at
anchors x and dyadic scales r, the Gaussian-weighted variance orthogonal to the best-fit
affine n-plane, divided by r^2 and the neighborhood mass. Small and decaying with r means
locally flat. Without a target n, n defaults to the number of eigenvalues of the global
covariance above their mean. One local PCA per anchor and scale; use a few thousand points.

Args:
    x: Points (N, D).
    n: Tangent dimension, 1 <= n < D.
    n_anchors: Anchor points per scale.
    n_scales: Dyadic scales below the anchor diameter.
    chunk: Anchors per batch.
    seed: Anchor sampling seed.

Returns:
    value: beta_2 at the middle scale.
    extras: n, and beta2_scale{i}, trace_scale{i}, local_id_scale{i}, r_scale{i}, with
        scale 0 the coarsest.
