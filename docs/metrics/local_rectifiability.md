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

The empirical beta-number of UR-JEPA (Le et al., 2026, arXiv:2606.01443,
Eq. 23-24): at each anchor x and dyadic scale r_k = 2^-k r_max, the
Gaussian-weighted centered scatter matrix S_r(x) with weights
exp(-|z - x|^2 / 2r^2), and beta_2(x, r) = (1/r^2) sum_{j>n} sigma_j^2 /
sum_j w_r(z_j - x): the kernel-weighted variance orthogonal to the
best-fit affine n-plane, normalized by the neighborhood mass. UR-JEPA
minimizes it as a regularizer toward a uniformly n-rectifiable measure;
here it is read as a diagnostic.
Small and decaying with r means locally flat and n-dimensional; large
and flat across scales means isotropic; near zero together with a
near-zero scatter trace means collapse. The count of eigenvalues above
their mean is a per-scale local-dimension estimate. Cost is a local PCA
per anchor and scale, so subsample to a few thousand points. UR-JEPA
fixes n as the target dimension of its regularizer; as a diagnostic with
no target, n defaults to the number of eigenvalues of the global centered
covariance above their mean (a participation count), recorded in the
extras, so profiles across layers of different width stay comparable.

Args:
    x: Points (N, D).
    n: Tangent dimension tested, 1 <= n < D; default as described above.
    n_anchors: Anchor points per scale.
    n_scales: Dyadic scales below the anchor diameter.
    chunk: Anchors per batch.
    seed: Anchor sampling seed.

Returns:
    value: beta_2 at the middle scale.
    extras: n (tangent dimension used), beta2_scale{i}, trace_scale{i}, local_id_scale{i}, r_scale{i};
        scale 0 is the coarsest.
