# `gaussianity`

Distance from an isotropic Gaussian along random directions (Epps-Pulley, KS or VISReg shape).

- Input: `points`
- Canonical preprocessing: `center`
- Tags: none
- Shared cache: none
- arXiv: https://arxiv.org/abs/2511.08544
- Cite: `balestriero2025lejepa` (LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics (2025)); `epps1983normality` (A test for normality based on the empirical characteristic function (1983)); `wu2026visreg` (VISReg: Variance-Invariance-Sketching Regularization for JEPA training (2026))

## Definition, protocol and pitfalls

Distance of the point cloud from an isotropic Gaussian along random 1-D projections.

All statistics use the same num_directions random directions; method chooses the value.
"epps_pulley" is SIGReg's statistic (LeJEPA; Balestriero and LeCun, 2025, arXiv:2511.08544): the
weighted squared deviation of the empirical characteristic function from exp(-t^2/2) on 17
points over [-5, 5], divided by N here so values compare across N. "ks" is the mean
Kolmogorov-Smirnov distance from N(0, 1). "swd" is the shape term of VISReg (Wu et al., 2026,
arXiv:2606.02572, Alg. 1) on the standardized cloud, insensitive to scale drift.

The value is a distance to this one target and does not score representation quality.
"epps_pulley" and "ks" compare the centered projections with N(0, 1) including their scale, so
an L2-normalized cloud, whose projections have variance about 1/D, scores far from zero;
standardize first or read "swd". At small D a cloud spread uniformly on the sphere also scores
away from zero, because its projections have compact support. The finite number of directions
adds estimator variance.

SIGReg tests the raw embeddings, mean included, while this function centers by default;
center=False gives the SIGReg value. The grid follows the LeJEPA paper (17 points on [-5, 5]),
whereas the released code integrates [0, 3] with doubled weights. On an exactly Gaussian cloud
epps_pulley_total is about 1.06 rather than 0, the finite-N bias term of LeJEPA's Theorem 6.

Args:
    x: Points (N, D).
    method: "epps_pulley", "ks" or "swd".
    num_directions: Random unit directions.
    seed: Seed of the directions.
    center: Mean-center before projecting ("swd" always standardizes).

Returns:
    value: the chosen statistic; lower is closer to an isotropic Gaussian.
    extras: epps_pulley, epps_pulley_total (times N), ks, swd_shape, swd_center, swd_scale.
