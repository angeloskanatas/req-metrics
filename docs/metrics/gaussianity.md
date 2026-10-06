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
"epps_pulley" is SIGReg's statistic (Balestriero and LeCun, 2025, arXiv:2511.08544): the
weighted squared deviation of the empirical characteristic function from exp(-t^2/2) on 17
points over [-5, 5], divided by N here so values compare across N. "ks" is the mean
Kolmogorov-Smirnov distance from N(0, 1). "swd" is the shape term of VISReg (Wu et al., 2026,
arXiv:2606.02572, Alg. 1) on the standardized cloud, insensitive to scale drift. LeJEPA
argues the isotropic Gaussian is optimal for downstream risk (Sec. 3); its model selection
(Sec. 6.2) uses the LeJEPA training loss, not this statistic on other encoders.

Args:
    x: Points (N, D).
    method: "epps_pulley", "ks" or "swd".
    num_directions: Random unit directions.
    seed: Seed of the directions.
    center: Mean-center before projecting ("swd" always standardizes).

Returns:
    value: the chosen statistic; lower is closer to an isotropic Gaussian.
    extras: epps_pulley, epps_pulley_total (times N), ks, swd_shape, swd_center, swd_scale.
