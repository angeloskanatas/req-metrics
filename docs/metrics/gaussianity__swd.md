# `gaussianity/swd`

Sliced-Wasserstein shape distance from an isotropic Gaussian (VISReg); standardizes internally, so no registry preprocessing.

- Input: `points`
- Canonical preprocessing: `none`
- Tags: none
- Shared cache: none
- Origin: https://arxiv.org/abs/2606.02572
- Cite: `wu2026visreg` (VISReg: Variance-Invariance-Sketching Regularization for JEPA training (2026))

## Definition, protocol and pitfalls

Distance of the point cloud from an isotropic Gaussian via 1-D projections.

SIGReg from LeJEPA (Balestriero and LeCun, 2025, arXiv:2511.08544)
projects the embeddings onto random unit directions and compares each
univariate marginal with N(0, 1); by Cramer-Wold, matching every
marginal matches the joint. "epps_pulley" follows their reference
implementation exactly: the empirical characteristic function on a
17-point grid over [-5, 5], squared deviation from exp(-t^2/2) weighted
by the same Gaussian, trapezoid-integrated, then multiplied by N (Epps
and Pulley, 1983, Biometrika). Here the per-sample statistic (divided
by N) is returned so values are comparable across sample counts; the
LeJEPA-scale total is in the extras. The cloud is centered but not
rescaled, so on raw encoder features the statistic mixes scale with
shape; read layer trends within a run. "ks" replaces the statistic by
the Kolmogorov-Smirnov distance. "swd" follows VISReg (Wu et al., 2026,
arXiv:2606.02572, Algorithm 1), which separates the three effects:
center = mean squared coordinate of the mean, scale = mean squared
deviation of the per-dimension standard deviation from 1, shape = mean
squared 2-Wasserstein distance between the sorted projections of the
standardized cloud and the standard-normal quantiles i/(N+1); shape is
returned as the value and stays informative under scale drift.

Args:
    x: Points (N, D).
    method: "epps_pulley", "ks" or "swd".
    num_directions: Number of random unit directions (LeJEPA default 256).
    seed: Seed for the directions.
    center: Mean-center before projecting (ignored by "swd", which always does).

Returns:
    value: the statistic; lower is closer to an isotropic Gaussian.
    extras: epps_pulley_total for "epps_pulley"; center and scale for "swd".
