# `gaussianity`

Distance from an isotropic Gaussian along random directions (Epps-Pulley, KS or VISReg shape).

- Input: `points`
- Canonical preprocessing: `center`
- Tags: none
- Shared cache: none
- Origin: https://arxiv.org/abs/2511.08544
- Cite: `balestriero2025lejepa` (LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics (2025)); `epps1983normality` (A test for normality based on the empirical characteristic function (1983)); `wu2026visreg` (VISReg: Variance-Invariance-Sketching Regularization for JEPA training (2026))

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
by N) is reported so values are comparable across sample counts; the
LeJEPA-scale total is in the extras. The cloud is centered but not
rescaled, so on raw encoder features the statistic mixes scale with
shape; read layer trends within a run. "ks" is the mean
Kolmogorov-Smirnov distance of the same projections from N(0, 1), one of
the univariate tests LeJEPA compares. "swd" follows VISReg (Wu et al.,
2026, arXiv:2606.02572, Algorithm 1), which separates three effects:
center = mean squared coordinate of the mean, scale = mean squared
deviation of the per-dimension standard deviation from 1, shape = mean
squared 2-Wasserstein distance between the sorted projections of the
standardized cloud and the standard-normal quantiles i/(N+1); shape stays
informative under scale drift. All three statistics use the same
directions and are in the extras of every call; method chooses the value.
LeJEPA argues that an isotropic Gaussian is the embedding distribution that
minimizes downstream prediction risk (their Sec. 3); their label-free model
selection (Sec. 6.2) uses the full training loss of LeJEPA runs, not this
statistic measured on other encoders.

Args:
    x: Points (N, D).
    method: "epps_pulley", "ks" or "swd" (the VISReg shape distance).
    num_directions: Number of random unit directions (LeJEPA default 256).
    seed: Seed for the directions.
    center: Mean-center before projecting for "epps_pulley" and "ks"; "swd" always
        standardizes.

Returns:
    value: the chosen statistic; lower is closer to an isotropic Gaussian.
    extras: epps_pulley, epps_pulley_total, ks, swd_shape, swd_center, swd_scale.
