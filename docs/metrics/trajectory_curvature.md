# `trajectory_curvature`

Mean turning angle of a frame trajectory, oriented angle in [0, pi].

- Input: `trajectory`
- Canonical preprocessing: `none`
- Tags: paper-canonical
- Shared cache: none
- Cite: `henaff2019perceptual` (Perceptual straightening of natural videos (2019)); `DBLP:conf/nips/HosseiniF23` (Large language models implicitly learn to straighten neural sentence
                  trajectories to construct a predictive representation of natural language (2023)); `kanatas2026goodlayer` (What Makes a Good Layer? Assessing the Layer-Wise Intrinsic Properties of Music Foundation Models (2026))

## Definition, protocol and pitfalls

Mean turning angle between successive displacement vectors of a frame trajectory.

Henaff, Goris and Simoncelli (2019, Nature Neuroscience); Hosseini and Fedorenko (2023,
NeurIPS). With v_t = z_{t+k} - z_t, c_t = arccos(v_t . v_{t+k} / |v_t| |v_{t+k}|) in [0, pi],
averaged over t: 0 for a straight trajectory, pi/2 for a random walk, 2pi/3 for independent
frames (k = 1). convention="abs" folds angles to [0, pi/2], as in the code of Skean et al. (2025); above
pi/2 the folded reading reverses the ranking, so the conventions are not comparable.
normalize="path_length" divides each angle by the two step lengths (RECURVE, Shin et al.,
2024, Def. 3.2) and is not scale-free. All readings are in the extras.

Args:
    z: One sample's frames (T, D), time-ordered; T >= 2k + 1.
    k: Frame gap of the displacements.
    convention: "signed" or "abs".
    normalize: "none" or "path_length".

Returns:
    value: mean curvature in radians (per unit length when normalized).
    extras: signed, abs, signed_degrees, abs_degrees, path_length_normalized, mean_cos,
        n_angles, n_zero_steps.
