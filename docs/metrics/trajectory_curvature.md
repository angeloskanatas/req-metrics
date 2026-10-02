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

Discrete curvature of a trajectory (Henaff, Goris and Simoncelli, 2019,
Nature Neuroscience; Hosseini and Fedorenko, 2023, NeurIPS): with
displacements v_t = z_{t+k} - z_t, the angle c_t = arccos(v_t . v_{t+k} /
|v_t||v_{t+k}|) in [0, pi], averaged over t. Zero for a straight
trajectory, invariant to the scale of the representation, and the
definition of Kanatas et al. (2026). Reference values for
k = 1: independent frames give 2pi/3 (120 degrees), a random walk pi/2,
and the transformer layers of Kanatas et al. (2026) 101 to 115 degrees.

convention="abs" folds the range to [0, pi/2] by taking the absolute
cosine, as the Skean et al. (2025, ICML) implementation does; because
consecutive displacements in audio encoders mostly point backwards, the
two conventions are near-perfect rank inversions of each other across
layers in Kanatas et al. (2026) (Spearman -0.91 to -0.98), whose correlation
analysis uses the folded variant negated. normalize="path_length" divides each
angle by the sum of the two displacement lengths (RECURVE, Shin et al.,
2024, NeurIPS, Definition 3.2), the turning rate per unit length used for
boundary detection; it is no longer scale-free and in Kanatas et al. (2026)
it removed the cross-layer signal. All three readings come from the same
angles and are in the extras of every call; convention and normalize only
choose the value. The extras also report the mean cosine itself, the
"straightness" maximized by Niu et al. (2024, NeurIPS) and Wang et al.
(2026, ICML).

Args:
    z: One clip's frames, shape (T, D), time-ordered; T >= 2k + 1.
    k: Frame gap of the displacement vectors; the angle spans 2k frames.
    convention: "signed" (angle in [0, pi]) or "abs" (folded to [0, pi/2]).
    normalize: "none" or "path_length".

Returns:
    value: mean curvature in radians (or radians per unit length).
    extras: signed and abs (both conventions, radians) with signed_degrees and abs_degrees,
        path_length_normalized (the RECURVE reading), mean_cos, n_angles, n_zero_steps.
