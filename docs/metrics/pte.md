# `pte`

Pitch-transposition equivariance, linear probe, phase distance (Kanatas et al., 2026).

- Input: `shifted`
- Canonical preprocessing: `none`
- Tags: paper-canonical
- Shared cache: none
- arXiv: https://arxiv.org/abs/2608.14819
- Cite: `kanatas2026goodlayer` (What Makes a Good Layer? Assessing the Layer-Wise Intrinsic Properties of Music Foundation Models (2026)); `DBLP:conf/ismir/KongLMWLH24` (STONE: Self-Supervised Tonality Estimator (2024)); `DBLP:journals/spl/LostanlenKMLH25` (Understanding Equivariant Self-Supervised Learning in Musical Pitch
                  Class Space (2025))

## Definition, protocol and pitfalls

Pitch-transposition equivariance: PTE = 1 - d/2 on held-out clips; the paper reports -d, the same
ranking.

A probe maps each representation to a 12-bin softmax key profile, projected on the circle of
fifths (DFT bin omega = 7). The probe is trained on (clip, k-semitone transposition) pairs so that the cross-power of the two
projections has phase -2 pi omega k / 12. It is scored by d, the RMS chordal distance of the unit-normalized
cross-power to that target (score="phase"), or of the raw cross-power (score="cpsd"). 1 means equivariant tonal content is linearly decodable,
about 0.29 random phase. Clips are split before shifting. Read with mean_abs_cpsd: a probe
whose cross-power stays near zero never left the uniform softmax. The MLP probe uses Xavier
initialization, an asymmetric output bias and temperature 0.5 to leave that fixed point.

Args:
    z: Representations of the original clips (N, D).
    shifted: Semitone shift k (nonzero) -> representations (N, D), rows aligned with z;
        positive k raises the pitch; reversing every sign leaves the score unchanged, since a
        reflection of the key bins conjugates the target phase and the probe absorbs it.
    probe: "linear" or "mlp".
    score: "phase" or "cpsd".
    center: Subtract the training-split mean of the originals from the originals and every
        shifted copy before probing; the function class is unchanged for a probe with a bias,
        the optimisation is not.
    hidden_units: MLP widths.
    temperature: Softmax temperature; default 0.5 for the MLP probe, 1.0 for the linear one.
    epochs, lr, weight_decay, batch_size, patience: Training protocol.
    val_fraction, test_fraction: Clip-level split fractions.
    seed: Split and initialization seed.
    device: Torch device; default CPU.

Returns:
    value: PTE from the chosen distance.
    extras: phase_rmse, cpsd_rmse, mean_abs_cpsd, val_rmse, train_rmse, epochs_trained,
        rmse_k{k} per shift, n_train, n_test.
