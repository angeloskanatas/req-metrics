# `pte/cpsd`

PTE scored by the complex cross-power distance including magnitude (used for MusicGen-L).

- Input: `shifted`
- Canonical preprocessing: `none`
- Tags: paper-figure-config
- Shared cache: none
- Origin: https://arxiv.org/abs/2608.14819
- Cite: `kanatas2026goodlayer` (What Makes a Good Layer? Assessing the Layer-Wise Intrinsic Properties of Music Foundation Models (2026)); `DBLP:conf/ismir/KongLMWLH24` (STONE: Self-Supervised Tonality Estimator (2024))

## Definition, protocol and pitfalls

Pitch-transposition equivariance of a layer: PTE = 1 - d/2 on held-out clips.

A probe h maps a clip's representation to a 12-bin softmax key profile,
projected onto the circle of fifths by the DFT at omega = 7. Applied
independently to a clip and its k-semitone audio transposition, the
cross-power of the two projections should have phase -2 pi omega k / 12
for an equivariant layer; the probe is trained toward that target over
all shifts, and scored on held-out clips by d, the root-mean-square
chordal distance between the unit-normalized cross-power and its target
(score="phase", the definition of Kanatas et al. (2026)), or by the distance of the
raw cross-power including its magnitude (score="cpsd"). PTE = 1 - d/2 in
[0, 1]: 1 means transposition-equivariant tonal content is linearly
decodable, about 0.29 means random phase. Protocol of Kanatas et al. (2026): 10,000
clips split 70/15/15 by clip before shifting, 11 nonzero shifts, Adam at
1e-3 with weight decay 1e-3, batches of 256 (clip, shift) pairs with
mixed shifts, up to 200 epochs with early stopping on the validation loss
after 15 flat epochs, best validation state restored. The MLP probe uses
Xavier initialization, and for the shared method an asymmetric output
bias and softmax temperature 0.5, which break the uniform-softmax fixed
point where the gradient vanishes (the mitigation of Theorem III.2's
caveat that the objective's convexity does not transfer to network
weights).

Read PTE together with mean_abs_cpsd: a probe whose cross-power magnitude
stays near zero never trained, and its layer variation then correlates
with any well-layered task; a working threshold of 0.3 separates trained from
untrained probes on music encoders. The shared method measures transport along the pitch
axis, not tonal content per se. About 2,000 clips are not enough at
omega = 7; 10,000 to 20,000 are.

Args:
    z: Unshifted representations, shape (N, D).
    shifted: Map from semitone shift k (nonzero integers) to representations, each (N, D),
        rows aligned with z. Positive k means the audio was shifted up in pitch.
    method: "shared" (Kanatas et al., 2026) or "concat" (paired-probe decodability, not in that paper).
    probe: "linear" (paper) or "mlp".
    score: "phase" (paper) or "cpsd".
    hidden_units: MLP widths.
    temperature: Softmax temperature; default 0.5 for shared MLP, 1.0 otherwise.
    epochs, lr, weight_decay, batch_size, patience: Training protocol.
    val_fraction, test_fraction: Clip-level split fractions.
    seed: Split and initialization seed.
    device: Torch device; default CPU.

Returns:
    value: PTE from the chosen distance.
    extras: phase_rmse, cpsd_rmse, mean_abs_cpsd (or mean_abs_z_pred for concat), val_rmse,
        train_rmse, epochs_trained, rmse_k{k} per shift, n_train, n_test.
