"""Pitch-transposition equivariance (PTE) of a layer, scored on the circle of fifths.

Introduced in Kanatas et al. (2026, ISMIR, arXiv:2608.14819) as a frozen-
representation diagnostic adapted from the cross-power spectral density
objective of STONE (Kong et al., 2024, ISMIR); the algebra of that objective
is in Lostanlen et al. (2025, IEEE Signal Processing Letters), whose Theorem
III.2 shows that for omega coprime with 12 its global minima are exactly the
pairs of key-signature profiles related by a shift of k pitch classes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from req_metrics._types import InputKind, MetricResult, Preprocess
from req_metrics.registry import register_metric

OMEGA = 7  # circle of fifths: the DFT bin of the 12-bin key signature profile


class _ZTransform(nn.Module):
    """12-bin probability vector -> complex point Z(y) = sum_q y[q] exp(2 pi i omega q / 12)."""

    alpha: Tensor

    def __init__(self, omega: int = OMEGA):
        super().__init__()
        self.omega = omega / 12
        self.register_buffer("alpha", torch.exp(1j * 2 * torch.pi * self.omega * torch.arange(12)))

    def forward(self, y: Tensor) -> Tensor:
        """Complex projection of a pitch-class profile onto the circle of the chosen DFT bin."""
        return torch.matmul(torch.complex(y, torch.zeros_like(y)), self.alpha)


def _head(input_dim: int, hidden_units: Sequence[int]) -> nn.Sequential:
    if not hidden_units:
        return nn.Sequential(nn.Linear(input_dim, 12))
    layers, prev = [], input_dim
    for h in hidden_units:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, 12))
    return nn.Sequential(*layers)


class _Probe(nn.Module):
    """Probe from embeddings to a 12-bin softmax key profile, projected on the circle of fifths.

    The same head maps the clip and its transposition; the cross-power
    Z(y) Z(y')^* of the two projections is compared with exp(-2 pi i omega k / 12).
    The negative sign is the waveform pitch-shift-up convention of Kanatas et al.
    (2026); STONE's CQT crop shifts pitch classes the other way and uses the
    positive sign.
    """

    def __init__(self, input_dim: int, probe: str, hidden_units: Sequence[int], temperature: float):
        super().__init__()
        self.temperature = temperature
        hu = list(hidden_units) if probe == "mlp" else []
        self.head = _head(input_dim, hu)
        self.z = _ZTransform()
        if probe == "mlp":
            with torch.no_grad():
                for m in self.head.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        nn.init.zeros_(m.bias)
                last = self.head[-1]
                assert isinstance(last, nn.Linear)
                last.bias.data[0] = 2.0  # breaks the uniform-softmax fixed point (|Z| = 0, zero gradient)

    def profile(self, x: Tensor) -> Tensor:
        """Softmax pitch-class profile of the probe head."""
        return torch.softmax(self.head(x) / self.temperature, dim=-1)

    def cross_power(self, z_orig: Tensor, z_shifted: Tensor) -> Tensor:
        """Cross-power spectral density between the original and shifted circle projections."""
        return self.z(self.profile(z_orig)) * self.z(self.profile(z_shifted)).conj()

    def target(self, k: Tensor) -> Tensor:
        """Expected unit-modulus phase for a shift of k semitones."""
        return torch.exp(-1j * 2 * torch.pi * self.z.omega * k)

    def forward(self, z_orig: Tensor, z_shifted: Tensor, k: Tensor) -> Tensor:
        """Mean squared distance between the target phase and the measured cross-power."""
        return (self.target(k) - self.cross_power(z_orig, z_shifted)).abs().pow(2).mean()


def pte(
    z: Tensor,
    shifted: Mapping[int, Tensor],
    *,
    probe: str = "linear",
    score: str = "phase",
    hidden_units: Sequence[int] = (512, 256),
    temperature: float | None = None,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    batch_size: int = 256,
    patience: int = 15,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    device: torch.device | str | None = None,
) -> MetricResult:
    """Pitch-transposition equivariance of a layer: PTE = 1 - d/2 on held-out clips.

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
    decodable, about 0.29 means random phase. Protocol of Kanatas et al. (2026): a
    linear probe, 10,000 clips split 70/15/15 by clip before shifting, 11 nonzero
    shifts, Adam at 1e-3, up to 200 epochs with early stopping. The remaining
    defaults here are weight decay 1e-3, batches of 256 (clip, shift) pairs with
    mixed shifts, early stopping after 15 flat epochs and the best validation state
    restored. The MLP probe (probe="mlp") uses
    Xavier initialization, an asymmetric output bias and softmax temperature
    0.5, which break the uniform-softmax fixed
    point where the gradient vanishes (the mitigation of Theorem III.2's
    caveat that the objective's convexity does not transfer to network
    weights).

    Read PTE together with mean_abs_cpsd: a probe whose cross-power magnitude
    stays near zero never left the uniform softmax, and its phase error is then
    noise rather than a measurement. The metric measures transport along the pitch
    axis, not tonal content per se. The phase distance of the definition and the
    distance of the raw cross-power (the STONE loss, magnitude included) are both
    in the extras of every run; `score` chooses which one is the value.

    Args:
        z: Unshifted representations, shape (N, D).
        shifted: Map from semitone shift k (nonzero integers) to representations, each (N, D),
            rows aligned with z. Positive k means the audio was shifted up in pitch.
        probe: "linear" (paper) or "mlp".
        score: "phase" (paper) or "cpsd".
        hidden_units: MLP widths.
        temperature: Softmax temperature; default 0.5 for the MLP probe, 1.0 for the linear probe.
        epochs, lr, weight_decay, batch_size, patience: Training protocol.
        val_fraction, test_fraction: Clip-level split fractions.
        seed: Split and initialization seed.
        device: Torch device; default CPU.

    Returns:
        value: PTE from the chosen distance.
        extras: phase_rmse, cpsd_rmse, mean_abs_cpsd, val_rmse,
            train_rmse, epochs_trained, rmse_k{k} per shift, n_train, n_test.
    """
    if z.ndim != 2:
        raise ValueError(f"expected (N, D), got shape {tuple(z.shape)}")
    ks = sorted(int(k) for k in shifted)
    if not ks or any(k == 0 for k in ks):
        raise ValueError("shifted must map nonzero semitone shifts to (N, D) tensors")
    for k in ks:
        if tuple(shifted[k].shape) != tuple(z.shape):
            raise ValueError(f"shifted[{k}] has shape {tuple(shifted[k].shape)}, expected {tuple(z.shape)}")
    if probe not in ("linear", "mlp") or score not in ("phase", "cpsd"):
        raise ValueError("probe in {linear, mlp}, score in {phase, cpsd}")
    if temperature is None:
        temperature = 0.5 if probe == "mlp" else 1.0
    dev = torch.device(device) if device is not None else torch.device("cpu")

    n, d = z.shape
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_test, n_val = max(int(test_fraction * n), 1), max(int(val_fraction * n), 1)
    test_idx, val_idx, train_idx = perm[:n_test], perm[n_test : n_test + n_val], perm[n_test + n_val :]
    n_train = len(train_idx)
    if n_train < batch_size:
        raise ValueError(f"{n_train} training clips are fewer than the batch size {batch_size}")

    def take(idx):
        return torch.as_tensor(z[idx], dtype=torch.float32, device=dev)

    def take_shifted(idx):
        return torch.stack(
            [torch.as_tensor(shifted[k][idx], dtype=torch.float32, device=dev) for k in ks]
        )  # (n_k, len(idx), D)

    z_tr, z_va, z_te = take(train_idx), take(val_idx), take(test_idx)
    s_tr, s_va, s_te = take_shifted(train_idx), take_shifted(val_idx), take_shifted(test_idx)
    k_tensor = torch.tensor(ks, dtype=torch.float32, device=dev)
    n_k = len(ks)

    torch.manual_seed(seed)
    model = _Probe(d, probe, hidden_units, temperature).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def split_loss(z_part: Tensor, s_part: Tensor) -> float:
        return (
            sum(
                model(z_part, s_part[i], torch.full((z_part.shape[0],), float(k), device=dev)).item()
                for i, k in enumerate(ks)
            )
            / n_k
        )

    best_val, best_state, flat, last = float("inf"), None, 0, 0
    n_pairs = n_train * n_k
    for epoch in range(epochs):
        last = epoch
        model.train()
        order = torch.randperm(n_pairs, device=dev)  # mixed shifts within every batch
        for i in range(0, n_pairs, batch_size):
            flat_idx = order[i : i + batch_size]
            s_idx, k_idx = flat_idx % n_train, flat_idx // n_train
            loss = model(z_tr[s_idx], s_tr[k_idx, s_idx], k_tensor[k_idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = split_loss(z_va, s_va)
        if val < best_val:
            best_val, flat = val, 0
            best_state = {name: p.clone() for name, p in model.state_dict().items()}
        else:
            flat += 1
            if flat >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    extras: dict[str, float] = {}
    with torch.no_grad():
        phase_mse = cpsd_mse = mag = 0.0
        for i, k in enumerate(ks):
            kk = torch.full((n_test,), float(k), device=dev)
            cp = model.cross_power(z_te, s_te[i])
            tgt = model.target(kk)
            mse_k = (tgt - cp).abs().pow(2).mean().item()
            cpsd_mse += mse_k
            extras[f"rmse_k{k}"] = math.sqrt(mse_k)
            unit = cp / (cp.abs() + 1e-8)
            phase_mse += (tgt - unit).abs().pow(2).mean().item()
            mag += cp.abs().mean().item()
        phase_rmse, cpsd_rmse = math.sqrt(phase_mse / n_k), math.sqrt(cpsd_mse / n_k)
        extras.update(
            {
                "phase_rmse": phase_rmse,
                "cpsd_rmse": cpsd_rmse,
                "mean_abs_cpsd": mag / n_k,
                "val_rmse": math.sqrt(split_loss(z_va, s_va)),
                "train_rmse": math.sqrt(split_loss(z_tr, s_tr)),
                "epochs_trained": float(last + 1),
                "n_train": float(n_train),
                "n_test": float(n_test),
            }
        )
    d_score = phase_rmse if score == "phase" else cpsd_rmse
    return MetricResult(1.0 - d_score / 2.0, extras)


_S = InputKind.SHIFTED
register_metric(
    "pte",
    inputs=_S,
    preprocess=Preprocess(),
    citation=("kanatas2026goodlayer", "DBLP:conf/ismir/KongLMWLH24", "DBLP:journals/spl/LostanlenKMLH25"),
    arxiv="2608.14819",
    tags=("paper-canonical",),
    description="Pitch-transposition equivariance, linear probe, phase distance (Kanatas et al., 2026).",
)(pte)
