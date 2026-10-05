"""Metrics of one sample's frame trajectory (time-ordered rows)."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from req_metrics._types import InputKind, MetricResult, Preprocess
from req_metrics.registry import register_metric


def trajectory_curvature(z: Tensor, *, k: int = 1, convention: str = "signed", normalize: str = "none") -> MetricResult:
    """Mean turning angle between successive displacement vectors of a frame trajectory.

    Henaff, Goris and Simoncelli (2019, Nature Neuroscience); Hosseini and Fedorenko (2023,
    NeurIPS). With v_t = z_{t+k} - z_t, c_t = arccos(v_t . v_{t+k} / |v_t| |v_{t+k}|) in [0, pi],
    averaged over t: 0 for a straight trajectory, pi/2 for a random walk, 2pi/3 for independent
    frames (k = 1). convention="abs" folds angles to [0, pi/2], as in Skean et al. (2025); above
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
    """
    if z.ndim != 2:
        raise ValueError(f"expected (T, D), got shape {tuple(z.shape)}")
    if k < 1:
        raise ValueError("k must be >= 1")
    if z.shape[0] < 2 * k + 1:
        raise ValueError(f"need at least {2 * k + 1} frames for k={k}, got {z.shape[0]}")
    zd = z.double()
    v = zd[k:] - zd[:-k]
    a, b = v[:-k], v[k:]
    na, nb = a.norm(dim=1), b.norm(dim=1)
    valid = (na > 0) & (nb > 0)
    if valid.sum() < 1:
        raise ValueError("every displacement pair has a zero-length step")
    dots = (a[valid] * b[valid]).sum(dim=1)
    cos = (dots / (na[valid] * nb[valid])).clamp(-1.0, 1.0)
    if convention not in ("signed", "abs"):
        raise ValueError(f"unknown convention {convention!r}")
    if normalize not in ("none", "path_length"):
        raise ValueError(f"unknown normalize {normalize!r}")
    signed, folded, lengths = torch.arccos(cos), torch.arccos(cos.abs()), na[valid] + nb[valid]
    angles = signed if convention == "signed" else folded
    value = float((angles / lengths).mean()) if normalize == "path_length" else float(angles.mean())
    extras = {
        "signed": float(signed.mean()),
        "abs": float(folded.mean()),
        "signed_degrees": math.degrees(float(signed.mean())),
        "abs_degrees": math.degrees(float(folded.mean())),
        "path_length_normalized": float((angles / lengths).mean()),
        "mean_cos": float(cos.mean()),
        "n_angles": float(valid.sum()),
        "n_zero_steps": float((~valid).sum()),
    }
    if normalize == "none":
        extras["degrees"] = math.degrees(value)
    return MetricResult(value, extras)


_T = InputKind.TRAJECTORY
register_metric(
    "trajectory_curvature",
    inputs=_T,
    preprocess=Preprocess(),
    citation=("henaff2019perceptual", "DBLP:conf/nips/HosseiniF23", "kanatas2026goodlayer"),
    tags=("paper-canonical",),
    description="Mean turning angle of a frame trajectory, oriented angle in [0, pi].",
)(trajectory_curvature)
