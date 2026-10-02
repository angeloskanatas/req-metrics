"""Metrics of one clip's frame trajectory (time-ordered rows)."""

from __future__ import annotations

import math
from functools import partial

import torch
from torch import Tensor

from req_metrics._types import InputKind, MetricResult, Preprocess
from req_metrics.registry import register_metric


def trajectory_curvature(z: Tensor, *, k: int = 1, convention: str = "signed", normalize: str = "none") -> MetricResult:
    """Mean turning angle between successive displacement vectors of a frame trajectory.

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
    it removed the cross-layer signal. The extras report the mean cosine itself, the
    "straightness" maximized by Niu et al. (2024, NeurIPS) and Wang et al.
    (2026, ICML).

    Args:
        z: One clip's frames, shape (T, D), time-ordered; T >= 2k + 1.
        k: Frame gap of the displacement vectors; the angle spans 2k frames.
        convention: "signed" (angle in [0, pi]) or "abs" (folded to [0, pi/2]).
        normalize: "none" or "path_length".

    Returns:
        value: mean curvature in radians (or radians per unit length).
        extras: degrees (value in degrees for normalize="none"), mean_cos, n_angles, n_zero_steps.
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
    if convention == "abs":
        angles = torch.arccos(cos.abs())
    elif convention == "signed":
        angles = torch.arccos(cos)
    else:
        raise ValueError(f"unknown convention {convention!r}")
    if normalize == "path_length":
        value = float((angles / (na[valid] + nb[valid])).mean())
        extras = {}
    elif normalize == "none":
        value = float(angles.mean())
        extras = {"degrees": math.degrees(value)}
    else:
        raise ValueError(f"unknown normalize {normalize!r}")
    extras.update(
        {"mean_cos": float(cos.mean()), "n_angles": float(valid.sum()), "n_zero_steps": float((~valid).sum())}
    )
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
register_metric(
    "trajectory_curvature/abs",
    inputs=_T,
    preprocess=Preprocess(),
    citation=("DBLP:conf/icml/SkeanAZPNLS25", "henaff2019perceptual"),
    arxiv="2502.02013",
    tags=("heldout-canonical",),
    description="Turning angle folded to [0, pi/2] by the absolute cosine (Skean implementation).",
)(partial(trajectory_curvature, convention="abs"))
register_metric(
    "trajectory_curvature/recurve",
    inputs=_T,
    preprocess=Preprocess(),
    citation=("shin2024recurve",),
    description="Turning angle per unit path length (RECURVE Definition 3.2).",
)(partial(trajectory_curvature, normalize="path_length"))
