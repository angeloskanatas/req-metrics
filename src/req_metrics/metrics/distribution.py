"""Distribution of a point cloud: distance from an isotropic Gaussian, sparsity and norms."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from req_metrics._types import InputKind, MetricResult, Preprocess
from req_metrics.preprocess import apply_preprocess
from req_metrics.registry import register_metric

_EP_GRID = (-5.0, 5.0, 17)  # LeJEPA reference implementation (Algorithm 1)


def _directions(d: int, m: int, seed: int, like: Tensor) -> Tensor:
    gen = torch.Generator().manual_seed(seed)  # CPU stream: the same directions on every device
    a = torch.randn(d, m, generator=gen, dtype=like.dtype).to(like.device)
    return a / a.norm(dim=0, keepdim=True)


def gaussianity(
    x: Tensor, *, method: str = "epps_pulley", num_directions: int = 256, seed: int = 0, center: bool = True
) -> MetricResult:
    """Distance of the point cloud from an isotropic Gaussian along random 1-D projections.

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
    """
    if method not in ("epps_pulley", "ks", "swd"):
        raise ValueError(f"unknown method {method!r}")
    if x.ndim != 2 or x.shape[0] < 4:
        raise ValueError(f"expected (N, D) with N >= 4, got shape {tuple(x.shape)}")
    x = x.double()
    n, d = x.shape
    dirs = _directions(d, num_directions, seed, x)
    u = apply_preprocess(x, Preprocess(center=center)) @ dirs  # (N, M)
    t = torch.linspace(*_EP_GRID, dtype=x.dtype, device=x.device)
    target = torch.exp(-0.5 * t**2)
    re = torch.stack([torch.cos(u * tk).mean(dim=0) for tk in t], dim=1)  # (M, T)
    im = torch.stack([torch.sin(u * tk).mean(dim=0) for tk in t], dim=1)
    ep = float(torch.trapezoid(((re - target) ** 2 + im**2) * target, t, dim=1).mean())
    us = u.sort(dim=0).values
    cdf = 0.5 * (1.0 + torch.erf(us / math.sqrt(2.0)))
    rank = torch.arange(1, n + 1, dtype=x.dtype, device=x.device).unsqueeze(1) / n
    ks = float(torch.maximum(rank - cdf, cdf - (rank - 1.0 / n)).max(dim=0).values.mean())
    mu = x.mean(dim=0)
    xc = x - mu
    std = xc.std(dim=0, unbiased=False).clamp_min(1e-8)
    p = ((xc / std) @ dirs).sort(dim=0).values
    q = torch.arange(1, n + 1, dtype=x.dtype, device=x.device) / (n + 1)
    shape = float((p - (math.sqrt(2.0) * torch.erfinv(2 * q - 1)).unsqueeze(1)).square().mean())
    extras = {
        "epps_pulley": ep,
        "epps_pulley_total": ep * n,
        "ks": ks,
        "swd_shape": shape,
        "swd_center": float(mu.square().mean()),
        "swd_scale": float((1.0 - std).square().mean()),
    }
    return MetricResult({"epps_pulley": ep, "ks": ks, "swd": shape}[method], extras)


def sparsity(x: Tensor) -> MetricResult:
    """Sparsity: Hoyer-type l1/l2 density and the fraction of active entries.

    Kuang et al. (2026, arXiv:2602.01456, appendix): m_l1 = E[||x||_1^2 / ||x||_2^2] / D, from
    1/D for a one-hot vector to 1 for equal magnitudes; m_l0 = E[||x||_0] / D, informative only
    after a rectifying nonlinearity.

    Args:
        x: Points (N, D).

    Returns:
        value: m_l1.
        extras: m_l0.
    """
    if x.ndim != 2:
        raise ValueError(f"expected (N, D), got shape {tuple(x.shape)}")
    x = x.double()
    d = x.shape[1]
    m_l0 = float((x != 0).double().sum(dim=1).mean() / d)
    l1 = x.abs().sum(dim=1)
    l2 = x.square().sum(dim=1)
    valid = l2 > 0
    if not valid.any():
        raise ValueError("all rows are zero")
    m_l1 = float((l1[valid] ** 2 / l2[valid]).mean() / d)
    return MetricResult(m_l1, {"m_l0": m_l0})


def embedding_norm(x: Tensor) -> MetricResult:
    """Mean Euclidean norm of the representations.

    Draganov et al. (2025, arXiv:2502.09252): pre-normalization norms govern convergence and
    shrink on unexpected samples. Take the representation before any L2 normalization.

    Args:
        x: Representations (N, D).

    Returns:
        value: mean norm.
        extras: std, median, min, max, cv (std over mean).
    """
    if x.ndim != 2 or x.shape[0] < 1:
        raise ValueError(f"expected (N, D), got shape {tuple(x.shape)}")
    norms = x.double().norm(dim=1)
    mean = float(norms.mean())
    std = float(norms.std(unbiased=False)) if norms.numel() > 1 else 0.0
    return MetricResult(
        mean,
        {
            "std": std,
            "median": float(norms.median()),
            "min": float(norms.min()),
            "max": float(norms.max()),
            "cv": std / mean if mean > 0 else float("nan"),
        },
    )


_P = InputKind.POINTS
register_metric(
    "gaussianity",
    inputs=_P,
    preprocess=Preprocess(center=True),
    citation=("balestriero2025lejepa", "epps1983normality", "wu2026visreg"),
    arxiv="2511.08544",
    description="Distance from an isotropic Gaussian along random directions (Epps-Pulley, KS or VISReg shape).",
)(gaussianity)
register_metric("sparsity", inputs=_P, preprocess=Preprocess(), citation=("kuang2026lpjepa",), arxiv="2602.01456")(
    sparsity
)
register_metric(
    "embedding_norm",
    inputs=InputKind.POINTS,
    preprocess=Preprocess(),
    citation=("draganov2025norms",),
    arxiv="2502.09252",
)(embedding_norm)
