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
    gen = torch.Generator(device=like.device).manual_seed(seed)
    a = torch.randn(d, m, generator=gen, device=like.device, dtype=like.dtype)
    return a / a.norm(dim=0, keepdim=True)


def gaussianity(
    x: Tensor, *, method: str = "epps_pulley", num_directions: int = 256, seed: int = 0, center: bool = True
) -> MetricResult:
    """Distance of the point cloud from an isotropic Gaussian via 1-D projections.

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
    """Fraction of active entries and a Hoyer-type l1/l2 density ratio.

    Rectified LpJEPA (Kuang et al., 2026, arXiv:2602.01456, appendix):
    m_l0 = E[||x||_0] / D, the fraction of nonzero entries, 0 for all-zero
    vectors and 1 for fully dense ones; m_l1 = E[||x||_1^2 / ||x||_2^2] / D,
    which is 1/D for a one-hot vector and 1 for a dense vector with equal
    magnitudes. Pre-activation transformer states are dense, so m_l0 is
    informative only after a rectifying nonlinearity; m_l1 varies
    continuously and is the returned value.

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
    """Mean Euclidean norm of the representations, with its spread.

    Draganov et al. (2025, arXiv:2502.09252) show that although cosine-based
    self-supervised objectives embed on a hypersphere, the norms of the
    pre-normalization embeddings govern convergence rates and encode the
    network's confidence, with smaller norms on unexpected samples. Tracked
    per layer during training the mean norm is a convergence monitor; the
    coefficient of variation separates a few outlier clips from a uniform
    rescaling. Take the representation before any L2 normalization.

    Args:
        x: Representations, shape (N, D).

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
