"""Embedding norms of a point cloud."""

from __future__ import annotations

from torch import Tensor

from req_metrics._types import InputKind, MetricResult, Preprocess
from req_metrics.registry import register_metric


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


register_metric(
    "embedding_norm",
    inputs=InputKind.POINTS,
    preprocess=Preprocess(),
    citation=("draganov2025norms",),
    arxiv="2502.09252",
)(embedding_norm)
