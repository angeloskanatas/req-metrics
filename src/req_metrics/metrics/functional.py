"""Functional sensitivity: the effective rank of a layer readout's input-output Jacobian."""

from __future__ import annotations

import torch
from torch import Tensor

from req_metrics._types import InputKind, MetricResult, Preprocess
from req_metrics.registry import register_metric


def jacobian_effective_rank(sketch: Tensor) -> MetricResult:
    """Jacobian effective rank (JER) from a sketch of the input-output Jacobian.

    Chung and Kim (2026, arXiv:2602.03282, Eq. 1): with s_i the k leading singular values of
    the Jacobian J(x) of a readout at input x, JER(x) = (sum s_i)^2 / sum s_i^2, averaged over
    inputs; at most k. The singular values are those of the sketch, e.g. B = Q^T J from
    jacobian_products, which estimates them by randomized range finding as the paper does
    (Sec. 4.1; k = 32, 5 power iterations, 100 inputs). It counts the input directions a
    readout responds to, a property of the model's function rather than of the
    representation's geometry, and its preferred direction depends on the task.

    Args:
        sketch: (B, k, M) per input, k rows whose singular values estimate those of J(x).

    Returns:
        value: mean JER over inputs.
        extras: std over inputs, n_probes, n_inputs.
    """
    if sketch.ndim != 3 or sketch.shape[1] < 2:
        raise ValueError(f"expected (B, k, M) with k >= 2, got shape {tuple(sketch.shape)}")
    s = torch.linalg.svdvals(sketch.double())  # (B, k)
    jer = s.sum(dim=1) ** 2 / s.square().sum(dim=1)
    std = float(jer.std(unbiased=False)) if jer.numel() > 1 else 0.0
    return MetricResult(
        float(jer.mean()), {"std": std, "n_probes": float(sketch.shape[1]), "n_inputs": float(sketch.shape[0])}
    )


register_metric(
    "jacobian_effective_rank",
    inputs=InputKind.JACOBIAN,
    preprocess=Preprocess(),
    citation=("chung2026globalgeometry",),
    arxiv="2602.03282",
    description="Effective rank of a layer readout's input-output Jacobian (Chung and Kim, 2026).",
)(jacobian_effective_rank)
