"""Functional sensitivity: the effective rank of a layer readout's input-output Jacobian."""

from __future__ import annotations

import torch
from torch import Tensor

from req_metrics._types import InputKind, MetricResult, Preprocess
from req_metrics.registry import register_metric


def jacobian_effective_rank(jvps: Tensor) -> MetricResult:
    """Jacobian effective rank (JER), from Jacobian-vector products.

    Chung and Kim (2026, arXiv:2602.03282, Eq. 1 and App. D): with s_i the singular
    values of the products J(x) v_1, ..., J(x) v_k of the Jacobian at input x with k
    random orthonormal input directions, JER(x) = (sum s_i)^2 / sum s_i^2, averaged over
    inputs; at most k. It counts the input directions a readout responds to, a property
    of the model's function rather than of the representation's geometry, and its
    preferred direction depends on the task.

    Args:
        jvps: (B, k, D) products for B inputs and k directions; D is the flattened readout size.

    Returns:
        value: mean JER over inputs.
        extras: std over inputs, n_probes, n_inputs.
    """
    if jvps.ndim != 3 or jvps.shape[1] < 2:
        raise ValueError(f"expected (B, k, D) with k >= 2, got shape {tuple(jvps.shape)}")
    s = torch.linalg.svdvals(jvps.double())  # (B, k)
    jer = s.sum(dim=1) ** 2 / s.square().sum(dim=1)
    std = float(jer.std(unbiased=False)) if jer.numel() > 1 else 0.0
    return MetricResult(
        float(jer.mean()), {"std": std, "n_probes": float(jvps.shape[1]), "n_inputs": float(jvps.shape[0])}
    )


register_metric(
    "jacobian_effective_rank",
    inputs=InputKind.JACOBIAN,
    preprocess=Preprocess(),
    citation=("chung2026globalgeometry",),
    arxiv="2602.03282",
    description="Effective rank of a layer readout's input-output Jacobian (Chung and Kim, 2026).",
)(jacobian_effective_rank)
