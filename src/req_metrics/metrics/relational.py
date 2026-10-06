"""Relational metrics: pairwise relations of the L2-normalized point cloud.

Self-clustering, uniformity, the normalized-output standard deviation and cosine
anisotropy describe how points spread on the unit sphere; they are the registry's
indicators of complete collapse and of concentration.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from req_metrics._types import InputKind, MetricResult, Preprocess
from req_metrics.preprocess import apply_preprocess, l2_normalize
from req_metrics.registry import register_metric


def _unit_rows(x: Tensor, center: bool) -> Tensor:
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError(f"expected (N, D) with N >= 2, got shape {tuple(x.shape)}")
    return apply_preprocess(x.double(), Preprocess(center=center, l2=True))


def self_clustering(x: Tensor, *, center: bool = False) -> MetricResult:
    """Self-clustering score: excess squared cosine over a uniform spherical cloud.

    Tsitsulin, Munkhoeva and Perozzi (2023, TAG-ML at ICML, arXiv:2305.16562, Def. 3.5): with Q
    the sum of squared cosines over all pairs of L2-normalized rows,
    (Q - N - N(N-1)/D) / (N^2 - N - N(N-1)/D), 0 for a uniform cloud and 1 for a single point.
    The paper writes Q as a Frobenius norm but states the expectation and maximum of its
    square, which is used here. Computed from the D x D second-moment matrix in O(N D^2).

    Args:
        x: Points (N, D).
        center: Mean-center before normalizing.

    Returns:
        value: score, 0 for a uniform cloud and 1 for a collapsed one; rows more spread than
            uniform score below 0 (orthonormal rows give -1/(D - 1)).
        extras: mean_squared_cosine over distinct pairs, and its uniform value 1/D.
    """
    w = _unit_rows(x, center)
    n, d = w.shape
    q = float((w.T @ w).square().sum())  # ||W W^T||_F^2
    expected = n + n * (n - 1) / d
    value = (q - expected) / (n * n - expected)
    return MetricResult(value, {"mean_squared_cosine": (q - n) / (n * (n - 1)), "uniform_mean_squared_cosine": 1.0 / d})


def uniformity(x: Tensor, *, t: float = 2.0, center: bool = False, chunk: int = 2048) -> MetricResult:
    """Uniformity: log mean pairwise Gaussian potential on the unit sphere.

    Wang and Isola (2020, ICML, arXiv:2005.10242, Sec. 4.1.2): log mean_{i<j} exp(-t ||u_i -
    u_j||^2) over L2-normalized rows, t = 2; lower is more uniform. Corollary 1 bounds it below
    by -2t + log 0F1(; D/2; t^2). Computed from Gram blocks in float64, O(N^2 D).

    Args:
        x: Points (N, D).
        t: Kernel scale.
        center: Mean-center before normalizing.
        chunk: Rows per Gram block.

    Returns:
        value: uniformity in nats.
        extras: lower_bound for this D and t, its large-D limit -2t, gap = value - lower_bound.
    """
    from scipy.special import hyp0f1

    u = _unit_rows(x, center)
    n, d = u.shape
    total = torch.zeros((), dtype=torch.float64, device=u.device)
    for start in range(0, n, chunk):
        block = u[start : start + chunk]
        sq = (2.0 - 2.0 * (block @ u.T)).clamp_min(0.0)  # squared distances to every point
        mask = torch.arange(n, device=u.device).unsqueeze(0) > torch.arange(
            start, start + block.shape[0], device=u.device
        ).unsqueeze(1)
        total = total + torch.exp(-t * sq)[mask].sum()
    value = float(torch.log(total / (n * (n - 1) / 2)))
    lower = -2.0 * t + math.log(float(hyp0f1(d / 2.0, t * t)))
    return MetricResult(value, {"lower_bound": lower, "lower_bound_large_d": -2.0 * t, "gap": value - lower})


def normalized_std(x: Tensor) -> MetricResult:
    """Mean per-channel standard deviation of the L2-normalized output.

    Chen and He (2021, CVPR, arXiv:2011.10566, Sec. 4.1): 0 under complete collapse and about
    1 / sqrt(D) for an isotropic Gaussian. It does not detect dimensional collapse.

    Args:
        x: Points (N, D).

    Returns:
        value: mean channel std of the unit-norm rows (N - 1 denominator).
        extras: reference 1 / sqrt(D) and the ratio value * sqrt(D).
    """
    u = _unit_rows(x, center=False)
    value = float(u.std(dim=0).mean())
    ref = 1.0 / math.sqrt(u.shape[1])
    return MetricResult(value, {"isotropic_reference": ref, "ratio_to_isotropic": value / ref})


def anisotropy_cosine(x: Tensor, *, center: bool = False) -> MetricResult:
    """Cosine anisotropy: mean cosine similarity between distinct samples.

    Ethayarajh (2019, EMNLP); Godey et al. (2024, EACL). Uncentered, it reflects the shared mean
    direction, which the spectral anisotropy of the centered matrix removes; Timkey and van
    Schijndel (2021, EMNLP) show it is often driven by a few rogue dimensions. Exact in O(N D)
    from the sum of the unit vectors.

    Args:
        x: Points (N, D).
        center: Mean-center before normalizing.

    Returns:
        value: mean off-diagonal cosine in [-1/(N-1), 1].
        extras: none.
    """
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError(f"expected (N, D) with N >= 2, got shape {tuple(x.shape)}")
    y = l2_normalize(apply_preprocess(x.double(), Preprocess(center=center)))
    n = y.shape[0]
    total = y.sum(dim=0)
    return MetricResult(float((total @ total - n) / (n * (n - 1))), {})


_P = InputKind.POINTS
register_metric(
    "self_clustering",
    inputs=_P,
    preprocess=Preprocess(l2=True),
    citation=("tsitsulin2023unsupervised", "arputharaj2026comparative"),
    arxiv="2305.16562",
    tags=("relational",),
)(self_clustering)
register_metric(
    "uniformity",
    inputs=_P,
    preprocess=Preprocess(l2=True),
    citation=("wang2020uniformity",),
    arxiv="2005.10242",
    tags=("relational",),
)(uniformity)
register_metric(
    "normalized_std",
    inputs=_P,
    preprocess=Preprocess(l2=True),
    citation=("chen2021simsiam",),
    arxiv="2011.10566",
    tags=("relational", "collapse-indicator"),
)(normalized_std)
register_metric(
    "anisotropy/cosine",
    inputs=_P,
    preprocess=Preprocess(l2=True),
    citation=("ethayarajh2019contextual", "DBLP:conf/eacl/GodeyCS24", "timkey2021rogue"),
    tags=("relational", "collapse-indicator"),
)(anisotropy_cosine)
