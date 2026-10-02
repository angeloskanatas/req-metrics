"""Relational metrics and collapse indicators on pooled vectors.

Pairwise-relation statistics of the L2-normalized point cloud, all label-free and
parameter-free: the self-clustering score of Tsitsulin et al. (2023), the
uniformity of Wang and Isola (2020) and the normalized-output standard deviation
of Chen and He (2021). They describe how points are spread on the unit sphere and
are the registry's indicators of complete collapse and of concentration.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from req_metrics._types import InputKind, MetricResult, Preprocess
from req_metrics.preprocess import apply_preprocess
from req_metrics.registry import register_metric


def _unit_rows(x: Tensor, center: bool) -> Tensor:
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError(f"expected (N, D) with N >= 2, got shape {tuple(x.shape)}")
    return apply_preprocess(x.double(), Preprocess(center=center, l2=True))


def self_clustering(x: Tensor, *, center: bool = False) -> MetricResult:
    """Self-clustering score: excess squared cosine over a uniform spherical cloud.

    Tsitsulin, Munkhoeva and Perozzi (2023, TAG-ML at ICML, arXiv:2305.16562,
    Def. 3.5): for L2-normalized rows W, the pairwise dot-product mass
    Q = sum_ij (w_i . w_j)^2 is compared with its expectation for N points
    uniform on the sphere, N + N(N-1)/D, and with its maximum N^2 at complete
    collapse: SelfCluster = (Q - N - N(N-1)/D) / (N^2 - N - N(N-1)/D), 0 for a
    uniform cloud and 1 for a single point. The paper writes Q as the Frobenius
    norm of W W^T; the expectation and maximum it states are those of the squared
    norm, which is what is used here so that collapse gives exactly 1. Computed
    through the D x D second-moment matrix, ||W^T W||_F^2 = ||W W^T||_F^2, in
    O(N D^2) (the reformulation of Arputharaj et al., 2026, TMLR). Arputharaj et
    al. report it as a reliable negative predictor of accuracy for
    self-supervised vision models, uninformative for supervised ones, and
    anti-correlated at -0.999 with diffusion spectral entropy under
    L2-normalization. The paper found its sign on graph embeddings consistent
    where spectral metrics flipped.

    Args:
        x: Points (N, D).
        center: Mean-center before normalizing (off in the source paper).

    Returns:
        value: SelfCluster, 0 (uniform) to 1 (collapsed).
        extras: mean_squared_cosine over distinct pairs, and its uniform value 1/D.
    """
    w = _unit_rows(x, center)
    n, d = w.shape
    q = float((w.T @ w).square().sum())  # ||W W^T||_F^2
    expected = n + n * (n - 1) / d
    value = (q - expected) / (n * n - expected)
    return MetricResult(value, {"mean_squared_cosine": (q - n) / (n * (n - 1)), "uniform_mean_squared_cosine": 1.0 / d})


def uniformity(x: Tensor, *, t: float = 2.0, center: bool = False, chunk: int = 2048) -> MetricResult:
    """Uniformity: log average pairwise Gaussian potential on the unit sphere.

    Wang and Isola (2020, ICML, arXiv:2005.10242, Sec. 4.1.2):
    L_uniform = log mean_{i<j} exp(-t ||u_i - u_j||^2) over L2-normalized
    features, t = 2 in the paper and its reference code. The uniform
    distribution on the sphere is its unique minimizer (Prop. 1); Corollary 1
    gives the range [-2t + log 0F1(; D/2; t^2), 0], the lower end reached only
    by a perfectly uniform encoder and 0 only by a constant one. Lower values
    mean points are spread more evenly. Pairwise distances come from the Gram
    matrix in row chunks, ||u - v||^2 = 2 - 2 u.v, in float64; O(N^2 D).

    Args:
        x: Points (N, D).
        t: Kernel scale of the Gaussian potential.
        center: Mean-center before normalizing (off in the source paper).
        chunk: Rows per Gram block.

    Returns:
        value: L_uniform in nats, in [lower_bound, 0].
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

    Chen and He (2021, CVPR, arXiv:2011.10566, Sec. 4.1): the std over samples of
    z / ||z||_2, averaged over channels, is 0 when all outputs collapse to one
    vector and about 1 / sqrt(D) when z is a zero-mean isotropic Gaussian, so it
    is the standard complete-collapse monitor of Siamese self-supervised
    learning (lightly's std_of_l2_normalized). It does not see dimensional
    collapse; pair it with a spectral metric. Sample std uses the N - 1
    denominator, as in that implementation.

    Args:
        x: Points (N, D).

    Returns:
        value: mean channel std of the unit-norm rows.
        extras: reference 1 / sqrt(D) and the ratio value * sqrt(D) (about 1 for an isotropic cloud).
    """
    u = _unit_rows(x, center=False)
    value = float(u.std(dim=0).mean())
    ref = 1.0 / math.sqrt(u.shape[1])
    return MetricResult(value, {"isotropic_reference": ref, "ratio_to_isotropic": value / ref})


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
    tags=("collapse-indicator",),
)(normalized_std)
