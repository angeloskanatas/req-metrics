"""Intrinsic-dimension estimators of a point cloud.

TwoNN and GRIDE are ports of DADApy (Glielmo et al., 2022, Patterns; Copyright 2021-2023
The DADApy Authors, Apache License 2.0; see NOTICE) on torch tensors. GRIDE, the MLE and
mLID read one shared Neighbors table; TwoNN builds its own two-neighbor table on the
distinct points, and the MST dimension uses spanning trees.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from req_metrics._types import InputKind, MetricResult, Preprocess
from req_metrics.neighbors import Neighbors
from req_metrics.registry import register_metric


def _neighbors(x: Tensor | Neighbors, k: int) -> Neighbors:
    if isinstance(x, Neighbors):
        if x.k < k:
            raise ValueError(f"Neighbors table has k={x.k}, estimator needs {k}")
        return x
    return Neighbors.from_points(_unique_rows(x), k)  # zero distances break the log-ratio estimators


def _unique_rows(x: Tensor) -> Tensor:
    return torch.unique(x, dim=0)


def twonn(x: Tensor, *, mu_fraction: float = 0.9, algorithm: str = "base") -> MetricResult:
    """TwoNN intrinsic dimension from the ratio of second- to first-neighbor distance.

    Facco et al. (2017, Scientific Reports): under local uniformity mu = r_2 / r_1 is Pareto with
    shape d. "base" fits -log(1 - F(mu)) = d log mu through the origin on the lowest mu_fraction
    of the ratios (the paper discards the top 10 percent); "ml" is d = (N - 1) / sum(log mu).
    Exact duplicate rows are removed first, which DADApy does not do by default; see
    extras["n_used"].

    Args:
        x: Points (N, D).
        mu_fraction: Fraction of the smallest ratios kept in the fit.
        algorithm: "base" or "ml".

    Returns:
        value: estimated dimension.
        extras: r (mean distance to the first two neighbors), n_used.
    """
    xu = _unique_rows(x)
    nb = Neighbors.from_points(xu, 2)
    mus = nb.ratios(2, 1)
    n = mus.numel()
    log_mus = torch.log(mus)
    if algorithm == "ml":
        d = float((n - 1) / log_mus.sum())
    elif algorithm == "base":
        n_eff = int(n * mu_fraction)
        xs = torch.sort(log_mus).values[:n_eff]
        ys = -torch.log(1.0 - torch.arange(1, n_eff + 1, dtype=xs.dtype, device=xs.device) / n)
        d = float((xs * ys).sum() / (xs * xs).sum())  # least-squares slope through the origin
    else:
        raise ValueError(f"unknown algorithm {algorithm!r}")
    return MetricResult(d, {"r": float(nb.distances[:, 1:3].mean()), "n_used": float(n)})


def _gride_single_scale(mus: Tensor, n1: int, n2: int, d0: float, d1: float, eps: float) -> tuple[float, float]:
    """GRIDE solver ported from DADApy (_filter_mus, _argmax_loglik, _fisher_info_scaling; Apache-2.0).

    Bisection on the derivative of the log-likelihood of the generalized ratios, then the
    inverse Fisher information as standard error. Ratios far in the tail (above
    median + 20 x the 95-50 percentile gap) are dropped before the fit, as upstream does.
    """
    mus = mus.clone()
    mus[mus == 1.0] += 10 * torch.finfo(mus.dtype).eps
    q95, q50 = torch.quantile(mus, 0.95), torch.quantile(mus, 0.5)
    kept = mus[mus < 20 * (q95 - q50) + q50]  # drop ratios from overlapping points
    n = kept.numel()
    log_kept = torch.log(kept)

    def neg_dloglik(d: float) -> float:
        one = (1.0 - kept ** (-d)).clamp_min(2 * eps)
        return float((((1 - n2 + n1) / one + n2 - 1.0) * log_kept).sum() - (n - 1) / d)

    lo, hi = d0, d1
    l_hi = neg_dloglik(hi)
    while abs(lo - hi) > eps:
        mid = (lo + hi) / 2.0
        if neg_dloglik(mid) * l_hi > 0:
            hi = mid
        else:
            lo = mid
    d = (lo + hi) / 2.0
    eps_f = 5 * torch.finfo(mus.dtype).eps
    one = (1.0 - mus ** (-d)).clamp_min(eps_f)
    log_mu = torch.log(mus)
    fisher = mus.numel() / d**2 + float(((n2 - n1 - 1) * (log_mu / one) ** 2 * mus ** (-d)).sum())
    return d, 1.0 / math.sqrt(fisher)


def gride(
    x: Tensor | Neighbors,
    *,
    scale: int = 8,
    range_max: int = 64,
    d0: float = 0.001,
    d1: float = 1000.0,
    eps: float = 1e-7,
) -> MetricResult:
    """GRIDE intrinsic dimension at doubling neighbor scales.

    Denti et al. (2022, Scientific Reports, arXiv:2104.13832, Eq. 13): the maximum-likelihood
    dimension from the ratio of the n2-th to the n1-th neighbor distance, at (n1, n2) = (k, 2k)
    for k = 1, 2, 4, ... up to range_max. The value is the estimate at n2 = scale; scale 8 uses
    the 8th and 4th neighbors. Bisection on the likelihood derivative and Fisher-information
    errors follow DADApy.

    Args:
        x: Points (N, D), duplicates removed first, or a Neighbors table with k >= range_max.
        scale: Outer neighbor rank of the returned estimate; a power of two.
        range_max: Largest outer rank.
        d0, d1: Bisection bounds on the dimension.
        eps: Bisection precision.

    Returns:
        value: dimension at the requested scale.
        extras: id_rank{n2}, err_rank{n2}, r_rank{n2} for every computed scale.
    """
    if scale & (scale - 1) or scale < 2:
        raise ValueError("scale must be a power of two >= 2")
    n_points = x.n if isinstance(x, Neighbors) else x.shape[0]
    max_rank = min(n_points - 1, range_max)
    if max_rank < scale:
        raise ValueError(f"scale {scale} exceeds the available neighbor rank {max_rank}")
    nb = _neighbors(x, max_rank)
    extras: dict[str, float] = {}
    value = float("nan")
    for i in range(int(math.log2(max_rank))):
        n1, n2 = 2**i, 2 ** (i + 1)
        d, err = _gride_single_scale(nb.ratios(n2, n1), n1, n2, d0, d1, eps)
        extras[f"id_rank{n2}"] = d
        extras[f"err_rank{n2}"] = err
        extras[f"r_rank{n2}"] = float(nb.distances[:, [n1, n2]].mean())
        if n2 == scale:
            value = d
    return MetricResult(value, extras)


def mle(x: Tensor | Neighbors, *, k_range: tuple[int, int] = (10, 20), unbiased: bool = False) -> MetricResult:
    """Levina-Bickel maximum-likelihood intrinsic dimension.

    Levina and Bickel (2004, NeurIPS, Eqs. 8-9): m_k(x) = [(1/(k-1)) sum_{j<k} log(T_k(x) /
    T_j(x))]^-1 from the neighbor distances T_j, averaged over points and over k in k_range
    (10 to 20 in the paper). unbiased=True divides by k - 2, which they note is asymptotically
    unbiased.

    Args:
        x: Points (N, D), duplicates removed first, or a Neighbors table with k >= k_range[1].
        k_range: Inclusive range of neighbor counts.
        unbiased: Use k - 2 in the denominator.

    Returns:
        value: estimated dimension.
        extras: id_k{k} for every k in the range.
    """
    k1, k2 = k_range
    if k1 < 2 or k2 < k1:
        raise ValueError("k_range must satisfy 2 <= k1 <= k2")
    nb = _neighbors(x, k2)
    log_d = torch.log(nb.distances[:, 1 : k2 + 1])  # (N, k2), column j-1 is T_j
    extras: dict[str, float] = {}
    for k in range(k1, k2 + 1):
        denom = (log_d[:, k - 1 : k] - log_d[:, : k - 1]).sum(dim=1)  # sum_j log(T_k / T_j)
        m_k = ((k - 2) if unbiased else (k - 1)) / denom
        extras[f"id_k{k}"] = float(m_k[torch.isfinite(m_k)].mean())
    return MetricResult(float(np.mean(list(extras.values()))), extras)


def mlid(x: Tensor | Neighbors, *, k: int = 64) -> MetricResult:
    """Geometric mean of per-point local intrinsic dimension (mLID).

    LID_i = mu_k / (w_k - mu_k) by the method of moments (Amsaleg et al., 2018), aggregated as
    the geometric mean, the Frechet mean of LDReg (Huang et al., 2024, ICLR, arXiv:2401.10474).
    w_k is the k-th neighbor distance and mu_k the mean of the k - 1 nearer ones, as in the
    LDReg code (lid_mom_est); the paper's text averages all k.

    Args:
        x: Points (N, D), duplicates removed first, or a Neighbors table with at least k neighbors.
        k: Neighborhood size.

    Returns:
        value: mLID.
        extras: frechet_var (variance of log LID), n_valid.
    """
    nb = _neighbors(x, k)
    k = min(k, nb.k)
    if k < 2:
        raise ValueError("need k >= 2")
    mu_k = nb.distances[:, 1:k].mean(dim=1)
    w_k = nb.distances[:, k]
    lids = mu_k / (w_k - mu_k + 1e-10)
    valid = torch.isfinite(lids) & (lids > 0)
    if valid.sum() < 2:
        raise ValueError("fewer than two valid local estimates")
    log_lids = torch.log(lids[valid])
    return MetricResult(
        float(torch.exp(log_lids.mean())), {"frechet_var": float(log_lids.var()), "n_valid": float(valid.sum())}
    )


def mst_dimension(x: Tensor, *, n_min: int | None = None, step: int | None = None, seed: int = 0) -> MetricResult:
    """Intrinsic dimension from the growth of minimum-spanning-tree length.

    Costa and Hero (2006); IdEst (Mordacq et al., 2026, arXiv:2606.03338, Alg. 1). The MST
    length of n points grows as n^((d-1)/d): log L is regressed on log n over subsamples of size
    n_min, n_min + step, ... and d = 1 / (1 - slope). Each MST is O(n^2) in memory; use a few
    thousand points.

    Args:
        x: Points (N, D).
        n_min: Smallest subsample size (default N/16, at least 8).
        step: Increment between subsample sizes (default N/16).
        seed: Subsampling seed.

    Returns:
        value: estimated dimension; nan if the slope leaves (0, 1).
        extras: slope, n_sizes.
    """
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import pdist, squareform

    z = x.detach().cpu().double().numpy()
    n = z.shape[0]
    if n < 16:
        raise ValueError("need at least 16 points")
    n_min = n_min or max(8, n // 16)
    step = step or max(1, n // 16)
    sizes = list(range(int(n_min), n, int(step)))
    if len(sizes) < 2:
        raise ValueError("fewer than two subsample sizes; lower n_min or step")
    rng = np.random.default_rng(seed)
    log_n, log_len = [], []
    for size in sizes:
        idx = rng.choice(n, size=size, replace=False)
        length = float(minimum_spanning_tree(squareform(pdist(z[idx]))).sum())
        if length > 0 and np.isfinite(length):
            log_n.append(math.log(size))
            log_len.append(math.log(length))
    slope = float(np.polyfit(log_n, log_len, 1)[0])
    value = 1.0 / (1.0 - slope) if 0.0 < slope < 1.0 else float("nan")
    return MetricResult(value, {"slope": slope, "n_sizes": float(len(log_n))})


_P = InputKind.POINTS
register_metric(
    "intrinsic_dimension",
    inputs=_P,
    preprocess=Preprocess(),
    citation=("DBLP:journals/corr/abs-1803-06992", "glielmo2022dadapy"),
    arxiv="1803.06992",
    tags=("paper-canonical",),
    description="TwoNN intrinsic dimension (Facco et al.).",
)(twonn)
register_metric(
    "intrinsic_dimension/gride",
    cache="neighbors",
    inputs=_P,
    preprocess=Preprocess(),
    citation=("denti2022gride", "glielmo2022dadapy", "DBLP:journals/corr/abs-1803-06992"),
    arxiv="2104.13832",
    description="GRIDE intrinsic dimension at the 8th-neighbor scale.",
)(gride)
register_metric(
    "intrinsic_dimension/mle",
    cache="neighbors",
    inputs=_P,
    preprocess=Preprocess(),
    citation=("levina2004mle",),
    description="Levina-Bickel MLE intrinsic dimension, k = 10..20.",
)(mle)
register_metric(
    "intrinsic_dimension/mlid",
    cache="neighbors",
    inputs=_P,
    preprocess=Preprocess(),
    citation=("huang2024ldreg", "amsaleg2018lid"),
    arxiv="2401.10474",
)(mlid)
register_metric(
    "intrinsic_dimension/mst",
    inputs=_P,
    preprocess=Preprocess(),
    max_items=2000,
    citation=("mordacq2026idest", "costa2006mst"),
    arxiv="2606.03338",
)(mst_dimension)
