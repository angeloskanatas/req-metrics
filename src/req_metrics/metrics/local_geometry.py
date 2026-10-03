"""Local geometry of a point cloud beyond dimension: neighborhood curvature and rectifiability."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from req_metrics._types import InputKind, MetricResult, Preprocess
from req_metrics.neighbors import Neighbors
from req_metrics.registry import register_metric


def neighborhood_curvature(
    x: Tensor, *, k: int = 64, neighbors: Neighbors | None = None, chunk: int | None = None
) -> MetricResult:
    """Neighborhood curvature: mean cosine between the unit edges to the k nearest neighbors.

    CurvSSL (Ghojogh et al., 2025, arXiv:2511.17426): per point, the mean pairwise cosine of the
    unit vectors to its k neighbors, averaged over points; near 0 for isotropic neighborhoods,
    toward 1 at boundaries. A Gaussian cloud scores about 0.2 at k = 32 in eight dimensions, so
    compare layers or runs rather than reading the value against zero. Points are processed in
    chunks, so memory is bounded by chunk * k * D float64 values.

    Args:
        x: Points (N, D); without neighbors, duplicates are removed first.
        k: Neighborhood size.
        neighbors: Precomputed Neighbors table of x with at least k neighbors.
        chunk: Points per batch; default keeps a batch near 2^25 values.

    Returns:
        value: mean pairwise neighbor cosine.
        extras: none.
    """
    if neighbors is None:
        x = torch.unique(x, dim=0)
    nb = neighbors if neighbors is not None else Neighbors.from_points(x, k)
    k = min(k, nb.k)
    if k < 2:
        raise ValueError("need k >= 2")
    xd = x.double()
    n, d = xd.shape
    chunk = chunk or max(1, 2**25 // (k * d))
    pair_cos = torch.empty(n, dtype=torch.float64, device=xd.device)
    for c0 in range(0, n, chunk):
        rows = slice(c0, min(c0 + chunk, n))
        edges = F.normalize(xd[nb.indices[rows, 1 : k + 1]] - xd[rows].unsqueeze(1), dim=2)  # (c, k, D)
        gram = torch.bmm(edges, edges.transpose(1, 2))  # (c, k, k)
        pair_cos[rows] = (gram.sum(dim=(1, 2)) - k) / (k * (k - 1))  # mean off-diagonal cosine
    valid = torch.isfinite(pair_cos)
    if valid.sum() < 2:
        raise ValueError("fewer than two valid points")
    return MetricResult(float(pair_cos[valid].mean()), {})


@torch.no_grad()
def local_rectifiability(
    x: Tensor, *, n: int | None = None, n_anchors: int = 256, n_scales: int = 6, chunk: int = 64, seed: int = 0
) -> MetricResult:
    """Multi-scale flatness of the cloud around an n-dimensional tangent plane.

    The empirical beta-number of UR-JEPA (Le et al., 2026, arXiv:2606.01443, Eqs. 23-24): at
    anchors x and dyadic scales r_k = 2^-k r_max, the Gaussian-weighted variance orthogonal to the
    best-fit affine n-plane, divided by r^2 and the neighborhood mass. Small and decaying with r
    means locally flat; near zero together with the trace means collapse. r_max is the largest
    anchor-to-point distance. The paper fixes n (Sec. 6.2); the default D / 8, at least 4,
    depends on the width only, so layers of equal width are compared at the same n. One local
    PCA per anchor and scale, in float32 with TF32 matmuls disabled; use a few thousand points.

    Args:
        x: Points (N, D).
        n: Tangent dimension, 1 <= n < D.
        n_anchors: Anchor points per scale.
        n_scales: Dyadic scales, the largest r_max.
        chunk: Anchors per batch.
        seed: Anchor sampling seed.

    Returns:
        value: beta_2 at scale index (n_scales - 1) // 2, r_max / 4 with the defaults.
        extras: n, and beta2_scale{i}, trace_scale{i}, local_id_scale{i} (eigenvalues of the
            local scatter above their mean), r_scale{i}, with scale 0 the coarsest.
    """
    if x.ndim != 2:
        raise ValueError(f"expected (N, D), got shape {tuple(x.shape)}")
    N, D = x.shape
    if N < 16 or D < 2:
        raise ValueError("need N >= 16 and D >= 2")
    n = min(D - 1, max(4, D // 8)) if n is None else n
    if not 1 <= n < D:
        raise ValueError("need 1 <= n < D")
    z = x.detach().float()
    g = torch.Generator().manual_seed(seed)  # CPU stream: the same anchors on every device
    anchors = torch.randperm(N, generator=g)[: min(n_anchors, N)].to(z.device)
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        r_max = torch.cdist(z[anchors], z).max().clamp_min(1e-6)
        scales = r_max * 2.0 ** (-torch.arange(n_scales, device=z.device))
        extras: dict[str, float] = {"n": float(n)}
        betas = []
        for s, r in enumerate(scales):
            r2 = (r * r).clamp_min(1e-12)
            b_sum = t_sum = id_sum = 0.0
            for c0 in range(0, len(anchors), chunk):
                xa = z[anchors[c0 : c0 + chunk]]  # (a, D)
                w = torch.exp(-torch.cdist(xa, z).square() / (2 * r2))  # (a, N) Gaussian kernel
                ws = w.sum(1, keepdim=True).clamp_min(1e-12)
                zc = z.unsqueeze(0) - (w @ z / ws).unsqueeze(1)  # (a, N, D) weighted-centered
                scatter = (zc * w.unsqueeze(-1)).transpose(1, 2) @ zc  # (a, D, D)
                eig = torch.linalg.eigvalsh(scatter.double()).clamp_min(0)  # ascending
                off = eig[:, : D - n].sum(1)  # residual off the top-n plane
                local_id = (eig > eig.sum(1, keepdim=True) / D).sum(1)
                b_sum += float((off / ws.squeeze(1).double() / r2.double()).sum())
                t_sum += float(eig.sum(1).sum())
                id_sum += float(local_id.sum())
            a = len(anchors)
            extras[f"beta2_scale{s}"] = b_sum / a
            extras[f"trace_scale{s}"] = t_sum / a
            extras[f"local_id_scale{s}"] = id_sum / a
            extras[f"r_scale{s}"] = float(r)
            betas.append(b_sum / a)
    finally:
        torch.set_float32_matmul_precision(precision)
    return MetricResult(betas[(len(betas) - 1) // 2], extras)


register_metric(
    "neighborhood_curvature",
    cache="neighbors_kw",
    inputs=InputKind.POINTS,
    preprocess=Preprocess(),
    citation=("ghojogh2025curvssl",),
    arxiv="2511.17426",
)(neighborhood_curvature)
register_metric(
    "local_rectifiability",
    inputs=InputKind.POINTS,
    preprocess=Preprocess(),
    max_items=2048,
    citation=("le2026urjepa",),
    arxiv="2606.01443",
    description="UR-JEPA beta-number flatness around an n-plane across dyadic scales (Eq. 24).",
)(local_rectifiability)
