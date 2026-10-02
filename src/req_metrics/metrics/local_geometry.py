"""Local geometry of a point cloud beyond dimension: neighborhood curvature and rectifiability."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from req_metrics._types import InputKind, MetricResult, Preprocess
from req_metrics.neighbors import Neighbors
from req_metrics.registry import register_metric


def neighborhood_curvature(x: Tensor, *, k: int = 64, neighbors: Neighbors | None = None) -> MetricResult:
    """Local bending of the cloud: mean cosine between unit edges to the k nearest neighbors.

    The discrete curvature score of CurvSSL (Ghojogh et al., 2025,
    arXiv:2511.17426): for each point, the unit vectors to its k nearest
    neighbors and the mean of their pairwise cosines, averaged over points.
    Near 0 when neighbors surround the point isotropically, toward 1 when
    they lie to one side (a boundary or a sharp bend), negative when they
    lie on opposite sides, as on a curve. A Gaussian cloud is not a null
    case: its outer points see neighbors biased toward the center and score
    around 0.2 at k = 32 in eight dimensions, so compare layers or runs
    rather than reading the value against zero. A point-cloud quantity,
    unrelated to the trajectory curvature of a frame sequence.

    Args:
        x: Points (N, D).
        k: Neighborhood size.
        neighbors: Precomputed Neighbors table with at least k neighbors.

    Returns:
        value: mean pairwise neighbor cosine.
        extras: none.
    """
    nb = neighbors if neighbors is not None else Neighbors.from_points(x, k)
    k = min(k, nb.k)
    if k < 2:
        raise ValueError("need k >= 2")
    xd = x.double()
    edges = F.normalize(xd[nb.indices[:, 1 : k + 1]] - xd.unsqueeze(1), dim=2)  # (N, k, D)
    gram = torch.bmm(edges, edges.transpose(1, 2))  # (N, k, k)
    pair_cos = (gram.sum(dim=(1, 2)) - k) / (k * (k - 1))  # mean off-diagonal cosine
    valid = torch.isfinite(pair_cos)
    if valid.sum() < 2:
        raise ValueError("fewer than two valid points")
    return MetricResult(float(pair_cos[valid].mean()), {})


@torch.no_grad()
def local_rectifiability(
    x: Tensor, *, n: int | None = None, n_anchors: int = 256, n_scales: int = 6, chunk: int = 64, seed: int = 0
) -> MetricResult:
    """Multi-scale flatness of the cloud around an n-dimensional tangent plane.

    The empirical beta-number of UR-JEPA (Le et al., 2026, arXiv:2606.01443,
    Eq. 23-24): at each anchor x and dyadic scale r_k = 2^-k r_max, the
    Gaussian-weighted centered scatter matrix S_r(x) with weights
    exp(-|z - x|^2 / 2r^2), and beta_2(x, r) = (1/r^2) sum_{j>n} sigma_j^2 /
    sum_j w_r(z_j - x): the kernel-weighted variance orthogonal to the
    best-fit affine n-plane, normalized by the neighborhood mass. UR-JEPA
    minimizes it as a regularizer toward a uniformly n-rectifiable measure;
    here it is read as a diagnostic.
    Small and decaying with r means locally flat and n-dimensional; large
    and flat across scales means isotropic; near zero together with a
    near-zero scatter trace means collapse. The count of eigenvalues above
    their mean is a per-scale local-dimension estimate. Cost is a local PCA
    per anchor and scale, so subsample to a few thousand points. UR-JEPA
    fixes n as the target dimension of its regularizer; as a diagnostic with
    no target, n defaults to the number of eigenvalues of the global centered
    covariance above their mean (a participation count), recorded in the
    extras, so profiles across layers of different width stay comparable.

    Args:
        x: Points (N, D).
        n: Tangent dimension tested, 1 <= n < D; default as described above.
        n_anchors: Anchor points per scale.
        n_scales: Dyadic scales below the anchor diameter.
        chunk: Anchors per batch.
        seed: Anchor sampling seed.

    Returns:
        value: beta_2 at the middle scale.
        extras: n (tangent dimension used), beta2_scale{i}, trace_scale{i}, local_id_scale{i}, r_scale{i};
            scale 0 is the coarsest.
    """
    if x.ndim != 2:
        raise ValueError(f"expected (N, D), got shape {tuple(x.shape)}")
    N, D = x.shape
    if N < 16 or D < 2:
        raise ValueError("need N >= 16 and D >= 2")
    if n is None:
        lam = torch.linalg.svdvals(x.double() - x.double().mean(dim=0))
        lam = lam.square()
        n = int(min(D - 1, max(1, int((lam > lam.mean()).sum()))))
    if not 1 <= n < D:
        raise ValueError("need 1 <= n < D")
    z = x.detach().float()
    g = torch.Generator(device=z.device).manual_seed(seed)
    anchors = torch.randperm(N, generator=g, device=z.device)[: min(n_anchors, N)]
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
    return MetricResult(betas[len(betas) // 2], extras)


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
