"""Clustering quality of a point cloud: k-means inertia and the Davies-Bouldin index."""

from __future__ import annotations

import torch
from torch import Tensor

from req_metrics._types import InputKind, MetricResult, Preprocess
from req_metrics.registry import register_metric


def _sq_dists(x: Tensor, c: Tensor, chunk: int) -> Tensor:
    """Squared Euclidean distances from every row of x to every row of c, (N, K)."""
    cn = c.square().sum(1)
    out = torch.empty((x.shape[0], c.shape[0]), dtype=x.dtype, device=x.device)
    for s in range(0, x.shape[0], chunk):
        b = x[s : s + chunk]
        out[s : s + chunk] = (b.square().sum(1, keepdim=True) - 2.0 * b @ c.T + cn).clamp_min(0.0)
    return out


def _kmeans_pp(x: Tensor, k: int, gen: torch.Generator, chunk: int) -> Tensor:
    """k-means++ seeding (Arthur and Vassilvitskii, 2007)."""
    n = x.shape[0]
    centers = [x[int(torch.randint(n, (1,), generator=gen).item())]]
    d2 = _sq_dists(x, centers[0][None], chunk)[:, 0]
    for _ in range(1, k):
        p = (d2 / d2.sum()).cpu() if d2.sum() > 0 else torch.full((n,), 1.0 / n)
        i = int(torch.multinomial(p, 1, generator=gen).item())
        centers.append(x[i])
        d2 = torch.minimum(d2, _sq_dists(x, x[i][None], chunk)[:, 0])
    return torch.stack(centers)


def kmeans(
    x: Tensor, k: int, *, iters: int = 100, tol: float = 1e-6, seed: int = 0, chunk: int = 8192
) -> tuple[Tensor, Tensor, float]:
    """Lloyd's k-means with k-means++ seeding.

    Empty clusters keep their center; iteration stops when the relative change of the inertia
    falls below tol.

    Returns:
        centers (K, D), labels (N,), inertia.
    """
    gen = torch.Generator().manual_seed(seed)
    c = _kmeans_pp(x, k, gen, chunk)
    prev = float("inf")
    for _ in range(iters):
        d2 = _sq_dists(x, c, chunk)
        dmin, labels = d2.min(1)
        inertia = float(dmin.sum())
        sums = torch.zeros_like(c).index_add_(0, labels, x)
        counts = torch.bincount(labels, minlength=k).to(x.dtype)
        nonempty = counts > 0
        c = torch.where(nonempty[:, None], sums / counts.clamp_min(1)[:, None], c)
        if prev != float("inf") and prev - inertia <= tol * prev:
            break
        prev = inertia
    d2 = _sq_dists(x, c, chunk)
    dmin, labels = d2.min(1)
    return c, labels, float(dmin.sum())


def davies_bouldin(x: Tensor, labels: Tensor) -> float:
    """Davies-Bouldin index (Davies and Bouldin, 1979).

    Mean over clusters of max_{j != i} (s_i + s_j) / d_ij, with s_i the mean distance to the
    centroid and d_ij the distance between centroids; coincident centroids contribute 0. Equals
    scikit-learn's davies_bouldin_score.
    """
    uniq, lab = torch.unique(labels, return_inverse=True)
    k = uniq.numel()
    if k < 2:
        return float("nan")
    counts = torch.bincount(lab, minlength=k).to(x.dtype)
    cent = torch.zeros((k, x.shape[1]), dtype=x.dtype, device=x.device).index_add_(0, lab, x) / counts[:, None]
    s = torch.zeros(k, dtype=x.dtype, device=x.device).index_add_(0, lab, (x - cent[lab]).norm(dim=1)) / counts
    d = torch.cdist(cent, cent, compute_mode="donot_use_mm_for_euclid_dist")
    if bool(torch.allclose(s, torch.zeros_like(s))) or bool(torch.allclose(d, torch.zeros_like(d))):
        return 0.0
    d = torch.where(d == 0, torch.full_like(d, float("inf")), d)
    d.fill_diagonal_(float("inf"))
    return float(((s[:, None] + s[None, :]) / d).max(1).values.mean())


def cluster_quality(
    x: Tensor, *, k: int = 1024, score: str = "davies_bouldin", iters: int = 100, seed: int = 0
) -> MetricResult:
    """Clustering quality under k-means: Davies-Bouldin index and inertia.

    Whetten et al. (2025, Interspeech, Sec. 3.1.1, Eqs. 2-3): k-means with k = 1024 and k-means++
    seeding on the frames of one layer, early in pretraining. In their study lower inertia goes
    with better recognition and the Davies-Bouldin index with worse, so read both, at equal
    layer, N and k. Full-batch Lloyd iterations with a seed replace the paper's mini-batch
    k-means.

    Args:
        x: Points (N, D), N well above k.
        k: Number of clusters.
        score: "davies_bouldin" or "inertia".
        iters: Maximum Lloyd iterations.
        seed: Seed of the k-means++ initialization.

    Returns:
        value: the chosen score.
        extras: davies_bouldin, inertia, inertia_per_point, k, n_empty_clusters.
    """
    if score not in ("davies_bouldin", "inertia"):
        raise ValueError(f"score must be 'davies_bouldin' or 'inertia', got {score!r}")
    if x.ndim != 2 or x.shape[0] <= k:
        raise ValueError(f"expected (N, D) with N > k = {k}, got shape {tuple(x.shape)}")
    xd = x.double()
    centers, labels, inertia = kmeans(xd, k, iters=iters, seed=seed)
    db = davies_bouldin(xd, labels)
    n_empty = int((torch.bincount(labels, minlength=k) == 0).sum())
    extras = {
        "davies_bouldin": db,
        "inertia": inertia,
        "inertia_per_point": inertia / x.shape[0],
        "k": float(k),
        "n_empty_clusters": float(n_empty),
    }
    return MetricResult(db if score == "davies_bouldin" else inertia, extras)


register_metric(
    "cluster_quality",
    inputs=InputKind.POINTS,
    preprocess=Preprocess(),
    citation=("whetten2025early",),
    max_items=100_000,
    description="k-means Davies-Bouldin index, with the inertia in the extras (Whetten et al., 2025).",
)(cluster_quality)
