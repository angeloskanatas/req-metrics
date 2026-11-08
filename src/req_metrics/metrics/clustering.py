"""Clustering quality of a point cloud: k-means inertia and the Davies-Bouldin index; the dense
representation structure estimator, which clusters each sample's tokens."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from req_metrics._types import InputKind, MetricResult, Preprocess
from req_metrics.preprocess import l2_normalize
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
    centroid and d_ij the distance between centroids; coincident centroids contribute 0.
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

    Whetten et al. (2025, Interspeech, Sec. 3.1.1, Eqs. 2-3) run k-means with k = 1024 and
    k-means++ seeding on the frames of one layer, early in pretraining. In their study a lower
    inertia goes with better recognition and a lower Davies-Bouldin index with worse recognition,
    so read the two together, at equal layer, N and k. Seeded full-batch Lloyd iterations replace
    the paper's mini-batch k-means.

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


_DSE_VARIANTS = ("released", "paper")


def _cluster_radius(points: Tensor, variant: str) -> float:
    """Sum of the singular values of the centered points, over sqrt(max(n - 1, d)) in the released code and
    sqrt(n - 1) in Eq. 5."""
    n, d = points.shape
    s = torch.linalg.svdvals(points - points.mean(0))
    return float(s.sum()) / math.sqrt(max(n - 1, d) if variant == "released" else n - 1)


def _erank(x: Tensor, normalized: bool) -> float:
    """exp of the Shannon entropy of the normalized singular values of x (uncentered); divided by min(n, d) when
    normalized."""
    s = torch.linalg.svdvals(x)
    p = s / s.sum()
    p = p[p > 0]
    rank = float(torch.exp(-(p * p.log()).sum()))
    return rank / min(x.shape) if normalized else rank


def _clusters(points: Tensor, k: int, iters: int, seed: int) -> tuple[Tensor, Tensor]:
    """k-means centers and labels with at most as many clusters as points."""
    centers, labels, _ = kmeans(points, min(k, points.shape[0]), iters=iters, seed=seed)
    return centers, labels


def dense_structure(
    x: Tensor,
    *,
    groups: Tensor | None = None,
    k_sample: int = 3,
    k_group: int = 24,
    group_size: int = 8,
    l2: bool = True,
    variant: str = "released",
    draws: int | None = None,
    iters: int = 20,
    seed: int = 0,
) -> MetricResult:
    """Dense representation structure estimator, DSE (Dai et al., 2025, NeurIPS, arXiv:2510.17299, Sec. 4.1, Eq. 5).

    A label-free estimate of how well a linear head can classify the tokens of a sample set, built from a
    nearest-class-mean error bound (their Thm. 2 and Cor. 5). It adds class separability on k-means
    pseudo-labels to the effective dimensionality of the tokens,

        DSE = M_inter - M_intra + lambda * M_dim.

    - M_intra, the radius of a cluster: the sum of the singular values of its centered tokens over
      sqrt(N_j - 1), averaged over clusters; computed on each sample's tokens with k_sample = 3 clusters and on
      the tokens of group_size = 8 samples with k_group = 24 clusters, and the two averaged (their App. D.4).
    - M_inter: each token's distance to the nearest center of another cluster, from the k_group clustering.
    - M_dim: the effective rank (Roy and Vetterli) of one token per sample, the samples' tokens drawn at
      random, averaged over draws; the samples should outnumber the dimensions.

    variant="released" (the default) follows the authors' code (github.com/EldercatSAM/SSL-Degradation,
    estimator.compute_DSE, commit b75fd51), which produced their published values and differs from Eq. 5 as
    printed: M_inter and both radii are divided by the mean distance of all tokens to their mean, the radius
    denominator is sqrt(max(N_j - 1, d)), M_inter is the mean over all tokens rather than over clusters, and
    M_dim is divided by min(samples, d). These make the value free of the scale and, for M_dim, of the width,
    so layers and models can be compared. variant="paper" follows Eq. 5 as printed. lambda is 1 here; the
    paper sets it across candidates to Std(M_inter - M_intra) / Std(M_dim), which recombine_dense_structure
    applies to the components of a set of checkpoints or layers. The authors L2-normalize the tokens (their
    README's --normalize, l2=True) and measure the last layer of each checkpoint on 2,048 ImageNet images in
    batches of 1,024. Other differences from the released code: its k-means++ seeding weights candidates by
    distance rather than squared distance, it scores a sample with no cluster of two or more tokens as 1.0
    where this function skips it, and it has no cap on the draws.

    Args:
        x: Tokens (N, D) with groups, or token fields (B, T, D) of B samples.
        groups: The sample of every row of a 2D x, (N,); compute() passes it at level="population".
        k_sample: Clusters per sample.
        k_group: Clusters per group of samples.
        group_size: Samples per group, taken in order.
        l2: Scale every token to unit length first.
        variant: "released" or "paper".
        draws: Number of one-token-per-sample draws for M_dim; None uses as many as the shortest sample
            has tokens, as the released code does.
        iters: Maximum Lloyd iterations (20 in the released code).
        seed: Seed of the k-means seeding and of the draws.

    Returns:
        value: DSE with lambda = 1.
        extras: m_inter, m_intra (the mean of m_intra_sample and m_intra_group), m_intra_sample,
            m_intra_group, m_dim, separability (m_inter - m_intra), scale (the mean distance to the mean),
            n_samples, n_groups, draws, n_samples_skipped.
    """
    if variant not in _DSE_VARIANTS:
        raise ValueError(f"variant must be one of {_DSE_VARIANTS}, got {variant!r}")
    if x.ndim == 3 and groups is None:
        b, t, _ = x.shape
        x, groups = x.reshape(b * t, -1), torch.arange(b).repeat_interleave(t)
    if x.ndim != 2 or groups is None:
        raise ValueError(
            "dense_structure needs the sample of every token: pass groups with an (N, D) tensor, a (B, T, D) "
            "tensor, or compute(..., level='population')"
        )
    if groups.shape[0] != x.shape[0]:
        raise ValueError(f"groups has {groups.shape[0]} entries for {x.shape[0]} tokens")
    xd = l2_normalize(x.double()) if l2 else x.double()
    groups = groups.to(xd.device)
    samples = [xd[groups == g] for g in torch.unique(groups)]
    if len(samples) < 2:
        raise ValueError("dense_structure needs at least two samples")
    released = variant == "released"
    scale = float((xd - xd.mean(0)).norm(dim=1).mean())

    sample_radii, skipped = [], 0
    for i, tokens in enumerate(samples):
        _, labels = _clusters(tokens, k_sample, iters, seed + i)
        radii = [_cluster_radius(tokens[labels == j], variant) for j in labels.unique() if (labels == j).sum() > 1]
        if radii:
            sample_radii.append(sum(radii) / len(radii))
        else:
            skipped += 1

    group_radii, inter = [], []
    for start in range(0, len(samples), group_size):
        tokens = torch.cat(samples[start : start + group_size])
        centers, labels = _clusters(tokens, k_group, iters, seed + start)
        for j in labels.unique():
            if (labels == j).sum() > 1:
                group_radii.append(_cluster_radius(tokens[labels == j], variant))
        d = torch.cdist(tokens, centers, compute_mode="donot_use_mm_for_euclid_dist")
        d[torch.arange(tokens.shape[0], device=d.device), labels] = float("inf")
        nearest_other = d.min(1).values
        if released:
            inter.append(float(nearest_other.mean()))
        else:
            inter.append(float(torch.stack([nearest_other[labels == j].mean() for j in labels.unique()]).mean()))
    if not sample_radii or not group_radii:
        raise ValueError("no cluster holds two or more tokens: too few tokens per sample for k_sample, k_group")

    gen = torch.Generator().manual_seed(seed)
    t_min = min(s.shape[0] for s in samples)
    n_draws = t_min if draws is None else max(1, min(int(draws), t_min))
    orders = [torch.randperm(s.shape[0], generator=gen)[:n_draws] for s in samples]
    m_dim = (
        sum(
            _erank(torch.stack([s[o[j]] for s, o in zip(samples, orders, strict=True)]), released)
            for j in range(n_draws)
        )
        / n_draws
    )

    norm = scale if released else 1.0
    m_inter = sum(inter) / len(inter) / norm
    m_intra_sample = sum(sample_radii) / len(sample_radii) / norm
    m_intra_group = sum(group_radii) / len(group_radii) / norm
    m_intra = (m_intra_sample + m_intra_group) / 2
    extras = {
        "m_inter": m_inter,
        "m_intra": m_intra,
        "m_intra_sample": m_intra_sample,
        "m_intra_group": m_intra_group,
        "m_dim": m_dim,
        "separability": m_inter - m_intra,
        "scale": scale,
        "n_samples": float(len(samples)),
        "n_groups": float(len(inter)),
        "draws": float(n_draws),
        "n_samples_skipped": float(skipped),
    }
    return MetricResult(m_inter - m_intra + m_dim, extras)


def recombine_dense_structure(components: Sequence[Mapping[str, float]]) -> tuple[list[float], float]:
    """DSE of a set of candidates with the paper's lambda (Dai et al., 2025, NeurIPS, arXiv:2510.17299, Eq. 5).

    lambda = Std(M_inter - M_intra) / Std(M_dim) over the candidates (checkpoints in the paper; layers, models
    or checkpoints here), so the two terms carry equal spread; the authors apply it to the saved components
    of all checkpoints after computing them (their README).

    Args:
        components: For each candidate, the extras of its dense_structure value (separability and m_dim).

    Returns:
        DSE per candidate, and lambda.
    """
    sep = torch.tensor([float(c["separability"]) for c in components], dtype=torch.float64)
    dim = torch.tensor([float(c["m_dim"]) for c in components], dtype=torch.float64)
    if sep.numel() < 2 or float(dim.std()) == 0.0:
        raise ValueError("lambda needs at least two candidates whose m_dim differ")
    lam = float(sep.std() / dim.std())
    return (sep + lam * dim).tolist(), lam


register_metric(
    "dense_structure",
    inputs=InputKind.TOKENS,
    preprocess=Preprocess(l2=True),
    citation=("dai2025structural",),
    arxiv="2510.17299",
    grouped=True,
    description="Dense representation structure estimator: k-means separability of each sample's tokens plus "
    "the effective rank of one token per sample (Dai et al., 2025).",
)(dense_structure)
