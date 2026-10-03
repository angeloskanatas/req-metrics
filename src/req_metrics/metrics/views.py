"""Metrics over augmented views of the same clips: views has shape (q, N, D)."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from req_metrics._types import InputKind, MetricResult, Preprocess
from req_metrics.preprocess import apply_preprocess, l2_normalize
from req_metrics.registry import register_metric


def _check_views(views: Tensor, min_q: int) -> tuple[int, int, int]:
    if views.ndim != 3 or views.shape[0] < min_q or views.shape[1] < 2:
        raise ValueError(f"expected (q >= {min_q}, N >= 2, D) views, got shape {tuple(views.shape)}")
    return int(views.shape[0]), int(views.shape[1]), int(views.shape[2])


def lidar(
    views: Tensor, *, delta: float = 1e-4, unbiased: bool = True, max_eigenvalues: int | None = None
) -> MetricResult:
    """LiDAR: effective rank of the LDA matrix, with clips as classes and their views as samples.

    Thilak et al. (2024, ICLR, arXiv:2312.04000, Eqs. 1-2; Eqs. 1-4 in arXiv v1): S_b is the
    scatter of the clips' view means and S_w the scatter of the views around their clip mean plus
    delta I; LiDAR is the exponential of the entropy of the normalized eigenvalues of
    S_w^{-1/2} S_b S_w^{-1/2}. The clean clip names the class and is not one of the q views. Use
    the training objective's own positives when monitoring one model (their Sec. 4.2) and one
    shared chain when comparing models. The denominators leave the value unchanged; an absolute
    delta makes it scale-dependent when within-clip variance approaches delta. Directions without
    clip signal keep eigenvalues of order 1/q, so compare at equal q and width, with n above the
    width (App. 11). The paper's epsilon is omitted.

    Args:
        views: Augmented representations (q, N, D), q >= 2.
        delta: Ridge added to S_w; the paper gives no value, 1e-4 is that of Skean et al. (2025).
        unbiased: Denominators n - 1 and n(q - 1), or n and nq.
        max_eigenvalues: Keep only the largest eigenvalues.

    Returns:
        value: LiDAR.
        extras: entropy, n_positive_eigenvalues.
    """
    q, n, d = _check_views(views, 2)
    a = views.double()
    class_means = a.mean(dim=0)  # (N, D)
    cm = class_means - class_means.mean(dim=0, keepdim=True)
    sigma_b = cm.T @ cm / ((n - 1) if unbiased else n)
    centered = (a - class_means.unsqueeze(0)).reshape(q * n, d)
    sigma_w = centered.T @ centered / ((n * (q - 1)) if unbiased else (n * q))
    sigma_w = sigma_w + delta * torch.eye(d, dtype=a.dtype, device=a.device)
    evals, evecs = torch.linalg.eigh(sigma_w)
    pos = evals > 0
    inv_sqrt = evecs[:, pos] @ torch.diag(evals[pos].pow(-0.5)) @ evecs[:, pos].T
    lam = torch.linalg.eigvalsh(inv_sqrt @ sigma_b @ inv_sqrt)
    lam = lam[lam > 0]
    if lam.numel() == 0:
        raise ValueError("the LDA matrix has no positive eigenvalue")
    if max_eigenvalues is not None and lam.numel() > max_eigenvalues:
        lam = lam[-max_eigenvalues:]
    p = lam / lam.sum()
    entropy = float(-(p * p.log()).sum())
    return MetricResult(math.exp(entropy), {"entropy": entropy, "n_positive_eigenvalues": float(lam.numel())})


def alignment(views: Tensor, *, alpha: float = 2.0) -> MetricResult:
    """Alignment: mean distance between L2-normalized views of the same clip, to the power alpha.

    Wang and Isola (2020, ICML, arXiv:2005.10242, Sec. 4.1.1), alpha = 2: 0 for perfectly aligned
    views, 2 for unrelated unit vectors in high dimension; averaged over all view pairs and clips.

    Args:
        views: Augmented representations (q, N, D), q >= 2.
        alpha: Distance exponent.

    Returns:
        value: alignment.
        extras: n_pairs.
    """
    q, n, d = _check_views(views, 2)
    u = torch.nn.functional.normalize(views.double(), dim=-1)
    total, pairs = 0.0, 0
    for a in range(q):
        for b in range(a + 1, q):
            total += float((u[a] - u[b]).norm(dim=-1).pow(alpha).mean())
            pairs += 1
    return MetricResult(total / pairs, {"n_pairs": float(pairs)})


def infonce(
    views: Tensor,
    *,
    temperature: float = 0.1,
    center: bool = True,
    l2: bool = True,
    symmetric: bool = False,
    anchor: int | None = None,
) -> MetricResult:
    """InfoNCE loss between augmented views of the same clips.

    van den Oord, Li and Vinyals (2018, arXiv:1807.03748, Eq. 4): the cross-entropy of
    identifying each clip's view b among all N clips' views b from its view a, with cosine
    logits over the temperature (rows centered and L2-normalized, as in Skean et al., 2025).
    Lower is more invariant to the augmentations. With q > 2 views the loss is averaged over
    the pairs a < b, the full graph of Tian et al. (2020, Eq. 8), or over the pairs (anchor,
    b), their core view (Eq. 7), for a non-exchangeable view such as a clean or global one.
    symmetric=True adds the reverse direction of each pair (Tian et al., Eq. 4). log N - L,
    the bound of van den Oord et al., cannot exceed log N and, for unit vectors with nearly
    orthogonal negatives, about 1 / temperature, even for identical views; compare values at
    equal N and temperature, and read contrastive_accuracy, which has no such ceiling.

    Args:
        views: Views (q, N, D), q >= 2.
        temperature: Softmax temperature.
        center: Mean-center each view over clips.
        l2: Scale rows to unit norm.
        symmetric: Average both directions of each pair.
        anchor: View paired with every other view; None pairs all views.

    Returns:
        value: mean loss in nats.
        extras: log_n_minus_loss, contrastive_accuracy (top-1 of the positive), n_pairs.
    """
    q, n, _ = _check_views(views, 2)
    if anchor is not None and not 0 <= anchor < q:
        raise ValueError(f"anchor must index one of the {q} views, got {anchor}")
    v = apply_preprocess(views.double(), Preprocess(center=center, l2=l2))
    if anchor is None:
        pairs = [(a, b) for a in range(q) for b in range(a + 1, q)]
    else:
        pairs = [(anchor, b) for b in range(q) if b != anchor]
    if symmetric:
        pairs += [(b, a) for a, b in pairs]
    labels = torch.arange(n, device=v.device)
    losses, hits = [], []
    for a, b in pairs:
        logits = v[a] @ v[b].T / temperature
        losses.append(float(torch.nn.functional.cross_entropy(logits, labels)))
        hits.append(float((logits.argmax(dim=1) == labels).double().mean()))
    loss = sum(losses) / len(losses)
    extras = {
        "log_n_minus_loss": math.log(n) - loss,
        "contrastive_accuracy": sum(hits) / len(hits),
        "n_pairs": float(len(pairs)),
    }
    return MetricResult(loss, extras)


def _normalized_gram(x: Tensor, kernel: str) -> Tensor:
    """Gram matrix with unit diagonal: cosine (linear on unit rows) or Gaussian with the median bandwidth."""
    if kernel == "linear":
        u = l2_normalize(x)
        return u @ u.T
    if kernel == "rbf":
        d2 = torch.cdist(x, x).square()
        sigma2 = torch.median(d2[d2 > 0]) if (d2 > 0).any() else torch.tensor(1.0, dtype=x.dtype)
        return torch.exp(-d2 / (2 * sigma2))
    raise ValueError(f"unknown kernel {kernel!r}")


def _renyi_matrix_entropy(k: Tensor, alpha: float) -> float:
    lam = torch.linalg.eigvalsh(k).clamp_min(0)
    p = lam / lam.sum()
    p = p[p > 0]
    if alpha == 1.0:
        return float(-(p * p.log()).sum())
    return float(torch.log((p**alpha).sum()) / (1.0 - alpha))


def dime(
    views: Tensor,
    *,
    alpha: float = 1.0,
    n_perm: int = 10,
    kernel: str = "linear",
    seed: int = 0,
    normalization: str = "raw",
    center: bool = False,
) -> MetricResult:
    """DiME: permuted minus paired matrix-based joint entropy of two views.

    Skean et al. (2023, arXiv:2301.08164, Sec. 2.2): with K_X and K_Y the normalized N x N Gram
    matrices of the two views, DiME is the mean joint entropy S_alpha(K_X o K_Y) under random
    re-pairings minus the paired joint entropy; zero for unrelated views. The paper uses Gaussian
    kernels and alpha = 1.01. The N x N definition is kept, since the Hadamard product does not
    survive a swap to D x D covariances, so cost grows as N^3 per permutation.

    Args:
        views: Two views (2, N, D).
        alpha: Renyi order.
        n_perm: Random re-pairings averaged.
        kernel: "linear" (cosine Gram) or "rbf" (median bandwidth).
        seed: Permutation seed.
        normalization: "raw" or "max" (divide by log N).
        center: Mean-center each view before the kernel, as Skean et al. (2025) do.

    Returns:
        value: DiME.
        extras: joint_entropy, permuted_joint_entropy.
    """
    q, n, _ = _check_views(views, 2)
    if q != 2:
        raise ValueError(f"dime takes exactly two views, got {q}")
    x, y = views[0].double(), views[1].double()
    if center:
        x, y = x - x.mean(dim=0, keepdim=True), y - y.mean(dim=0, keepdim=True)
    kx, ky = _normalized_gram(x, kernel), _normalized_gram(y, kernel)
    joint = _renyi_matrix_entropy(kx * ky, alpha)
    g = torch.Generator(device=x.device).manual_seed(seed)
    permuted = 0.0
    for _ in range(n_perm):
        idx = torch.randperm(n, generator=g, device=x.device)
        permuted += _renyi_matrix_entropy(kx * ky[idx][:, idx], alpha) / n_perm
    value = permuted - joint
    if normalization == "max":
        value /= math.log(n)
    elif normalization != "raw":
        raise ValueError(f"unknown normalization {normalization!r}")
    return MetricResult(value, {"joint_entropy": joint, "permuted_joint_entropy": permuted})


_V = InputKind.VIEWS
register_metric(
    "lidar",
    inputs=_V,
    preprocess=Preprocess(),
    citation=("DBLP:conf/iclr/Thilak0SDGNSL24",),
    arxiv="2312.04000",
    tags=("paper-canonical",),
)(lidar)
register_metric(
    "alignment",
    inputs=_V,
    preprocess=Preprocess(l2=True),
    citation=("wang2020uniformity",),
    arxiv="2005.10242",
    tags=("relational",),
)(alignment)
register_metric(
    "infonce",
    inputs=_V,
    preprocess=Preprocess(center=True, l2=True),
    citation=("DBLP:journals/corr/abs-1807-03748", "DBLP:conf/icml/SkeanAZPNLS25"),
    arxiv="1807.03748",
    tags=("paper-canonical",),
)(infonce)
register_metric(
    "dime",
    inputs=_V,
    preprocess=Preprocess(),
    max_items=3000,
    citation=("skean2023dime",),
    arxiv="2301.08164",
)(dime)
