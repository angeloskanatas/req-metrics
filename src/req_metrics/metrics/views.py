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
    """LiDAR: effective rank of the LDA matrix with clips as classes and views as samples.

    Thilak et al. (2024, ICLR, arXiv:2312.04000), Eq. 1-4. Each clean clip
    is a surrogate class and its q views the within-class samples:
    S_b is the scatter of the class means around the grand mean, S_w the
    scatter of the views around their class mean plus delta I, and
    LiDAR = exp(-sum p_i log p_i) over the eigenvalues of
    S_w^{-1/2} S_b S_w^{-1/2}, normalized by their sum. It counts the
    directions that separate clips after whitening the variability the
    augmentations induce, so it tracks the training objective's own
    invariances rather than raw covariance rank. The paper uses unbiased
    estimates (n-1 and n(q-1) denominators), recommends n above the feature
    width because rank(S_b) <= n, and finds q = 10 within one percent of
    q = 50. Computed in float64 with symmetric eigendecompositions; the
    epsilon the paper adds inside the logarithm is omitted, zero eigenvalues
    contributing nothing. The paper does not print delta; 1e-4 is the value
    of the reference implementation of Skean et al. (2025). Protocol of Kanatas et al. (2026):
    10,000 clips, 10 views, biased denominators, delta 1e-6, a shared
    augmentation chain across models with task-defining augmentations
    removed per task family, whereas the original work uses each method's
    own training augmentations. On autoregressive decoders the layer-wise
    correlation with downstream accuracy reverses sign in Kanatas et al. (2026).

    Args:
        views: Augmented representations, shape (q, N, D), q >= 2.
        delta: Ridge added to the within-class scatter.
        unbiased: Use n-1 and n(q-1) denominators (paper) or n and nq.
        max_eigenvalues: Keep only the largest eigenvalues, if set.

    Returns:
        value: LiDAR effective rank.
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

    Wang and Isola (2020, ICML, arXiv:2005.10242, Sec. 4.1.1):
    L_align = E ||f(x) - f(y)||^alpha over positive pairs, alpha = 2 in Wang and
    Isola (2020) and their reference code; 0 for a perfectly aligned encoder, 2 for unrelated
    unit vectors in high dimension. Averaged over all pairs of views and all
    clips. The companion of uniformity: the two together are the asymptotic
    form of the contrastive loss (Theorem 1), and encoders with low values of
    both perform best in that study. With two views that are a sample and its
    transformed copy, 1 - value / 2 is the cosine similarity that Plachouras et
    al. (2025, IJCNN) report as an invariance score over a transformation's
    parameter range.

    Args:
        views: Augmented representations, shape (q, N, D), q >= 2; rows are L2-normalized here.
        alpha: Distance exponent.

    Returns:
        value: L_align.
        extras: n_pairs of views averaged.
    """
    q, n, d = _check_views(views, 2)
    u = torch.nn.functional.normalize(views.double(), dim=-1)
    total, pairs = 0.0, 0
    for a in range(q):
        for b in range(a + 1, q):
            total += float((u[a] - u[b]).norm(dim=-1).pow(alpha).mean())
            pairs += 1
    return MetricResult(total / pairs, {"n_pairs": float(pairs)})


def infonce(views: Tensor, *, temperature: float = 0.1, center: bool = True, l2: bool = True) -> MetricResult:
    """Full-batch InfoNCE loss between two views of the same clips.

    van den Oord, Li and Vinyals (2018, arXiv:1807.03748), Eq. 4: the
    cross-entropy of identifying each clip's second view among all N second
    views, with logits the scaled similarities. Rows are centered and
    L2-normalized so logits are cosines over the temperature, the
    preprocessing of the Skean et al. (2025, ICML) reference implementation and of every
    stored result file of Kanatas et al. (2026). Lower loss means the layer is more invariant
    to the augmentations relative to clip identity. The bound of van den Oord et al.,
    I >= log N - L is reported in nats and as the fraction 1 - L / log N;
    for unrelated views the loss exceeds log N by about half the variance of
    the scaled similarities, so the bound can be negative. The temperature
    is a protocol constant that must be recorded: the
    reference implementation uses 0.1, the runs of Kanatas et al. (2026) 0.3. The shared
    augmentation chain contained a pitch shift, which confounds this metric
    on tonal tasks unless that augmentation is removed.

    Args:
        views: Two views, shape (2, N, D).
        temperature: Softmax temperature.
        center: Mean-center each view over clips.
        l2: Scale each row to unit norm.

    Returns:
        value: InfoNCE loss in nats.
        extras: mi_lower_bound (1 - L / log N), mi_bound_nats (log N - L).
    """
    q, n, _ = _check_views(views, 2)
    if q != 2:
        raise ValueError(f"infonce takes exactly two views, got {q}")
    v = apply_preprocess(views.double(), Preprocess(center=center, l2=l2))
    logits = v[0] @ v[1].T / temperature
    labels = torch.arange(n, device=logits.device)
    loss = float(torch.nn.functional.cross_entropy(logits, labels))
    return MetricResult(loss, {"mi_lower_bound": 1.0 - loss / math.log(n), "mi_bound_nats": math.log(n) - loss})


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
) -> MetricResult:
    """DiME: permuted minus paired matrix-based joint entropy of two views.

    Skean et al. (2023, arXiv:2301.08164): with K_X and K_Y the N x N
    normalized Gram matrices of the two views (unit diagonal), the joint
    entropy is S_alpha(K_X o K_Y) over the Hadamard product (their Sec. 2.2),
    and DiME is the expected joint entropy under random re-pairings of one
    view minus the paired joint entropy. It behaves like a mutual
    information between the views and is zero when they are unrelated. The
    paper uses Gaussian kernels with alpha = 1.01; the Skean et al. (2025)
    layer-wise reference implementation uses the linear Gram of the states. Both
    that implementation and the analysis code of Kanatas et al. (2026) replaced the N x N Gram
    matrices by D x D
    covariances whenever N > D; the Hadamard product does not commute with
    that swap, so those values are not the published quantity. This
    implementation follows the definition and therefore costs an N x N
    eigendecomposition per permutation; subsample to a few thousand clips.
    Excluded from the headline set of Kanatas et al. (2026) for low sign consistency.

    Args:
        views: Two views, shape (2, N, D).
        alpha: Renyi order; 1.0 gives von Neumann entropy.
        n_perm: Random re-pairings averaged for the baseline.
        kernel: "linear" (cosine Gram of unit rows) or "rbf" (median bandwidth).
        seed: Permutation seed.
        normalization: "raw" or "max" (divide by log N).

    Returns:
        value: DiME.
        extras: joint_entropy, permuted_joint_entropy.
    """
    q, n, _ = _check_views(views, 2)
    if q != 2:
        raise ValueError(f"dime takes exactly two views, got {q}")
    x, y = views[0].double(), views[1].double()
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
    tags=("computed-not-in-paper",),
)(dime)
