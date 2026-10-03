"""Token-field metrics of one clip's tokens (T, D), read before pooling.

High-norm outlier tokens, their channel concentration, the pairwise cosine structure, and its
drift during training.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from req_metrics._types import InputKind, MetricResult, Preprocess
from req_metrics.registry import register_metric


def _check_tokens(x: Tensor, min_tokens: int = 2) -> None:
    if x.ndim != 2 or x.shape[0] < min_tokens:
        raise ValueError(f"expected (T >= {min_tokens}, D) tokens, got shape {tuple(x.shape)}")


def token_norm_outliers(tokens: Tensor, *, factor: float = 3.0, cutoff: float | None = None) -> MetricResult:
    """Fraction of high-norm tokens, and the channel concentration of the largest one.

    Darcet et al. (2024, ICLR, arXiv:2309.16588) report high-norm tokens with an absolute cutoff
    (150 for DINOv2) that varies across models, so a token here is an outlier when its norm
    exceeds factor times the median. The extras describe the largest token's channel energies,
    after Jiang et al. (2025) and Sun et al. (2024).

    Args:
        tokens: One clip's tokens (T, D), class tokens removed.
        factor: Cutoff relative to the median norm.
        cutoff: Absolute cutoff overriding factor.

    Returns:
        value: fraction of outlier tokens.
        extras: max_over_median, median_norm, max_norm, channel_participation_ratio,
            top1_channel_mass.
    """
    _check_tokens(tokens)
    x = tokens.double()
    norms = x.norm(dim=1)
    median = norms.median()
    threshold = cutoff if cutoff is not None else factor * float(median)
    energy = x[norms.argmax()].square()
    pr = float(energy.sum().square() / energy.square().sum()) if energy.sum() > 0 else float("nan")
    return MetricResult(
        float((norms > threshold).double().mean()),
        {
            "max_over_median": float(norms.max() / median) if median > 0 else float("nan"),
            "median_norm": float(median),
            "max_norm": float(norms.max()),
            "channel_participation_ratio": pr,
            "top1_channel_mass": float(energy.max() / energy.sum()) if energy.sum() > 0 else float("nan"),
        },
    )


def token_cosine(tokens: Tensor, *, n_tokens: int = 64, seed: int = 0) -> MetricResult:
    """Mean cosine similarity between distinct tokens of one clip.

    Marouani et al. (2026, ICLR, arXiv:2602.08626): near 1 for a collapsed token field. Exact on
    up to n_tokens tokens sampled without replacement.

    Args:
        tokens: One clip's tokens (T, D), class tokens removed.
        n_tokens: Tokens sampled when T exceeds it.
        seed: Sampling seed.

    Returns:
        value: mean off-diagonal cosine.
        extras: n_tokens.
    """
    _check_tokens(tokens)
    x = tokens.double()
    if x.shape[0] > n_tokens:
        g = torch.Generator(device=x.device).manual_seed(seed)
        x = x[torch.randperm(x.shape[0], generator=g, device=x.device)[:n_tokens]]
    u = F.normalize(x, dim=1)
    n = u.shape[0]
    total = u.sum(dim=0)
    return MetricResult(float((total @ total - n) / (n * (n - 1))), {"n_tokens": float(n)})


def cls_patch_cosine(tokens_with_cls: Tensor, *, n_prefix: int = 1) -> MetricResult:
    """Mean cosine between the class token(s) and the patch tokens of one clip.

    Marouani et al. (2026, ICLR, arXiv:2602.08626) show class and patch tokens diverging at
    specific layers.

    Args:
        tokens_with_cls: The clip's full token sequence (n_prefix + T, D).
        n_prefix: Leading class or register tokens.

    Returns:
        value: mean cosine over (class, patch) pairs.
        extras: std over patches.
    """
    _check_tokens(tokens_with_cls, min_tokens=n_prefix + 1)
    x = tokens_with_cls.double()
    cls, patches = F.normalize(x[:n_prefix], dim=1), F.normalize(x[n_prefix:], dim=1)
    cos = cls @ patches.T
    return MetricResult(float(cos.mean()), {"std": float(cos.std(unbiased=False))})


def token_gram_drift(tokens: Tensor, reference: Tensor) -> MetricResult:
    """Drift of a clip's token cosine Gram matrix against a reference field.

    DINOv3's Gram anchoring term (Simeoni et al., 2025, arXiv:2508.10104, Sec. 4): ||X_S X_S^T -
    X_G X_G^T||_F^2 on L2-normalized tokens, divided by P^2 here so clips of different length
    compare. As a monitor, the reference is an earlier sweep of the same model. Called directly;
    compute() and compute_pairs() do not route it.

    Args:
        tokens: Current token field (T, D).
        reference: Reference field of the same clip (T, D_ref).

    Returns:
        value: mean squared difference of the two cosine Gram matrices.
        extras: none.
    """
    _check_tokens(tokens)
    if reference.ndim != 2 or reference.shape[0] != tokens.shape[0]:
        raise ValueError("reference must have the same number of tokens as tokens")
    gs = F.normalize(tokens.double(), dim=1)
    gr = F.normalize(reference.double(), dim=1)
    return MetricResult(float(((gs @ gs.T) - (gr @ gr.T)).square().mean()), {})


_T = InputKind.TOKENS
register_metric(
    "token_norm_outliers",
    inputs=_T,
    preprocess=Preprocess(),
    citation=("darcet2024registers", "jiang2025noregisters", "sun2024massive"),
    arxiv="2309.16588",
)(token_norm_outliers)
register_metric(
    "token_cosine", inputs=_T, preprocess=Preprocess(), citation=("marouani2026clspatch",), arxiv="2602.08626"
)(token_cosine)
register_metric(
    "cls_patch_cosine", inputs=_T, preprocess=Preprocess(), citation=("marouani2026clspatch",), arxiv="2602.08626"
)(cls_patch_cosine)
register_metric(
    "token_gram_drift",
    inputs=InputKind.PAIR,
    preprocess=Preprocess(),
    citation=("simeoni2025dinov3",),
    arxiv="2508.10104",
    description="Squared Frobenius drift of a clip's token cosine Gram matrix against a reference field.",
)(token_gram_drift)
