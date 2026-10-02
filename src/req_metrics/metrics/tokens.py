"""Token-population health of one clip's token field (T, D): norm outliers, cosine structure, Gram drift.

These read the token field of a transformer layer directly, before any pooling,
and watch for the artifacts reported in vision and language transformers:
high-norm outlier tokens, their concentration in a few channels, collapse of
the pairwise cosine structure, and drift of that structure during training.
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
    """Fraction of high-norm tokens, and how concentrated the largest token is in a few channels.

    Darcet et al. (2024, ICLR, arXiv:2309.16588) find that large ViTs
    repurpose a few low-information patches as high-norm tokens; they use an
    absolute cutoff on the token norm (150 for DINOv2) and observe that the value
    varies across models, so the default here is relative: a token is an
    outlier when its norm exceeds factor times the median token norm. Jiang
    et al. (2025, arXiv:2506.08010) trace the high norms to a sparse set of
    "register neurons", and Sun et al. (2024, COLM, arXiv:2402.17762)
    describe the analogous massive activations in language models as a few
    channels orders of magnitude above the median. The extras therefore
    report the participation ratio of the squared channel energies of the
    highest-norm token (near 1 when one channel carries it, near D when the
    energy is spread) and the share of its largest channel.

    Args:
        tokens: One clip's tokens, shape (T, D), class tokens removed.
        factor: Relative cutoff on the median norm.
        cutoff: Absolute norm cutoff overriding factor, as in Darcet et al.

    Returns:
        value: fraction of outlier tokens.
        extras: max_over_median, median_norm, max_norm, channel_participation_ratio, top1_channel_mass.
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

    The patch-to-patch similarity tracked by Marouani et al. (2026, ICLR,
    arXiv:2602.08626) around each block of a ViT, here on the layer output.
    Near 1 means the tokens are interchangeable (a collapsed field), near 0
    means they spread out. Computed exactly on up to n_tokens tokens sampled
    without replacement.

    Args:
        tokens: One clip's tokens, shape (T, D), class tokens removed.
        n_tokens: Tokens sampled when T exceeds it.
        seed: Sampling seed.

    Returns:
        value: mean off-diagonal cosine.
        extras: n_tokens used.
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

    Marouani et al. (2026, ICLR, arXiv:2602.08626) show that [CLS] and patch
    tokens diverge at specific layers inside each block although they share
    the same operators, and that disentangling them improves dense features.
    Pass the token sequence with its leading class token(s) as the model
    emits it; the mean is over all (class, patch) pairs.

    Args:
        tokens_with_cls: One clip's full token sequence, shape (n_prefix + T, D).
        n_prefix: Number of leading class or register tokens.

    Returns:
        value: mean cosine between class and patch tokens.
        extras: std over patches.
    """
    _check_tokens(tokens_with_cls, min_tokens=n_prefix + 1)
    x = tokens_with_cls.double()
    cls, patches = F.normalize(x[:n_prefix], dim=1), F.normalize(x[n_prefix:], dim=1)
    cos = cls @ patches.T
    return MetricResult(float(cos.mean()), {"std": float(cos.std(unbiased=False))})


def token_gram_drift(tokens: Tensor, reference: Tensor) -> MetricResult:
    """Squared Frobenius distance between the cosine Gram matrices of two token fields of the same clip.

    The quantity DINOv3's Gram anchoring regularizes (Simeoni et al., 2025,
    arXiv:2508.10104, Sec. 4): with X_S and X_G the (P, d) L2-normalized
    local features of the current and of a reference model, the loss is
    ||X_S X_S^T - X_G X_G^T||_F^2, which pins the similarity structure while
    letting the features move. Divided by P^2 here so clips of different
    length are comparable. As a monitor, the reference is an earlier sweep of
    the same model; rising drift late in training together with weakening
    dense probes is the degradation signature that motivated the loss. A
    two-field metric: call it directly with the two token fields; compute()
    and compute_pairs() do not route it.

    Args:
        tokens: Current token field, shape (T, D).
        reference: Reference token field of the same clip, shape (T, D_ref).

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
