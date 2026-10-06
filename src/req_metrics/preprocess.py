"""Row-wise preprocessing along the sample axis.

All functions treat dim -2 as the sample axis and dim -1 as features, so they
apply unchanged to (N, D) points, (T, D) trajectories and (q, N, D) views,
where each view is processed on its own.
"""

import torch
from torch import Tensor

from req_metrics._types import Preprocess


def center(x: Tensor) -> Tensor:
    """Subtract the per-feature mean over the sample axis.

    A cloud whose rows are all equal up to rounding becomes exactly zero, so a complete collapse
    is read as rank 0 by every spectral estimator rather than as the rounding residue of the mean.
    """
    xc = x - x.mean(dim=-2, keepdim=True)
    eps = torch.finfo(x.dtype).eps if x.is_floating_point() else 0.0
    scale = float(x.norm())
    if scale > 0.0 and float(xc.norm()) <= 1e3 * eps * scale:
        return torch.zeros_like(xc)
    return xc


def standardize(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Center and divide each feature by its standard deviation over the sample axis."""
    xc = center(x)
    return xc / xc.std(dim=-2, unbiased=False, keepdim=True).clamp_min(eps)


def l2_normalize(x: Tensor, eps: float = 1e-12) -> Tensor:
    """Scale each sample to unit Euclidean norm."""
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def apply_preprocess(x: Tensor, pre: Preprocess) -> Tensor:
    """Apply a Preprocess specification in the fixed order center, standardize, l2."""
    if x.ndim < 2:
        raise ValueError(f"expected at least 2 dims (samples, features), got shape {tuple(x.shape)}")
    if pre.standardize:
        x = standardize(x)
    elif pre.center:
        x = center(x)
    if pre.l2:
        x = l2_normalize(x)
    return x
