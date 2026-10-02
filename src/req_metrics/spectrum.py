"""Singular spectrum of a centered point cloud, computed once and shared.

Effective rank, spectral entropy, alpha-ReQ, spectral anisotropy, participation
ratio and eigenvalue early enrichment are all functions of this object.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Spectrum:
    """Descending singular values of the centered matrix and its shape.

    Singular values are computed in float64; the k = min(N, D) values are
    returned, so the D - k unused ambient directions carry zero variance and
    must be padded by metrics defined over all D directions.
    """

    singular_values: Tensor
    n: int
    d: int

    @classmethod
    def from_points(cls, x: Tensor, center: bool = True, dtype: torch.dtype = torch.float64) -> Spectrum:
        """Singular values of x on its own device. float64 by default for the tail of the
        spectrum; pass torch.float32 on GPUs with slow double precision when only the
        leading part of the spectrum matters."""
        if x.ndim != 2:
            raise ValueError(f"expected (N, D), got shape {tuple(x.shape)}")
        xd = x.to(dtype)
        if center:
            xd = xd - xd.mean(dim=0, keepdim=True)
        s = torch.linalg.svdvals(xd)
        return cls(singular_values=s, n=int(x.shape[0]), d=int(x.shape[1]))

    @property
    def eigenvalues(self) -> Tensor:
        """Eigenvalues of the scatter matrix, s^2 (covariance up to the 1/N factor)."""
        return self.singular_values.square()

    def normalized(self, kind: str = "singular") -> Tensor:
        """Spectrum as a probability vector: singular values or eigenvalues over their sum."""
        v = self.singular_values if kind == "singular" else self.eigenvalues
        total = v.sum()
        if total <= 0:
            raise ValueError("zero spectrum")
        return v / total

    def padded_eigenvalues(self) -> Tensor:
        """Eigenvalues over all D ambient directions, zeros for unused ones."""
        lam = self.eigenvalues
        if lam.numel() < self.d:
            lam = torch.nn.functional.pad(lam, (0, self.d - lam.numel()))
        return lam
