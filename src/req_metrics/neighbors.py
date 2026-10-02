"""Sorted nearest-neighbor distances of a point cloud, computed once and shared.

TwoNN, GRIDE, the Levina-Bickel MLE, mLID and the kNN curvature all read from
this table. Column 0 is the point itself (distance 0), so column j is the
j-th nearest neighbor, matching the convention of dadapy and of the
estimators' papers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _cdist_mode(dtype: torch.dtype) -> str:
    return "use_mm_for_euclid_dist_if_necessary" if dtype == torch.float64 else "donot_use_mm_for_euclid_dist"


@dataclass(frozen=True)
class Neighbors:
    """Ascending neighbor distances and indices, shape (N, k + 1), self at column 0."""

    distances: Tensor
    indices: Tensor

    @property
    def n(self) -> int:
        """Number of points."""
        return int(self.distances.shape[0])

    @property
    def k(self) -> int:
        """Number of neighbors per point, the point itself excluded."""
        return int(self.distances.shape[1]) - 1

    @classmethod
    def from_points(cls, x: Tensor, k: int, chunk: int = 1024, dtype: torch.dtype = torch.float64) -> Neighbors:
        """Exact Euclidean k nearest neighbors, on the device of x.

        In float64 the matmul expansion of the distances is used (BLAS, about
        three times faster) and agrees with the direct path to 1e-15 on the
        nearest-neighbor distances. In float32 the direct path is kept: the
        expansion loses the small distances that ratio estimators depend on,
        and under TF32 they are wrong. Rows are processed in
        chunks so memory stays at chunk x N.
        """
        if x.ndim != 2:
            raise ValueError(f"expected (N, D), got shape {tuple(x.shape)}")
        xd = x.to(dtype)
        n = xd.shape[0]
        k = min(k, n - 1)
        if k < 1:
            raise ValueError("need at least two points")
        dists, idxs = [], []
        for start in range(0, n, chunk):
            block = xd[start : start + chunk]
            d = torch.cdist(block, xd, compute_mode=_cdist_mode(xd.dtype))
            rows = torch.arange(block.shape[0], device=d.device)
            d[rows, start + rows] = -1.0  # pin the point itself to column 0 ahead of exact duplicates
            v, i = torch.topk(d, k + 1, dim=1, largest=False)
            v[:, 0] = 0.0
            dists.append(v)
            idxs.append(i)
        return cls(torch.cat(dists), torch.cat(idxs))

    def ratios(self, outer: int, inner: int) -> Tensor:
        """Per-point distance ratio r_outer / r_inner (GRIDE's generalized ratio)."""
        return self.distances[:, outer] / self.distances[:, inner]
