"""Sample-size diagnostics: does a metric value depend on how many items it saw?

convergence() recomputes a metric on repeated random subsets at several fractions and
reports the mean, the relative spread and the change toward the full sample, following the
subsample protocol of Ansuini et al. (2019) and Arputharaj et al. (2026).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from req_metrics.pipeline import _points_sweep
from req_metrics.registry import get_metric


@dataclass(frozen=True)
class ConvergenceRow:
    """One fraction of the sample: items used, mean and relative std over repeats, relative change from the previous fraction."""

    fraction: float
    n_items: int
    mean: float
    rel_std: float
    rel_change: float


@dataclass(frozen=True)
class Convergence:
    """Subsample curve of one metric on one point cloud."""

    metric: str
    rows: tuple[ConvergenceRow, ...]

    @property
    def converged(self) -> bool:
        """True when the last step changed the mean by under 2 percent and the spread is under 5 percent."""
        last = self.rows[-1]
        return abs(last.rel_change) < 0.02 and last.rel_std < 0.05

    def to_markdown(self) -> str:
        """The curve as a Markdown table."""
        head = f"| fraction | N | {self.metric} | rel. std | rel. change |\n|---|---|---|---|---|\n"
        return head + "\n".join(
            f"| {r.fraction:.2f} | {r.n_items} | {r.mean:.4g} | {100 * r.rel_std:.1f}% | {100 * r.rel_change:+.1f}% |"
            for r in self.rows
        )


def convergence(
    x: Tensor,
    metric: str,
    *,
    fractions: Sequence[float] = (0.125, 0.25, 0.5, 1.0),
    repeats: int = 5,
    seed: int = 0,
    params: dict[str, Any] | None = None,
) -> Convergence:
    """Recompute a point metric on random subsets of x at several fractions.

    Args:
        x: Points (N, D).
        metric: Registry name of a point metric.
        fractions: Fractions of N to evaluate, ascending; 1.0 uses every row once (no repeats).
        repeats: Random subsets per fraction below 1.0.
        seed: Subsampling seed.
        params: Estimator keyword arguments.

    Returns:
        Convergence with one row per fraction; rel_change of the first row is nan.
    """
    spec = get_metric(metric)
    n = x.shape[0]
    gen = torch.Generator().manual_seed(seed)
    rows: list[ConvergenceRow] = []
    prev = None
    for f in fractions:
        m = max(2, int(round(f * n)))
        reps = 1 if m >= n else repeats
        values = []
        for _ in range(reps):
            idx = torch.randperm(n, generator=gen)[:m] if m < n else torch.arange(n)
            values.append(float(_points_sweep(x[idx], [spec], {metric: dict(params or {})})[metric][0]))
        v = torch.tensor(values, dtype=torch.float64)
        mean = float(v.mean())
        std = float(v.std()) if reps > 1 else 0.0
        rel_change = float("nan") if prev is None or prev == 0 else (mean - prev) / abs(prev)
        rows.append(ConvergenceRow(f, m, mean, std / abs(mean) if mean else float("nan"), rel_change))
        prev = mean
    return Convergence(metric, tuple(rows))
