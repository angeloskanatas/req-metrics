"""Label-free selection rules on records: across runs or checkpoints at one layer, and across layers within a run.

The published rules differ in what they compare. RankMe (Garrido et al., 2023) and
LiDAR (Thilak et al., 2024) select the hyperparameter configuration with the highest
value, computed on the representation that will be used downstream, with ties broken
by the hyperparameter value. Aldeneh et al. (2024) find that effective rank tracks
downstream performance across checkpoints within a layer but cannot rank layers
against each other, so runs are compared at the same layer here. Kanatas et al.
(2026) rank the layers of one model by a metric and probe the top three. The
direction of a metric depends on the task family and the training paradigm (both
that paper and Arputharaj et al., 2026, report sign reversals), so it is an argument
of these functions, not a property of the metric.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from req_metrics.records import Records

_DIRECTIONS = ("max", "min", "target")


def value_at(records: Records, metric: str, layer: int) -> float:
    """The value of one metric at one layer of a run, nan when it was not computed."""
    for r in records:
        if r.metric == metric and r.layer == layer and r.layer_b is None:
            return float(r.value)
    return math.nan


def _score(value: float, direction: str, target: float | None) -> float:
    """Larger is better under every direction; nan sorts last."""
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction must be one of {_DIRECTIONS}, got {direction!r}")
    if math.isnan(value):
        return -math.inf
    if direction == "max":
        return value
    if direction == "min":
        return -value
    if target is None:
        raise ValueError("direction='target' needs a target value")
    return -abs(value - target)


def rank_runs(
    runs: Mapping[str, Records], metric: str, layer: int, *, direction: str = "max", target: float | None = None
) -> list[tuple[str, float]]:
    """Order runs or checkpoints by one metric at one layer, best first.

    Args:
        runs: Run or checkpoint name -> its records (one sweep each).
        metric: Registry name.
        layer: The layer compared across runs; the one that will be read downstream.
        direction: "max" (RankMe, LiDAR), "min", or "target" with `target` (alpha-ReQ near 1,
            Agrawal et al., 2022).
        target: Preferred value for direction="target".

    Returns:
        (name, value) pairs, best first; runs without the metric at that layer come last.
    """
    rows = [(name, value_at(rec, metric, layer)) for name, rec in runs.items()]
    return sorted(rows, key=lambda nv: _score(nv[1], direction, target), reverse=True)


def top_layers(
    records: Records, metric: str, k: int = 3, *, direction: str = "max", target: float | None = None
) -> list[int]:
    """The k layers of one run ranked by a metric, best first: the proxy-ranked shortlist of Kanatas et al. (2026).

    Args:
        records: One run's records.
        metric: Registry name.
        k: Number of layers to return.
        direction: The metric's correlation sign for the task family, "max" or "min", or
            "target" with `target`.
        target: Preferred value for direction="target".

    Returns:
        Layer indices, best first.
    """
    prof = records.profile(metric)
    ranked = sorted(prof, key=lambda lv: _score(lv[1], direction, target), reverse=True)
    return [layer for layer, _ in ranked[:k]]
