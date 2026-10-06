"""Label-free selection rules on records: across runs at one layer, and across layers of a run.

RankMe (Garrido et al., 2023) and LiDAR (Thilak et al., 2024) select the configuration with
the highest value on the representation used downstream. Aldeneh et al. (2024) find that
effective rank orders checkpoints within a layer but not layers, so runs are compared at one
layer. Kanatas et al. (2026) rank the layers of a model and probe the top three. A metric's
direction depends on the task family and paradigm, so it is an argument.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from req_metrics.records import Records

_DIRECTIONS = ("max", "min", "target")


def value_at(records: Records, metric: str, layer: int) -> float:
    """The value of one metric at one layer of a run, nan when it was not computed."""
    rows = [r for r in records if r.metric == metric and r.layer == layer and r.layer_b is None]
    if len(rows) > 1:
        raise ValueError(f"{len(rows)} records of {metric!r} at layer {layer}: pass one sweep of one model")
    return float(rows[0].value) if rows else math.nan


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
    records: Records,
    metric: str,
    k: int = 3,
    *,
    direction: str = "max",
    target: float | None = None,
    model: str | None = None,
) -> list[int]:
    """The k layers of one run ranked by a metric, best first.

    Args:
        records: One run's records.
        metric: Registry name.
        k: Number of layers to return.
        direction: "max", "min", or "target" with `target`.
        target: Preferred value for direction="target".
        model: Restrict to the records of one model label when the records hold several.

    Returns:
        Layer indices, best first.
    """
    prof = records.profile(metric, model)
    if not prof:
        raise ValueError(f"no records of {metric!r}" + (f" for model {model!r}" if model is not None else ""))
    if len({layer for layer, _ in prof}) != len(prof):
        raise ValueError("several records per layer: pass one sweep, and set model when the records hold several")
    ranked = sorted(prof, key=lambda lv: _score(lv[1], direction, target), reverse=True)
    return [layer for layer, _ in ranked[:k]]
