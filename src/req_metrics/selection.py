"""Label-free selection rules on records: across runs at one layer, and across layers of a run.

RankMe (Garrido et al., 2023, ICML, arXiv:2210.02885) and LiDAR (Thilak et al., 2024, ICLR,
arXiv:2312.04000) select the configuration with
the highest value on the representation used downstream. Aldeneh et al. (2025, ICASSP, arXiv:2409.10787) find that
effective rank orders checkpoints within a layer but not layers, so runs are compared at one
layer. Kanatas et al. (2026, ISMIR, arXiv:2608.14819) rank the layers of a model and probe the top three. Dai et al.
(2025, NeurIPS, arXiv:2510.17299) keep the best local maxima along training, so that one broad peak does not fill the
selection. A metric's direction depends on the task family and paradigm, so it is an argument.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

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
            Agrawal et al., 2022, NeurIPS).
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


def top_local_maxima(
    values: Sequence[float], t: int = 3, *, window: int = 2, direction: str = "max", target: float | None = None
) -> list[int]:
    """The t best local maxima of an ordered sequence of candidates, best first.

    Dai et al. (2025, NeurIPS, Alg. 1, with their released model_selection.py): candidate i is kept when its
    score is the best of the candidates i - window .. i + window (clipped at the ends, ties kept), and the
    kept candidates are ranked by score, earlier first on ties. They apply it to checkpoints in training
    order with window 2 and t 3, on their dense structure estimator; the order can as well be depth.

    Args:
        values: One value per candidate, in order (checkpoints by training step, or layers by depth).
        t: Number of candidates to return.
        window: Half-width of the neighborhood a local maximum must win.
        direction: "max", "min", or "target" with `target`.
        target: Preferred value for direction="target".

    Returns:
        Indices into values, best first; nan values are never selected.
    """
    scores = [_score(float(v), direction, target) for v in values]
    kept = [
        i
        for i, sc in enumerate(scores)
        if sc != -math.inf and sc == max(scores[max(0, i - window) : min(len(scores), i + window + 1)])
    ]
    return sorted(kept, key=lambda i: (-scores[i], i))[:t]
