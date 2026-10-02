"""Named protocols: the metric set, variants and parameters of a published analysis.

A protocol is data, not code: lists of (metric, params) per input kind, the
fixed sample size, and notes on what the original record files left
unrecorded. Protocols are named after their paper (first author and year, as
a citation key) and looked up with get(); compute() is called once per input
kind with the matching names and params.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Protocol:
    """The analysis recipe of one paper.

    Attributes:
        name: Citation-style key, e.g. "kanatas2026".
        paper: Title, venue and identifier of the paper.
        pooled, frames, views, shifted: (metric, params) pairs per input kind.
        n_items: Clips per model in the published analysis.
        notes: Protocol facts and fields the original records left unrecorded.
    """

    name: str
    paper: str
    pooled: tuple[tuple[str, dict], ...]
    frames: tuple[tuple[str, dict], ...]
    views: tuple[tuple[str, dict], ...]
    shifted: tuple[tuple[str, dict], ...]
    n_items: int
    notes: tuple[str, ...] = field(default_factory=tuple)

    def names(self, kind: str) -> list[str]:
        """Registry names of the metrics of one input kind ("pooled", "frames", "views", "shifted")."""
        return [m for m, _ in getattr(self, kind)]

    def params(self, kind: str) -> dict[str, dict]:
        """Metric name -> estimator parameters for one input kind."""
        return {m: dict(p) for m, p in getattr(self, kind)}


KANATAS2026 = Protocol(
    name="kanatas2026",
    paper="Kanatas et al., What Makes a Good Layer? Assessing the Layer-Wise Intrinsic Properties of Music "
    "Foundation Models, ISMIR 2026, arXiv:2608.14819",
    pooled=(
        ("intrinsic_dimension/gride", {"scale": 8, "range_max": 8192}),  # the ID row of the correlation analysis
        ("intrinsic_dimension", {}),  # TwoNN, named in the methods
        ("effective_rank", {"spectrum": "singular", "center": True}),
        ("anisotropy", {"center": True, "l2": True}),
        ("alpha_req", {}),  # computed, dropped from the text
    ),
    frames=(
        (
            "trajectory_curvature",
            {"k": 1, "convention": "signed"},
        ),  # the methods definition; the correlation analysis read extras["abs"], negated
    ),
    views=(
        ("lidar", {"delta": 1e-6, "unbiased": False}),  # 10 views per clip
        (
            "infonce",
            {"temperature": 0.3, "center": True, "l2": True},
        ),  # 2 views; temperature from the run configuration
    ),
    shifted=(
        ("pte", {"probe": "linear", "score": "phase"}),  # 11 nonzero shifts; cpsd_rmse in the extras
    ),
    n_items=10000,
    notes=(
        "Section 3.2: 10,000 15-second clips, one per track; pooled vectors are time means for encoders and the "
        "final token for autoregressive decoders.",
        "Frame-level curvature runs used 1,000 to 5,000 clips per model, not 10,000.",
        "The InfoNCE temperature and the LiDAR view count were not recorded in the result files; the values here "
        "come from the run configuration and the paper text.",
        "The PTE row of the correlation analysis selects, per model, the configuration among {linear, mlp} x "
        "{phase, cpsd} with the largest |rho|; 'pte' reproduces the canonical linear/phase score of the text.",
    ),
)

_PROTOCOLS: dict[str, Protocol] = {KANATAS2026.name: KANATAS2026}


def get(name: str) -> Protocol:
    """Protocol by citation key, e.g. get("kanatas2026")."""
    try:
        return _PROTOCOLS[name]
    except KeyError:
        raise KeyError(f"unknown protocol {name!r}; see list_protocols()") from None


def list_protocols() -> list[str]:
    """Names of the registered protocols, by citation key."""
    return sorted(_PROTOCOLS)
