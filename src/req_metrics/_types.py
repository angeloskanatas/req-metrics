"""Core types shared by estimators, registry and pipeline."""

from dataclasses import dataclass, field
from enum import Enum


class InputKind(str, Enum):
    """What an estimator consumes. Shapes use N clips, T frames, D dims, q views."""

    POINTS = "points"  # (N, D)
    TRAJECTORY = "trajectory"  # (T, D), time-ordered frames of one clip
    VIEWS = "views"  # (q, N, D), q augmented views of the same N clips
    SHIFTED = "shifted"  # (N, D) plus {semitones: (N, D)} transposed copies
    TOKENS = "tokens"  # (N, T, D)
    PAIR = "pair"  # two point clouds (N, D_a), (N, D_b) describing the same N items
    JACOBIAN = "jacobian"  # (B, k, D) Jacobian-vector products: B inputs, k input directions


@dataclass(frozen=True)
class Preprocess:
    """Row-wise preprocessing applied along the sample axis before an estimator.

    Order of application: center, standardize, l2. Estimators whose definition
    is on the covariance center internally and declare Preprocess() here.
    """

    center: bool = False
    standardize: bool = False
    l2: bool = False

    def describe(self) -> str:
        """Short text of the preprocessing steps, as stored in records."""
        parts = [n for n in ("center", "standardize", "l2") if getattr(self, n)]
        return "+".join(parts) if parts else "none"


@dataclass
class MetricResult:
    """One estimator output: the headline value and named secondary quantities."""

    value: float
    extras: dict[str, float] = field(default_factory=dict)
