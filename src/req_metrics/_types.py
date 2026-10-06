"""Core types shared by estimators, registry and pipeline."""

from dataclasses import dataclass, field
from enum import Enum


class InputKind(str, Enum):
    """What an estimator consumes. Shapes use N samples, T tokens, D dims, q views."""

    POINTS = "points"  # (N, D)
    TRAJECTORY = "trajectory"  # (T, D), time-ordered tokens of one sample
    VIEWS = "views"  # (q, N, D), q augmented views of the same N samples
    SHIFTED = "shifted"  # (N, D) plus {semitones: (N, D)} transposed copies
    TOKENS = "tokens"  # (T, D), the token field of one sample
    PAIR = "pair"  # two point clouds (N, D_a), (N, D_b) describing the same N items
    TOKEN_PAIR = "token_pair"  # (T, D) token field and a (T, D_ref) reference field of the same sample
    JACOBIAN = "jacobian"  # (B, k, M) Jacobian sketches: B inputs, k rows per input


@dataclass(frozen=True)
class Preprocess:
    """Preprocessing of a (..., N, D) tensor before an estimator.

    center subtracts the feature means over the N samples, standardize also divides
    each feature by its standard deviation, l2 scales each sample to unit norm;
    applied in that order.
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
