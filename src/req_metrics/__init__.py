"""Representation-quality metrics for layer-wise analysis of pretrained encoders.

Estimators are pure functions on tensors that return a MetricResult. The
registry records, for each metric, its input contract, canonical preprocessing
and source citation. Populations (pooled, frames, tokens) and I/O belong to the
pipeline, not to the estimators.
"""

__version__ = "0.0.1"


from req_metrics._types import InputKind, MetricResult, Preprocess
from req_metrics.metrics.spectral import alpha_req, anisotropy_cosine, anisotropy_spectral, effective_rank, eigenvalue_early_enrichment, gaussianity, matrix_entropy, participation_ratio, sparsity, spectral_entropy
from req_metrics.neighbors import Neighbors
from req_metrics.preprocess import apply_preprocess, center, l2_normalize, standardize
from req_metrics.registry import MetricSpec, get_metric, list_metrics, register_metric
from req_metrics.spectrum import Spectrum

__all__ = [
    "InputKind",
    "MetricResult",
    "Preprocess",
    "Spectrum",
    "Neighbors",
    "MetricSpec",
    "register_metric",
    "get_metric",
    "list_metrics",
    "apply_preprocess",
    "center",
    "l2_normalize",
    "standardize",
    "effective_rank",
    "spectral_entropy",
    "matrix_entropy",
    "alpha_req",
    "anisotropy_spectral",
    "anisotropy_cosine",
    "participation_ratio",
    "eigenvalue_early_enrichment",
    "gaussianity",
    "sparsity",
]
