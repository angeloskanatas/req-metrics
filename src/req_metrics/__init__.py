"""Label-free representation-quality metrics for layer-wise analysis of pretrained models and for monitoring during training.

Estimators are pure functions on tensors that return a MetricResult. The
registry records, for each metric, its input contract, canonical preprocessing
and source citation. Populations (pooled, frames, tokens) and I/O belong to the
pipeline, not to the estimators.
"""

__version__ = "0.1.0"

from req_metrics import protocols
from req_metrics._types import InputKind, MetricResult, Preprocess
from req_metrics.layouts import (
    frame_tokens_to_pooled,
    grid_to_pooled,
    grid_to_tokens,
    grid_to_trajectory,
    stack_clips,
    strip_prefix_tokens,
)
from req_metrics.metrics.clustering import cluster_quality
from req_metrics.metrics.compare import cka, information_imbalance, neighborhood_overlap, svcca
from req_metrics.metrics.dimension import gride, mle, mlid, mst_dimension, twonn
from req_metrics.metrics.equivariance import pte
from req_metrics.metrics.local_geometry import local_rectifiability, neighborhood_curvature
from req_metrics.metrics.norms import embedding_norm
from req_metrics.metrics.relational import normalized_std, self_clustering, uniformity
from req_metrics.metrics.spectral import (
    alpha_req,
    anisotropy_cosine,
    anisotropy_spectral,
    effective_rank,
    eigenvalue_early_enrichment,
    gaussianity,
    matrix_entropy,
    participation_ratio,
    sparsity,
    spectral_entropy,
)
from req_metrics.metrics.tokens import cls_patch_cosine, token_cosine, token_gram_drift, token_norm_outliers
from req_metrics.metrics.trajectory import trajectory_curvature
from req_metrics.metrics.views import alignment, dime, infonce, lidar
from req_metrics.monitor import (
    LayerMonitor,
    OnlineBuffer,
    csv_sink,
    json_sink,
    make_pooler,
    metric_key_prefix,
    monitor_loader,
    resolve_layers,
    tensorboard_sink,
    wandb_sink,
)
from req_metrics.neighbors import Neighbors
from req_metrics.pipeline import choose_indices, compute, compute_pairs
from req_metrics.preprocess import apply_preprocess, center, l2_normalize, standardize
from req_metrics.records import Record, Records
from req_metrics.registry import MetricSpec, get_metric, list_metrics, register_metric
from req_metrics.selection import rank_runs, top_layers, value_at
from req_metrics.spectrum import Spectrum
from req_metrics.stability import Convergence, convergence
from req_metrics.view_construction import ShiftSpec, ViewSpec, make_shifted, make_views, stack_views

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
    "twonn",
    "gride",
    "mle",
    "mlid",
    "mst_dimension",
    "neighborhood_curvature",
    "local_rectifiability",
    "information_imbalance",
    "trajectory_curvature",
    "lidar",
    "infonce",
    "dime",
    "ViewSpec",
    "make_views",
    "stack_views",
    "ShiftSpec",
    "make_shifted",
    "pte",
    "token_norm_outliers",
    "token_cosine",
    "cls_patch_cosine",
    "token_gram_drift",
    "embedding_norm",
    "Record",
    "Records",
    "compute",
    "compute_pairs",
    "choose_indices",
    "protocols",
    "LayerMonitor",
    "OnlineBuffer",
    "metric_key_prefix",
    "csv_sink",
    "json_sink",
    "tensorboard_sink",
    "wandb_sink",
    "resolve_layers",
    "make_pooler",
    "monitor_loader",
    "grid_to_trajectory",
    "grid_to_tokens",
    "grid_to_pooled",
    "frame_tokens_to_pooled",
    "strip_prefix_tokens",
    "stack_clips",
    "alignment",
    "normalized_std",
    "self_clustering",
    "uniformity",
    "convergence",
    "Convergence",
    "neighborhood_overlap",
    "cka",
    "svcca",
    "cluster_quality",
    "rank_runs",
    "top_layers",
    "value_at",
]
