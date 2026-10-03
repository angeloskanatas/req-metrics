# Changelog

All notable changes to this project are documented here. The format follows
Keep a Changelog; versions follow Semantic Versioning.

## [Unreleased]

- One record per computation: `effective_rank/variance`, `trajectory_curvature/abs`,
  `trajectory_curvature/recurve` and `pte/cpsd` are removed; their values are in the extras of
  `effective_rank`, `trajectory_curvature` and `pte`, and the `spectrum`, `convention`, `normalize`
  and `score` arguments choose the value. The unpublished concatenated-probe method of `pte` is
  removed. Frames records carry the mean of the per-clip extras. The site export writes every
  variant the site stores from those extras. 37 metrics.
- `cluster_quality`: k-means Davies-Bouldin index and inertia (Whetten et al., 2025), equal to
  scikit-learn's on the same labels.
- `cka` (linear, with the debiased estimator) and `svcca` as layer-pair metrics, with parity
  against the authors' reference code; `compute_pairs(metric="cka" | "svcca", params=...)`
  computes per-layer summaries once and accepts two checkpoints as A and B. 40 metrics.

## [0.1.0] - 2026-10-02

First public release.

- 41 label-free representation-quality metrics in ten groups: spectral (effective rank,
  spectral and matrix entropy, alpha-ReQ, anisotropy, participation ratio with a
  finite-sample bias correction, eigenvalue early enrichment, Gaussianity, sparsity),
  intrinsic dimension (TwoNN, GRIDE, Levina-Bickel MLE, mLID, MST dimension), local
  geometry (kNN curvature, local rectifiability), relational
  (self-clustering, uniformity, normalized standard deviation), trajectory curvature
  (signed, absolute, RECURVE), view metrics (LiDAR, InfoNCE, DiME, alignment),
  pitch-transposition equivariance, layer-pair metrics (information imbalance,
  neighborhood overlap), token-field metrics and embedding norm. Every metric is
  registered with its input contract, canonical preprocessing and citation.
- Pipeline: `compute()` over pooled vectors, per-clip frames or token clouds, with
  shared spectra and neighbor tables, seeded subsetting and per-metric item caps;
  `compute_pairs()`; `Records` with JSON, CSV and pandas export; `convergence()`
  sample-size curves; selection rules across runs and across layers (`rank_runs`, `top_layers`); parameter-free readouts for token grids and frame sequences;
  `protocols.get("kanatas2026")`.
- Monitoring: `LayerMonitor` on a fixed seeded subset through forward hooks,
  `OnlineBuffer` over training batches, CSV, JSON, TensorBoard and Weights & Biases
  sinks, and a PyTorch Lightning callback with epoch and step schedules.
- Documentation: metric reference, design notes, views note, one card per metric,
  bibliography, two example scripts and a benchmark script.
