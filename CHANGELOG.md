# Changelog

All notable changes to this project are documented here. The format follows
Keep a Changelog; versions follow Semantic Versioning.

## [0.2.0] - 2026-10-03

### Added
- `cluster_quality`: k-means inertia and Davies-Bouldin index (Whetten et al., 2025).
- `cka` (linear, optionally debiased) and `svcca`, tested against the authors' reference code.
  `compute_pairs(metric="cka")` or `"svcca"` computes them for every layer pair, within one model
  or between two checkpoints.

### Changed
- Frames records include the mean of each numeric per-clip extra.
- Monitoring sinks use the global step as the x-axis, and in profile-plot legends, when records
  carry it.

### Removed
- Four registry names. Their values are extras of the base metric, selected by an argument:

  | Removed | Use |
  |---|---|
  | `effective_rank/variance` | `effective_rank`, `spectrum="variance"` |
  | `trajectory_curvature/abs` | `trajectory_curvature`, `convention="abs"` |
  | `trajectory_curvature/recurve` | `trajectory_curvature`, `normalize="path_length"` |
  | `pte/cpsd` | `pte`, `score="cpsd"` |

- The `method` argument of `pte`. The same probe always maps the clip and its transposition.

### Fixed
- `json_sink` overwrote files when epoch and step schedules were combined or training-batch
  monitoring was on. Files are now named `epoch_<e>_step_<s>.json`, with an `online_` prefix for
  training-batch records.

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
