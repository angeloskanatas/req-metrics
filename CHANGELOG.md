# Changelog

All notable changes to this project are documented here. The format follows
Keep a Changelog; versions follow Semantic Versioning.

## [Unreleased]

First release in preparation.

### Added

- 38 metrics in twelve groups, each registered with its input contract, canonical preprocessing and
  citation: spectral (effective rank, matrix-based entropy, alpha-ReQ, anisotropy, participation
  ratio with the corrections of Chun et al., eigenvalue early enrichment), intrinsic dimension
  (TwoNN, GRIDE, Levina-Bickel MLE, mLID, MST), local geometry (kNN curvature, local
  rectifiability), relational (cosine anisotropy, self-clustering, uniformity, normalized standard
  deviation), clustering (k-means inertia, Davies-Bouldin index), distribution (Gaussianity,
  sparsity, embedding norm), trajectory curvature, views (LiDAR, InfoNCE over any number of views,
  DiME, alignment), pitch-transposition equivariance, representation pairs (information imbalance,
  neighborhood overlap, cycle k-NN consistency, CKA, SVCCA, RSA), token fields and the Jacobian
  effective rank.
- `compute()` at three levels (one vector per sample; each sample's tokens, aggregated over samples;
  the tokens of all samples as one cloud), with shared spectra and neighbor tables, seeded
  subsetting, per-metric item caps, a `device` argument and `Records` with JSON, CSV and pandas
  export; `convergence()` sample-size curves; `rank_runs` and `top_layers`; parameter-free readouts
  for token grids and token sequences; `protocols.get("kanatas2026")`.
- `compute_pairs()` between every layer of two representations of the same items (layers,
  checkpoints, models or modalities): several pair metrics per call on one seeded subsample, one
  rank or neighbor table per layer, cosine neighbors and the Jaccard normalization of the overlap on
  request.
- `LayerMonitor` on a fixed seeded subset in eval mode, with view passes from a live loader or in
  train mode for masked objectives, the Jacobian effective rank of every layer's readout, the drift
  of every layer against its first or previous sweep, `OnlineBuffer` over training batches, CSV,
  JSON, TensorBoard and Weights & Biases sinks, and a PyTorch Lightning callback with epoch and step
  schedules.
- Seeded draws come from CPU generators, so a seed gives the same value on CPU and GPU tensors.
- Input validation and failure records: non-finite inputs, integer layer keys, `n_items >= 2`, the
  `ViewSpec` view count and aligned pitch-shifted copies are checked up front; every estimator
  failure is a nan record with `extras["error"]`, in `compute_pairs` as well, and sample-level
  aggregates report skipped samples in `extras["first_error"]`. `describe(name)` prints a metric's
  card; `Records` has a repr, slicing and `to_markdown`; JSON files write `null` for nan.
- Lightning callback: per-layer scalars in `trainer.callback_metrics` on every rank, so
  `ModelCheckpoint` and `EarlyStopping` select by them; callback state (history, schedule, drift
  reference) saved in checkpoints and restored on resume; one sweep when a step trigger falls on an
  epoch end; sinks receive the global step; the callback, the monitor and the sinks are picklable.
- Dashboard: layer indices in logging keys zero-padded to the depth's width; depth-profile charts
  keep at most eight sweeps spread from first to latest; `spectra=` on the monitor and the callback
  adds per-layer charts of the log singular spectrum on Weights & Biases, on the same sweeps as the
  profiles; chart tables name their axes and legend.
- Sweep records carry `delta_first` and `drawdown` extras (change since the first sweep, fall below
  the running peak), the trajectory summaries of De Melo Costa et al. (2026); logged through
  `log_extras`.
- Spectral metrics record `rank_cap` and `compute` warns once when a layer has fewer points than
  dimensions.
- `uniformity` adds the `excess` extra (0 uniform to 1 collapsed) and documents the D-dependence of
  the raw value; `gaussianity` documents that it measures distance to the isotropic Gaussian target,
  including scale.
- `LayerMonitor` warns once when a batch of N items yields a different number of rows at the first
  hooked layer (clips or crops flattened into the batch, a layer run several times per batch).
- Degenerate inputs: a collapsed cloud centers to exactly zero (effective rank 0, errors for the
  undefined spectral estimators), neighbor tables and spectra refuse non-finite values, GRIDE, MLE
  and mLID report insufficient distinct points, the pair pipeline ranks ties as the imbalance
  estimator does, and LiDAR ignores rounding-noise eigenvalues.
