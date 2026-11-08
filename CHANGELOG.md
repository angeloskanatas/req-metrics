# Changelog

All notable changes to this project are documented here. The format follows
Keep a Changelog; versions follow Semantic Versioning.

## [Unreleased]

## [0.1.0] - 2026-11-08

First public release.

### Added

- 39 metrics in eleven groups, each registered with its input contract, canonical preprocessing and
  citation: spectral (effective rank, matrix-based entropy, alpha-ReQ, anisotropy, participation
  ratio with the corrections of Chun et al., eigenvalue early enrichment), intrinsic dimension
  (TwoNN, GRIDE, Levina-Bickel MLE, mLID, MST), relational (cosine anisotropy, self-clustering,
  uniformity, normalized standard deviation), clustering (k-means inertia and Davies-Bouldin index,
  dense representation structure), distribution (Gaussianity, sparsity, embedding norm), trajectory
  curvature, views (LiDAR, InfoNCE over any number of views, DiME, alignment), pitch-transposition
  equivariance, representation pairs (information imbalance, neighborhood overlap, cycle k-NN
  consistency, CKA, SVCCA, RSA, orthogonal Procrustes distance, ridge predictivity), token fields
  and the Jacobian effective rank.
- `compute()` at three levels (one vector per sample; each sample's tokens, aggregated over samples;
  the tokens of all samples as one cloud), with shared spectra and neighbor tables, seeded
  subsampling, per-metric item caps and a `device` argument; `Records` with JSON, CSV, Markdown and
  pandas export.
- `compute_pairs()` between every layer of two representations of the same items (layers,
  checkpoints, models or modalities), with one rank or neighbor table per layer; the deconfounded CKA
  and RSA of Cui et al. (2022) through `confound=`.
- `req_metrics.calibration`: permutation-null calibration of a score, of a maximum over layer pairs
  and of every cell of a layer-pair grid (Gröger et al., 2026), with whole-group permutations for
  the frames of one clip, and the half-sample error bars of Acevedo et al. (2026).
- `convergence()` sample-size curves, `rank_runs()`, `top_layers()` and `top_local_maxima()` (Dai et
  al., 2025), parameter-free readouts for token grids and sequences, and
  `protocols.get("kanatas2026")`.
- `LayerMonitor` on a fixed seeded subset in eval mode: augmented-view passes, the Jacobian effective
  rank of every layer's readout, layer drift against the first or the previous sweep, `OnlineBuffer`
  over training batches, and CSV, JSON, TensorBoard and Weights & Biases sinks.
- A PyTorch Lightning callback with epoch and step schedules. Per-layer scalars are written to
  `trainer.callback_metrics`, so `ModelCheckpoint` and `EarlyStopping` can select by them, and the
  callback state is saved in checkpoints and restored on resume.
- Weights & Biases depth profiles and per-layer singular-spectrum charts; `delta_first` and
  `drawdown` sweep extras (De Melo Costa et al., 2026).
- Input validation and failure records: invalid inputs are rejected before computing, a failed
  estimator is recorded as nan with `extras["error"]`, and warnings flag identical views and layers
  with fewer samples than dimensions.
- Seeded draws come from CPU generators, so a seed gives the same value on CPU and GPU tensors.
