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
