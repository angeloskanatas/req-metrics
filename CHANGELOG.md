# Changelog

All notable changes to this project are documented here. The format follows
Keep a Changelog; versions follow Semantic Versioning.

## [Unreleased]

First release in preparation.

- 36 label-free metrics in twelve groups: spectral (effective rank, matrix-based entropy, alpha-ReQ,
  anisotropy, participation ratio with the bias corrections of Chun et al., eigenvalue early
  enrichment), intrinsic dimension (TwoNN, GRIDE, Levina-Bickel MLE, mLID, MST dimension), local
  geometry (kNN curvature, local rectifiability), relational (cosine anisotropy, self-clustering,
  uniformity, normalized standard deviation), clustering (k-means inertia and Davies-Bouldin index),
  distribution (Gaussianity, sparsity, embedding norm), trajectory curvature, view metrics (LiDAR,
  InfoNCE over any number of views, DiME, alignment), pitch-transposition equivariance, layer-pair
  metrics (information imbalance, neighborhood overlap, CKA, SVCCA), token-field metrics and the
  Jacobian effective rank. Each is registered with its input contract, canonical preprocessing and
  citation.
- Pipeline: `compute()` at three levels (one vector per sample; each sample's tokens, with the
  values aggregated over samples; the tokens of all samples as one cloud), with shared spectra and
  neighbor tables, seeded subsetting and per-metric item caps; the population token cloud is drawn
  without concatenating samples, and token-field metrics that pair tokens within a sample run per
  sample at both token levels; neighbor estimators, TwoNN included, drop duplicate rows and share
  one table per layer, and at the population level report the sample count and the share of
  neighbors from the same sample; `device=` moves each layer, sample or population token cloud to
  the computing device one at a time, and MPS maps to the CPU; `compute_pairs()`, with one rank
  table per layer when a model is compared with itself; `Records` with JSON, CSV and pandas export;
  `convergence()` sample-size curves; selection rules across runs and layers (`rank_runs`,
  `top_layers`); parameter-free readouts for token grids and token sequences, with prefix tokens
  dropped by `n_prefix`; `protocols.get("kanatas2026")`.
- Seeded draws (projection directions, anchors, permutations, token samples) come from CPU
  generators, so a seed gives the same value on CPU and GPU tensors.
- Monitoring: `LayerMonitor` on a fixed seeded subset in eval mode, with a sample cap and, at the
  population level, a separate token cap; view passes that read a live loader, so random crops are
  redrawn per view, and run in train mode for objectives whose positives come from masking inside
  the model, with buffers restored; view passes held on the CPU and computed one layer at a time on
  the model's device; the Jacobian effective rank of every layer's readout by randomized range
  finding; `OnlineBuffer` over training batches; CSV, JSON, TensorBoard and Weights & Biases sinks,
  with selected extras; and a PyTorch Lightning callback with epoch and step schedules.
