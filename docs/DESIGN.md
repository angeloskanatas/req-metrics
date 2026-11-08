# Design notes

This document records the design decisions that apply across metrics. Per-metric details are in
`METRICS.md` and the metric cards.

## 1. Inclusion rule: published metrics only

A metric enters the registry only if a published paper defines it as a representation measure,
loss statistic or estimator. It is registered with its input contract, canonical preprocessing and
citation keys. Where a reference implementation deviates from the published definition, the
docstring says so and the published definition takes precedence. A deviation is kept only when it
is itself published, and then as a named variant.

## 2. Estimators, inputs and records

Estimators are plain functions on tensors, with no model, audio or framework code. Inputs are
declared by kind (points, trajectory, views, shifted copies, tokens, pairs). The level (sequence,
sample, population) is a pipeline argument, not a property of the estimator. Readouts must be
parameter-free, because metrics on label-trained poolers would measure the probe. The caller
builds augmented views and pitch-shifted copies with their own encoder and augmentation callables.
`ViewSpec` and `ShiftSpec` are stored with every result, because a score can only be interpreted
together with how the views were made. Every record carries the preprocessing, the parameters, the
sample size, the seed, the pooling label and the library version, since a score without them
cannot be compared.

## 2b. Readouts and layouts

Spectrogram-patch encoders give an (F, T, D) token grid per sample. The layout is chosen
explicitly and recorded in the `pooling` label:

- `grid_to_trajectory` concatenates the frequency patches of each time step into a (T, F * D)
  trajectory (the frequency-preserving readout of MSM-MAE).
- `grid_to_tokens` flattens all patches into the token set used at the sample and population
  levels.
- `grid_to_pooled` gives one vector per sample by `gap`, `freq_concat_mean`
  (frequency-concatenated time mean) or block-`partitioned` means.

Frame-sequence encoders need no layout. Their (T, D) output is already a trajectory, and
`frame_tokens_to_pooled` pools it (time mean, or the last token for causal decoders). A 2D grid
can therefore be read in three ways, and the record says which was used: as the patch cloud (all
patches as points, the vision convention for token-level geometry), as the frequency-concatenated
trajectory (time steps as points in F * D dimensions, which keeps the frequency axis that tonal
tasks need), or as one pooled vector per sample. Point metrics accept all three. Trajectory
curvature needs a time-ordered sequence, and the token-field metrics read the patch cloud of one
sample. All readouts are parameter-free (Section 2). During training the same readouts are
available by name through `make_pooler`.

## 3. Efficiency

Memory is bounded by chunking: kNN tables and rank tables are built in row chunks, never as N x N
tensors. One singular spectrum per (layer, preprocessing) and one neighbor table per layer, sized
to the largest requested k, are shared by every estimator that reads them. Spectra, kNN and LDA
run in float64. `Spectrum.from_points` and `Neighbors.from_points` take a dtype for GPUs with slow
double precision. Pairwise distances use the matmul expansion in float64 and the direct path in
float32, because in float32 the expansion loses the small distances that ratio estimators depend
on.

Computation runs on the device of the inputs, or on the one given to `compute(..., device=)`. Each
layer, sample or population token cloud is moved there after subsetting, one at a time. MPS maps
to the CPU, since it has no float64. The sample and population levels take per-sample token
sequences, so a corpus of memory-mapped samples is read one sample at a time, and the population
token cloud is drawn before any sample is concatenated. The information-imbalance matrix builds
one rank table per layer and gathers it for every other layer, which costs O(L N^2 D) instead of
O(L^2 N^2 D). GRIDE needs a neighbor table with `range_max` neighbors, so keep that modest unless
you are reproducing the published ID row at 8,192. Sharding over layers, models or partitions is
left to the caller.

Estimators whose cost grows faster than N log N (the MST dimension, DiME)
carry a `max_items` cap in the registry, and `compute(..., limits=)` overrides it per metric.
Above the cap, the pipeline draws a seeded subsample, builds that subsample's own spectrum or
neighbor table, and records the count in `extras["n_items_used"]` next to the full size.

Wall-clock times depend on the machine and on BLAS threading, so costs are given here as classes,
and `scripts/benchmark.py` measures every registered metric on the machine at hand. Spectral
metrics cost one SVD per (layer, preprocessing). The neighbor-based estimators share one chunked
kNN table, O(N^2 D), and uniformity, computed from Gram blocks, has the same cost. The MST
dimension builds dense distance matrices over subsamples up to N (O(N^2) memory). DiME
eigendecomposes N x N Gram matrices for each of its re-pairings (O(N^3)). Both carry item caps.
Gaussianity
computes all three of its statistics on every call, with one Kolmogorov-Smirnov test per random
direction. View metrics scale with q, and PTE trains a probe.

## 4. Monitoring

At every sweep, `LayerMonitor` evaluates a fixed, seeded subset of samples in eval mode with the
current weights. A change between two sweeps is then due to the model, and the curves are
comparable with post-hoc runs on checkpoints. This costs `n_items` forward passes per sweep, times
`1 + q` with view metrics. `OnlineBuffer` avoids the extra forward passes. It is a ring buffer of
the most recent training-batch outputs of every hooked layer and measures the training-time
representation (augmented inputs, train-mode layers, weights that moved while the buffer filled).
Its records are therefore tagged `extras["source"] = "training-batches"` and logged under
`online_metrics/`, never mixed with fixed-subset records. Monitoring callbacks that read a queue of
training batches, as in stable-pretraining (arXiv:2511.19484), measure the same training-time
representation. Centered spectral and neighbor metrics cannot see representations collapsing onto
one shared vector. `normalized_std` and `anisotropy/cosine` can, so a monitoring run logs one of
them beside the others.

With `drift_metrics`, every sweep also runs pair metrics between the current representation of
each layer and the same layer at the first sweep (or at the previous sweep), on the same
monitoring items in the same order. Raghu et al. (2017, Sec. 4.1) read layers this way: they
compare every layer during training with a fixed state and find that layers settle from the
bottom up. The reference is held once on the CPU in float32. A sweep costs one pair metric per
layer rather than a layer map, and the records are logged under `drift_metrics/` with the
reference step in their extras.

Under distributed training, the sweep runs on global rank zero and its scalars are broadcast to
every rank. The monitoring subset is a re-iterable loader. A one-shot iterator, or any loader with
`cache_batches=True`, is materialized once on the CPU, so every sweep sees the same batches
without decoding them again. The q view passes read the live loader instead, so a dataset that
draws a random crop per item gives every view its own crop, as the positives of the objective do.
With `views_in_train_mode`, buffers such as BatchNorm running statistics are restored after those
passes. View passes are held on the CPU, and the (q, N, D) stack of each layer moves to the device
of the hooked layers for its metrics.

Selection during training goes through Lightning's own machinery. The scalars of every sweep are
written to `trainer.callback_metrics` on every rank (rank zero computes, the strategy
broadcasts), and `ModelCheckpoint` and `EarlyStopping` read the quantity they monitor from there.
Lightning's `LearningRateMonitor` publishes its values the same way. The published evidence
supports comparing runs or checkpoints at one layer and equal step, which is the rule of RankMe
and LiDAR, and stopping on collapse, since `normalized_std` and `anisotropy/cosine` fall when
representations converge onto one vector. It does not support a universal early-stopping
criterion within one run. The effective rank of a language model expands and then contracts while
downstream quality keeps improving (Li et al., 2025, NeurIPS, arXiv:2509.23024), so the checkpoint
with the maximum rank can be the wrong one. The callback state (sweep history, schedule
bookkeeping, drift reference) is saved in the checkpoint and restored on resume, and the callback,
the monitor and the sinks are picklable for spawn-based launchers.

To monitor an EMA target or any other branch, point `model_attr` at it, with one callback per
branch. Every record carries the epoch and the global step. The W&B profile plots draw one line
for each sweep so far, so the evolution of a depth profile can be read off one chart.

Weights & Biases rejects `step=` values below its internal counter, so no sink passes `step=`.
Instead, `wandb_sink` logs a `monitor/step` metric and binds `layer_metrics/*`, `profiles/*` and
the `online_*` families to it with `define_metric`. The Lightning path follows
`WandbLogger.log_metrics` (the `trainer/global_step` key). Every scalar sink uses the keys of
`layer_scalars`, `layer_metrics/<metric>_layer_<l>`. For sample and population records the level
is appended to the metric name, and selected extras are logged as `<metric>_<extra>`.
`tensorboard_sink` writes the scalars through `torch.utils.tensorboard`. The CSV and JSON sinks
keep every field and extra, and any callable of `(records, step)` is accepted as a sink.

On the dashboard, the per-layer scalars are native panels, one per layer and metric, which the
dashboard can compare across runs. Layer indices are zero-padded to the number of digits of the
depth, so the panels sort by depth rather than lexicographically. The depth profiles
(`profiles/<metric>`) draw the metric against the layer with one line per sweep. At most eight
lines are drawn, spread from the first sweep to the latest, so a long run stays readable and its
first and current profiles are always on the chart. The spectra (`spectra/layer_<l>`, with
`spectra > 0`) draw the log10 of each layer's singular values over the largest against their
index, with one line for each sweep the profiles chose. The plateau and cliff of dimensional
collapse and the shrinking spectrum of complete collapse are visible directly, where the scalars
only show that something moved. Both kinds of chart are logged as tables whose columns name the
axes and the legend. Spectra are kept in the monitor's side store, not in the records, since the
records hold only tabular data.

## 4b. Layers, schedule and selection

All hooked layers come from one forward pass and spectral metrics cost milliseconds per layer, so
the monitor reads every block by default. Layer-wise training studies find the fast changes early
and in every layer: an expansion followed by a compression of anisotropy and intrinsic dimension
across pretraining checkpoints (Razzhigaev et al., 2024, EACL), an intrinsic-dimension transition
near 10^3 steps in nearly all layers (Lee et al., 2025, ACL), effective-rank phases on the final
layer (Li et al., 2025), and layers that predict downstream performance best without being the
ones read downstream (Whetten et al., 2025, Interspeech). The callback therefore accepts a
step-based schedule (`sweep_steps`, typically log-spaced, or `every_n_steps`) next to the epoch
period, and stamps every record with the epoch and the global step. Per-metric item caps
(`max_items` in the registry, `limits=` in `compute`) bound the quadratic estimators at large N.

The selection helpers follow published rules. RankMe (Garrido et al., 2023) and LiDAR (Thilak et
al., 2024) pick, within one method, the configuration with the highest value on the representation
that is used downstream. Aldeneh et al. (2024) track RankMe-t across checkpoints within a layer
and find that it does not rank layers against each other. Whetten et al. (2025) read one layer
early in pretraining to predict the outcome. `selection.rank_runs` and `selection.top_layers`
encode these rules with the direction as an argument, because the sign of a metric depends on the
task family (Kanatas et al., 2026) and on the architecture class and training objective
(Arputharaj et al., 2026), and within one run the relation need not be monotone (Li et al., 2025).

Label-free ranking is the option when there are no labels. With a few hundred labeled examples, a
probe trained on one early checkpoint predicts the final outcome more accurately (Whetten et al.,
2025, Interspeech, arXiv:2501.05966; Liu et al., 2026, arXiv:2604.01025), and it is the reference
to check a label-free order against. The rule applies at a fixed scale. There the training loss is
uninformative for joint-embedding objectives, and for language models compared across recipe
variants it is inversely related to downstream quality (Zeng et al., 2025, arXiv:2504.12491).
Along the scale and token trajectory of one autoregressive recipe, the loss itself tracks prompted
performance (Du et al., 2024, arXiv:2403.15796).

## 5. Metrics beyond the paper

Beyond the metrics of Kanatas et al. (2026), the library includes:

- self-clustering (Tsitsulin et al., 2023), NESum as an anisotropy extra (He and Ozay, 2022),
  uniformity and alignment (Wang and Isola, 2020), the normalized standard deviation (Chen and He,
  2021) and the bias-corrected participation ratio (Chun et al., 2026)
- the intrinsic-dimension caveats of Schulte and Rügamer (2026) as card text, and the taxonomy and
  the alpha-ReQ fit-range caveat of Arputharaj et al. (2026)
- the k-means inertia and Davies-Bouldin index of Whetten et al. (2025)
- as pair metrics: neighborhood overlap (Doimo et al., 2020), linear CKA (Kornblith et al., 2019),
  the held-out ridge predictivity of encoding models (Tuckute et al., 2023; Pepino et al., 2025),
  the orthogonal Procrustes distance (Ding et al., 2021), SVCCA (Raghu et al., 2017),
  representational similarity analysis (Kriegeskorte et al., 2008), the cycle k-NN consistency
  (Huh et al., 2024, ICML, arXiv:2405.07987), and the deconfounded CKA and RSA of Cui et al. (2022,
  NeurIPS, arXiv:2202.00095)
- the dense representation structure estimator of Dai et al. (2025, NeurIPS, arXiv:2510.17299), in
  the computation of its released code, with Eq. 5 as printed as an option, and its local-maxima
  selection rule
- as helpers: the permutation-null calibration of Gröger et al. (2026, ICML, arXiv:2602.14486) and
  the half-sample error bars of Acevedo et al. (2026, TMLR, arXiv:2505.17101)
- the Jacobian effective rank of Chung and Kim (2026), which needs the model and is therefore
  computed by the monitor

## 6. Sample size

N is part of the protocol, and several metrics depend on it:

- Entropic effective rank keeps rising with N. RankMe used 25,600 samples, 10,000 samples give
  over 95 percent of the asymptote at width 2048, and Tsitsulin et al. (2023) need 16,384 for a
  0.95 approximation.
- The plug-in participation ratio is biased by about PR/N.
- Nearest-neighbor ID estimates carry a few percent of subsample noise (a median of 4.5 percent
  over 90 percent subsamples in Arputharaj et al., 2026).
- Even anisotropy moves when N is not much larger than D, because sampling inflates the leading
  eigenvalue.

`convergence()` recomputes a metric on repeated subsets at several fractions, through the same
preprocessing and caches as `compute()`. N can then be chosen per metric, and cells compared at
unequal N can be recognized. For the metrics that depend on N, published values are not comparable
across different N. The protocol of Kanatas et al. (2026) used 10,000 clips of 15 seconds, one per
track.
