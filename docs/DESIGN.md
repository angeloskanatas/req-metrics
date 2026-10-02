# Design notes

The decisions behind the library. The per-metric facts are in `METRICS.md` and the cards; this
document is about the whole.

## 1. Inclusion rule: published metrics only

A metric enters the registry only if a published paper defines it as a
representation measure, loss statistic or estimator, and it is registered
with its input contract, canonical preprocessing and citation keys.
Instruments without a published definition are not registered. Where a
reference implementation deviates from the published definition, the
docstring says so and the published definition takes precedence; the deviation is kept
only when it is itself published and then as a named variant.

## 2. Pure estimators, declared inputs, recorded protocol

Estimators are plain functions on tensors with no model, audio or framework
code. Inputs are declared by kind (points, trajectory, views, shifted copies,
tokens, pairs) and populations (pooled, frames, tokens) are a pipeline
argument, not a property of the estimator. Readouts must be parameter-free,
because metrics on label-trained poolers would measure the probe. Augmented
views and pitch-shifted copies are inputs the caller builds with their own
encoder and augmentation callables; `ViewSpec` and `ShiftSpec` are stored with
every result because the score is only interpretable together with how the
views were made. Every record carries the preprocessing, the parameters, the
sample size, the seed, the pooling label and the library version; the result
files of Kanatas et al. (2026) recorded none of the number of views, the
InfoNCE temperature or the LiDAR ridge, and such scores cannot be compared.

## 2b. Readouts and layouts

Spectrogram-patch encoders yield an (F, T, D) token grid per clip. The layout
is chosen explicitly and recorded in the `pooling` label: `grid_to_trajectory`
concatenates the frequency patches of each time step into a (T, F * D)
trajectory (the frequency-preserving readout of MSM-MAE), `grid_to_tokens`
flattens all patches for the "tokens" population, and `grid_to_pooled` gives
one vector per clip by `gap`, `freq_concat_mean` (frequency-concatenated time mean) or
block-`partitioned` means. All readouts are parameter-free: metrics on
label-trained poolers would measure the probe, not the representation.
Frame-sequence encoders need nothing: their (T, D) output is already a
trajectory, pooled by `frame_tokens_to_pooled` (time mean, or the last token
for causal decoders). A 2D grid therefore has three valid readings and the
record says which was used: the patch cloud (all patches as points, the
vision convention for token-level geometry), the frequency-concatenated
trajectory (time steps as points in F * D dimensions, which keeps the
frequency axis that tonal tasks need) and one pooled vector per clip. Point
metrics accept all three; the trajectory curvature needs a time-ordered
sequence; the token-field metrics read the patch cloud of one clip. During
training the same readouts are available by name through `make_pooler`.

## 3. Efficiency

Memory-bounded chunking throughout: kNN tables and rank tables are built in
row chunks, never as N x N tensors. One singular spectrum per (layer,
preprocessing) and one neighbor table per layer, sized to the largest
requested k, are shared by every estimator that reads them. Spectra, kNN and
LDA run in float64 with a dtype switch for GPUs with slow double precision;
in float64 the matmul expansion of pairwise distances is used, in float32 the
direct path, because the expansion loses the small distances that ratio
estimators depend on. Computation stays on the device of the input tensors.
The information-imbalance matrix builds one rank table per layer and gathers
it for every other layer, O(L N^2 D) instead of O(L^2 N^2 D). GRIDE needs a
neighbor table with `range_max` neighbors, so keep that modest unless
reproducing the published ID row at 8,192. Sharding over layers, models or
partitions belongs to the caller, which composes with any scheduler.
Estimators whose cost grows faster than N log N (the MST dimension, local
rectifiability, DiME) carry a `max_items` cap in the registry; `compute(...,
limits=)` overrides it per metric. Above the cap the pipeline draws a seeded
subsample, builds that subsample's own spectrum or neighbor table, and records
the count in `extras["n_items_used"]` next to the population size.
Cost classes rather than timings, since wall-clock numbers depend on the machine and
on BLAS threading. Spectral metrics are one SVD per (layer, preprocessing). The
neighbor-based estimators share one chunked kNN table, O(N^2 D), as does
uniformity. The MST dimension builds dense distance matrices over subsamples up to
N (O(N^2) memory); DiME eigendecomposes N x N Gram matrices for each of its
re-pairings (O(N^3)); local rectifiability is linear in N per anchor set but
repeated over scales. Those three carry item caps. The Kolmogorov-Smirnov
Gaussianity runs one test per dimension. View metrics scale with q and PTE trains
a probe. `scripts/benchmark.py` measures every registered metric on the machine at
hand.

## 4. Monitoring

`LayerMonitor` evaluates a fixed, seeded subset of clips in eval mode with
the current weights at every sweep, so a change between two sweeps is due to
the model and the curves are comparable with post-hoc runs on checkpoints.
This costs `n_items` forward passes per sweep, times `1 + q` with view
metrics. `OnlineBuffer` is the alternative without extra forward passes, a ring buffer of the most
recent training-batch outputs of every hooked layer: no extra passes, but it
measures the training-time representation (augmented inputs, train-mode
layers, weights that moved while the buffer filled), so its records are tagged
`extras["source"] = "training-batches"` and logged under `online_metrics/`,
never mixed with fixed-subset records. Monitoring callbacks that read a queue
of training batches, as in stable-pretraining (arXiv:2511.19484), measure that
training-time representation; the fixed subset is what makes a sweep
comparable with the previous sweep and with a post-hoc run on a checkpoint.
Under distributed training the callback runs on
global rank zero only. The monitoring subset is a re-iterable loader; a
one-shot iterator, or any loader with `cache_batches=True`, is materialized
once on the CPU so every sweep sees the same batches without decoding again.
An EMA target or any other branch is monitored by pointing `model_attr` at
it, one callback per branch. Every record carries the epoch and the global
step, and the W&B profile plots draw all sweeps so far, one line per step, so
the evolution of a depth profile is read off one chart.

Logging: Weights & Biases rejects `step=` values below its internal counter,
so no sink passes `step=`; `wandb_sink` logs a `monitor/step` metric and binds
`layer_metrics/*`, `profiles/*` and the `online_*` families to it with
`define_metric`, and the Lightning path follows `WandbLogger.log_metrics`
(the `trainer/global_step` key). `tensorboard_sink` writes
`layer_metrics/<metric>/layer_<l>` through `torch.utils.tensorboard`; CSV and
JSON sinks and any callable of `(records, step)` are accepted.

## 4b. Which layers, and when

All hooked layers come from one forward pass and spectral metrics cost
milliseconds per layer, so the monitor reads every block by default. The
training-dynamics studies support that: Razzhigaev et al. (2024, EACL) track
anisotropy and intrinsic dimension at every internal layer across pretraining
checkpoints and find an expansion followed by a compression; Lee et al. (2025,
ACL) find an intrinsic-dimension phase transition near 10^3 steps for
virtually all layers of Pythia models; Li et al. (2025) describe warmup,
entropy-seeking and compression-seeking phases on the last-token state of the
last layer; Whetten et al. (2025, Interspeech) find that the layers whose rank
predicts downstream speech performance are the first and last rather than the
task-optimal middle ones. Because the early changes are fast, the callback
accepts a step-based schedule (`sweep_steps`, typically log-spaced, or
`every_n_steps`) next to the epoch period, and stamps every record with the
epoch and the global step. Per-metric item caps (`max_items` in the registry,
`limits=` in `compute`) keep the quadratic estimators bounded at large N.

The speech studies that use these metrics for monitoring and selection fix the
layer and the cadence. Aldeneh et al. (2024) compute RankMe-t per layer on
10,000 utterances of the pretraining data at checkpoints every 20,000 steps,
and find that it tracks downstream performance across checkpoints within a
layer but cannot rank layers against each other. Whetten et al. read layer 12
for ASR and layer 8 for speaker verification at 50,000 steps, with k-means
inertia and the Davies-Bouldin index next to RankMe-t, to predict the outcome
at 200,000 steps. RankMe (Garrido et al., 2023) selects the hyperparameter
configuration with the highest rank on 25,600 samples of the representation
that is used downstream; LiDAR (Thilak et al., 2024) applies the same rule.
Post hoc, Skean et al. (2025), Kanatas et al. (2026) and the comparative study
of Arputharaj et al. (2026) relate layer profiles to downstream performance.
`selection.rank_runs` and `selection.top_layers` encode the selection rules
with the direction as an argument, since both Kanatas et al. and Arputharaj et
al. report sign reversals across task families and training paradigms.

## 5. What was considered and not adopted

Each candidate was assessed from its source paper and, where it exists, its
reference code. Adopted beyond the paper's own set: self-clustering
(Tsitsulin et al., 2023), NESum as an anisotropy extra (He and Ozay, 2022),
uniformity and alignment (Wang and Isola, 2020), the normalized standard
deviation (Chen and He, 2021), the bias-corrected participation ratio (Chun et
al., 2026), neighborhood overlap (Doimo et al., 2020), the intrinsic-dimension
caveats of Schulte and Rügamer (2026) as card text, and the taxonomy and the
alpha-ReQ fit-range caveat of Arputharaj et al. (2026). Not adopted: the
condition number and coherence of Tsitsulin et al. (sign reversals across
datasets, and a correlation with accuracy that Arputharaj et al. trace to OLS
conditioning), diffusion spectral entropy (Liao et al., 2024; a bandwidth in
absolute embedding units, non-commercial reference code, and near-perfect
collinearity with self-clustering), and the layer-pair candidates of Jiang et
al. (2026), which have no code and no peer review yet. Candidates for a next
version, each needing a full read of its source first: the k-means inertia and
Davies-Bouldin index of Whetten et al. (2025), the parameter- and
representation-prediction equivariance probes of Plachouras et al. (2025),
and the dense-representation structure estimator of Dai et al. (NeurIPS
2025, arXiv:2510.17299). The reasons, with the tables they rest on, are in
`docs/METRICS.md`, section 6.

## 6. Sample size

N is part of the protocol. Entropic effective rank keeps rising with N
(RankMe used 25,600 samples; 10,000 gives over 95 percent of the asymptote at
width 2048), the plug-in participation ratio is biased by about PR/N,
nearest-neighbour ID estimates carry a few percent of subsample noise, and
even anisotropy moves when N is not much larger than D. `convergence()`
recomputes a metric on repeated subsets at several fractions through the same
preprocessing and caches as `compute()`, so N can be chosen per metric and
cells compared at unequal N can be recognised. Published values are not
comparable across different N for the N-dependent metrics; the protocol of
Kanatas et al. (2026) used 10,000 clips of 15 seconds, one per track, after
checking that depth profiles had converged.
