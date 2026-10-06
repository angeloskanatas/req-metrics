# Design notes

Design decisions that apply across metrics. Per-metric details are in `METRICS.md`
and the metric cards.

## 1. Inclusion rule: published metrics only

A metric enters the registry only if a published paper defines it as a
representation measure, loss statistic or estimator, and it is registered with its
input contract, canonical preprocessing and citation keys. Instruments without a
published definition are not registered. Where a reference implementation deviates
from the published definition, the docstring says so and the published definition
takes precedence; the deviation is kept only when it is itself published and then as
a named variant.

## 2. Pure estimators, declared inputs, recorded protocol

Estimators are plain functions on tensors with no model, audio or framework code.
Inputs are declared by kind (points, trajectory, views, shifted copies, tokens,
pairs) and the level (sequence, sample, population) is a pipeline argument, not a
property of the estimator. Readouts must be parameter-free, because metrics on
label-trained poolers would measure the probe. Augmented views and pitch-shifted
copies are inputs the caller builds with their own encoder and augmentation
callables; `ViewSpec` and `ShiftSpec` are stored with every result because the score
is only interpretable together with how the views were made. Every record carries
the preprocessing, the parameters, the sample size, the seed, the pooling label and
the library version, because a score without them cannot be compared.

## 2b. Readouts and layouts

Spectrogram-patch encoders yield an (F, T, D) token grid per sample. The layout is
chosen explicitly and recorded in the `pooling` label: `grid_to_trajectory`
concatenates the frequency patches of each time step into a (T, F * D) trajectory
(the frequency-preserving readout of MSM-MAE), `grid_to_tokens` flattens all patches
into the token set of the sample and population levels, and `grid_to_pooled` gives
one vector per sample by `gap`, `freq_concat_mean` (frequency-concatenated time mean)
or block-`partitioned` means. All readouts are parameter-free: metrics on
label-trained poolers would measure the probe, not the representation.
Frame-sequence encoders need nothing: their (T, D) output is already a trajectory,
pooled by `frame_tokens_to_pooled` (time mean, or the last token for causal
decoders). A 2D grid therefore has three valid readings and the record says which
was used: the patch cloud (all patches as points, the vision convention for
token-level geometry), the frequency-concatenated trajectory (time steps as points
in F * D dimensions, which keeps the frequency axis that tonal tasks need) and one
pooled vector per sample. Point metrics accept all three; the trajectory curvature
needs a time-ordered sequence; the token-field metrics read the patch cloud of one
sample. During training the same readouts are available by name through `make_pooler`.

## 3. Efficiency

Memory-bounded chunking throughout: kNN tables and rank tables are built in row
chunks, never as N x N tensors. One singular spectrum per (layer, preprocessing) and
one neighbor table per layer, sized to the largest requested k, are shared by every
estimator that reads them. Spectra, kNN and LDA run in float64;
`Spectrum.from_points` and `Neighbors.from_points` take a dtype for GPUs with slow
double precision. In float64 the matmul expansion of pairwise distances is used, in
float32 the direct path, because the expansion loses the small distances that ratio
estimators depend on.

Computation runs on the device of the inputs, or on `compute(..., device=)`, to
which each layer, sample or population token cloud is moved after subsetting, one at
a time; MPS maps to the CPU, since it has no float64. The sample and population
levels take per-sample token sequences, so a corpus of memory-mapped samples is read
one sample at a time, and the population token cloud is drawn before any sample is
concatenated. The information-imbalance matrix builds one rank table per layer and
gathers it for every other layer, O(L N^2 D) instead of O(L^2 N^2 D). GRIDE needs a
neighbor table with `range_max` neighbors, so keep that modest unless reproducing
the published ID row at 8,192. Sharding over layers, models or partitions belongs to
the caller, which composes with any scheduler. Estimators whose cost grows faster
than N log N (the MST dimension, local rectifiability, DiME) carry a `max_items` cap
in the registry; `compute(..., limits=)` overrides it per metric. Above the cap the
pipeline draws a seeded subsample, builds that subsample's own spectrum or neighbor
table, and records the count in `extras["n_items_used"]` next to the full size.

Cost classes rather than timings, since wall-clock numbers depend on the machine and
on BLAS threading. Spectral metrics are one SVD per (layer, preprocessing). The
neighbor-based estimators share one chunked kNN table, O(N^2 D), as does uniformity.
The MST dimension builds dense distance matrices over subsamples up to N (O(N^2)
memory); DiME eigendecomposes N x N Gram matrices for each of its re-pairings
(O(N^3)); local rectifiability is linear in N per anchor set but repeated over
scales. Those three carry item caps. Gaussianity computes all three statistics on
every call, with one Kolmogorov-Smirnov test per random direction. View metrics
scale with q and PTE trains a probe. `scripts/benchmark.py` measures every
registered metric on the machine at hand.

## 4. Monitoring

`LayerMonitor` evaluates a fixed, seeded subset of samples in eval mode with the
current weights at every sweep, so a change between two sweeps is due to the model
and the curves are comparable with post-hoc runs on checkpoints. This costs
`n_items` forward passes per sweep, times `1 + q` with view metrics. `OnlineBuffer`
is the alternative without extra forward passes: a ring buffer of the most recent
training-batch outputs of every hooked layer. It measures the training-time
representation (augmented inputs, train-mode layers, weights that moved while the
buffer filled), so its records are tagged `extras["source"] = "training-batches"`
and logged under `online_metrics/`, never mixed with fixed-subset records. Centered
spectral and neighbor metrics cannot see representations collapsing onto one shared
vector; `normalized_std` and `anisotropy/cosine` can, so a monitoring run logs one
of them beside the others.

With `drift_metrics`, every sweep also runs pair metrics between each layer's current
representation and the same layer at the first sweep (or at the previous sweep), on
the same monitoring items in the same order. This is the reading of Raghu et al.
(2017, Sec. 4.1), who compare every layer during training with a fixed state and find
that layers settle from the bottom up. The reference is held once on the CPU in
float32; a sweep costs one pair metric per layer rather than a layer map, and the
records are logged under `drift_metrics/` with the reference step in their extras.

Monitoring callbacks that read a queue of training batches, as in stable-pretraining
(arXiv:2511.19484), measure that training-time representation; the fixed subset is
what makes a sweep comparable with the previous sweep and with a post-hoc run on a
checkpoint. Under distributed training the sweep runs on global rank zero and its scalars are
broadcast to every rank.
The monitoring subset is a re-iterable loader; a one-shot iterator, or any loader
with `cache_batches=True`, is materialized once on the CPU so every sweep sees the
same batches without decoding again. The q view passes read the live loader instead,
so a dataset that draws a random crop per item gives every view its own crop, as the
objective's positives do; with `views_in_train_mode`, buffers such as BatchNorm
running statistics are restored after those passes. View passes are held on the CPU,
and each layer's (q, N, D) stack moves to the device of the hooked layers for its
metrics.

Selection during training goes through Lightning's own machinery: every sweep's
scalars are written to `trainer.callback_metrics` on every rank (rank zero computes,
the strategy broadcasts), which is where `ModelCheckpoint` and `EarlyStopping` read the
quantity they monitor; Lightning's `LearningRateMonitor` publishes the same way. The
published evidence supports two uses of this: comparing runs or checkpoints at one layer
and equal step, the rule of RankMe and LiDAR, and stopping on collapse, since
`normalized_std` and `anisotropy/cosine` fall when representations converge onto one
vector. It does not support a universal early-stopping criterion within one run: the
effective rank of a language model expands and then contracts while downstream quality
keeps improving (Li et al., 2025), so a maximum-rank checkpoint can be the wrong one. The
callback state (sweep history, schedule bookkeeping, drift reference) is saved in the
checkpoint and restored on resume, and the callback, the monitor and the sinks are
picklable for spawn-based launchers.

An EMA target or any other branch is monitored by pointing `model_attr` at it, one
callback per branch. Every record carries the epoch and the global step, and the W&B
profile plots draw all sweeps so far, one line per step, so the evolution of a depth
profile is read off one chart.

Logging: Weights & Biases rejects `step=` values below its internal counter, so no
sink passes `step=`; `wandb_sink` logs a `monitor/step` metric and binds
`layer_metrics/*`, `profiles/*` and the `online_*` families to it with
`define_metric`, and the Lightning path follows `WandbLogger.log_metrics` (the
`trainer/global_step` key). Every scalar sink uses the keys of `layer_scalars`,
`layer_metrics/<metric>_layer_<l>`, with the level appended to the metric name for
sample and population records and selected extras as `<metric>_<extra>`;
`tensorboard_sink` writes them through `torch.utils.tensorboard`. CSV and JSON sinks
keep every field and extra, and any callable of `(records, step)` is accepted.

Three views, each for a different reading. The per-layer scalars are native panels, one
per layer and metric, which dashboards compare across runs; layer indices are
zero-padded to the depth's width so the panels sort by depth rather than
lexicographically. The depth profiles (`profiles/<metric>`) draw metric against layer
with one line per sweep, at most eight lines spread from the first sweep to the latest,
so a long run stays readable and its first and current profiles are always on the
chart. The spectra (`spectra/layer_<l>`, with `spectra > 0`) draw the log10 of each
layer's singular values over the largest against their index, one line per sweep, the
sweeps the profiles chose: the plateau-and-cliff of dimensional collapse and the
shrinking spectrum of complete collapse are read off directly, where the scalars only
say that something moved. Both chart kinds are logged as tables whose columns name the
axes and the legend. Spectra live in the monitor's side store, not in the records, since
the records are the tabular data plane.

## 4b. Which layers, and when

All hooked layers come from one forward pass and spectral metrics cost milliseconds
per layer, so the monitor reads every block by default. Layer-wise training studies
find the fast changes early and in every layer: an expansion followed by a
compression of anisotropy and intrinsic dimension across pretraining checkpoints
(Razzhigaev et al., 2024, EACL), an intrinsic-dimension transition near 10^3 steps in
nearly all layers (Lee et al., 2025, ACL), effective-rank phases on the final layer
(Li et al., 2025), and layers that predict downstream performance best without being
the ones read downstream (Whetten et al., 2025, Interspeech). The callback therefore
accepts a step-based schedule (`sweep_steps`, typically log-spaced, or
`every_n_steps`) next to the epoch period and stamps every record with the epoch and
the global step; per-metric item caps (`max_items` in the registry, `limits=` in
`compute`) bound the quadratic estimators at large N.

Selection follows the published rules. RankMe (Garrido et al., 2023) and LiDAR
(Thilak et al., 2024) pick, within one method, the configuration with the highest
value on the representation that is used downstream; Aldeneh et al. (2024) track
RankMe-t across checkpoints within a layer and find that it does not rank layers
against each other; Whetten et al. (2025) read one layer early in pretraining to
predict the outcome. `selection.rank_runs` and `selection.top_layers` encode these
rules with the direction as an argument, because the sign of a metric depends on the
task family (Kanatas et al., 2026) and on the architecture class and training
objective (Arputharaj et al., 2026), and within one run the relation need not be
monotone (Li et al., 2025).

## 5. What was considered and not adopted

Each candidate was assessed from its source paper and, where it exists, its
reference code. Adopted beyond the paper's own set: self-clustering (Tsitsulin et
al., 2023), NESum as an anisotropy extra (He and Ozay, 2022), uniformity and
alignment (Wang and Isola, 2020), the normalized standard deviation (Chen and He,
2021), the bias-corrected participation ratio (Chun et al., 2026), neighborhood
overlap (Doimo et al., 2020), the intrinsic-dimension caveats of Schulte and Rügamer
(2026) as card text, the taxonomy and the alpha-ReQ fit-range caveat of Arputharaj
et al. (2026), the k-means inertia and Davies-Bouldin index of Whetten et al.
(2025), linear CKA (Kornblith et al., 2019), SVCCA (Raghu et al., 2017),
representational similarity analysis (Kriegeskorte et al., 2008) and the cycle k-NN
consistency (Huh et al., 2024) as pair metrics,
and the Jacobian effective rank of Chung and Kim (2026), which needs the model and
is therefore computed by the monitor. Not adopted: the condition number of Tsitsulin
et al. (its correlation with accuracy reverses sign across their datasets, and
Arputharaj et al. trace it to OLS conditioning), their coherence (the least stable
of their metrics under subsampling), diffusion spectral entropy (Liao et al., 2024;
a bandwidth in absolute embedding units, non-commercial reference code, and
near-perfect collinearity with self-clustering), the layer-pair measures of Jiang et
al. (2026), which have no code and no peer review yet, the dense-representation
structure estimator of Dai et al. (2025), whose paper and released code define
different quantities, the parameter- and representation-prediction probes of
Plachouras et al. (2025), which train a probe for every layer, transformation and
evaluation, Task Priors (Patel and Balestriero, 2025), whose prior kernel and
temperature have no selection rule, and Q-Score (Kalibhat et al., 2024), a
per-sample misclassification predictor whose authors do not extend it to ViT
encoders. The pair measures of the cross-model literature that were not adopted are
listed in `docs/METRICS.md`, section 9; the reasons for the rest are in its section
6.

## 6. Sample size

N is part of the protocol. Entropic effective rank keeps rising with N (RankMe used
25,600 samples; 10,000 gives over 95 percent of the asymptote at width 2048;
Tsitsulin et al., 2023, need 16,384 for a 0.95 approximation), the plug-in
participation ratio is biased by about PR/N, nearest-neighbor ID estimates carry a
few percent of subsample noise (4.5 percent median over 90 percent subsamples in
Arputharaj et al., 2026), and even anisotropy moves when N is not much larger than
D, because sampling inflates the leading eigenvalue. `convergence()` recomputes a
metric on repeated subsets at several fractions through the same preprocessing and
caches as `compute()`, so N can be chosen per metric and cells compared at unequal N
can be recognized. Published values are not comparable across different N for the
N-dependent metrics; the protocol of Kanatas et al. (2026) used 10,000 clips of 15
seconds, one per track.
