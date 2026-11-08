# req-metrics

Label-free metrics of representation quality, for analyzing the layers of trained models and for
monitoring self-supervised training.

[![tests](https://github.com/angeloskanatas/req-metrics/actions/workflows/ci.yml/badge.svg)](https://github.com/angeloskanatas/req-metrics/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/req-metrics.svg)](https://pypi.org/project/req-metrics/)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/angeloskanatas/req-metrics/blob/main/LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://github.com/angeloskanatas/req-metrics/blob/main/pyproject.toml)
[![arXiv](https://img.shields.io/badge/arXiv-2608.14819-b31b1b.svg)](https://arxiv.org/abs/2608.14819)

When you use a pretrained model as a frozen feature extractor, you have to decide which layer to
read. When you train a self-supervised model, you want to know whether the run is healthy and which
checkpoint to keep. Both questions are usually answered by training labeled probes. The metrics in
this library answer them from the embeddings alone, with a few thousand unlabeled samples.

req-metrics has 39 metrics. They cover the spectrum of the embeddings (effective rank or RankMe,
α-ReQ, anisotropy), intrinsic dimension (TwoNN, GRIDE, MLE), neighborhoods and clusters, invariance
and discriminability over augmented views (LiDAR, InfoNCE, DiME), pitch-transposition equivariance
(PTE), and the similarity between two representations (CKA, SVCCA, information imbalance). Each one
is a PyTorch function on tensors, so it works with embeddings from any model.

On top of the metrics, `rq.compute` draws a seeded subsample, shares spectra and neighbor tables
between metrics, and works at three levels: one vector per sample, the tokens of each sample, or the
tokens of all samples. It returns records that keep the parameters, sample size and citation of every
value.

The library was written for the paper [What Makes a Good Layer?](https://arxiv.org/abs/2608.14819)
(ISMIR 2026) and includes more metrics than the paper uses.

## Installation

Requires Python 3.10+ and PyTorch 2.1+.

```bash
pip install req-metrics
pip install "req-metrics[lightning]"   # with the Lightning callback
```

The development version:

```bash
pip install "req-metrics @ git+https://github.com/angeloskanatas/req-metrics.git"
```

## Quick start

### Analyzing a trained model

Pass the embeddings of each layer as a dictionary keyed by layer index. The `model` and `pooling`
labels are stored in the record.

```python
import req_metrics as rq

layers = {0: z0, 1: z1, 2: z2}                       # (N, D) per layer: one vector per sample
rec = rq.compute(layers, ["effective_rank", "intrinsic_dimension/gride", "anisotropy/spectral", "self_clustering"],
                 n_items=10000, seed=42, model="my-encoder", pooling="time-mean")
rec.profile("effective_rank")                         # [(layer, value), ...]
print(rec.to_markdown())                              # layers x metrics; also to_csv, to_json, to_pandas

tokens = {0: [t0_s0, t0_s1, ...], 1: [...]}          # (T_i, D) frames or patches per sample
rq.compute(tokens, ["trajectory_curvature", "effective_rank"], level="sample", n_items=2000)
rq.compute(tokens, ["effective_rank"], level="population", n_tokens=10000)

views = {0: v0, 1: v1}                                # (q, N, D) augmented views of the same samples
rq.compute(views, ["lidar", "infonce"], views=rq.ViewSpec(source="shared", augmentations=("PitchShift(-4..4)",), q=10))

rq.compute(shifted_layers, ["pte"], shifts=rq.ShiftSpec("waveform pitch shift", semitones=tuple(range(1, 12))))
rq.compute_pairs(layers, metrics=["information_imbalance", "cka"])          # every pair of layers
rq.compute_pairs(audio_layers, text_layers, metrics=["neighborhood_overlap"],  # two models or modalities
                 params={"neighborhood_overlap": {"k": 10, "l2": True}}, model="audio", model_b="text")
rq.calibrate(rq.cka, a, b, groups=clip_ids)          # against permutations of whole clips (Gröger et al., arXiv:2602.14486)
rq.jackknife(rq.information_imbalance, a, b)          # mean and spread over five half-samples (Acevedo et al., arXiv:2505.17101)
rq.compute(tokens, ["dense_structure"], level="population")  # needs the sample of every token

rq.convergence(z1, "effective_rank").to_markdown()   # the value against the sample size
rq.top_layers(rec, "intrinsic_dimension/gride", k=3)
p = rq.protocols.get("kanatas2026")                  # the metrics and parameters of a published analysis
rq.compute(layers, p.names("sequence"), params=p.params("sequence"), n_items=p.n_items)
```

Single metrics can also be called directly and return `MetricResult(value, extras)`, for example
`rq.effective_rank(x)`, `rq.lidar(views)` or `rq.pte(z, shifted)`. The functions in
`req_metrics.layouts` turn spectrogram-patch grids and frame sequences into points, trajectories or
token sets.

### Monitoring training

The Lightning callback hooks the blocks of the model and keeps a fixed subset of the training set
for monitoring. At the start of training and then on a schedule, it computes every metric on every
layer and logs one scalar per layer and metric (`layer_metrics/<metric>_layer_<l>`). On Weights &
Biases it also plots the depth profile at each sweep, and with `spectra=256` the singular spectrum of
every layer.

```python
from req_metrics.integrations.lightning import LayerMonitorCallback

trainer = pl.Trainer(callbacks=[LayerMonitorCallback(
    ["effective_rank", "intrinsic_dimension/mlid", "anisotropy/spectral"], layers="backbone.blocks", pool="cls",
    n_items=5000, every_n_epochs=5, sweep_steps=(100, 300, 1000, 3000, 10000),
    model_attr="backbone",
    view_metrics=["lidar"], augment=my_augment, q=10,   # views: the objective's own positives
    online=True)])                                      # also the training-batch buffer, as a collapse indicator
```

Without Lightning, `LayerMonitor` does the same with a forward function and a data loader. With
`drift_metrics`, each layer is also compared with its state at the first (or the previous) sweep, as
in SVCCA (Raghu et al., NeurIPS 2017):

```python
mon = rq.LayerMonitor(model.blocks, pool="cls", metrics=["effective_rank", "intrinsic_dimension/mlid"],
                      n_items=5000, drift_metrics=["cka"])
mon.sweep(model, monitor_loader, step=epoch, sinks=[rq.wandb_sink(history=mon.history), rq.csv_sink("sweeps.csv")])
mon.profiles("effective_rank")                       # {step: [(layer, value), ...]}
mon.profiles("cka")                                  # drift of every layer from its first sweep
```

A few things to keep in mind when monitoring:

- Spectral and neighbor metrics that center the embeddings miss a collapse onto a single shared
  vector. Log `normalized_std` (Chen and He, 2021) or `anisotropy/cosine` next to them.
- Each record also holds `delta_first`, the change since the first sweep, and `drawdown`, the drop
  below the best value so far. Absolute collapse thresholds vary by dataset (De Melo Costa et al.,
  2026), so a threshold on the drawdown is easier to set. Log it with `log_extras=("drawdown",)`.
- A hooked layer must return one row per item. If the trainer flattens the frames of a clip into
  the batch, pool them back into a (B, T, D) trajectory and use `level="sample"`.
- Block outputs are taken before the final normalization layer, which can change the spectrum.
  Compare the last block with the normalized output.
- Spectral metrics warn when a layer has fewer samples than dimensions, and view metrics warn when
  two passes return identical views.

The per-layer values are also written to `trainer.callback_metrics`, so `EarlyStopping` and
`ModelCheckpoint` can use them:

```python
EarlyStopping(monitor="layer_metrics/normalized_std_layer_12", mode="max", divergence_threshold=0.2 / D**0.5)
ModelCheckpoint(monitor="layer_metrics/effective_rank_layer_12", mode="max", save_top_k=1)  # only where rank tracks your probe
```

With `mode="max"`, `divergence_threshold` stops training when the value falls below it. For
`normalized_std`, 1/sqrt(D) is the value of an embedding that has not collapsed. Layer indices in the
keys are zero-padded (`layer_07` in a 12-block model). Between sweeps the last value is kept, so
match the checkpoint schedule to the sweep schedule. A resumed run restores the sweep history and the
drift reference from the checkpoint.

### Ranking runs and layers

- `rq.rank_runs({name: records}, "effective_rank", layer=12)` orders runs or checkpoints by a
  metric at one layer. This is the selection rule of RankMe and LiDAR. Use the layer you read
  downstream.
- `rq.top_layers` ranks the layers of one model.
- `rq.top_local_maxima(values, t=3, window=2)` keeps the best local maxima over checkpoints or
  layers, the rule Dai et al. (NeurIPS 2025) use with `dense_structure`.

The direction is an argument, because whether higher is better depends on the task and on how the
model was trained. If you have a few hundred labeled examples, check the ranking against a probe
(`docs/DESIGN.md`, section 4b). Near-equal values should be read as ties.

`pool` can be `cls`, `mean`, `max`, `last`, `tokens`, a grid readout (`gap`, `freq_concat_mean`,
`partitioned`, `freq_concat`, `freq_mean`, with `pool_kwargs={"grid": (F, T)}`) or any function
without parameters. `augment` is any function you supply. Results can go to `csv_sink`, `json_sink`,
`tensorboard_sink`, `wandb_sink` or any function of `(records, step)`. Monitoring and post-hoc
analysis produce records with the same schema. `docs/DESIGN.md` explains the choices behind the fixed
subset, the training-batch buffer, the sweep schedule and reading every layer.

## Metrics

| Group | Metrics | Input |
|---|---|---|
| Spectral | `alpha_req`, `anisotropy/spectral`, `effective_rank`, `eigenvalue_early_enrichment`, `matrix_entropy`, `participation_ratio` | `(N, D)` points |
| Intrinsic dimension | `intrinsic_dimension/twonn`, `intrinsic_dimension/gride`, `intrinsic_dimension/mle`, `intrinsic_dimension/mlid`, `intrinsic_dimension/mst` | `(N, D)` points |
| Relational | `anisotropy/cosine`, `normalized_std`, `self_clustering`, `uniformity` | `(N, D)` points |
| Clustering | `cluster_quality`, `dense_structure` | `(N, D)` points; `dense_structure` takes the token fields of many samples at the population level |
| Trajectory | `trajectory_curvature` | `(T, D)` per sample, time-ordered |
| Views | `alignment`, `dime`, `infonce`, `lidar` | `(q, N, D)` augmented views; `dime` takes q = 2 |
| Equivariance | `pte` | embeddings of transformed copies (pitch shifts in the paper) |
| Representation pairs | `cka`, `cycle_knn`, `information_imbalance`, `neighborhood_overlap`, `procrustes`, `ridge_predictivity`, `rsa`, `svcca` | two representations of the same items: layers, checkpoints, models or modalities |
| Functional | `jacobian_effective_rank` | the model and its inputs, during monitoring |
| Token fields | `cls_patch_cosine`, `token_cosine`, `token_gram_drift`, `token_norm_outliers` | `(T, D)` token fields per sample; `token_gram_drift` also takes a reference field and is called directly |
| Distribution | `embedding_norm`, `gaussianity`, `sparsity` | `(N, D)` points |

`rq.list_metrics()`, `rq.get_metric(name)` and `rq.describe(name)` show the input, preprocessing,
citation and card of each metric. Different estimators of the same quantity share a prefix and are
named after their method, for example `intrinsic_dimension/twonn` and `intrinsic_dimension/gride`.
None of them is the default. Options such as a bias correction are keyword arguments, and the
returned value follows the option you chose. Some metrics also report the other options in `extras`.
The participation ratio, for instance, keeps its uncorrected estimate and the three corrected ones,
and curvature reports both the signed and the folded angle.

## Documentation

- `docs/METRICS.md`: the definition, protocol and citation of every metric.
- `docs/metrics/`: one card per metric, generated from the registry.
- `docs/VIEWS.md`: how augmented views and pitch-shifted copies are passed in.
- `docs/DESIGN.md`: the design choices, the cost of each metric and the sample sizes.
- `examples/`: from per-layer embedding files to records, and correlating layer profiles with
  per-layer downstream scores.

## Reproducing the paper

`rq.protocols.get("kanatas2026")` fixes the metric variants, parameters and sample sizes used in
the paper. The per-layer results of every model and task are on the
[project page](https://angeloskanatas.github.io/music-fms-layer-eval/), and `Records.to_atlas_json`
writes records in the same format.

## Related tools

- [DADApy](https://github.com/sissa-data-science/DADApy) and
  [scikit-dimension](https://github.com/scikit-learn-contrib/scikit-dimension): intrinsic-dimension estimators.
- [reptrix](https://github.com/BARL-SSL/reptrix): RankMe, α-ReQ and LiDAR for PyTorch models.
- [stable-pretraining](https://github.com/galilai-group/stable-pretraining): Lightning callbacks for RankMe
  and LiDAR on a queue of training batches.
- [lightly](https://github.com/lightly-ai/lightly): a collapse indicator for self-supervised training.
- [information_flow](https://github.com/OFSkean/information_flow) (Skean et al., ICML 2025): layer-wise
  entropy, curvature and InfoNCE for language models.
- [synesis](https://github.com/chrispla/synesis) (Plachouras et al., IJCNN 2025): probe-based
  informativeness, equivariance, invariance and disentanglement.

## Development

```bash
git clone https://github.com/angeloskanatas/req-metrics.git && cd req-metrics
pip install -e ".[dev]"
pre-commit install
ruff check src tests examples scripts && ruff format --check src tests examples scripts
pytest
```

`CONTRIBUTING.md` explains how to add a metric.

## Citation

If you use req-metrics, please cite the paper below and the original paper of each metric you
report. Every metric's citation keys point into `docs/references.bib`, and `CITATION.cff` cites the
software.

```bibtex
@inproceedings{kanatas2026goodlayer,
  title     = {What Makes a Good Layer? Assessing the Layer-Wise Intrinsic Properties of Music Foundation Models},
  author    = {Kanatas, Angelos-Nikolaos and Kong, Yuexuan and Alonso-Jim{\'e}nez, Pablo and Serra, Xavier and Bogdanov, Dmitry},
  booktitle = {Proceedings of the International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2026},
  note      = {arXiv:2608.14819}
}
```

## License

Apache License 2.0. Code adapted from other projects is credited in `NOTICE`.
