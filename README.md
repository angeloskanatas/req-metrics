# req-metrics

[![tests](https://github.com/angeloskanatas/req-metrics/actions/workflows/ci.yml/badge.svg)](https://github.com/angeloskanatas/req-metrics/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/angeloskanatas/req-metrics/blob/main/LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://github.com/angeloskanatas/req-metrics/blob/main/pyproject.toml)

Label-free representation-quality metrics for layer-wise analysis of pretrained
models and for monitoring during training: intrinsic dimension, effective rank,
anisotropy, self-clustering, uniformity, trajectory curvature, LiDAR, InfoNCE,
alignment, pitch-transposition equivariance and more, as pure functions on
embedding tensors. Every result records the preprocessing, the sample size and
the citation of the metric it reports.

Post hoc, such metrics characterize the layers of a trained model and predict
which layer transfers. During training, they flag collapse and select
checkpoints and hyperparameters without labels. The inputs are embedding
tensors of any modality; the documentation says clips for the items of a
corpus, and only the pitch-transposition equivariance metric is specific to
music. The literature behind both uses is listed in `docs/DESIGN.md`.

The registry holds 41 metrics in ten groups: spectral, intrinsic dimension, local
geometry, relational, trajectory, views, equivariance, layer pairs, token fields
and norms. Each metric has a published definition and is checked against the
implementation it was adopted from. The same estimators run post hoc on
extracted embeddings and, through forward hooks, on every layer of a model
while it trains.

## Installation

Python 3.10 or later and PyTorch 2.1 or later.

```bash
pip install "req-metrics @ git+https://github.com/angeloskanatas/req-metrics.git@v0.1.0"
pip install "req-metrics[lightning] @ git+https://github.com/angeloskanatas/req-metrics.git@v0.1.0"   # with the Lightning callback
```

For development, with the test, lint and Lightning dependencies:

```bash
git clone https://github.com/angeloskanatas/req-metrics.git && cd req-metrics
pip install -e ".[dev]"
pytest
```

## Quick start

### Post-hoc analysis of a trained model

Layers are plain tensors keyed by layer index; how they were extracted is
recorded through the `model` and `pooling` labels.

```python
import req_metrics as rq

layers = {0: z0, 1: z1, 2: z2}                      # (N, D) pooled embeddings per layer, rows = clips
rec = rq.compute(layers, ["effective_rank", "intrinsic_dimension/gride", "anisotropy", "self_clustering"],
                 n=10000, seed=42, model="my-encoder", pooling="time-mean")
rec.profile("effective_rank")                        # [(layer, value), ...]
rec.to_csv("my-encoder.csv")                         # or to_json, to_pandas

frames = {0: [t0_clip0, t0_clip1, ...], 1: [...]}   # (T_i, D) per clip, time-ordered
rq.compute(frames, ["trajectory_curvature"], population="frames", n=2000)

views = {0: v0, 1: v1}                               # (q, N, D) augmented views of the same clips
rq.compute(views, ["lidar", "infonce"], views=rq.ViewSpec(source="shared", augmentations=("PitchShift(-4..4)",), q=10))

rq.compute(shifted_layers, ["pte"], shifts=rq.ShiftSpec("waveform pitch shift", semitones=tuple(range(1, 12))))
rq.compute_pairs(layers, k=1)                        # information imbalance between all layer pairs
rq.compute_pairs(layers, metric="neighborhood_overlap")

rq.convergence(z1, "effective_rank").to_markdown()  # does the value depend on N? subsample curve
rq.top_layers(rec, "intrinsic_dimension/gride", k=3)  # the proxy-ranked layer shortlist of Kanatas et al. (2026)
p = rq.protocols.get("kanatas2026")                 # metric variants and parameters of a published protocol
rq.compute(layers, p.names("pooled"), params=p.params("pooled"), n=p.n_items)
```

Single estimators are plain functions returning `MetricResult(value, extras)`:
`rq.effective_rank(x)`, `rq.lidar(views)`, `rq.pte(z, shifted)`. Spectrogram-patch
grids and frame sequences are turned into points, trajectories or token clouds by
the parameter-free readouts in `req_metrics.layouts`.

### Monitoring during training

The same records, produced while a model trains, for watching collapse and
geometry during a run and for comparing runs or checkpoints without labels.
With PyTorch Lightning it is one callback. It finds the blocks, draws a fixed
monitoring subset of the training set, sweeps every layer at the start of
training and on the schedule you give, and logs to the trainer's logger
(`layer_metrics/<metric>_layer_<l>`, plus layer-profile plots on Weights & Biases).

```python
from req_metrics.integrations.lightning import LayerMonitorCallback

trainer = pl.Trainer(callbacks=[LayerMonitorCallback(
    ["effective_rank", "mlid", "anisotropy"], layers="backbone.blocks", pool="cls",
    n_items=5000, every_n_epochs=5, sweep_steps=(100, 300, 1000, 3000, 10000),
    model_attr="backbone",
    view_metrics=["lidar"], augment=my_augment, q=10,   # views: the objective's own positives
    online=True)])                                      # also the training-batch buffer, as a collapse alarm
```

Without Lightning, `LayerMonitor` does the same with a forward callable and a loader:

```python
mon = rq.LayerMonitor(model.blocks, pool="cls", metrics=["effective_rank", "mlid"], n_items=5000)
mon.sweep(model, monitor_loader, step=epoch, sinks=[rq.wandb_sink(history=mon.history), rq.csv_sink("sweeps.csv")])
mon.profiles("effective_rank")                       # {step: [(layer, value), ...]}
```

Selection follows the published rules: `rq.rank_runs({name: records},
"effective_rank", layer=12)` orders runs or checkpoints by a metric at the layer
read downstream, as RankMe and LiDAR do, and `rq.top_layers` ranks the layers of
one run. The direction is an argument because a metric's sign depends on the
task family and the training paradigm.

`pool` is `cls`, `mean`, `max`, `last`, `frames`, a grid readout (`gap`,
`freq_concat_mean`, `partitioned`, `freq_concat`, with `pool_kwargs={"grid": (F, T)}`)
or any parameter-free callable; `augment` is yours. Sinks are `csv_sink`,
`json_sink`, `tensorboard_sink`, `wandb_sink` or any callable of `(records, step)`.
Monitoring records have the same schema as post-hoc records. The rationale for
the fixed subset, the training-batch buffer, the sweep schedule and reading every
layer is in `docs/DESIGN.md`.

## Metrics

| Group | Metrics | Input |
|---|---|---|
| Spectral | `alpha_req`, `anisotropy`, `anisotropy/cosine`, `effective_rank`, `effective_rank/variance`, `eigenvalue_early_enrichment`, `gaussianity`, `gaussianity/ks`, `gaussianity/swd`, `matrix_entropy`, `participation_ratio`, `participation_ratio/corrected`, `sparsity`, `spectral_entropy` | `(N, D)` points |
| Intrinsic dimension | `intrinsic_dimension`, `intrinsic_dimension/gride`, `intrinsic_dimension/mle`, `mlid`, `mst_dimension` | `(N, D)` points |
| Local geometry | `local_rectifiability`, `neighborhood_curvature` | `(N, D)` points |
| Relational | `normalized_std`, `self_clustering`, `uniformity` | `(N, D)` points |
| Trajectory | `trajectory_curvature`, `trajectory_curvature/abs`, `trajectory_curvature/recurve` | `(T, D)` per clip, time-ordered |
| Views | `alignment`, `dime`, `infonce`, `lidar` | `(q, N, D)` augmented views |
| Equivariance | `pte`, `pte/cpsd`, `pte/mlp` | embeddings of pitch-shifted copies |
| Layer pairs | `information_imbalance`, `neighborhood_overlap` | two layers |
| Token fields | `cls_patch_cosine`, `token_cosine`, `token_gram_drift`, `token_norm_outliers` | `(T, D)` token fields per clip |
| Norms | `embedding_norm` | `(N, D)` points |

`rq.list_metrics()` and `rq.get_metric(name)` expose the registry: input contract,
preprocessing, citation keys and item cap of every metric. Metric names with a
slash are variants of the base metric.

## Documentation

- `docs/METRICS.md`: definition, canonical protocol and citation of every metric, the
  verification against reference implementations, and what was considered and not adopted.
- `docs/metrics/`: one card per metric, generated from the registry.
- `docs/VIEWS.md`: how augmented views and pitch-shifted copies enter.
- `docs/DESIGN.md`: the decisions behind the library, cost classes, sample size.
- `examples/`: per-layer memmaps to records, and correlation of layer profiles with
  per-layer downstream scores (Spearman and depth-controlled partial Spearman).

## Reproducing Kanatas et al. (2026)

`rq.protocols.get("kanatas2026")` pins the metric variants, parameters and sample
sizes of the paper per input kind. The layer-wise results and raw per-layer records
are on the companion site, https://angeloskanatas.github.io/music-fms-layer-eval/,
and `Records.to_atlas_json` writes new records in its format.

## Related tools

DADApy and scikit-dimension implement intrinsic-dimension estimators; reptrix
RankMe, alpha-ReQ and LiDAR for PyTorch models; stable-pretraining Lightning
callbacks for RankMe and LiDAR on a queue of training batches; lightly a single
collapse indicator; the harness of Skean et al. (2025) layer-wise entropy,
curvature and InfoNCE for language models; synesis (Plachouras et al., 2025)
probe-based informativeness, equivariance, invariance and disentanglement. The
comparative study of Arputharaj, Jönsson and Eilertsen (TMLR 2026) evaluates seven
label-free metrics on 260 vision models.

req-metrics collects these families in one registry with recorded protocol and
provenance, shares the singular spectrum and the neighbor table across
estimators, separates the population a metric sees from the estimator, and
produces the same records during training and post hoc, for every layer. Any
extractor can supply the embeddings; Essentia's `TensorflowPredictMAEST`, for
example, exposes intermediate layers through its `output` parameter.

## Development

```bash
pip install -e ".[dev]"
pre-commit install
ruff check src tests examples scripts && ruff format --check .
pytest
```

See `CONTRIBUTING.md` for how a metric is added. Agentic coding tools were used
as a programming aid. Every metric follows its source paper and reference
implementation and is checked in `tests/`; responsibility for the code rests
with the author.

## Citation

If you use req-metrics, cite the toolkit (`CITATION.cff`), the paper below, and
the original paper of every metric you report; each registry entry's citation
keys point into `docs/references.bib`.

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

Apache License 2.0. Code ported from other projects is attributed in `NOTICE`.
