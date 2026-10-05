# req-metrics

[![tests](https://github.com/angeloskanatas/req-metrics/actions/workflows/ci.yml/badge.svg)](https://github.com/angeloskanatas/req-metrics/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/angeloskanatas/req-metrics/blob/main/LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://github.com/angeloskanatas/req-metrics/blob/main/pyproject.toml)

req-metrics computes label-free metrics of learned representations: effective rank,
intrinsic dimension, anisotropy, LiDAR, InfoNCE, pitch-transposition equivariance
and thirty-one others, as functions on embedding tensors. It serves two uses: layer-wise
analysis of trained models, and monitoring during training, where the same metrics
flag collapse and compare runs and checkpoints without labels.

Estimators are PyTorch functions on tensors. The pipeline adds shared spectra and
neighbor tables, seeded subsampling, three levels (one vector per sample, the tokens
of each sample, the tokens of all samples) and records that carry the estimator
parameters, preprocessing, sample size and citation of every value. Each metric
follows a published definition and is checked against its reference implementation.
Inputs are embeddings of any modality; only `pte` is specific to music.

## Installation

Python 3.10 or later and PyTorch 2.1 or later.

```bash
pip install "req-metrics @ git+https://github.com/angeloskanatas/req-metrics.git"
pip install "req-metrics[lightning] @ git+https://github.com/angeloskanatas/req-metrics.git"   # with the Lightning callback
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

layers = {0: z0, 1: z1, 2: z2}                      # (N, D) per layer, one vector per sample (sequence level)
rec = rq.compute(layers, ["effective_rank", "intrinsic_dimension/gride", "anisotropy", "self_clustering"],
                 n=10000, seed=42, model="my-encoder", pooling="time-mean")
rec.profile("effective_rank")                        # [(layer, value), ...]
rec.to_csv("my-encoder.csv")                         # or to_json, to_pandas

tokens = {0: [t0_s0, t0_s1, ...], 1: [...]}         # (T_i, D) frames or patches per sample
rq.compute(tokens, ["trajectory_curvature", "effective_rank"], level="sample", n=2000)  # per sample, averaged
rq.compute(tokens, ["effective_rank"], level="population", n=10000)  # tokens of all samples as one cloud

views = {0: v0, 1: v1}                               # (q, N, D) augmented views of the same samples
rq.compute(views, ["lidar", "infonce"], views=rq.ViewSpec(source="shared", augmentations=("PitchShift(-4..4)",), q=10))

rq.compute(shifted_layers, ["pte"], shifts=rq.ShiftSpec("waveform pitch shift", semitones=tuple(range(1, 12))))
rq.compute_pairs(layers, k=1)                        # information imbalance between all layer pairs
rq.compute_pairs(layers, metric="cka")              # or "svcca", "rsa", "neighborhood_overlap"
rq.compute_pairs(audio_layers, text_layers, metric="neighborhood_overlap", k=10, params={"l2": True},
                 model="audio-encoder", model_b="text-encoder")  # two models or modalities, paired items

rq.convergence(z1, "effective_rank").to_markdown()  # does the value depend on N? subsample curve
rq.top_layers(rec, "intrinsic_dimension/gride", k=3)  # the k layers ranked best by a metric
p = rq.protocols.get("kanatas2026")                 # metric variants and parameters of a published protocol
rq.compute(layers, p.names("sequence"), params=p.params("sequence"), n=p.n_items)
```

Single estimators are plain functions returning `MetricResult(value, extras)`:
`rq.effective_rank(x)`, `rq.lidar(views)`, `rq.pte(z, shifted)`. Spectrogram-patch
grids and frame sequences are turned into points, trajectories or token clouds by
the parameter-free readouts in `req_metrics.layouts`.

### Monitoring during training

During training the same records come from forward hooks, to watch collapse and
geometry within a run and to compare runs or checkpoints without labels. With
PyTorch Lightning it is one callback. It finds the blocks, draws a fixed
monitoring subset of the training set, sweeps every layer at the start of
training and on the schedule you give, and logs to the trainer's logger
(`layer_metrics/<metric>_layer_<l>`, plus layer-profile plots on Weights & Biases).

```python
from req_metrics.integrations.lightning import LayerMonitorCallback

trainer = pl.Trainer(callbacks=[LayerMonitorCallback(
    ["effective_rank", "intrinsic_dimension/mlid", "anisotropy"], layers="backbone.blocks", pool="cls",
    n_items=5000, every_n_epochs=5, sweep_steps=(100, 300, 1000, 3000, 10000),
    model_attr="backbone",
    view_metrics=["lidar"], augment=my_augment, q=10,   # views: the objective's own positives
    online=True)])                                      # also the training-batch buffer, as a collapse indicator
```

Without Lightning, `LayerMonitor` does the same with a forward callable and a loader:

```python
mon = rq.LayerMonitor(model.blocks, pool="cls", metrics=["effective_rank", "intrinsic_dimension/mlid"], n_items=5000)
mon.sweep(model, monitor_loader, step=epoch, sinks=[rq.wandb_sink(history=mon.history), rq.csv_sink("sweeps.csv")])
mon.profiles("effective_rank")                       # {step: [(layer, value), ...]}
```

Centered spectral and neighbor metrics cannot see representations collapsing onto one
shared vector; `normalized_std` (Chen and He, 2021) and `cosine_anisotropy` can, so log
one of them beside the others.

Selection follows the published rules: `rq.rank_runs({name: records},
"effective_rank", layer=12)` orders runs or checkpoints by a metric at the layer
read downstream, as RankMe and LiDAR do, and `rq.top_layers` ranks the layers of
one run. The direction is an argument because a metric's sign depends on the
task family and the training paradigm.

`pool` is `cls`, `mean`, `max`, `last`, `tokens`, a grid readout (`gap`,
`freq_concat_mean`, `partitioned`, `freq_concat`, `freq_mean`, with
`pool_kwargs={"grid": (F, T)}`) or any parameter-free callable; `augment` is any
callable you supply. Sinks are `csv_sink`, `json_sink`, `tensorboard_sink`,
`wandb_sink` or any callable of `(records, step)`. Monitoring records have the same
schema as post-hoc records. The rationale for the fixed subset, the training-batch
buffer, the sweep schedule and reading every layer is in `docs/DESIGN.md`.

## Metrics

| Group | Metrics | Input |
|---|---|---|
| Spectral | `alpha_req`, `anisotropy`, `effective_rank`, `eigenvalue_early_enrichment`, `matrix_entropy`, `participation_ratio` | `(N, D)` points |
| Intrinsic dimension | `intrinsic_dimension/twonn`, `intrinsic_dimension/gride`, `intrinsic_dimension/mle`, `intrinsic_dimension/mlid`, `intrinsic_dimension/mst` | `(N, D)` points |
| Local geometry | `local_rectifiability`, `neighborhood_curvature` | `(N, D)` points |
| Relational | `cosine_anisotropy`, `normalized_std`, `self_clustering`, `uniformity` | `(N, D)` points |
| Clustering | `cluster_quality` | `(N, D)` points |
| Trajectory | `trajectory_curvature` | `(T, D)` per sample, time-ordered |
| Views | `alignment`, `dime`, `infonce`, `lidar` | `(q, N, D)` augmented views |
| Equivariance | `pte` | embeddings of pitch-shifted copies |
| Representation pairs | `cka`, `information_imbalance`, `neighborhood_overlap`, `rsa`, `svcca` | two representations of the same items: layers, checkpoints, models or modalities |
| Functional | `jacobian_effective_rank` | the model and its inputs, during monitoring |
| Token fields | `cls_patch_cosine`, `token_cosine`, `token_gram_drift`, `token_norm_outliers` | `(T, D)` token fields per sample |
| Distribution | `embedding_norm`, `gaussianity`, `sparsity` | `(N, D)` points |

`rq.list_metrics()` and `rq.get_metric(name)` expose the registry: input contract,
preprocessing, citation keys and item cap of every metric. Estimators of one quantity
share a prefix and are named by their method (`intrinsic_dimension/twonn`,
`intrinsic_dimension/gride`); none is a default. Settings of one computation, such as
a bias correction or a spectrum convention, are arguments, and every alternative is in
the extras of the record.

## Documentation

- `docs/METRICS.md`: definition, canonical protocol and citation of every metric,
  the verification against reference implementations, and what was considered and
  not adopted.
- `docs/metrics/`: one card per metric, generated from the registry.
- `docs/VIEWS.md`: how augmented views and pitch-shifted copies enter.
- `docs/DESIGN.md`: the decisions behind the library, cost classes, sample size.
- `examples/`: per-layer memmaps to records, and correlation of layer profiles with
  per-layer downstream scores (Spearman and depth-controlled partial Spearman).

## Reproducing Kanatas et al. (2026)

`rq.protocols.get("kanatas2026")` pins the metric variants, parameters and sample
sizes used in the paper. Its layer-wise results and raw per-layer records are on the
companion site, https://angeloskanatas.github.io/music-fms-layer-eval/, and
`Records.to_atlas_json` writes records in that format.

## Related tools

- [DADApy](https://github.com/sissa-data-science/DADApy) and
  [scikit-dimension](https://github.com/scikit-learn-contrib/scikit-dimension): intrinsic-dimension estimators.
- [reptrix](https://github.com/BARL-SSL/reptrix): RankMe, alpha-ReQ and LiDAR for PyTorch models.
- [stable-pretraining](https://github.com/galilai-group/stable-pretraining): Lightning callbacks for RankMe
  and LiDAR on a queue of training batches.
- [lightly](https://github.com/lightly-ai/lightly): a collapse indicator for self-supervised training.
- [information_flow](https://github.com/OFSkean/information_flow) (Skean et al., ICML 2025,
  arXiv:2502.02013): layer-wise entropy, curvature and InfoNCE for language models.
- [synesis](https://github.com/chrispla/synesis) (Plachouras et al., IJCNN 2025, arXiv:2505.06224):
  probe-based informativeness, equivariance, invariance and disentanglement.
- Arputharaj, Jönsson and Eilertsen (TMLR 2026, arXiv:2608.23182): a comparative
  study of seven label-free metrics on 260 vision models.

req-metrics differs in holding these families in one registry with recorded protocol,
sharing spectra and neighbor tables across estimators, and producing the same records
during training and post hoc.

## Development

```bash
pip install -e ".[dev]"
pre-commit install
ruff check src tests examples scripts && ruff format --check .
pytest
```

See `CONTRIBUTING.md` for how a metric is added.

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
