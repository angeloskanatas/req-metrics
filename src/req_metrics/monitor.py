"""Layer-wise monitoring during training, framework-independent.

Two modes. LayerMonitor registers forward hooks on the given layer modules,
runs a fixed monitoring set through a forward callable in eval mode, pools each
layer's output, and hands the per-layer tensors to compute(); the same records
come out as in a post-hoc run, so curves logged during training are directly
comparable with curves computed on stored embeddings. OnlineBuffer hooks the
same modules during the ordinary training forward passes and keeps the most
recent rows, at no extra cost, as a collapse indicator on the training-time
representation. Which modules are layers, how a block output becomes one
vector per clip, and how an augmented view is drawn are arguments.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from req_metrics.pipeline import compute
from req_metrics.records import Record, Records
from req_metrics.view_construction import ViewSpec

Pooler = Callable[[Any], Tensor]


def _first_tensor(out: Any) -> Tensor:
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)):
        for o in out:
            if torch.is_tensor(o):
                return o
    raise TypeError(f"hooked module returned no tensor: {type(out).__name__}")


def resolve_layers(model: nn.Module, layers: str | Sequence[nn.Module] | None = None) -> list[nn.Module]:
    """The layer modules to hook: an explicit sequence, an attribute path, or the model's main block list.

    With a string, follows dotted attributes from the model (e.g. "encoder.layers",
    "blocks") and returns that container's children. With None, picks the first
    nn.ModuleList or nn.Sequential with at least two children whose name contains
    "blocks", "layers", "layer" or "encoder", in that order of preference.
    """
    if layers is not None and not isinstance(layers, str):
        return list(layers)
    if isinstance(layers, str):
        node: Any = model
        for part in layers.split("."):
            node = getattr(node, part)
        return list(node.children()) if isinstance(node, nn.Module) else list(node)
    candidates = [
        (name, m)
        for name, m in model.named_modules()
        if isinstance(m, (nn.ModuleList, nn.Sequential)) and len(list(m.children())) >= 2
    ]
    for key in ("blocks", "layers", "layer", "encoder"):
        for name, m in candidates:
            if key in name.lower():
                return list(m.children())
    if candidates:
        return list(candidates[0][1].children())
    raise ValueError("no block list found; pass layers explicitly")


def make_pooler(
    pool: str | Pooler,
    *,
    grid: tuple[int, int] | None = None,
    time_axis: int = 1,
    n_prefix: int = 0,
    freq_chunks: int = 1,
    time_chunks: int = 1,
) -> Pooler:
    """A pooling callable from a name, for (B, L, D) block outputs.

    Sequence readouts: "cls" (token 0), "mean" (over all tokens), "max", "last" (the
    final token, for causal decoders), "frames" (no pooling; (B, T, D) for the frames
    and tokens populations). Grid readouts, for spectrogram-patch encoders whose L
    tokens are an (F, T) grid given as `grid`: "gap" (mean over patches), "freq_concat_mean"
    (frequency patches concatenated, then the mean over time, (B, F * D); the readout
    of MSM-MAE and M2D), "partitioned" (block means over freq_chunks x time_chunks
    regions, Gu et al., 2026), "freq_concat" (the (B, T, F * D) trajectory for the
    frames population) and "freq_mean" (the (B, T, D) trajectory). The grid readouts
    are the batched form of the functions in `layouts`.

    Args:
        pool: Readout name, or a callable mapping one block output to (B, D) or (B, T, D).
        grid: (F, T) patch counts of the grid readouts; the first `n_prefix` tokens are dropped.
        time_axis: 1 when tokens are ordered frequency-major (index f * T + t), 0 when
            time-major (index t * F + f).
        n_prefix: Leading class or register tokens to drop before a grid readout.
        freq_chunks, time_chunks: Block counts of "partitioned".
    """
    if callable(pool):
        return pool
    if pool == "cls":
        return lambda out: out[:, 0] if out.ndim == 3 else out
    if pool == "mean":
        return lambda out: out.mean(dim=1) if out.ndim == 3 else out
    if pool == "max":
        return lambda out: out.amax(dim=1) if out.ndim == 3 else out
    if pool == "last":
        return lambda out: out[:, -1] if out.ndim == 3 else out
    if pool == "frames":
        return lambda out: out
    if pool in ("gap", "freq_concat_mean", "partitioned", "freq_concat", "freq_mean"):
        if grid is None:
            raise ValueError(f"pooling {pool!r} needs grid=(F, T)")
        f_count, t_count = grid

        def to_grid(out: Tensor) -> Tensor:  # (B, F, T, D)
            tokens = out[:, n_prefix:]
            if tokens.shape[1] != f_count * t_count:
                raise ValueError(
                    f"expected {f_count * t_count} grid tokens after {n_prefix} prefix tokens, got {tokens.shape[1]}"
                )
            if time_axis == 1:
                return tokens.reshape(tokens.shape[0], f_count, t_count, -1)
            return tokens.reshape(tokens.shape[0], t_count, f_count, -1).transpose(1, 2)

        if pool == "gap":
            return lambda out: to_grid(out).mean(dim=(1, 2))
        if pool == "freq_concat_mean":
            return lambda out: to_grid(out).mean(dim=2).flatten(1)
        if pool == "freq_concat":
            return lambda out: to_grid(out).transpose(1, 2).flatten(2)
        if pool == "freq_mean":
            return lambda out: to_grid(out).mean(dim=1)

        def partitioned(out: Tensor) -> Tensor:  # same bins and block order as layouts.grid_to_pooled
            g = to_grid(out).permute(0, 3, 1, 2)  # (B, D, F, T)
            blocks = torch.nn.functional.adaptive_avg_pool2d(g, (freq_chunks, time_chunks))
            return blocks.permute(0, 2, 3, 1).flatten(1)  # (B, fc * tc * D)

        return partitioned
    raise ValueError(f"unknown pooling {pool!r}")


def monitor_loader(dataset, n_items: int, batch_size: int = 32, seed: int = 0):
    """A fixed, shuffle-free DataLoader over a seeded random subset of a dataset, for repeatable sweeps."""
    from torch.utils.data import DataLoader, Subset

    n = len(dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))[: min(n_items, n)].sort().values
    return DataLoader(Subset(dataset, idx.tolist()), batch_size=batch_size, shuffle=False, num_workers=0)


def metric_key_prefix(record: Record) -> tuple[str, str]:
    """(scalar prefix, profile prefix) for logging keys: online_* for training-batch records, layer_metrics/profiles otherwise."""
    if record.extras.get("source") == "training-batches":
        return "online_metrics", "online_profiles"
    return "layer_metrics", "profiles"


class OnlineBuffer:
    """Ring buffers of pooled block outputs captured from the ordinary training forward passes.

    The monitoring mode without extra forward passes. Hooks on the layer modules copy
    the pooled output of every training-mode forward into a per-layer ring buffer that
    keeps the most recent n_items rows, and compute() runs point metrics on the buffers.
    What is measured is the training-time representation: augmented inputs, train-mode
    layers (dropout, batch statistics, masking) and weights that moved while the window
    filled. Records carry extras["source"] = "training-batches" and the sinks log them
    under online_metrics/, so they never mix with fixed-subset records. Use it as a
    collapse indicator; use LayerMonitor for curves that compare across sweeps, runs and
    checkpoints. Forward hooks fire again on activation recomputation (gradient
    checkpointing), which duplicates rows; blocks skipped by stochastic depth fire
    less often, and all buffers are read at the smallest filled count. Eval-mode
    forwards (validation, LayerMonitor sweeps) are ignored.

    Args:
        layer_modules: Modules whose forward outputs are the layers, in depth order.
        pool: A readout name (see make_pooler) or a callable mapping one block output to (B, D).
        pool_kwargs: Keyword arguments of make_pooler for a named readout.
        n_items: Rows kept per layer (the most recent ones).
        device: Buffer device; default is the device of the hooked output. Stored as float32.
    """

    def __init__(
        self,
        layer_modules: Sequence[nn.Module],
        pool: str | Pooler,
        n_items: int = 5000,
        device: torch.device | str | None = None,
        *,
        pool_kwargs: Mapping[str, Any] | None = None,
    ):
        self.layer_modules = list(layer_modules)
        self.pool = make_pooler(pool, **dict(pool_kwargs or {}))
        self.n_items = n_items
        self.device = device
        self.buffers: dict[int, Tensor] = {}
        self.pointer: dict[int, int] = {}
        self.seen: dict[int, int] = {}
        self.handles: list = []
        self.history: list[tuple[int, Records]] = []

    def attach(self) -> None:
        """Register the forward hooks (idempotent)."""
        if self.handles:
            return
        self.handles = [m.register_forward_hook(self._make_hook(i)) for i, m in enumerate(self.layer_modules)]

    def detach(self) -> None:
        """Remove the forward hooks."""
        for h in self.handles:
            h.remove()
        self.handles = []

    def _make_hook(self, i: int):
        def hook(module, inputs, output):
            if not module.training:
                return
            z = self.pool(_first_tensor(output)).detach()
            if z.ndim != 2:
                raise TypeError(f"online pooling must return (B, D), got shape {tuple(z.shape)}")
            z = z.to(dtype=torch.float32, device=self.device or z.device)
            if i not in self.buffers:
                self.buffers[i] = torch.empty((self.n_items, z.shape[1]), dtype=torch.float32, device=z.device)
                self.pointer[i], self.seen[i] = 0, 0
            buf, n, b = self.buffers[i], self.n_items, z.shape[0]
            if b >= n:
                buf.copy_(z[-n:])
                self.pointer[i] = 0
            else:
                start = self.pointer[i]
                first = min(b, n - start)
                buf[start : start + first] = z[:first]
                if first < b:
                    buf[: b - first] = z[first:]
                self.pointer[i] = (start + b) % n
            self.seen[i] += b

        return hook

    def layers(self) -> dict[int, Tensor]:
        """Layer index -> (m, D) tensor of the most recent rows, m the smallest filled count over layers.

        Rows are not aligned across layers. Blocks skipped by stochastic depth (layer
        drop) fire their hook less often, so their buffers fill more slowly; every
        buffer is cut to the common count so the records share one N.
        """
        counts = [min(self.seen[i], self.n_items) for i in self.buffers if self.seen[i] > 0]
        if not counts:
            return {}
        m = min(counts)
        return {i: buf[:m] for i, buf in self.buffers.items() if self.seen[i] > 0}

    def compute(
        self,
        metrics: Sequence[str],
        step: int,
        sinks: Sequence[Callable[[Records, int], None]] = (),
        *,
        params: Mapping[str, Mapping[str, Any]] | None = None,
        model: str | None = None,
        pooling: str | None = None,
        corpus: str | None = None,
        seed: int = 0,
    ) -> Records:
        """Run point metrics on the current buffers, tag the records as training-batch records, and pass them to the sinks."""
        layers = self.layers()
        if not layers:
            return Records()
        rec = compute(
            layers,
            list(metrics),
            population="pooled",
            seed=seed,
            params=params,
            model=model,
            pooling=pooling,
            corpus=corpus,
        )
        for r in rec:
            r.extras["step"] = step
            r.extras["source"] = "training-batches"
        self.history.append((step, rec))
        for sink in sinks:
            sink(rec, step)
        return rec


class LayerMonitor:
    """Collect per-layer representations of a fixed monitoring set and compute metrics on them.

    Args:
        layer_modules: Modules whose forward outputs are the layers, in depth order
            (e.g. the blocks of a transformer or the stages of a CNN).
        pool: A readout name (see make_pooler: "cls", "mean", "max", "last", "frames", and the
            grid readouts "gap", "freq_concat_mean", "partitioned", "freq_concat", "freq_mean" with
            pool_kwargs such as grid=(F, T)), or a callable mapping one block output to a
            tensor: (B, D) one vector per clip for population "pooled", or (B, T, D) frames
            for "frames" and "tokens". The pooling must be parameter-free.
        pool_kwargs: Keyword arguments of make_pooler for a named readout.
        metrics: Registry names of point or trajectory metrics; all of one input kind.
        n_items: Clips of the monitoring set to use; the loader is consumed until reached.
        population: "pooled", "frames" or "tokens", matching what pool returns; the frames and
            tokens populations need clips of equal length within a sweep (batches are concatenated).
        params: Metric name -> estimator keyword arguments.
        model, pooling, corpus: Labels written into every record.
        view_metrics: Registry names of view metrics to compute when `augment` is given.
        augment: augment(batch, generator) -> batch; applied on q extra passes over the
            loader to build (q, N, D) views per layer. Keep the loader order fixed across passes.
        q: Views per clip for the view metrics.
        view_spec: Description of the view construction, recorded with the view records.
        seed: Subsampling and augmentation seed.
    """

    def __init__(
        self,
        layer_modules: Sequence[nn.Module],
        pool: str | Pooler,
        metrics: Sequence[str],
        *,
        pool_kwargs: Mapping[str, Any] | None = None,
        n_items: int = 10000,
        population: str = "pooled",
        params: Mapping[str, Mapping[str, Any]] | None = None,
        model: str | None = None,
        pooling: str | None = None,
        corpus: str | None = None,
        view_metrics: Sequence[str] = (),
        augment: Callable[[Any, torch.Generator], Any] | None = None,
        q: int = 2,
        view_spec: ViewSpec | None = None,
        seed: int = 0,
    ):
        self.layer_modules = list(layer_modules)
        self.pool = make_pooler(pool, **dict(pool_kwargs or {}))
        self.metrics = list(metrics)
        self.n_items = n_items
        self.population = population
        self.params = {k: dict(v) for k, v in (params or {}).items()}
        self.labels: dict[str, Any] = {"model": model, "pooling": pooling, "corpus": corpus}
        self.view_metrics = list(view_metrics)
        self.augment = augment
        self.q = q
        self.view_spec = view_spec or (ViewSpec(source="objective", q=q, seed=seed) if augment is not None else None)
        self.seed = seed
        self.history: list[tuple[int, Records]] = []

    @torch.no_grad()
    def collect(
        self,
        forward: Callable[[Any], Any],
        loader: Iterable,
        augment: Callable[[Any, torch.Generator], Any] | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[int, Any]:
        """One pass over the loader with hooks: layer index -> pooled (N, D) tensor or list of (T_i, D) frames."""
        buffers: dict[int, list[Tensor]] = {i: [] for i in range(len(self.layer_modules))}

        def make_hook(i: int):
            def hook(module, inputs, output):
                buffers[i].append(self.pool(_first_tensor(output)).detach())

            return hook

        handles = [m.register_forward_hook(make_hook(i)) for i, m in enumerate(self.layer_modules)]
        try:
            seen = 0
            for batch in loader:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                if augment is not None:
                    x = augment(x, generator if generator is not None else torch.Generator().manual_seed(self.seed))
                forward(x)
                seen += int(buffers[0][-1].shape[0]) if buffers[0] else 0
                if seen >= self.n_items:
                    break
        finally:
            for h in handles:
                h.remove()
        layers: dict[int, Any] = {}
        for i, chunks in buffers.items():
            if not chunks:
                continue
            cat = torch.cat(chunks, dim=0)[: self.n_items]
            layers[i] = cat if self.population == "pooled" else [c for c in cat]  # frames: list of (T, D)
        return layers

    @torch.no_grad()
    def sweep(
        self,
        forward: Callable[[Any], Any],
        loader: Iterable,
        step: int,
        sinks: Sequence[Callable[[Records, int], None]] = (),
    ) -> Records:
        """Collect representations, compute metrics, record the step, and pass the records to the sinks.

        Args:
            forward: Runs the model on one input batch so the hooked modules fire; its return value is ignored.
            loader: Iterable of batches (tensor, or tuple whose first element is the input), fixed order.
            step: Training step or epoch written into every record's extras["step"].
            sinks: Callables receiving (records, step): csv_sink, json_sink, tensorboard_sink,
                wandb_sink, or any callable of your own.
        """
        layers = self.collect(forward, loader)
        rec = compute(
            layers,
            self.metrics,
            population=self.population,
            n=self.n_items,
            seed=self.seed,
            params=self.params,
            **self.labels,
        )
        if self.view_metrics and self.augment is not None:
            if self.population != "pooled":
                raise ValueError("view metrics need population='pooled'")
            passes = []
            for p in range(self.q):
                gen = torch.Generator().manual_seed(self.seed + p)
                passes.append(self.collect(forward, loader, augment=self.augment, generator=gen))
            view_layers = {l: torch.stack([passes[p][l] for p in range(self.q)], dim=0) for l in passes[0]}
            rec.extend(
                compute(
                    view_layers,
                    self.view_metrics,
                    n=self.n_items,
                    seed=self.seed,
                    params=self.params,
                    views=self.view_spec,
                    **self.labels,
                )
            )
        for r in rec:
            r.extras["step"] = step
        self.history.append((step, rec))
        for sink in sinks:
            sink(rec, step)
        return rec

    def profiles(self, metric: str) -> dict[int, list[tuple[int, float]]]:
        """step -> [(layer, value), ...] for one metric across all sweeps so far."""
        return {step: rec.profile(metric) for step, rec in self.history}


def csv_sink(path: str | Path) -> Callable[[Records, int], None]:
    """Append every sweep's rows to one CSV file (header written once)."""
    path = Path(path)

    def sink(rec: Records, step: int) -> None:
        new = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(
                    [
                        "step",
                        "layer",
                        "layer_b",
                        "metric",
                        "value",
                        "population",
                        "pooling",
                        "n_items",
                        "dim",
                        "n_views",
                        "params",
                        "views",
                        "extras",
                    ]
                )
            for r in rec:
                w.writerow(
                    [
                        step,
                        r.layer,
                        r.layer_b,
                        r.metric,
                        r.value,
                        r.population,
                        r.pooling,
                        r.n_items,
                        r.dim,
                        r.n_views,
                        json.dumps(r.params, default=float),
                        r.views,
                        json.dumps(r.extras, default=float),
                    ]
                )

    return sink


def json_sink(directory: str | Path) -> Callable[[Records, int], None]:
    """Write one JSON file per sweep: <directory>/step_<step>.json."""
    directory = Path(directory)

    def sink(rec: Records, step: int) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        rec.to_json(directory / f"step_{step}.json")

    return sink


def tensorboard_sink(log_dir: str | Path) -> Callable[[Records, int], None]:
    """Write layer_metrics/<metric>/layer_<l> scalars to TensorBoard event files (torch.utils.tensorboard)."""
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(str(log_dir))

    def sink(rec: Records, step: int) -> None:
        for r in rec:
            if r.layer_b is None and r.value == r.value:
                writer.add_scalar(
                    f"{metric_key_prefix(r)[0]}/{r.metric.replace('/', '_')}/layer_{r.layer}", r.value, step
                )
        writer.flush()

    return sink


def _profile_series(
    rec: Records, history: Sequence[tuple[int, Records]] | None, metric: str
) -> tuple[list, list, list]:
    """xs (layers), ys (one profile per sweep) and keys for a line-series plot; history, when given, adds earlier sweeps."""
    prof = rec.profile(metric)
    xs = [l for l, _ in prof]
    ys, keys = [], []
    for step, past in history or []:
        p = dict(past.profile(metric))
        if p and all(l in p for l in xs) and p != dict(prof):
            ys.append([p[l] for l in xs])
            keys.append(f"step {step}")
    ys.append([v for _, v in prof])
    keys.append(f"step {rec[0].extras.get('step', '')}")
    return xs, ys, keys


def wandb_sink(
    run=None,
    line_series: bool = True,
    step_metric: str = "monitor/step",
    history: Sequence[tuple[int, Records]] | None = None,
) -> Callable[[Records, int], None]:
    """Log to Weights & Biases: layer_metrics/<metric>_layer_<l> scalars and profiles/<metric> line series
    (online_metrics/ and online_profiles/ for training-batch records).

    The sweep step is logged as its own metric and declared the x-axis of every
    layer_metrics/* and profiles/* key with wandb.define_metric, so sweeps interleave
    with the run's other logging without passing an explicit `step=` (which W&B
    requires to be monotone across all log calls). Metric names have "/" replaced by
    "_" in keys. Uses the active run unless one is given. With history (a
    LayerMonitor's or OnlineBuffer's .history), every profile plot shows all sweeps so
    far, one line per step, so the depth profile's evolution is read off one chart.
    """
    state = {"defined": False}

    def sink(rec: Records, step: int) -> None:
        import wandb

        target = run if run is not None else wandb
        if not state["defined"]:
            target.define_metric(step_metric)
            for key in ("layer_metrics/*", "profiles/*", "online_metrics/*", "online_profiles/*"):
                target.define_metric(key, step_metric=step_metric)
            state["defined"] = True
        log: dict[str, Any] = {step_metric: step}
        for r in rec:
            if r.layer_b is None and r.value == r.value:  # skip nan
                log[f"{metric_key_prefix(r)[0]}/{r.metric.replace('/', '_')}_layer_{r.layer}"] = r.value
        if line_series and len(rec):
            profile_prefix = metric_key_prefix(rec[0])[1]
            for metric in sorted({r.metric for r in rec if r.layer_b is None}):
                xs, ys, keys = _profile_series(rec, history, metric)
                log[f"{profile_prefix}/{metric.replace('/', '_')}"] = wandb.plot.line_series(
                    xs=xs, ys=ys, keys=keys, title=metric, xname="layer"
                )
        target.log(log)

    return sink
