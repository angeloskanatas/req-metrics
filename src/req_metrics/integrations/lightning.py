"""PyTorch Lightning callback around LayerMonitor. Lightning is imported only here.

Given the metrics, and optionally the block list and the pooling, the callback
finds the blocks and builds a fixed monitoring subset of the training set. It
runs a sweep at the start of training and every n epochs and logs per-layer
scalars to the trainer's logger, with layer-profile line plots when the logger
is Weights & Biases. With online=True it also keeps ring
buffers of the training forward passes and logs the same point metrics on
them under online_metrics/, without extra forward passes. The per-layer scalars are
also written to trainer.callback_metrics, so ModelCheckpoint and EarlyStopping can
select by them. Under distributed training the sweeps run on global rank zero and
their scalars are broadcast to every rank.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import torch.nn as nn

from req_metrics.monitor import (
    LayerMonitor,
    OnlineBuffer,
    Pooler,
    layer_scalars,
    metric_key,
    metric_key_prefix,
    monitor_loader,
    resolve_layers,
)
from req_metrics.records import Record, Records, _upgrade
from req_metrics.view_construction import ViewSpec

_Base: Any
try:
    import lightning.pytorch as pl

    _Base = pl.Callback
except ImportError:  # pragma: no cover
    _Base = object


def _default_batch_input(batch: Any) -> Any:
    """Element 0 of a tuple or list batch, else the batch itself; dict batches need batch_input."""
    if isinstance(batch, Mapping):
        raise TypeError("the batch is a dict; pass batch_input= to select the model input")
    return batch[0] if isinstance(batch, (tuple, list)) else batch


class _Inputs:
    """Re-iterable view of the monitoring batches as 1-tuples of model inputs; view metrics read it q + 1 times."""

    def __init__(self, batches: Iterable[Any], batch_input: Callable[[Any], Any]):
        self.batches, self.batch_input = batches, batch_input

    def __iter__(self):
        return ((self.batch_input(b),) for b in self.batches)


class LayerMonitorCallback(_Base):
    """Sweep layer-wise metrics during a Lightning fit.

    Args:
        metrics: Registry names of point or trajectory metrics.
        layers: Block list to hook: attribute path from the LightningModule ("backbone.blocks"),
            a sequence of modules, or None to find the main block list of the called module automatically.
        pool: A readout name (see make_pooler) or a callable; pool_kwargs carries its options, e.g.
            {"grid": (F, T)} for the grid readouts of spectrogram-patch encoders.
        level: "sequence", "sample" or "population".
        n_items: Monitoring-set size, drawn once from the training dataset with seed.
        n_tokens: Tokens drawn at the population level; see LayerMonitor.
        every_n_epochs: Sweep period in epochs (None disables epoch sweeps); a sweep also runs at
            the start of training.
        model_attr: Attribute of the LightningModule to call on the input (e.g. "backbone");
            None calls the module itself.
        batch_input: Extracts the model input from a batch; default takes element 0 of a
            tuple or list and passes a tensor through.
        loader: Optional callable (trainer, pl_module) -> iterable replacing the automatic
            monitoring subset.
        view_metrics, augment, q, view_spec, views_in_train_mode: View metrics computed from q
            augmented passes with augment(batch, generator); see LayerMonitor. Only the monitored
            module and its layers change mode during a sweep, so an augmentation module of the
            LightningModule that acts in train mode only keeps acting.
        jacobian_items, jacobian_probes, jacobian_power_iters, jacobian_input, jacobian_forward:
            Jacobian effective rank of every layer's readout on the first jacobian_items
            monitoring inputs; see LayerMonitor. Off by default.
        drift_metrics, drift_reference: Pair metrics between every layer's reference state (the
            first sweep, or the previous one) and its current state on the monitoring items; see
            LayerMonitor. Logged under drift_metrics/. Off by default.
        params, model, pooling, corpus: Forwarded to LayerMonitor and recorded.
        online: Also hook the training forward passes into an OnlineBuffer and compute
            online_metrics (default: the same point metrics) on the most recent online_n_items
            rows at the end of every sweep epoch; the records are tagged
            extras["source"] = "training-batches" and logged under online_metrics/.
        online_metrics, online_n_items, online_device: See OnlineBuffer; defaults follow
            metrics, n_items and the output device.
        every_n_steps, sweep_steps: Step-based schedule in addition to the epoch one: sweep every
            n optimizer steps, or at the listed global steps (e.g. a log-spaced list such as
            (100, 300, 1000, 3000, 10000) to resolve the early phases of training). Every record
            carries extras["epoch"] and extras["global_step"].
        cache_batches: Materialize the monitoring subset once (CPU tensors) so later sweeps skip
            decoding and see the same inputs, also when the dataset's __getitem__ draws a random
            crop; the q view passes still read the live loader, so each view draws its own crop.
            A loader that is a one-shot iterator is always materialized, and its views share it.
        sinks: Extra callables receiving (records, global step): csv_sink, json_sink,
            tensorboard_sink, wandb_sink or your own.
        spectra: Leading singular values per layer kept at every sweep for the spectra/ charts of a
            W&B logger; see LayerMonitor. 0 disables.
        callback_metrics: Also write the per-layer scalars (the keys of layer_scalars) to
            trainer.callback_metrics on every rank, so ModelCheckpoint(monitor=
            "layer_metrics/<metric>_layer_<l>") and EarlyStopping select by them as by a validation
            loss; layer indices are zero-padded to the depth's width (layer_07 in a 12-block model).
            Between sweeps the last value stands, so align their cadence with the sweep schedule.
        log: Log per-layer scalars through trainer.logger (TensorBoard, CSV, MLflow, W&B, ...);
            adds layer-profile plots when the logger is W&B.
        log_extras: Numeric extras logged next to the values, e.g. ("frechet_var",).
        batch_size: Batch size of the automatic monitoring loader.
        seed: Subset and augmentation seed.
    """

    def __init__(
        self,
        metrics: Sequence[str],
        *,
        layers: str | Sequence[nn.Module] | None = None,
        pool: str | Pooler = "mean",
        level: str = "sequence",
        n_items: int = 5000,
        n_tokens: int | None = None,
        every_n_epochs: int | None = 1,
        model_attr: str | None = None,
        batch_input: Callable[[Any], Any] | None = None,
        loader: Callable[[Any, Any], Iterable] | None = None,
        view_metrics: Sequence[str] = (),
        augment: Callable | None = None,
        q: int = 10,
        view_spec: ViewSpec | None = None,
        views_in_train_mode: bool = False,
        jacobian_items: int = 0,
        jacobian_probes: int = 32,
        jacobian_power_iters: int = 5,
        jacobian_input: Callable[[Any], Any] | None = None,
        jacobian_forward: Callable[[Any], Any] | None = None,
        drift_metrics: Sequence[str] = (),
        drift_reference: str = "first",
        params=None,
        model: str | None = None,
        pooling: str | None = None,
        corpus: str | None = None,
        sinks: Sequence[Callable[[Records, int], None]] = (),
        log: bool = True,
        log_extras: Sequence[str] = (),
        batch_size: int = 32,
        seed: int = 0,
        online: bool = False,
        online_metrics: Sequence[str] | None = None,
        online_n_items: int | None = None,
        online_device=None,
        every_n_steps: int | None = None,
        sweep_steps: Sequence[int] | None = None,
        cache_batches: bool = False,
        pool_kwargs: Mapping[str, Any] | None = None,
        callback_metrics: bool = True,
        spectra: int = 0,
    ):
        super().__init__()
        self.spectra = spectra
        if view_metrics and augment is None:
            raise ValueError(
                "view metrics need augment; pass an identity callable when the view loader draws the views"
            )
        self.callback_metrics = callback_metrics
        self._pending_state: dict[str, Any] | None = None
        self.pool_kwargs = dict(pool_kwargs or {})
        self.every_n_steps, self.sweep_steps, self.cache_batches = every_n_steps, set(sweep_steps or ()), cache_batches
        self._done_steps: set[int] = set()
        self.online_enabled = online
        self.online_metrics = list(online_metrics) if online_metrics is not None else list(metrics)
        self.online_n_items = online_n_items if online_n_items is not None else n_items
        self.online_device = online_device
        self.online: OnlineBuffer | None = None
        self.metrics, self.layers, self.pool, self.level = list(metrics), layers, pool, level
        self.n_items, self.every_n_epochs, self.model_attr = n_items, every_n_epochs, model_attr
        self.n_tokens = n_tokens
        self.batch_input = batch_input or _default_batch_input
        self._loader_fn = loader
        self._loader: Iterable[Any] | None = None
        self._live: Iterable[Any] | None = None
        self.view_metrics, self.augment, self.q, self.view_spec = list(view_metrics), augment, q, view_spec
        self.views_in_train_mode = views_in_train_mode
        self.jacobian_items, self.jacobian_probes = jacobian_items, jacobian_probes
        self.jacobian_power_iters = jacobian_power_iters
        self.jacobian_input, self.jacobian_forward = jacobian_input, jacobian_forward
        self.drift_metrics, self.drift_reference = list(drift_metrics), drift_reference
        self.labels: dict[str, Any]
        self.params, self.labels = (
            params,
            {
                "model": model,
                "pooling": pooling if pooling is not None else (pool if isinstance(pool, str) else None),
                "corpus": corpus,
            },
        )
        self.sinks, self.log, self.batch_size, self.seed = list(sinks), log, batch_size, seed
        self.log_extras = tuple(log_extras)
        self.monitor: LayerMonitor | None = None

    def _target(self, pl_module):
        return getattr(pl_module, self.model_attr) if self.model_attr else pl_module

    def _ensure(self, trainer, pl_module) -> None:
        if self.monitor is None:
            root = (
                pl_module if isinstance(self.layers, str) else self._target(pl_module)
            )  # paths start at the LightningModule
            self.monitor = LayerMonitor(
                resolve_layers(root, self.layers),
                self.pool,
                self.metrics,
                pool_kwargs=self.pool_kwargs,
                n_items=self.n_items,
                n_tokens=self.n_tokens,
                level=self.level,
                params=self.params,
                view_metrics=self.view_metrics,
                augment=self.augment,
                q=self.q,
                view_spec=self.view_spec,
                views_in_train_mode=self.views_in_train_mode,
                jacobian_items=self.jacobian_items,
                jacobian_probes=self.jacobian_probes,
                jacobian_power_iters=self.jacobian_power_iters,
                jacobian_input=self.jacobian_input,
                jacobian_forward=self.jacobian_forward,
                drift_metrics=self.drift_metrics,
                drift_reference=self.drift_reference,
                spectra=self.spectra,
                seed=self.seed,
                **self.labels,
            )
            self._restore(self.monitor)
        if self._loader is None:
            if self._loader_fn is not None:
                batches: Iterable[Any] = self._loader_fn(trainer, pl_module)
            else:
                src = trainer.train_dataloader
                if callable(src):
                    src = src()
                dataset = getattr(src, "dataset", None)
                if dataset is None:
                    raise ValueError("cannot find the training dataset; pass loader=")
                try:
                    batches = monitor_loader(dataset, self.n_items, self.batch_size, self.seed)
                except TypeError:
                    raise ValueError("the training dataset has no length; pass loader= with a fixed subset") from None
            one_shot = iter(batches) is batches
            self._live = None if one_shot else batches
            if self.cache_batches or one_shot:  # a one-shot iterator is kept for later sweeps
                batches = [
                    tuple(t.cpu() if hasattr(t, "cpu") else t for t in b)
                    if isinstance(b, (tuple, list))
                    else (b.cpu() if hasattr(b, "cpu") else b)
                    for b in batches
                ]
            self._loader = batches

    def _logger_sink(self, trainer):
        """Per-layer scalars through every Lightning logger of the trainer; layer-profile plots on W&B,
        one line per sweep so far.

        Steps follow Lightning's own convention for W&B: no explicit `step=`, the global
        step travels as the "trainer/global_step" key, so the sweep never collides with
        the run's step counter.
        """

        def sink(rec: Records, epoch: int) -> None:
            loggers = list(getattr(trainer, "loggers", None) or ([trainer.logger] if trainer.logger else []))
            if not loggers or not len(rec):
                return
            scalars = {**layer_scalars(rec, self.log_extras), "epoch": int(trainer.current_epoch)}
            for logger in loggers:
                logger.log_metrics(scalars, step=trainer.global_step)
                exp = getattr(logger, "experiment", None)
                if type(logger).__name__ == "WandbLogger" and exp is not None:
                    import wandb

                    from req_metrics.monitor import _profile_series, _spectrum_plots

                    source = (
                        self.online
                        if (self.online is not None and rec[0].extras.get("source") == "training-batches")
                        else self.monitor
                    )
                    history = source.history if source is not None else None
                    plots: dict[str, Any] = {"trainer/global_step": trainer.global_step}
                    for metric in sorted({r.metric for r in rec if r.layer_b is None}):
                        first = rec.where(metric=metric)[0]
                        xs, ys, keys = _profile_series(rec, history, metric)
                        plots[f"{metric_key_prefix(first)[1]}/{metric_key(first)}"] = wandb.plot.line_series(
                            xs=xs, ys=ys, keys=keys, title=metric, xname="layer"
                        )
                    if self.monitor is not None and source is self.monitor and rec[0].extras.get("source") is None:
                        plots.update(_spectrum_plots(self.monitor.spectra, rec, wandb))
                    exp.log(plots)

        return sink

    def _sinks(self, trainer):
        return list(self.sinks) + ([self._logger_sink(trainer)] if self.log else [])

    def _stamp(self, trainer, rec: Records) -> None:
        for r in rec:
            r.extras["epoch"] = int(trainer.current_epoch)
            r.extras["global_step"] = int(trainer.global_step)

    def _publish(self, trainer, scalars: dict[str, float] | None) -> None:
        """Write a sweep's scalars to trainer.callback_metrics on every rank, where ModelCheckpoint and
        EarlyStopping read the quantity they monitor; rank zero's values are broadcast first."""
        if not self.callback_metrics:
            return
        strategy = getattr(trainer, "strategy", None)
        if strategy is not None and hasattr(strategy, "broadcast"):
            scalars = strategy.broadcast(scalars, src=0)
        metrics = getattr(trainer, "callback_metrics", None)
        if scalars and metrics is not None:
            import torch

            device = getattr(strategy, "root_device", None)
            metrics.update({k: torch.tensor(v, device=device) for k, v in scalars.items()})

    def _online_compute(self, trainer) -> None:
        if not self.online_enabled:
            return
        scalars = None
        if self.online is not None and getattr(trainer, "is_global_zero", True):
            step = int(trainer.global_step)
            rec = self.online.compute(self.online_metrics, step, (), params=self.params, seed=self.seed, **self.labels)
            self._stamp(trainer, rec)
            for sink in self._sinks(trainer):
                sink(rec, step)
            scalars = layer_scalars(rec, self.log_extras)
        self._publish(trainer, scalars)

    def _sweep(self, trainer, pl_module) -> None:
        scalars = None
        if getattr(trainer, "is_global_zero", True):
            self._ensure(trainer, pl_module)
            assert self.monitor is not None and self._loader is not None
            target = self._target(pl_module)  # the monitor switches only this module and the hooked layers
            device = next(pl_module.parameters()).device
            if device.type == "cuda":
                import torch

                torch.cuda.empty_cache()
            forward = lambda x: target(x.to(device) if hasattr(x, "to") else x)  # noqa: E731
            views = _Inputs(self._live, self.batch_input) if self._live is not None else None
            step = int(trainer.global_step)
            rec = self.monitor.sweep(
                forward, _Inputs(self._loader, self.batch_input), step, (), module=target, view_loader=views
            )
            self._stamp(trainer, rec)
            for sink in self._sinks(trainer):
                sink(rec, step)
            scalars = layer_scalars(rec, self.log_extras)
        self._publish(trainer, scalars)

    def on_train_start(self, trainer, pl_module) -> None:
        if self.online_enabled and self.online is None and getattr(trainer, "is_global_zero", True):
            self._ensure(trainer, pl_module)
            assert self.monitor is not None
            self.online = OnlineBuffer(
                self.monitor.layer_modules,
                self.pool,
                self.online_n_items,
                self.online_device,
                pool_kwargs=self.pool_kwargs,
            )
            self.online.attach()
        if trainer.current_epoch == 0:
            self._done_steps.add(int(trainer.global_step))
            self._sweep(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch + 1
        step = int(trainer.global_step)
        if self.every_n_epochs and epoch % self.every_n_epochs == 0 and step not in self._done_steps:
            self._done_steps.add(step)  # a step trigger on the same global step sweeps once
            self._online_compute(trainer)
            self._sweep(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = int(trainer.global_step)
        due = step in self.sweep_steps or (
            self.every_n_steps is not None and step > 0 and step % self.every_n_steps == 0
        )
        if due and step not in self._done_steps:
            self._done_steps.add(step)
            self._online_compute(trainer)
            self._sweep(trainer, pl_module)

    def on_train_end(self, trainer, pl_module) -> None:
        if self.online is not None:
            self.online.detach()

    @property
    def state_key(self) -> str:
        """Distinct per monitored module, so several callbacks can keep their state in one checkpoint."""
        generate = getattr(self, "_generate_state_key", None)
        return generate(model_attr=self.model_attr) if generate is not None else type(self).__name__

    def state_dict(self) -> dict[str, Any]:
        """The sweep history, the schedule bookkeeping and the drift reference, so a resumed run continues the
        curves and keeps comparing with its first sweep. The reference is one float32 copy of the monitoring
        set per hooked layer; the online buffer is not saved and refills."""
        state: dict[str, Any] = {"done_steps": sorted(self._done_steps)}
        if self.monitor is not None:
            state["history"] = [(step, [r.to_dict() for r in rec]) for step, rec in self.monitor.history]
            state["reference"] = self.monitor._reference
            state["reference_step"] = self.monitor._reference_step
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._done_steps = set(state_dict.get("done_steps", ()))
        self._pending_state = state_dict
        if self.monitor is not None:
            self._restore(self.monitor)

    def _restore(self, monitor: LayerMonitor) -> None:
        state, self._pending_state = self._pending_state, None
        if not state:
            return
        monitor.history = [
            (step, Records(Record(**_upgrade(r)) for r in rows)) for step, rows in state.get("history", [])
        ]
        monitor._reference = state.get("reference")
        monitor._reference_step = state.get("reference_step")
