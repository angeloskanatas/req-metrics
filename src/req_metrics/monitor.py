"""Layer-wise monitoring during training, framework-independent.

Two modes. LayerMonitor registers forward hooks on the given layer modules,
runs a fixed monitoring set through a forward callable in eval mode, pools each
layer's output, and hands the per-layer tensors to compute(); the same records
come out as in a post-hoc run, so curves logged during training are directly
comparable with curves computed on stored embeddings. OnlineBuffer hooks the
same modules during the ordinary training forward passes and keeps the most
recent rows, without extra forward passes, as a collapse indicator on the
training-time representation. Which modules are layers, how a block output becomes one
vector per sample, and how an augmented view is drawn are arguments.
"""

from __future__ import annotations

import contextlib
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
    final token, for causal decoders), "tokens" (no pooling; the (B, T, D) tokens after
    the first `n_prefix`, for the sample and population levels). Grid readouts, for
    spectrogram-patch encoders whose L tokens are an (F, T) grid given as `grid`: "gap"
    (mean over patches), "freq_concat_mean" (frequency patches concatenated, then the
    mean over time, (B, F * D); the readout of MSM-MAE and M2D), "partitioned" (block
    means over freq_chunks x time_chunks regions, Gu et al., 2026), "freq_concat" (the
    (B, T, F * D) trajectory for the sample level) and "freq_mean" (the (B, T, D)
    trajectory). The grid readouts are the batched form of the functions in `layouts`.

    Args:
        pool: Readout name, or a callable mapping one block output to (B, D) or (B, T, D).
        grid: (F, T) patch counts of the grid readouts; the first `n_prefix` tokens are dropped.
        time_axis: 1 when tokens are ordered frequency-major (index f * T + t), 0 when
            time-major (index t * F + f).
        n_prefix: Leading class or register tokens to drop before "tokens" or a grid readout.
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
    if pool == "tokens":
        return lambda out: out[:, n_prefix:] if out.ndim == 3 else out
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


def metric_key(record: Record) -> str:
    """The logging name of a record's metric: "/" replaced by "_", and the level appended unless "sequence"."""
    name = record.metric.replace("/", "_")
    return name if record.level in (None, "sequence") else f"{name}_{record.level}"


def layer_scalars(rec: Records, extras: Sequence[str] = ()) -> dict[str, float]:
    """Logging scalars of the single-layer records: <prefix>/<metric>_layer_<l>, and
    <prefix>/<metric>_<extra>_layer_<l> for each named numeric extra. Non-finite values are skipped."""
    out: dict[str, float] = {}
    for r in rec:
        if r.layer_b is not None:
            continue
        stem = f"{metric_key_prefix(r)[0]}/{metric_key(r)}"
        if r.value == r.value:
            out[f"{stem}_layer_{r.layer}"] = r.value
        for e in extras:
            v = r.extras.get(e)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v:
                out[f"{stem}_{e}_layer_{r.layer}"] = float(v)
    return out


class OnlineBuffer:
    """Ring buffers of pooled block outputs captured from the training forward passes.

    Monitoring without extra forward passes: hooks copy the pooled output of every train-mode
    forward into a per-layer buffer of the most recent n_items rows, and compute() runs point
    metrics on them. This measures the training-time representation (augmented inputs,
    dropout, masking, moving weights), so use it as a collapse indicator and LayerMonitor for
    comparable curves. Records carry extras["source"] = "training-batches". Every train-mode
    forward of the hooked modules enters the buffer, so an encoder run on two inputs per step
    (e.g. context and target) mixes both; eval-mode forwards are ignored; gradient checkpointing
    duplicates rows; buffers are read at the smallest filled count, which stochastic depth lowers.

    Args:
        layer_modules: Modules whose outputs are the layers, in depth order.
        pool: A readout name (see make_pooler) or a callable mapping one block output to (B, D).
        pool_kwargs: Keyword arguments of make_pooler for a named readout.
        n_items: Rows kept per layer.
        device: Buffer device; default the device of the hooked output. Stored as float32.
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
            level="sequence",
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


@contextlib.contextmanager
def _forward_mode_attention():
    """Math attention kernels for the duration: the fused kernels and the MHA fast path have no forward-mode derivative."""
    mha = getattr(torch.backends, "mha", None)
    fast = mha.get_fastpath_enabled() if mha is not None else None
    if mha is not None:
        mha.set_fastpath_enabled(False)
    try:
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            kernels = sdpa_kernel(SDPBackend.MATH)
        except ImportError:  # torch < 2.3
            kernels = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
        with kernels:
            yield
    finally:
        if mha is not None and fast is not None:
            mha.set_fastpath_enabled(fast)


def jacobian_products(
    forward: Callable[[Tensor], Any],
    x: Tensor,
    modules: Sequence[nn.Module],
    pool: Pooler,
    *,
    num_probes: int = 32,
    power_iters: int = 5,
    seed: int = 0,
) -> dict[int, Tensor]:
    """Randomized range-finder sketches of the Jacobian of every module's readout.

    Chung and Kim (2026, Sec. 4.1 and App. F) estimate the leading singular values of the
    Jacobian J of a readout with respect to the input by randomized range finding (Halko et
    al., 2011): Y = J Omega for k orthonormal input directions Omega per input (QR of a seeded
    Gaussian draw), power_iters rounds of subspace iteration, Q an orthonormal basis of Y, and
    B = Q^T J, whose singular values estimate the k largest of J. The paper uses k = 32 and 5
    rounds. The first products Y come from one forward-mode pass per direction for all modules;
    each round and the final B take k forward-mode (J v) or reverse-mode (J^T u) passes per
    module. Inputs in a batch must not interact (eval-mode normalization).

    Args:
        forward: Runs the model on x so the modules fire; derivatives are taken with respect to x.
        x: Inputs, (B, ...).
        modules: Modules whose outputs are read out, in depth order.
        pool: Readout applied to each module output, as in the sweep.
        num_probes: Directions k per input, capped at the input size.
        power_iters: Rounds of subspace iteration.
        seed: Probe seed.

    Returns:
        Module index -> (B, k, M) rows of B = Q^T J, M the input size; k is capped at the readout size.
    """
    from torch.func import jvp, vjp

    b, size = x.shape[0], x[0].numel()
    k = min(num_probes, size)
    gen = torch.Generator().manual_seed(seed)
    omega = torch.stack(
        [torch.linalg.qr(torch.randn(size, k, generator=gen, dtype=torch.float64))[0].T for _ in range(b)]
    )

    def readouts(inp: Tensor) -> tuple[Tensor, ...]:
        out: dict[int, Tensor] = {}
        handles = [
            m.register_forward_hook(lambda _m, _i, o, i=i: out.__setitem__(i, pool(_first_tensor(o))))
            for i, m in enumerate(modules)
        ]
        try:
            forward(inp)
        finally:
            for h in handles:
                h.remove()
        missing = [i for i in range(len(modules)) if i not in out]
        if missing:
            raise RuntimeError(f"modules {missing} did not run in the forward pass")
        return tuple(out[i].reshape(b, -1) for i in range(len(modules)))

    def orth(rows: Tensor) -> Tensor:  # (B, r, n) -> orthonormal rows spanning the same space
        return torch.linalg.qr(rows.double().transpose(1, 2))[0].transpose(1, 2)

    def apply_j(v: Tensor) -> tuple[Tensor, ...]:  # (B, r, M) -> per module (B, r, D_i)
        cols = [jvp(readouts, (x,), (v[:, j].to(x.dtype).reshape(x.shape),))[1] for j in range(v.shape[1])]
        return tuple(torch.stack([c[i] for c in cols], dim=1) for i in range(len(modules)))

    out: dict[int, Tensor] = {}
    with _forward_mode_attention():
        linearized = vjp(readouts, x)  # (outputs, pullback); indexed, since vjp may also return aux
        outputs, pullback = linearized[0], linearized[1]

        def apply_jt(i: int, u: Tensor) -> Tensor:  # (B, r, D_i) -> (B, r, M) for module i
            rows = []
            for j in range(u.shape[1]):
                cot = tuple(u[:, j].to(o.dtype) if m == i else torch.zeros_like(o) for m, o in enumerate(outputs))
                rows.append(pullback(cot)[0].reshape(b, -1))
            return torch.stack(rows, dim=1)

        y = apply_j(omega)
        for i in range(len(modules)):
            q = orth(y[i])
            for _ in range(power_iters):
                q = orth(apply_j(orth(apply_jt(i, q)))[i])
            out[i] = apply_jt(i, q).detach()
    return out


class LayerMonitor:
    """Collect per-layer representations of a fixed monitoring set and compute metrics on them.

    Args:
        layer_modules: Modules whose forward outputs are the layers, in depth order
            (e.g. the blocks of a transformer or the stages of a CNN).
        pool: A readout name (see make_pooler: "cls", "mean", "max", "last", "tokens", and the
            grid readouts "gap", "freq_concat_mean", "partitioned", "freq_concat", "freq_mean" with
            pool_kwargs such as grid=(F, T)), or a callable mapping one block output to a
            tensor: (B, D) one vector per sample at the sequence level, or (B, T, D) tokens at
            the sample and population levels. The pooling must be parameter-free.
        pool_kwargs: Keyword arguments of make_pooler for a named readout.
        metrics: Registry names of point or trajectory metrics; all of one input kind.
        n_items: Samples of the monitoring set to use; the loader is consumed until reached. The
            sample and population levels hold every sample's tokens (on the CPU), so a few hundred
            samples is the usual scale there. The q view passes are also held on the CPU; metrics
            run on the device of the hooked layers, one layer at a time.
        n_tokens: At the population level, the tokens drawn for the point metrics; None uses all
            tokens of the n_items samples.
        level: "sequence", "sample" or "population", matching what pool returns; the sample and
            population levels need samples of equal length within a sweep (batches are concatenated).
        params: Metric name -> estimator keyword arguments.
        model, pooling, corpus: Labels written into every record.
        view_metrics: Registry names of view metrics to compute when `augment` is given.
        augment: augment(batch, generator) -> batch; applied on q extra passes over the
            loader to build (q, N, D) views per layer. Keep the loader order fixed across passes.
            The views should follow the objective's positive construction (Thilak et al., 2024):
            when positives differ by random crops, pass a view_loader to sweep() that draws them.
            augment runs in the mode its own modules are in; sweep() switches only the model and
            the hooked layers.
        q: Views per sample for the view metrics.
        view_spec: Description of the view construction, recorded with the view records.
        views_in_train_mode: Run the q augmented passes with the model in train mode, for
            objectives whose positives come from stochasticity inside the model, such as token
            masking; the views then follow the objective's own positive construction (Thilak et
            al., 2024). Set dropout to zero, or it adds to the views. Buffers such as BatchNorm
            running statistics are restored after these passes.
        jacobian_items: Inputs from the start of the monitoring set on which the Jacobian
            effective rank of every layer's readout is computed; 0 disables it. Chung and Kim
            (2026) use 100 inputs.
        jacobian_probes: Random orthonormal input directions per input (32 in Chung and Kim, 2026).
        jacobian_power_iters: Rounds of subspace iteration (5 in Chung and Kim, 2026); each
            costs 2 * jacobian_probes passes per layer (see jacobian_products).
        jacobian_input: Maps a batch input to the tensor the Jacobian is taken with respect to,
            without differentiation (e.g. waveform to spectrogram); default the input itself.
        jacobian_forward: Runs the model on that tensor; default the sweep's forward.
        seed: Subsampling, augmentation and probe seed.
    """

    def __init__(
        self,
        layer_modules: Sequence[nn.Module],
        pool: str | Pooler,
        metrics: Sequence[str],
        *,
        pool_kwargs: Mapping[str, Any] | None = None,
        n_items: int = 10000,
        n_tokens: int | None = None,
        level: str = "sequence",
        params: Mapping[str, Mapping[str, Any]] | None = None,
        model: str | None = None,
        pooling: str | None = None,
        corpus: str | None = None,
        view_metrics: Sequence[str] = (),
        augment: Callable[[Any, torch.Generator], Any] | None = None,
        q: int = 10,
        view_spec: ViewSpec | None = None,
        views_in_train_mode: bool = False,
        jacobian_items: int = 0,
        jacobian_probes: int = 32,
        jacobian_power_iters: int = 5,
        jacobian_input: Callable[[Tensor], Tensor] | None = None,
        jacobian_forward: Callable[[Tensor], Any] | None = None,
        seed: int = 0,
    ):
        self.layer_modules = list(layer_modules)
        self.pool = make_pooler(pool, **dict(pool_kwargs or {}))
        self.metrics = list(metrics)
        self.n_items = n_items
        self.n_tokens = n_tokens
        self.level = level
        self.params = {k: dict(v) for k, v in (params or {}).items()}
        self.labels: dict[str, Any] = {"model": model, "pooling": pooling, "corpus": corpus}
        self.view_metrics = list(view_metrics)
        self.augment = augment
        self.q = q
        self.view_spec = view_spec or (ViewSpec(source="objective", q=q, seed=seed) if augment is not None else None)
        self.views_in_train_mode = views_in_train_mode
        self.jacobian_items, self.jacobian_probes = jacobian_items, jacobian_probes
        self.jacobian_power_iters = jacobian_power_iters
        self.jacobian_input, self.jacobian_forward = jacobian_input, jacobian_forward
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
        """One pass over the loader with hooks: layer index -> (N, D) tensor or list of (T_i, D) token tensors.

        Token and augmented passes are kept on the CPU; clean sequence-level passes on the layers' device.
        """
        buffers: dict[int, list[Tensor]] = {i: [] for i in range(len(self.layer_modules))}
        on_device = self.level == "sequence" and augment is None

        def make_hook(i: int):
            def hook(module, inputs, output):
                z = self.pool(_first_tensor(output)).detach()
                buffers[i].append(z if on_device else z.cpu())

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
            layers[i] = cat if self.level == "sequence" else [c for c in cat]  # tokens: list of (T, D)
        return layers

    @torch.no_grad()
    def sweep(
        self,
        forward: Callable[[Any], Any],
        loader: Iterable,
        step: int,
        sinks: Sequence[Callable[[Records, int], None]] = (),
        module: nn.Module | None = None,
        view_loader: Iterable | None = None,
    ) -> Records:
        """Collect representations, compute metrics, record the step, and pass the records to the sinks.

        Args:
            forward: Runs the model on one input batch so the hooked modules fire; its return value is ignored.
            loader: Iterable of batches (tensor, or tuple whose first element is the input), fixed order.
            step: Training step or epoch written into every record's extras["step"].
            sinks: Callables receiving (records, step): csv_sink, json_sink, tensorboard_sink,
                wandb_sink, or any callable of your own.
            module: The model forward runs, when forward is not itself a module.
            view_loader: Iterable read by the q augmented passes instead of loader: the same samples
                in the same order, e.g. a DataLoader whose dataset draws a random crop per item
                while loader is a cached list of one draw.

        The model (module, or forward when it is a module) and the hooked layers run in eval
        mode, except the augmented passes with views_in_train_mode; every submodule's previous
        mode is restored afterwards.
        """
        views = bool(self.view_metrics) and self.augment is not None
        if ((views and view_loader is None) or self.jacobian_items > 0) and iter(loader) is loader:
            raise ValueError("view metrics and the Jacobian read the loader again; pass a list or a DataLoader")
        if views and view_loader is not None and self.q > 1 and iter(view_loader) is view_loader:
            raise ValueError("the q view passes read view_loader q times; pass a list or a DataLoader")
        model = module if module is not None else (forward if isinstance(forward, nn.Module) else None)
        roots = ([model] if model is not None else []) + list(self.layer_modules)
        modes = [(m, m.training) for root in roots for m in root.modules()]
        for root in roots:
            root.eval()
        try:
            return self._sweep(forward, loader, step, sinks, roots, view_loader)
        finally:
            for m, was_training in modes:
                m.training = was_training

    def _sweep(
        self,
        forward: Callable[[Any], Any],
        loader: Iterable,
        step: int,
        sinks: Sequence[Callable[[Records, int], None]],
        roots: Sequence[nn.Module],
        view_loader: Iterable | None = None,
    ) -> Records:
        layers = self.collect(forward, loader)
        device = self._device()
        rec = compute(
            layers,
            self.metrics,
            level=self.level,
            n=self.n_tokens if self.level == "population" else self.n_items,
            seed=self.seed,
            params=self.params,
            device=device,
            **self.labels,
        )
        if self.view_metrics and self.augment is not None:
            if self.level != "sequence":
                raise ValueError("view metrics need level='sequence'")
            passes = []
            saved = (
                {id(b): (b, b.clone()) for root in roots for b in root.buffers()} if self.views_in_train_mode else {}
            )
            for root in roots if self.views_in_train_mode else ():
                root.train()
            try:
                for p in range(self.q):
                    gen = torch.Generator().manual_seed(self.seed + p)
                    source = view_loader if view_loader is not None else loader
                    passes.append(self.collect(forward, source, augment=self.augment, generator=gen))
            finally:
                for root in roots:
                    root.eval()
                for buf, value in saved.values():
                    buf.copy_(value)
            view_layers = {l: torch.stack([passes[p][l] for p in range(self.q)], dim=0) for l in passes[0]}
            rec.extend(
                compute(
                    view_layers,
                    self.view_metrics,
                    n=self.n_items,
                    seed=self.seed,
                    params=self.params,
                    views=self.view_spec,
                    device=device,
                    **self.labels,
                )
            )
        if self.jacobian_items > 0:
            rec.extend(self._jacobian(forward, loader))
        for r in rec:
            r.extras["step"] = step
        self.history.append((step, rec))
        for sink in sinks:
            sink(rec, step)
        return rec

    def _device(self) -> torch.device | None:
        """The device of the hooked layers' parameters, or None for parameter-free layers."""
        return next((p.device for m in self.layer_modules for p in m.parameters()), None)

    def _jacobian(self, forward: Callable[[Any], Any], loader: Iterable) -> Records:
        chunks, seen = [], 0
        for batch in loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            chunks.append(x)
            seen += int(x.shape[0])
            if seen >= self.jacobian_items:
                break
        x = torch.cat(chunks)[: self.jacobian_items]
        x = x.to(self._device() or x.device)
        if self.jacobian_input is not None:
            x = self.jacobian_input(x)
        products = jacobian_products(
            self.jacobian_forward or forward, x, self.layer_modules, self.pool, num_probes=self.jacobian_probes,
            power_iters=self.jacobian_power_iters, seed=self.seed,
        )  # fmt: skip
        return compute(products, ["jacobian_effective_rank"], level=self.level, seed=self.seed, **self.labels)

    def profiles(self, metric: str) -> dict[int, list[tuple[int, float]]]:
        """step -> [(layer, value), ...] for one metric across all sweeps so far."""
        return {step: rec.profile(metric) for step, rec in self.history}


def _x_step(rec: Records, step: int) -> int:
    """The x value of a sweep: its global step when the records carry one (the Lightning callback), else step."""
    return int(rec[0].extras.get("global_step", step)) if len(rec) else step


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
                        "level",
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
                        r.level,
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
    """Write one JSON file per sweep: <directory>/step_<step>.json.

    Records stamped with the epoch and the global step (as the Lightning callback
    does) are written to epoch_<e>_step_<s>.json, so epoch and step schedules never
    share a file name; training-batch records get the prefix online_, and records at the
    sample or population level the level name.
    """
    directory = Path(directory)

    def sink(rec: Records, step: int) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        ex = rec[0].extras if len(rec) else {}
        prefix = "online_" if ex.get("source") == "training-batches" else ""
        if len(rec) and rec[0].level not in (None, "sequence"):
            prefix += f"{rec[0].level}_"
        if "epoch" in ex and "global_step" in ex:
            name = f"{prefix}epoch_{ex['epoch']}_step_{ex['global_step']}.json"
        else:
            name = f"{prefix}step_{step}.json"
        rec.to_json(directory / name)

    return sink


def tensorboard_sink(log_dir: str | Path, extras: Sequence[str] = ()) -> Callable[[Records, int], None]:
    """Write the layer_scalars of every sweep to TensorBoard event files (torch.utils.tensorboard).

    The x value is the global step when the records carry one, else the step argument.
    extras names numeric extras logged next to the values, e.g. ("frechet_var",).
    """
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(str(log_dir))

    def sink(rec: Records, step: int) -> None:
        for key, value in layer_scalars(rec, extras).items():
            writer.add_scalar(key, value, _x_step(rec, step))
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
            keys.append(f"step {past[0].extras.get('global_step', step)}")
    ys.append([v for _, v in prof])
    keys.append(f"step {rec[0].extras.get('global_step', rec[0].extras.get('step', ''))}")
    return xs, ys, keys


def wandb_sink(
    run=None,
    line_series: bool = True,
    step_metric: str = "monitor/step",
    history: Sequence[tuple[int, Records]] | None = None,
    extras: Sequence[str] = (),
) -> Callable[[Records, int], None]:
    """Log to Weights & Biases: layer_metrics/<metric>_layer_<l> scalars and profiles/<metric> line series
    (online_metrics/ and online_profiles/ for training-batch records).

    The sweep step is logged as its own metric and declared the x-axis of every
    layer_metrics/* and profiles/* key with wandb.define_metric, so sweeps interleave
    with the run's other logging without passing an explicit `step=` (which W&B
    requires to be monotone across all log calls); its value is the global step when
    the records carry one, so epoch and step schedules share one axis. Metric names have "/" replaced by
    "_" in keys. Uses the active run unless one is given. With history (a
    LayerMonitor's or OnlineBuffer's .history), every profile plot shows all sweeps so
    far, one line per step, so the depth profile's evolution is read off one chart.
    extras names numeric extras logged next to the values, e.g. ("frechet_var",).
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
        log: dict[str, Any] = {step_metric: _x_step(rec, step), **layer_scalars(rec, extras)}
        if line_series and len(rec):
            profile_prefix = metric_key_prefix(rec[0])[1]
            for metric in sorted({r.metric for r in rec if r.layer_b is None}):
                xs, ys, keys = _profile_series(rec, history, metric)
                log[f"{profile_prefix}/{metric_key(rec.where(metric=metric)[0])}"] = wandb.plot.line_series(
                    xs=xs, ys=ys, keys=keys, title=metric, xname="layer"
                )
        target.log(log)

    return sink
