"""LayerMonitor on a toy stack of blocks: sequence and sample levels, views, sinks, Lightning adapter."""

import json
import tempfile
import unittest
import warnings
from pathlib import Path

import torch
import torch.nn as nn

import req_metrics as rq


class Toy(nn.Module):
    """Three residual blocks over (B, T, D) token sequences."""

    def __init__(self, d=16, t=12):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(d, d), nn.GELU()) for _ in range(3)])
        self.t = t

    def forward(self, x):  # x: (B, D) waveform stand-in -> (B, T, D) tokens
        h = x.unsqueeze(1).repeat(1, self.t, 1) + 0.1 * torch.arange(self.t).view(1, -1, 1)
        for b in self.blocks:
            h = h + b(h)
        return h


class Clips(nn.Module):
    """Blocks over frames: a (B, T, D) clip is flattened to B * T rows before the blocks, as video and
    world-model trainers do."""

    def __init__(self, d=16):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(d, d) for _ in range(2)])

    def forward(self, x):  # (B, T, D) -> (B * T, D)
        h = x.reshape(-1, x.shape[-1])
        for b in self.blocks:
            h = h + torch.tanh(b(h))
        return h


def loader(n=120, d=16, batch=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    data = torch.randn(n, d, generator=g)
    return [(data[i : i + batch], torch.zeros(batch)) for i in range(0, n, batch)]


class MonitorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = Toy().eval()

    def test_pooled_sweep_matches_compute(self):
        mon = rq.LayerMonitor(
            self.model.blocks,
            pool=lambda out: out.mean(dim=1),
            metrics=["effective_rank", "anisotropy/spectral"],
            n_items=100,
            model="toy",
            pooling="time-mean",
        )
        rec = mon.sweep(self.model, loader(), step=0)
        self.assertEqual(len(rec), 6)
        self.assertEqual(rec[0].extras["step"], 0)
        self.assertEqual(rec[0].n_items, 100)
        layers = mon.collect(self.model, loader())
        direct = rq.compute(layers, ["effective_rank"], n_items=100)
        self.assertAlmostEqual(
            rec.where(metric="effective_rank", layer=2)[0].value, direct.where(layer=2)[0].value, places=9
        )
        mon.sweep(self.model, loader(), step=1)
        self.assertEqual(sorted(mon.profiles("effective_rank")), [0, 1])

    def test_sample_level_and_trajectory_metric(self):
        mon = rq.LayerMonitor(
            self.model.blocks, pool=lambda out: out, metrics=["trajectory_curvature"], n_items=60, level="sample"
        )
        rec = mon.sweep(self.model, loader(), step=3)
        self.assertEqual(rec[0].level, "sample")
        self.assertEqual(rec[0].extras["n_items"], 60)

    def test_views_from_augment_callable(self):
        aug = lambda x, gen: x + 0.3 * torch.randn(x.shape, generator=gen)
        mon = rq.LayerMonitor(
            self.model.blocks,
            pool=lambda out: out.mean(dim=1),
            metrics=["effective_rank"],
            n_items=100,
            view_metrics=["lidar", "infonce"],
            augment=aug,
            q=2,
            view_spec=rq.ViewSpec(source="objective", augmentations=("gaussian noise 0.3",), q=2),
        )
        rec = mon.sweep(self.model, loader(), step=0)
        self.assertEqual(len(rec.where(metric="lidar")), 3)
        self.assertEqual(rec.where(metric="lidar")[0].n_views, 2)
        self.assertIn("gaussian noise", rec.where(metric="infonce")[0].views)

    def test_sinks(self):
        mon = rq.LayerMonitor(
            self.model.blocks, pool=lambda out: out.mean(dim=1), metrics=["effective_rank"], n_items=80
        )
        with tempfile.TemporaryDirectory() as d:
            calls = []
            defined = []
            fake = type(
                "W",
                (),
                {
                    "log": lambda self, log: calls.append((log["monitor/step"], sorted(log))),
                    "define_metric": lambda self, name, step_metric=None: defined.append((name, step_metric)),
                },
            )()
            import sys
            import types

            stub = types.ModuleType("wandb")
            stub.plot = types.SimpleNamespace(plot_table=lambda **kw: kw)
            stub.Table = lambda data, columns: {"data": data, "columns": columns}
            sys.modules["wandb"] = stub
            try:
                mon.sweep(
                    self.model,
                    loader(),
                    step=5,
                    sinks=[rq.csv_sink(Path(d) / "m.csv"), rq.json_sink(Path(d) / "j"), rq.wandb_sink(fake)],
                )
                mon.sweep(self.model, loader(), step=6, sinks=[rq.csv_sink(Path(d) / "m.csv")])
            finally:
                del sys.modules["wandb"]
            lines = (Path(d) / "m.csv").read_text().strip().splitlines()
            self.assertEqual(len(lines), 1 + 3 + 3)
            self.assertTrue((Path(d) / "j" / "step_5.json").exists())
            self.assertEqual(json.loads((Path(d) / "j" / "step_5.json").read_text())[0]["metric"], "effective_rank")
            self.assertEqual(calls[0][0], 5)
            self.assertEqual(
                defined[:3],
                [("monitor/step", None), ("layer_metrics/*", "monitor/step"), ("profiles/*", "monitor/step")],
            )
            self.assertEqual(len(defined), 8)  # the step metric and seven key families
            self.assertIn("layer_metrics/effective_rank_layer_0", calls[0][1])
            self.assertIn("profiles/effective_rank", calls[0][1])

    def test_tensorboard_sink(self):
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        except ImportError:
            self.skipTest("tensorboard not installed")
        mon = rq.LayerMonitor(self.model.blocks, pool="mean", metrics=["effective_rank"], n_items=80)
        with tempfile.TemporaryDirectory() as d:
            mon.sweep(self.model, loader(), step=2, sinks=[rq.tensorboard_sink(d)])
            self.assertTrue(any(f.startswith("events.out.tfevents") for f in Path(d).iterdir() for f in [f.name]))

    def test_online_buffer_captures_train_mode_only_and_wraps(self):
        buf = rq.OnlineBuffer(self.model.blocks, pool="mean", n_items=50)
        buf.attach()
        self.model.train()
        for x, _ in loader(n=120, batch=40):  # 3 batches of 40 -> ring of 50 wraps
            self.model(x)
        self.model.eval()
        self.model(torch.randn(40, 16))  # eval-mode forward must be ignored
        buf.detach()
        self.assertEqual(buf.seen[0], 120)
        self.assertEqual(buf.layers()[2].shape, (50, 16))
        self.assertEqual(buf.pointer[0], 120 % 50)
        self.model.train()
        self.model(torch.randn(40, 16))
        self.assertEqual(buf.seen[0], 120)  # detached: no capture
        rec = buf.compute(["effective_rank"], step=7, pooling="mean")
        self.assertEqual(len(rec), 3)
        self.assertEqual(rec[0].n_items, 50)
        self.assertEqual(rec[0].extras["source"], "training-batches")
        self.assertEqual(rec[0].extras["step"], 7)
        self.assertEqual(rq.metric_key_prefix(rec[0]), ("online_metrics", "online_profiles"))
        self.assertEqual(buf.history[0][0], 7)

    def test_online_buffer_reads_at_common_count_when_a_block_is_skipped(self):
        """Stochastic depth: a block that fires less often must not shorten the others' index range."""
        blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        buf = rq.OnlineBuffer(list(blocks), pool=lambda out: out, n_items=50)
        buf.attach()
        try:
            blocks.train()
            for step in range(6):
                x = torch.randn(8, 4)
                blocks[0](x)
                if step % 3:  # block 1 skipped every third step
                    blocks[1](x)
        finally:
            buf.detach()
        layers = buf.layers()
        self.assertEqual({tuple(v.shape) for v in layers.values()}, {(32, 4)})
        rec = buf.compute(["effective_rank"], step=1)
        self.assertEqual({r.n_items for r in rec}, {32})

    def test_sweep_runs_in_eval_mode_and_restores_modes(self):
        model = nn.Sequential(nn.Linear(16, 16), nn.Dropout(0.5), nn.Linear(16, 16), nn.Dropout(0.5))
        model.train()
        model[3].eval()  # a submodule left in eval by the user stays in eval
        mon = rq.LayerMonitor([model[1], model[3]], pool=lambda out: out, metrics=["effective_rank"], n_items=120)
        first = mon.sweep(model, loader(), step=0)
        second = mon.sweep(model, loader(), step=1)
        self.assertEqual([r.value for r in first], [r.value for r in second])  # no dropout noise
        self.assertTrue(model.training and model[1].training and not model[3].training)

    def test_jacobian_in_a_sweep_leaves_the_online_buffer_untouched(self):
        model = nn.Sequential(nn.Linear(16, 16), nn.GELU(), nn.Linear(16, 16))
        buf = rq.OnlineBuffer([model[0], model[2]], pool=lambda out: out, n_items=200)
        buf.attach()
        mon = rq.LayerMonitor(
            [model[0], model[2]], pool=lambda out: out, metrics=["effective_rank"], n_items=120, jacobian_items=6
        )
        try:
            rec = mon.sweep(model, loader(), step=0)
        finally:
            buf.detach()
        jer = [r for r in rec if r.metric == "jacobian_effective_rank"]
        self.assertEqual([r.layer for r in jer], [0, 1])
        self.assertTrue(all(1.0 <= r.value <= 16.0 and r.n_items == 6 for r in jer))
        self.assertEqual(buf.seen.get(0, 0), 0)  # eval-mode passes are not training batches

    def test_views_in_train_mode_switch_only_the_augmented_passes(self):
        modes = []

        class Probe(nn.Module):
            def forward(self, x):
                modes.append(self.training)
                return x

        model = nn.Sequential(Probe(), nn.Linear(16, 16))
        model.eval()
        mon = rq.LayerMonitor(
            [model[1]],
            pool=lambda out: out,
            metrics=["effective_rank"],
            view_metrics=["lidar"],
            augment=lambda x, g: x + 0.1 * torch.randn(x.shape, generator=g),
            q=2,
            n_items=120,
            views_in_train_mode=True,
        )
        mon.sweep(model, loader(), step=0)
        self.assertEqual(modes, [False] * 3 + [True] * 6)  # 3 batches plain, then 2 views x 3 batches
        self.assertFalse(model.training)

    def test_train_mode_views_restore_batchnorm_statistics(self):
        model = nn.Sequential(nn.Linear(16, 16), nn.BatchNorm1d(16)).eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        mon = rq.LayerMonitor(
            [model[1]],
            pool=lambda out: out,
            metrics=["effective_rank"],
            view_metrics=["lidar"],
            augment=lambda x, g: x + 0.1 * torch.randn(x.shape, generator=g),
            q=2,
            n_items=120,
            views_in_train_mode=True,
        )
        mon.sweep(model, loader(), step=0)
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]), msg=k)

    def test_view_passes_read_the_view_loader(self):
        firsts = []
        mon = rq.LayerMonitor(
            self.model.blocks,
            pool="mean",
            metrics=["effective_rank"],
            view_metrics=["lidar"],
            augment=lambda x, g: firsts.append(float(x[0, 0])) or x + 0.1 * torch.randn(x.shape, generator=g),
            q=2,
            n_items=120,
        )
        live = loader(seed=1)  # stands in for a DataLoader that redraws the crops
        mon.sweep(self.model, loader(), step=0, view_loader=live)
        self.assertEqual(firsts, [float(b[0][0, 0]) for b in live] * 2)
        with self.assertRaisesRegex(ValueError, "view_loader q times"):
            mon.sweep(self.model, loader(), step=1, view_loader=iter(live))

    def test_keys_separate_levels_and_carry_extras(self):
        pooled = rq.LayerMonitor(self.model.blocks, pool="mean", metrics=["intrinsic_dimension/mlid"], n_items=120,
                                 params={"intrinsic_dimension/mlid": {"k": 16}})  # fmt: skip
        sample = rq.LayerMonitor(self.model.blocks, pool="tokens", metrics=["effective_rank"], n_items=120,
                                 level="sample")  # fmt: skip
        keys = rq.layer_scalars(pooled.sweep(self.model, loader(), step=0), extras=("frechet_var",))
        self.assertIn("layer_metrics/intrinsic_dimension_mlid_layer_0", keys)
        self.assertIn("layer_metrics/intrinsic_dimension_mlid_frechet_var_layer_0", keys)
        self.assertIn(
            "layer_metrics/effective_rank_sample_layer_0", rq.layer_scalars(sample.sweep(self.model, loader(), 0))
        )
        with tempfile.TemporaryDirectory() as d:
            sample.sweep(self.model, loader(), step=3, sinks=[rq.json_sink(d)])
            self.assertTrue((Path(d) / "sample_step_3.json").exists())

    def test_population_level_caps_tokens_separately(self):
        mon = rq.LayerMonitor(self.model.blocks, pool="tokens", metrics=["effective_rank"], n_items=40,
                              n_tokens=100, level="population")  # fmt: skip
        rec = mon.sweep(self.model, loader(), step=0)
        self.assertEqual({r.n_items for r in rec}, {100})  # 40 samples x 12 tokens pooled, 100 drawn

    def test_dead_layer_logs_zero_effective_rank(self):
        model = nn.Sequential(nn.Linear(16, 16), nn.ReLU())
        nn.init.zeros_(model[0].weight)
        nn.init.constant_(model[0].bias, -1.0)  # every unit dead: exact zeros
        rec = rq.LayerMonitor([model[1]], pool=lambda out: out, metrics=["effective_rank"], n_items=120).sweep(
            model, loader(), step=0
        )
        self.assertEqual(rq.layer_scalars(rec), {"layer_metrics/effective_rank_layer_0": 0.0})

    def test_drift_against_the_first_or_the_previous_sweep(self):
        mon = rq.LayerMonitor(
            self.model.blocks,
            pool="mean",
            metrics=["effective_rank"],
            drift_metrics=["cka", "cycle_knn"],
            params={"cycle_knn": {"k": 5}},
            n_items=120,
        )
        first = mon.sweep(self.model, loader(), step=0)
        self.assertEqual([r.metric for r in first], ["effective_rank"] * 3)  # the reference sweep has no drift rows
        same = mon.sweep(self.model, loader(), step=1)
        drift = [r for r in same if r.extras.get("source") == "drift"]
        self.assertEqual(sorted({r.metric for r in drift}), ["cka", "cycle_knn"])
        self.assertEqual(len(drift), 6)
        for r in drift:
            self.assertIsNone(r.layer_b)
            self.assertEqual(r.extras["reference_step"], 0)
            self.assertEqual(r.n_items, 120)
        self.assertTrue(all(abs(r.value - 1.0) < 1e-6 for r in drift if r.metric == "cka"))
        self.assertEqual([r.params for r in drift if r.metric == "cycle_knn"][0], {"k": 5})
        keys = rq.layer_scalars(same)
        self.assertIn("drift_metrics/cka_layer_2", keys)
        self.assertIn("layer_metrics/effective_rank_layer_2", keys)
        with torch.no_grad():
            self.model.blocks[1][0].weight.add_(0.5 * torch.randn(16, 16))
        moved = {r.layer: r.value for r in mon.sweep(self.model, loader(), step=2) if r.metric == "cka"}
        self.assertAlmostEqual(moved[0], 1.0, places=6)  # block 0 is upstream of the change
        self.assertLess(moved[1], 0.999)
        self.assertEqual([r.extras["reference_step"] for r in mon.history[-1][1] if r.metric == "cka"], [0, 0, 0])
        prev = rq.LayerMonitor(
            self.model.blocks, pool="mean", metrics=["effective_rank"], drift_metrics=["cka"],
            drift_reference="previous", n_items=120,
        )  # fmt: skip
        prev.sweep(self.model, loader(), step=0)
        with torch.no_grad():
            self.model.blocks[2][0].weight.add_(0.5 * torch.randn(16, 16))
        a = {r.layer: r.value for r in prev.sweep(self.model, loader(), step=1) if r.metric == "cka"}
        b = {r.layer: r.value for r in prev.sweep(self.model, loader(), step=2) if r.metric == "cka"}
        self.assertLess(a[2], 0.999)
        self.assertAlmostEqual(b[2], 1.0, places=6)  # nothing moved between sweeps 1 and 2
        self.assertEqual([r.extras["reference_step"] for r in prev.history[-1][1] if r.metric == "cka"], [1, 1, 1])

    def test_view_metrics_need_augment(self):
        with self.assertRaisesRegex(ValueError, "augment"):
            rq.LayerMonitor(self.model.blocks, pool="mean", metrics=["effective_rank"], view_metrics=["lidar"])

    def test_drift_needs_pair_metrics_at_the_sequence_level(self):
        with self.assertRaisesRegex(ValueError, "pair metric"):
            rq.LayerMonitor(
                self.model.blocks, pool="mean", metrics=["effective_rank"], drift_metrics=["effective_rank"]
            )
        with self.assertRaisesRegex(ValueError, "sequence"):
            rq.LayerMonitor(
                self.model.blocks, pool="tokens", metrics=["effective_rank"], level="sample", drift_metrics=["cka"]
            )
        with self.assertRaisesRegex(ValueError, "drift_reference"):
            rq.LayerMonitor(
                self.model.blocks,
                pool="mean",
                metrics=["effective_rank"],
                drift_metrics=["cka"],
                drift_reference="last",
            )

    def test_view_metrics_reject_a_one_shot_loader(self):
        mon = rq.LayerMonitor(
            self.model.blocks,
            pool="mean",
            metrics=["effective_rank"],
            view_metrics=["lidar"],
            augment=lambda x, g: x + 0.1 * torch.randn(x.shape, generator=g),
            q=2,
            n_items=120,
        )
        with self.assertRaisesRegex(ValueError, "read the loader again"):
            mon.sweep(self.model, iter(loader()), step=0)

    def test_warns_once_when_items_yield_several_rows(self):
        model = Clips().eval()
        clips = [(torch.randn(10, 4, 16), torch.zeros(10)) for _ in range(3)]
        mon = rq.LayerMonitor(model.blocks, pool=lambda out: out, metrics=["effective_rank"], n_items=60)
        with self.assertWarnsRegex(RuntimeWarning, "10 items produced 40 rows"):
            mon.sweep(model, clips, step=0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)  # once per monitor
            mon.sweep(model, clips, step=1)
        dicts = [{"pixels": c[0]} for c in clips]  # dict batches count their first tensor
        mon = rq.LayerMonitor(model.blocks, pool=lambda out: out, metrics=["effective_rank"], n_items=60)
        with self.assertWarnsRegex(RuntimeWarning, "each item produced 4 rows"):
            mon.sweep(lambda b: model(b["pixels"]), dicts, step=0)

    def test_warns_when_a_layer_runs_twice_per_batch(self):
        mon = rq.LayerMonitor(
            self.model.blocks, pool=lambda out: out.mean(dim=1), metrics=["effective_rank"], n_items=100
        )
        with self.assertWarnsRegex(RuntimeWarning, "ran 2 times"):
            mon.sweep(lambda x: (self.model(x), self.model(x)), loader(), step=0)

    def test_no_layout_warning_for_one_row_per_item(self):
        mon = rq.LayerMonitor(
            self.model.blocks, pool=lambda out: out.mean(dim=1), metrics=["effective_rank"], n_items=100
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            mon.sweep(self.model, loader(), step=0)

    def test_compute_warns_when_an_estimator_fails(self):
        views = {0: torch.randn(3, 200, 8)}
        with self.assertWarns(RuntimeWarning):
            rec = rq.compute(views, ["dime"])  # dime takes exactly two views
        self.assertIn("error", rec[0].extras)

    def test_compute_rejects_layers_of_unequal_length(self):
        with self.assertRaises(ValueError):
            rq.compute({0: torch.randn(40, 4), 1: torch.randn(30, 4)}, ["effective_rank"])

    def test_online_buffer_large_batch_keeps_last_rows(self):
        buf = rq.OnlineBuffer(self.model.blocks, pool="mean", n_items=10)
        buf.attach()
        self.model.train()
        self.model(torch.randn(25, 16))
        buf.detach()
        self.assertEqual(buf.layers()[0].shape, (10, 16))
        self.assertEqual(buf.pointer[0], 0)
        self.assertEqual(buf.seen[0], 25)

    def test_lightning_adapter_skips_non_zero_rank(self):
        try:
            from req_metrics.integrations.lightning import LayerMonitorCallback
        except ImportError:
            self.skipTest("lightning not installed")
        cb = LayerMonitorCallback(
            ["effective_rank"],
            layers="blocks",
            pool="mean",
            n_items=80,
            loader=lambda trainer, module: loader(),
            log=False,
            online=True,
        )
        trainer = type("T", (), {"current_epoch": 0, "logger": None, "global_step": 0, "is_global_zero": False})()
        cb.on_train_start(trainer, self.model)
        cb.on_train_epoch_end(trainer, self.model)
        self.assertIsNone(cb.monitor)
        self.assertIsNone(cb.online)

    def test_lightning_adapter_logs_to_every_logger(self):
        try:
            from req_metrics.integrations.lightning import LayerMonitorCallback
        except ImportError:
            self.skipTest("lightning not installed")
        calls: list[tuple[int, dict]] = []

        class FakeLogger:
            def __init__(self, idx):
                self.idx = idx

            def log_metrics(self, metrics, step=None):
                calls.append((self.idx, metrics))

        loggers = [FakeLogger(0), FakeLogger(1)]
        trainer = type("T", (), {"current_epoch": 0, "logger": loggers[0], "loggers": loggers, "global_step": 3})()
        cb = LayerMonitorCallback(
            ["effective_rank"], layers="blocks", pool="mean", n_items=80, loader=lambda t, m: loader()
        )
        cb.on_train_start(trainer, self.model)
        self.assertEqual(sorted(i for i, _ in calls), [0, 1])
        self.assertIn("layer_metrics/effective_rank_layer_0", calls[0][1])

    def test_lightning_adapter_with_explicit_loader(self):
        try:
            from req_metrics.integrations.lightning import LayerMonitorCallback
        except ImportError:
            self.skipTest("lightning not installed")
        cb = LayerMonitorCallback(
            ["effective_rank"],
            layers="blocks",
            pool="mean",
            n_items=80,
            every_n_epochs=2,
            loader=lambda trainer, module: loader(),
            log=False,
        )
        trainer = type("T", (), {"current_epoch": 0, "logger": None, "global_step": 0})()
        cb.on_train_start(trainer, self.model)
        trainer.current_epoch, trainer.global_step = 1, 6
        cb.on_train_epoch_end(trainer, self.model)  # epoch 2 -> sweep, labelled with the global step
        trainer.current_epoch, trainer.global_step = 2, 9
        cb.on_train_epoch_end(trainer, self.model)  # epoch 3 -> skip
        self.assertEqual([s for s, _ in cb.monitor.history], [0, 6])


class GridPoolerTests(unittest.TestCase):
    """make_pooler's grid readouts equal the per-clip layouts functions, for both token orders."""

    def setUp(self):
        torch.manual_seed(0)
        self.b, self.f, self.t, self.d = 3, 4, 5, 6
        self.grid = torch.randn(self.b, self.f, self.t, self.d)  # (B, F, T, D)

    def _tokens(self, time_axis, n_prefix=0):
        g = self.grid if time_axis == 1 else self.grid.transpose(1, 2)  # token index f*T+t or t*F+f
        toks = g.reshape(self.b, self.f * self.t, self.d)
        return torch.cat([torch.randn(self.b, n_prefix, self.d), toks], dim=1) if n_prefix else toks

    def test_grid_readouts_match_layouts(self):
        from req_metrics.monitor import make_pooler

        for time_axis in (1, 0):
            toks = self._tokens(time_axis, n_prefix=1)
            kw = dict(grid=(self.f, self.t), time_axis=time_axis, n_prefix=1)
            for mode, ref in [
                ("gap", lambda g: rq.grid_to_pooled(g, mode="gap")),
                ("freq_concat_mean", lambda g: rq.grid_to_pooled(g, mode="freq_concat_mean")),
                ("partitioned", lambda g: rq.grid_to_pooled(g, mode="partitioned", freq_chunks=2, time_chunks=2)),
                ("freq_concat", lambda g: rq.grid_to_trajectory(g, mode="freq_concat")),
                ("freq_mean", lambda g: rq.grid_to_trajectory(g, mode="freq_mean")),
            ]:
                got = make_pooler(mode, freq_chunks=2, time_chunks=2, **kw)(toks)
                want = torch.stack([ref(self.grid[i]) for i in range(self.b)])
                self.assertTrue(torch.allclose(got, want, atol=1e-6), (mode, time_axis))

    def test_grid_readout_needs_grid_and_checks_token_count(self):
        from req_metrics.monitor import make_pooler

        with self.assertRaises(ValueError):
            make_pooler("freq_concat_mean")
        with self.assertRaises(ValueError):
            make_pooler("gap", grid=(4, 5))(torch.randn(2, 19, 6))

    def test_buffer_and_callback_accept_pool_kwargs(self):
        from req_metrics.monitor import OnlineBuffer

        blocks = nn.ModuleList([nn.Linear(6, 6)])
        buf = OnlineBuffer(list(blocks), pool="freq_concat_mean", n_items=8, pool_kwargs={"grid": (4, 5)})
        self.assertEqual(tuple(buf.pool(torch.randn(2, 20, 6)).shape), (2, 24))
        try:
            from req_metrics.integrations.lightning import LayerMonitorCallback
        except ImportError:
            self.skipTest("lightning not installed")
        cb = LayerMonitorCallback(
            ["effective_rank"], layers=list(blocks), pool="freq_concat_mean", pool_kwargs={"grid": (4, 5)}
        )
        self.assertEqual(cb.pool_kwargs, {"grid": (4, 5)})


if __name__ == "__main__":
    unittest.main()
