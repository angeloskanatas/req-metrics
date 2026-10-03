"""Item caps, cumulative profile plots and the step-based monitoring schedule."""

import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn

import req_metrics as rq
from req_metrics.monitor import _profile_series


class Toy(nn.Module):
    def __init__(self, d=12):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(d, d) for _ in range(2)])

    def forward(self, x):
        for b in self.blocks:
            x = torch.tanh(b(x))
        return x


def loader(n=64, d=12, batch=16):
    g = torch.Generator().manual_seed(0)
    for _ in range(n // batch):
        yield torch.randn(batch, d, generator=g)


class ItemCapTests(unittest.TestCase):
    def test_registry_caps_declared_for_superlinear_estimators(self):
        self.assertEqual(rq.get_metric("intrinsic_dimension/mst").max_items, 2000)
        self.assertIsNone(rq.get_metric("effective_rank").max_items)

    def test_limits_subsample_and_record_count(self):
        x = torch.randn(300, 8)
        rec = rq.compute({0: x}, ["effective_rank", "anisotropy"], limits={"effective_rank": 100})
        by = {r.metric: r for r in rec}
        self.assertEqual(by["effective_rank"].extras["n_items_used"], 100)
        self.assertNotIn("n_items_used", by["anisotropy"].extras)
        self.assertEqual(by["effective_rank"].n_items, 300)  # the population size is still recorded
        again = rq.compute({0: x}, ["effective_rank"], limits={"effective_rank": 100})
        self.assertEqual(again[0].value, by["effective_rank"].value)  # the subsample is seeded

    def test_limits_apply_to_tokens(self):
        clips = [torch.randn(50, 8) for _ in range(4)]
        rec = rq.compute({0: clips}, ["effective_rank"], population="tokens", limits={"effective_rank": 60})
        self.assertEqual(rec[0].extras["n_items_used"], 60)


class ProfileHistoryTests(unittest.TestCase):
    def test_profile_series_accumulates_sweeps(self):
        model = Toy()
        mon = rq.LayerMonitor(model.blocks, pool=lambda out: out, metrics=["effective_rank"], n_items=64)
        r1 = mon.sweep(model, loader(), step=1)
        with torch.no_grad():
            model.blocks[0].weight.mul_(3.0)
        r2 = mon.sweep(model, loader(), step=2)
        xs, ys, keys = _profile_series(r2, mon.history, "effective_rank")
        self.assertEqual(xs, [0, 1])
        self.assertEqual(len(ys), 2)
        self.assertEqual(keys, ["step 1", "step 2"])
        self.assertEqual(ys[0], [v for _, v in r1.profile("effective_rank")])
        xs, ys, keys = _profile_series(r1, None, "effective_rank")
        self.assertEqual(len(ys), 1)

    def test_wandb_sink_with_history_plots_every_sweep(self):
        model = Toy()
        mon = rq.LayerMonitor(model.blocks, pool=lambda out: out, metrics=["effective_rank"], n_items=64)
        logged = []
        fake = type("W", (), {"log": lambda self, d: logged.append(d), "define_metric": lambda self, *a, **k: None})()
        stub = types.ModuleType("wandb")
        stub.plot = types.SimpleNamespace(line_series=lambda **kw: kw)
        sys.modules["wandb"] = stub
        try:
            sink = rq.wandb_sink(fake, history=mon.history)
            mon.sweep(model, loader(), step=1, sinks=[sink])
            with torch.no_grad():
                model.blocks[1].weight.mul_(2.0)
            mon.sweep(model, loader(), step=2, sinks=[sink])
        finally:
            del sys.modules["wandb"]
        self.assertEqual(len(logged[0]["profiles/effective_rank"]["ys"]), 1)
        self.assertEqual(len(logged[1]["profiles/effective_rank"]["ys"]), 2)
        self.assertEqual(logged[1]["profiles/effective_rank"]["keys"], ["step 1", "step 2"])


class StepScheduleTests(unittest.TestCase):
    def _callback(self, **kw):
        try:
            from req_metrics.integrations.lightning import LayerMonitorCallback
        except ImportError:
            self.skipTest("lightning not installed")
        return LayerMonitorCallback(
            ["effective_rank"], layers="blocks", pool="mean", n_items=64, loader=lambda t, m: loader(), log=False, **kw
        )

    def _trainer(self):
        return type("T", (), {"current_epoch": 0, "logger": None, "global_step": 0})()

    def test_sweep_steps_and_every_n_steps(self):
        model = Toy()
        cb = self._callback(every_n_epochs=None, sweep_steps=(2,), every_n_steps=5)
        trainer = self._trainer()
        cb.on_train_start(trainer, model)
        for step in range(1, 12):
            trainer.global_step = step
            cb.on_train_batch_end(trainer, model, None, None, step - 1)
            cb.on_train_batch_end(trainer, model, None, None, step - 1)  # a repeated step sweeps once
        trainer.current_epoch = 0
        cb.on_train_epoch_end(trainer, model)  # epoch sweeps disabled
        self.assertEqual([s for s, _ in cb.monitor.history], [0, 2, 5, 10])
        last = cb.monitor.history[-1][1][0]
        self.assertEqual(last.extras["global_step"], 10)
        self.assertEqual(last.extras["epoch"], 0)

    def test_mixed_schedules_keep_every_sweep_in_the_sinks(self):
        model = Toy()
        xs = []
        fake = type(
            "W",
            (),
            {
                "log": lambda self, log: xs.append(log["monitor/step"]),
                "define_metric": lambda self, name, step_metric=None: None,
            },
        )()
        sys.modules["wandb"] = types.ModuleType("wandb")
        try:
            with tempfile.TemporaryDirectory() as d:
                sinks = [rq.json_sink(d), rq.wandb_sink(fake, line_series=False)]
                cb = self._callback(every_n_epochs=1, sweep_steps=(2,), sinks=sinks)
                trainer = self._trainer()
                cb.on_train_start(trainer, model)
                trainer.global_step = 2
                cb.on_train_batch_end(trainer, model, None, None, 1)  # step sweep, label 2
                trainer.global_step = 6
                cb.on_train_epoch_end(trainer, model)  # epoch sweep, label 1
                trainer.current_epoch, trainer.global_step = 1, 12
                cb.on_train_epoch_end(trainer, model)  # epoch sweep, label 2
                names = sorted(f.name for f in Path(d).glob("*.json"))
        finally:
            del sys.modules["wandb"]
        self.assertEqual(
            names, ["epoch_0_step_0.json", "epoch_0_step_2.json", "epoch_0_step_6.json", "epoch_1_step_12.json"]
        )
        self.assertEqual(xs, [0, 2, 6, 12])

    def test_cache_batches_materializes_once(self):
        model = Toy()
        calls = []

        def counting(t, m):
            calls.append(1)
            return loader()

        try:
            from req_metrics.integrations.lightning import LayerMonitorCallback
        except ImportError:
            self.skipTest("lightning not installed")
        cb = LayerMonitorCallback(
            ["effective_rank"], layers="blocks", pool="mean", n_items=64, loader=counting, log=False, cache_batches=True
        )
        trainer = self._trainer()
        cb.on_train_start(trainer, model)
        for e in range(2):
            trainer.current_epoch = e
            cb.on_train_epoch_end(trainer, model)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(cb.monitor.history), 3)
        self.assertEqual(cb.monitor.history[-1][1][0].n_items, 64)


if __name__ == "__main__":
    unittest.main()
