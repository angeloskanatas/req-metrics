"""The README quick start on synthetic stand-ins. Update this file together with the README."""

import tempfile
import unittest
import warnings
from pathlib import Path

import torch
import torch.nn as nn

import req_metrics as rq

try:
    import lightning.pytorch as pl

    HAVE_PL = True
except ImportError:  # pragma: no cover
    HAVE_PL = False


def _failed(rec):
    return [(r.metric, r.layer, r.extras.get("error")) for r in rec.rows if r.value != r.value]


class PostHocExampleTests(unittest.TestCase):
    def test_post_hoc_block(self):
        torch.manual_seed(0)
        base = torch.randn(600, 8)
        z0, z1, z2 = (base @ torch.randn(8, 24) + 0.3 * (i + 1) * torch.randn(600, 24) for i in range(3))
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)  # any failed estimator fails the test
            layers = {0: z0, 1: z1, 2: z2}
            rec = rq.compute(
                layers,
                ["effective_rank", "intrinsic_dimension/gride", "anisotropy", "self_clustering"],
                n=10000,
                seed=42,
                model="my-encoder",
                pooling="time-mean",
            )
            self.assertEqual(len(rec.profile("effective_rank")), 3)
            with tempfile.TemporaryDirectory() as d:
                rec.to_csv(Path(d) / "my-encoder.csv")
            tokens = {l: [torch.cumsum(torch.randn(40, 24), 0) for _ in range(10)] for l in range(2)}
            rq.compute(tokens, ["trajectory_curvature", "effective_rank"], level="sample", n=2000)
            rq.compute(tokens, ["effective_rank"], level="population", n=10000)
            views = {l: torch.stack([z + 0.1 * torch.randn_like(z) for _ in range(10)]) for l, z in layers.items()}
            spec = rq.ViewSpec(source="shared", augmentations=("PitchShift(-4..4)",), q=10)
            rq.compute(views, ["lidar", "infonce"], views=spec)
            shifted = {0: (z0, {k: z0 + 0.05 * k * torch.randn_like(z0) for k in range(1, 12)})}
            rq.compute(
                shifted,
                ["pte"],
                shifts=rq.ShiftSpec("waveform pitch shift", semitones=tuple(range(1, 12))),
                params={"pte": {"epochs": 3}},
            )
            for kw in ({"k": 1}, {"metric": "neighborhood_overlap"}, {"metric": "cka"}, {"metric": "svcca"}):
                self.assertEqual(len(rq.compute_pairs(layers, **kw).rows), 9)
            rq.convergence(z1, "effective_rank").to_markdown()
            self.assertEqual(len(rq.top_layers(rec, "intrinsic_dimension/gride", k=3)), 3)
            p = rq.protocols.get("kanatas2026")
            self.assertEqual(
                _failed(rq.compute(layers, p.names("sequence"), params=p.params("sequence"), n=p.n_items)), []
            )


class _Backbone(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d, 2, 32, batch_first=True) for _ in range(2)])

    def forward(self, x):  # (B, T, D) frames -> (B, 1 + T, D) tokens
        h = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1)
        for b in self.blocks:
            h = b(h)
        return h


def _augment(batch, generator):
    return batch + 0.1 * torch.randn(batch.shape, generator=generator)


class MonitoringExampleTests(unittest.TestCase):
    @unittest.skipUnless(HAVE_PL, "lightning not installed")
    def test_lightning_block(self):
        from req_metrics.integrations.lightning import LayerMonitorCallback

        class Lit(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.backbone = _Backbone()

            def training_step(self, batch, idx):
                return self.backbone(batch[0])[:, 0].square().mean()

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.01)

        torch.manual_seed(0)
        dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(128, 6, 16)), batch_size=32)
        cb = LayerMonitorCallback(
            ["effective_rank", "intrinsic_dimension/mlid", "anisotropy"],
            layers="backbone.blocks",
            pool="cls",
            n_items=96,
            every_n_epochs=1,
            sweep_steps=(2,),
            model_attr="backbone",
            view_metrics=["lidar"],
            augment=_augment,
            q=4,
            online=True,
            params={"intrinsic_dimension/mlid": {"k": 16}},
            log=False,
        )
        trainer = pl.Trainer(
            max_epochs=1,
            callbacks=[cb],
            accelerator="cpu",
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
            enable_model_summary=False,
        )
        trainer.fit(Lit(), dl)
        last = cb.monitor.history[-1][1]
        self.assertEqual(
            {r.metric for r in last}, {"effective_rank", "intrinsic_dimension/mlid", "anisotropy", "lidar"}
        )
        self.assertEqual(_failed(last), [])

    def test_standalone_monitor_block(self):
        torch.manual_seed(0)
        model = _Backbone()
        data = torch.utils.data.TensorDataset(torch.randn(96, 6, 16))
        batches = [(b[0],) for b in rq.monitor_loader(data, 96, 32, 0)]
        mon = rq.LayerMonitor(
            model.blocks,
            pool="cls",
            metrics=["effective_rank", "intrinsic_dimension/mlid"],
            n_items=96,
            params={"intrinsic_dimension/mlid": {"k": 16}},
        )
        with tempfile.TemporaryDirectory() as d:
            for epoch in range(2):
                mon.sweep(model, batches, step=epoch, sinks=[rq.csv_sink(Path(d) / "sweeps.csv")])
        profiles = mon.profiles("effective_rank")
        self.assertEqual(profiles[0], profiles[1])  # same weights, eval mode: identical sweeps


if __name__ == "__main__":
    unittest.main()
