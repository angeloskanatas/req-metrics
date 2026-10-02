"""End-to-end: a real Lightning Trainer fit with the plug-and-play callback and a CSV logger."""

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

import req_metrics as rq

try:
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import CSVLogger

    HAVE_PL = True
except ImportError:  # pragma: no cover
    HAVE_PL = False


@unittest.skipUnless(HAVE_PL, "lightning not installed")
class LightningPlugAndPlayTests(unittest.TestCase):
    def test_fit_with_callback(self):
        from req_metrics.integrations.lightning import LayerMonitorCallback

        class Encoder(nn.Module):
            def __init__(self, d=16, t=8):
                super().__init__()
                self.embed = nn.Linear(d, d)
                self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(d, d), nn.GELU()) for _ in range(3)])
                self.t = t

            def forward(self, x):
                h = self.embed(x).unsqueeze(1).repeat(1, self.t, 1)
                for b in self.blocks:
                    h = h + b(h)
                return h

        class Lit(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.backbone = Encoder()
                self.head = nn.Linear(16, 1)

            def training_step(self, batch, idx):
                x, y = batch
                return nn.functional.mse_loss(self.head(self.backbone(x).mean(1)).squeeze(-1), y)

            def validation_step(self, batch, idx):
                x, y = batch
                self.log("val_loss", nn.functional.mse_loss(self.head(self.backbone(x).mean(1)).squeeze(-1), y))

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.01)

        torch.manual_seed(0)
        ds = torch.utils.data.TensorDataset(torch.randn(96, 16), torch.randn(96))
        dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True)
        with tempfile.TemporaryDirectory() as d:
            cb = LayerMonitorCallback(
                ["effective_rank", "anisotropy"],
                layers="backbone.blocks",
                pool="mean",
                n_items=64,
                every_n_epochs=1,
                model_attr="backbone",
                model="toy",
                batch_size=16,
                online=True,
                online_n_items=80,
            )
            trainer = pl.Trainer(
                max_epochs=2,
                logger=CSVLogger(d, name="run"),
                enable_progress_bar=False,
                enable_checkpointing=False,
                enable_model_summary=False,
                callbacks=[cb],
                accelerator="cpu",
                log_every_n_steps=1,
                num_sanity_val_steps=0,
            )
            trainer.fit(Lit(), dl, torch.utils.data.DataLoader(ds, batch_size=32))
            self.assertEqual([s for s, _ in cb.monitor.history], [0, 1, 2])
            rec = cb.monitor.history[-1][1]
            self.assertEqual(len(rec), 6)
            self.assertEqual(rec[0].n_items, 64)
            self.assertEqual(rec[0].pooling, "mean")
            self.assertEqual([s for s, _ in cb.online.history], [1, 2])  # online records at both epoch ends
            self.assertEqual(cb.online.seen[0], 192)  # 2 epochs x 96 training rows, validation and sweeps ignored
            on = cb.online.history[-1][1]
            self.assertEqual(on[0].n_items, 80)
            self.assertEqual(on[0].extras["source"], "training-batches")
            self.assertEqual(cb.online.handles, [])  # detached at train end
            metrics_csv = next(Path(d).rglob("metrics.csv")).read_text()
            self.assertIn("layer_metrics/effective_rank_layer_2", metrics_csv)
            self.assertIn("online_metrics/effective_rank_layer_2", metrics_csv)

    def test_resume_from_checkpoint(self):
        """A run resumed from a checkpoint sweeps on its own schedule and stamps the resumed epochs."""
        from req_metrics.integrations.lightning import LayerMonitorCallback

        class Lit(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Linear(8, 8) for _ in range(2)])
                self.head = nn.Linear(8, 1)

            def forward(self, x):
                for b in self.blocks:
                    x = torch.tanh(b(x))
                return x

            def training_step(self, batch, idx):
                x, y = batch
                return nn.functional.mse_loss(self.head(self(x)).squeeze(-1), y)

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.01)

        torch.manual_seed(0)
        ds = torch.utils.data.TensorDataset(torch.randn(64, 8), torch.randn(64))
        dl = torch.utils.data.DataLoader(ds, batch_size=16)
        with tempfile.TemporaryDirectory() as d:
            common = dict(layers="blocks", pool="mean", n_items=32, every_n_epochs=1, model="toy", batch_size=16)
            cb1 = LayerMonitorCallback(["effective_rank"], **common)
            t1 = pl.Trainer(
                max_epochs=1,
                default_root_dir=d,
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                callbacks=[cb1],
                accelerator="cpu",
            )
            t1.fit(Lit(), dl)
            ckpt = str(Path(d) / "last.ckpt")
            t1.save_checkpoint(ckpt)
            self.assertEqual([s for s, _ in cb1.monitor.history], [0, 1])
            cb2 = LayerMonitorCallback(["effective_rank"], **common)
            t2 = pl.Trainer(
                max_epochs=3,
                default_root_dir=d,
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                callbacks=[cb2],
                accelerator="cpu",
            )
            t2.fit(Lit(), dl, ckpt_path=ckpt)
            self.assertEqual([s for s, _ in cb2.monitor.history], [2, 3])  # no sweep at start of a resumed run
            self.assertEqual([r.extras["epoch"] for _, rec in cb2.monitor.history for r in rec[:1]], [1, 2])

    def test_resolve_layers_and_poolers(self):
        m = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))

        class Wrap(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Module()
                self.encoder.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(5)])
                self.other = m

        w = Wrap()
        self.assertEqual(len(rq.resolve_layers(w)), 5)  # prefers a list named "layers"
        self.assertEqual(len(rq.resolve_layers(w, "encoder.layers")), 5)
        self.assertEqual(len(rq.resolve_layers(w, [w.other[0], w.other[2]])), 2)
        out = torch.randn(3, 7, 4)
        self.assertTrue(torch.equal(rq.make_pooler("cls")(out), out[:, 0]))
        self.assertTrue(torch.equal(rq.make_pooler("last")(out), out[:, -1]))
        self.assertTrue(torch.allclose(rq.make_pooler("mean")(out), out.mean(1)))
        dl = rq.monitor_loader(
            torch.utils.data.TensorDataset(torch.arange(50.0).unsqueeze(1)), n_items=20, batch_size=8, seed=1
        )
        xs = torch.cat([b[0] for b in dl])
        self.assertEqual(len(xs), 20)
        self.assertTrue(torch.all(xs[1:] > xs[:-1]))


if __name__ == "__main__":
    unittest.main()
