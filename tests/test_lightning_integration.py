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
                ["effective_rank", "anisotropy/spectral"],
                layers="backbone.blocks",
                pool="mean",
                n_items=64,
                every_n_epochs=1,
                model_attr="backbone",
                model="toy",
                batch_size=16,
                online=True,
                online_n_items=80,
                drift_metrics=["cka"],
                sinks=[rq.json_sink(Path(d) / "sweeps")],
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
            self.assertEqual([s for s, _ in cb.monitor.history], [0, 6, 12])  # global steps: 96 rows / 16
            rec = cb.monitor.history[-1][1]
            self.assertEqual(len(rec), 9)  # two point metrics and the CKA drift, three layers each
            self.assertEqual([r.extras["reference_step"] for r in rec if r.metric == "cka"], [0, 0, 0])
            self.assertEqual(rec[0].n_items, 64)
            self.assertEqual(rec[0].pooling, "mean")
            self.assertEqual([s for s, _ in cb.online.history], [6, 12])  # online records at both epoch ends
            self.assertEqual(cb.online.seen[0], 192)  # 2 epochs x 96 training rows, validation and sweeps ignored
            on = cb.online.history[-1][1]
            self.assertEqual(on[0].n_items, 80)
            self.assertEqual(on[0].extras["source"], "training-batches")
            self.assertEqual(cb.online.handles, [])  # detached at train end
            metrics_csv = next(Path(d).rglob("metrics.csv")).read_text()
            self.assertIn("layer_metrics/effective_rank_layer_2", metrics_csv)
            self.assertIn("drift_metrics/cka_layer_2", metrics_csv)
            names = sorted(f.name for f in (Path(d) / "sweeps").glob("*.json"))
            self.assertEqual(len(names), 5)  # 3 sweeps and 2 online computations, none overwritten
            self.assertEqual(sum(n.startswith("online_") for n in names), 2)
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
            common = dict(
                layers="blocks",
                pool="mean",
                n_items=32,
                every_n_epochs=1,
                model="toy",
                batch_size=16,
                drift_metrics=["cka"],
            )
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
            self.assertEqual([s for s, _ in cb1.monitor.history], [0, 4])
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
            # the history is restored from the checkpoint; no sweep at the start of a resumed run
            self.assertEqual([s for s, _ in cb2.monitor.history], [0, 4, 8, 12])
            self.assertEqual([r.extras["epoch"] for _, rec in cb2.monitor.history[2:] for r in rec[:1]], [1, 2])
            drift = [r for r in cb2.monitor.history[-1][1] if r.metric == "cka"]
            self.assertEqual([r.extras["reference_step"] for r in drift], [0, 0])  # still against the first sweep

    def test_checkpoint_and_early_stopping_select_by_a_layer_metric(self):
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

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
                return torch.optim.SGD(self.parameters(), lr=0.05)

        torch.manual_seed(0)
        ds = torch.utils.data.TensorDataset(torch.randn(64, 8), torch.randn(64))
        dl = torch.utils.data.DataLoader(ds, batch_size=16)
        key = "layer_metrics/effective_rank_layer_1"
        with tempfile.TemporaryDirectory() as d:
            cb = LayerMonitorCallback(["effective_rank"], layers="blocks", pool="mean", n_items=32, batch_size=16)
            ckpt = ModelCheckpoint(dirpath=d, monitor=key, mode="max", save_top_k=1)
            trainer = pl.Trainer(
                max_epochs=3,
                default_root_dir=d,
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                callbacks=[cb, ckpt],
                accelerator="cpu",
            )
            trainer.fit(Lit(), dl)
            values = {s: rq.value_at(rec, "effective_rank", 1) for s, rec in cb.monitor.history if s > 0}
            self.assertAlmostEqual(float(ckpt.best_model_score), max(values.values()), places=6)
            self.assertIn(key, trainer.callback_metrics)
            stop = EarlyStopping(monitor=key, mode="max", patience=0, min_delta=1e9)  # stops at the first check
            trainer = pl.Trainer(
                max_epochs=5,
                default_root_dir=d,
                logger=False,
                enable_progress_bar=False,
                enable_checkpointing=False,
                enable_model_summary=False,
                callbacks=[LayerMonitorCallback(["effective_rank"], layers="blocks", pool="mean", n_items=32), stop],
                accelerator="cpu",
            )
            trainer.fit(Lit(), dl)
            self.assertLess(trainer.current_epoch, 5)

    def test_callback_monitor_and_sinks_pickle(self):
        import pickle

        from req_metrics.integrations.lightning import LayerMonitorCallback

        with tempfile.TemporaryDirectory() as d:
            cb = LayerMonitorCallback(
                ["effective_rank"], layers="blocks", pool="tokens", level="population", n_items=16,
                sinks=[rq.csv_sink(Path(d) / "s.csv"), rq.json_sink(d), rq.tensorboard_sink(d)],
            )  # fmt: skip
            pickle.dumps(cb)
            blocks = nn.ModuleList([nn.Linear(8, 8) for _ in range(2)])
            mon = rq.LayerMonitor(blocks, pool="gap", pool_kwargs={"grid": (2, 4)}, metrics=["effective_rank"])
            pickle.dumps(mon)
            pickle.dumps(rq.OnlineBuffer(blocks, "mean"))
            self.assertEqual(
                pickle.loads(pickle.dumps(rq.make_pooler("tokens", n_prefix=1)))(torch.ones(2, 3, 4)).shape, (2, 2, 4)
            )

    def test_fit_with_view_metrics(self):
        from req_metrics.integrations.lightning import LayerMonitorCallback

        class Lit(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(16, 16), nn.GELU()) for _ in range(2)])

            def forward(self, x):
                for b in self.blocks:
                    x = x + b(x)
                return x

            def training_step(self, batch, idx):
                return self(batch[0]).square().mean()

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.01)

        torch.manual_seed(0)
        dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(96, 16)), batch_size=16)
        cb = LayerMonitorCallback(
            ["effective_rank"],
            layers="blocks",
            pool=lambda out: out,
            n_items=64,
            batch_size=16,
            view_metrics=["lidar"],
            augment=lambda x, g: x + 0.1 * torch.randn(x.shape, generator=g),
            q=3,
            log=False,
        )
        trainer = pl.Trainer(
            max_epochs=1,
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            callbacks=[cb],
            accelerator="cpu",
        )
        trainer.fit(Lit(), dl)
        lidar = [r for _, rec in cb.monitor.history for r in rec if r.metric == "lidar"]
        self.assertEqual(len(lidar), 2 * 2)  # two sweeps, two layers
        self.assertTrue(all(r.value == r.value for r in lidar))

    def test_train_mode_augmentation_stays_active_during_sweeps(self):
        from req_metrics.integrations.lightning import LayerMonitorCallback

        aug_modes, backbone_modes = [], []

        class Aug(nn.Module):  # acts in train mode only, as GPU augmentation chains often do
            def forward(self, x, gen):
                aug_modes.append(self.training)
                return x + 0.1 * torch.randn(x.shape, generator=gen) if self.training else x

        class Probe(nn.Module):
            def forward(self, x):
                backbone_modes.append(self.training)
                return x

        class Lit(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.aug = Aug()
                self.backbone = nn.Sequential(Probe(), nn.Linear(8, 8), nn.Linear(8, 8))

            def training_step(self, batch, idx):
                return self.backbone(batch[0]).square().mean()

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.01)

        torch.manual_seed(0)
        lit = Lit()
        dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(64, 8)), batch_size=16)
        cb = LayerMonitorCallback(
            ["effective_rank"],
            layers=[lit.backbone[1], lit.backbone[2]],
            pool=lambda out: out,
            n_items=64,
            every_n_epochs=1,
            model_attr="backbone",
            view_metrics=["lidar"],
            augment=lambda x, g: lit.aug(x, g),
            q=3,
            log=False,
        )
        trainer = pl.Trainer(
            max_epochs=1,
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            callbacks=[cb],
            accelerator="cpu",
        )
        trainer.fit(lit, dl)
        self.assertTrue(aug_modes and all(aug_modes))  # the augmentation module kept acting
        self.assertIn(False, backbone_modes)  # sweeps read the backbone in eval mode
        self.assertIn(True, backbone_modes)  # training steps in train mode

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
        self.assertTrue(torch.equal(rq.make_pooler("tokens")(out), out))
        self.assertTrue(torch.equal(rq.make_pooler("tokens", n_prefix=1)(out), out[:, 1:]))
        dl = rq.monitor_loader(
            torch.utils.data.TensorDataset(torch.arange(50.0).unsqueeze(1)), n_items=20, batch_size=8, seed=1
        )
        xs = torch.cat([b[0] for b in dl])
        self.assertEqual(len(xs), 20)
        self.assertTrue(torch.all(xs[1:] > xs[:-1]))


if __name__ == "__main__":
    unittest.main()
