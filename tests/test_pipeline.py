"""compute() over synthetic layers: caching equivalence, populations, subsetting, records I/O, protocol."""

import math
import tempfile
import unittest
from pathlib import Path

import torch

import req_metrics as rq


def pooled_layers(n=400, d=16, n_layers=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(n, d, generator=g)
    return {l: base * (1 + 0.5 * l) + 0.3 * l * torch.randn(n, d, generator=g) for l in range(n_layers)}


class PooledTests(unittest.TestCase):
    def test_records_match_direct_calls_and_share_caches(self):
        layers = pooled_layers()
        mets = [
            "effective_rank",
            "anisotropy",
            "alpha_req",
            "participation_ratio",
            "intrinsic_dimension/mlid",
            "intrinsic_dimension/gride",
            "neighborhood_curvature",
            "gaussianity",
            "sparsity",
        ]
        rec = rq.compute(
            layers,
            mets,
            model="toy",
            pooling="time-mean",
            params={"intrinsic_dimension/mlid": {"k": 32}, "intrinsic_dimension/gride": {"scale": 4}},
        )
        self.assertEqual(len(rec), 3 * len(mets))
        r = rec.where(metric="effective_rank", layer=2)[0]
        self.assertAlmostEqual(r.value, rq.effective_rank(layers[2]).value, places=9)
        self.assertEqual(r.preprocess, "center")
        self.assertEqual(r.depth, 1.0)
        self.assertEqual(r.model, "toy")
        self.assertAlmostEqual(
            rec.where(metric="anisotropy", layer=1)[0].value, rq.anisotropy_spectral(layers[1]).value, places=9
        )
        self.assertAlmostEqual(
            rec.where(metric="intrinsic_dimension/mlid", layer=0)[0].value, rq.mlid(layers[0], k=32).value, places=9
        )
        self.assertEqual(rec.where(metric="intrinsic_dimension/mlid", layer=0)[0].params, {"k": 32})
        self.assertAlmostEqual(
            rec.where(metric="intrinsic_dimension/gride", layer=0)[0].value,
            rq.gride(layers[0], scale=4).value,
            places=9,
        )
        self.assertAlmostEqual(
            rec.where(metric="neighborhood_curvature", layer=0)[0].value,
            rq.neighborhood_curvature(layers[0]).value,
            places=9,
        )
        self.assertEqual([l for l, _ in rec.profile("effective_rank")], [0, 1, 2])

    def test_subsetting_is_shared_across_layers_and_seeded(self):
        layers = pooled_layers(n=300)
        groups = [i // 3 for i in range(300)]  # 100 groups of 3 clips
        a = rq.compute(layers, ["effective_rank"], n=50, seed=7, group_ids=groups)
        b = rq.compute(layers, ["effective_rank"], n=50, seed=7, group_ids=groups)
        self.assertEqual(a[0].n_items, 50)
        self.assertEqual([r.value for r in a], [r.value for r in b])
        idx = rq.choose_indices(300, None, 1, groups)
        self.assertEqual(len(idx), 100)
        self.assertEqual(len({groups[i] for i in idx}), 100)

    def test_failures_are_recorded_not_raised(self):
        layers = {0: torch.randn(30, 4)}  # too few points for the alpha fit window
        rec = rq.compute(layers, ["alpha_req", "effective_rank"])
        bad = rec.where(metric="alpha_req")[0]
        self.assertTrue(math.isnan(bad.value))
        self.assertIn("error", bad.extras)
        self.assertFalse(math.isnan(rec.where(metric="effective_rank")[0].value))

    def test_mixed_kinds_rejected(self):
        with self.assertRaises(ValueError):
            rq.compute(pooled_layers(), ["effective_rank", "lidar"])


class FramesAndTokensTests(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(1)
        self.clips = {l: [torch.randn(60 + 5 * (i % 4), 8, generator=g).cumsum(0) for i in range(40)] for l in range(2)}

    def test_frames_aggregate_per_clip(self):
        rec = rq.compute(self.clips, ["trajectory_curvature"], population="frames", n=20, seed=0, keep_per_clip=True)
        r = rec.where(layer=0)[0]
        self.assertEqual(r.extras["n_items"], 20)
        self.assertEqual(len(r.extras["per_clip"]), 20)
        self.assertAlmostEqual(r.value, sum(r.extras["per_clip"]) / 20, places=9)
        self.assertAlmostEqual(r.value, math.pi / 2, delta=0.15)  # random walks
        rec2 = rq.compute(self.clips, ["effective_rank"], population="frames", n=20)
        self.assertEqual(rec2.where(layer=1)[0].population, "frames")

    def test_tokens_pool_all_frames(self):
        rec = rq.compute(self.clips, ["effective_rank"], population="tokens", n=1000)
        self.assertEqual(rec[0].n_items, 1000)
        with self.assertRaises(ValueError):
            rq.compute(self.clips, ["trajectory_curvature"], population="tokens")


class ViewsShiftsPairsTests(unittest.TestCase):
    def test_views_and_shifts(self):
        g = torch.Generator().manual_seed(2)
        base = torch.randn(300, 10, generator=g)
        vl = {l: base.unsqueeze(0) + 0.2 * torch.randn(3, 300, 10, generator=g) for l in range(2)}
        spec = rq.ViewSpec(source="shared", augmentations=("noise",), q=3, seed=2)
        rec = rq.compute(vl, ["lidar", "infonce"], views=spec, params={"infonce": {"temperature": 0.3}})
        self.assertEqual(len(rec), 4)
        self.assertEqual(rec[0].n_views, 3)
        self.assertIn("shared", rec[0].views)
        keys = torch.randint(0, 12, (400,), generator=g)
        w = torch.randn(12, 12, generator=g)
        emb = lambda kk: torch.nn.functional.one_hot(kk % 12, 12).float() @ w
        sl = {0: (emb(keys), {1: emb(keys + 1), 5: emb(keys + 5)})}
        rec = rq.compute(
            sl,
            ["pte"],
            shifts=rq.ShiftSpec("waveform pitch shift", (1, 5)),
            params={"pte": {"epochs": 5, "batch_size": 64}},
        )
        self.assertEqual(rec[0].n_views, 2)
        self.assertIn("pitch shift", rec[0].shifts)

    def test_pairs_matrix(self):
        layers = pooled_layers(n=200, d=6, n_layers=3)
        rec = rq.compute_pairs(layers, k=1, model="toy")
        self.assertEqual(len(rec), 9)
        diag = rec.where(layer=1, layer_b=1)[0]
        self.assertAlmostEqual(diag.value, 2.0 / 200, places=9)
        off = rec.where(layer=0, layer_b=2)[0]
        self.assertGreater(off.value, diag.value)


class RecordsIOTests(unittest.TestCase):
    def test_json_csv_roundtrip(self):
        rec = rq.compute(
            pooled_layers(n=120, d=6, n_layers=2), ["effective_rank", "anisotropy"], model="toy", corpus="synthetic"
        )
        with tempfile.TemporaryDirectory() as d:
            j = rec.to_json(Path(d) / "r.json")
            c = rec.to_csv(Path(d) / "r.csv")
            back = rq.Records.from_json(j)
            self.assertEqual(len(back), 4)
            self.assertEqual(back[0].metric, rec[0].metric)
            self.assertEqual(back[0].tags, rec[0].tags)
            self.assertTrue(c.read_text().startswith("model,layer,layer_b,depth,metric,value"))
        try:
            import pandas  # noqa: F401

            df = rec.to_pandas()
            self.assertIn("extra_entropy", df.columns)
        except ImportError:
            pass


class AtlasExportTests(unittest.TestCase):
    def test_site_record_format(self):
        layers = pooled_layers(n=150, d=8, n_layers=3)
        rec = rq.compute(
            layers,
            ["effective_rank", "anisotropy", "intrinsic_dimension/gride", "alpha_req"],
            params={"intrinsic_dimension/gride": {"scale": 4}},
            corpus="synthetic",
        )
        with tempfile.TemporaryDirectory() as d:
            p = rec.to_atlas_json(Path(d) / "metrics.json", model="toy", space="pooled embeddings")
            import json

            payload = json.loads(p.read_text())
        self.assertEqual(payload["model"], "toy")
        self.assertEqual(payload["schema_version"], 1)
        by = {(r["metric"], r["variant"]): r for r in payload["records"]}
        self.assertIn(("id", "gride_k4"), by)
        self.assertIn(("anisotropy", "spectral"), by)
        self.assertIn(("alpha", "default"), by)
        self.assertEqual(by[("effective_rank", "default")]["n_layers"], 3)
        self.assertEqual(
            by[("effective_rank", "default")]["layers"], [r.value for r in rec.where(metric="effective_rank")]
        )
        self.assertIn("paper-canonical", by[("anisotropy", "spectral")]["tags"])
        self.assertEqual(by[("alpha", "default")]["corpus"], "synthetic")


class ProtocolTests(unittest.TestCase):
    def test_kanatas2026_is_consistent_with_the_registry(self):
        p = rq.protocols.get("kanatas2026")
        for kind in ("pooled", "frames", "views", "shifted"):
            for name in p.names(kind):
                rq.get_metric(name)
        self.assertEqual(p.params("pooled")["intrinsic_dimension/gride"]["scale"], 8)
        rec = rq.compute(pooled_layers(n=150, d=8), ["effective_rank", "anisotropy"], params=p.params("pooled"))
        self.assertEqual(rec[0].params, {"spectrum": "singular", "center": True})


if __name__ == "__main__":
    unittest.main()
