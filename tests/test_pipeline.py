"""compute() over synthetic layers: caching equivalence, levels, subsetting, records I/O, protocol."""

import inspect
import json
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

    def test_preprocessing_arguments_reach_cached_estimators(self):
        x = pooled_layers()[1] + 3.0
        cases = [
            ("anisotropy", {"l2": False}, rq.anisotropy_spectral(x, l2=False), "center"),
            ("effective_rank", {"center": False}, rq.effective_rank(x, center=False), "none"),
            ("matrix_entropy", {"center": True}, rq.matrix_entropy(x, center=True), "center"),
            ("gaussianity", {"center": False}, rq.gaussianity(x, center=False), "none"),
            ("participation_ratio", {}, rq.participation_ratio(x), "center"),
        ]
        for name, kw, direct, pre in cases:
            r = rq.compute({0: x}, [name], params={name: kw})[0]
            self.assertAlmostEqual(r.value, direct.value, places=9, msg=name)
            self.assertEqual((r.preprocess, r.params), (pre, kw), msg=name)
        both = rq.compute({0: x}, ["effective_rank", "anisotropy"], params={"anisotropy": {"l2": False}})
        self.assertAlmostEqual(both[0].value, rq.effective_rank(x).value, places=9)
        self.assertAlmostEqual(both[1].value, cases[0][2].value, places=9)

    def test_registry_preprocessing_matches_estimator_defaults(self):
        for name in rq.list_metrics():
            spec = rq.get_metric(name)
            accepted = inspect.signature(spec.fn).parameters
            for f in ("center", "standardize", "l2"):
                if f in accepted:
                    self.assertEqual(accepted[f].default, getattr(spec.preprocess, f), msg=f"{name}.{f}")

    def test_subsetting_is_shared_across_layers_and_seeded(self):
        layers = pooled_layers(n=300)
        groups = [i // 3 for i in range(300)]  # 100 groups of 3 clips
        a = rq.compute(layers, ["effective_rank"], n_items=50, seed=7, group_ids=groups)
        b = rq.compute(layers, ["effective_rank"], n_items=50, seed=7, group_ids=groups)
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


class SharedNeighborTableTests(unittest.TestCase):
    def test_twonn_reads_the_shared_table(self):
        x = pooled_layers(n=300, d=10, n_layers=1)[0]
        x = torch.cat([x, x[:5]])  # duplicates are dropped before the table is built
        rec = rq.compute({0: x}, ["intrinsic_dimension/twonn", "intrinsic_dimension/gride", "intrinsic_dimension/mlid"])
        direct = rq.twonn(x)
        self.assertEqual(rec[0].value, direct.value)
        self.assertEqual(rec[0].extras["n_distinct"], 300)
        self.assertEqual(rec[0].extras["n_used"], direct.extras["n_used"])


class FramesAndTokensTests(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(1)
        self.clips = {l: [torch.randn(60 + 5 * (i % 4), 8, generator=g).cumsum(0) for i in range(40)] for l in range(2)}

    def test_sample_level_aggregates_per_sample(self):
        rec = rq.compute(self.clips, ["trajectory_curvature"], level="sample", n_items=20, seed=0, keep_per_sample=True)
        r = rec.where(layer=0)[0]
        self.assertEqual(r.extras["n_items"], 20)
        self.assertEqual(len(r.extras["per_sample"]), 20)
        self.assertAlmostEqual(r.value, sum(r.extras["per_sample"]) / 20, places=9)
        self.assertAlmostEqual(r.value, math.pi / 2, delta=0.15)  # random walks
        rec2 = rq.compute(self.clips, ["effective_rank"], level="sample", n_items=20)
        self.assertEqual(rec2.where(layer=1)[0].level, "sample")

    def test_population_level_pools_all_tokens(self):
        rec = rq.compute(self.clips, ["effective_rank"], level="population", n_tokens=1000)
        self.assertEqual(rec[0].n_items, 1000)
        with self.assertRaises(ValueError):
            rq.compute(self.clips, ["trajectory_curvature"], level="population")

    def test_population_records_carry_the_neighbor_regime(self):
        g = torch.Generator().manual_seed(5)
        centers = 20 * torch.randn(30, 6, generator=g)
        tight = {0: [c + 0.01 * torch.randn(100, 6, generator=g) for c in centers]}
        mets = ["intrinsic_dimension/twonn", "intrinsic_dimension/mle", "effective_rank"]
        by = {r.metric: r for r in rq.compute(tight, mets, level="population", n_tokens=3000)}
        self.assertEqual(by["effective_rank"].extras["n_samples"], 30)
        self.assertNotIn("same_sample_fraction", by["effective_rank"].extras)
        self.assertEqual(by["intrinsic_dimension/twonn"].extras["same_sample_fraction"], 1.0)  # k = 2 < 100 per sample
        self.assertEqual(by["intrinsic_dimension/mle"].extras["same_sample_fraction"], 1.0)  # k = 20 < 100
        single = {0: [c.unsqueeze(0) for c in centers]}  # one frame per clip: every neighbor is another clip
        rec = rq.compute(single, ["intrinsic_dimension/twonn"], level="population")
        self.assertEqual(rec[0].extras["same_sample_fraction"], 0.0)

    def test_device_argument_gives_the_same_records(self):
        mets = ["effective_rank", "intrinsic_dimension/twonn"]
        for level, cap in (("sample", {"n_items": 10}), ("population", {"n_tokens": 500})):
            base = rq.compute(self.clips, mets, level=level, seed=3, **cap)
            moved = rq.compute(self.clips, mets, level=level, seed=3, device="cpu", **cap)
            self.assertEqual([r.value for r in base], [r.value for r in moved], msg=level)


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
        rec = rq.compute_pairs(layers, metrics=["information_imbalance"], model="toy")  # k defaults to 1
        self.assertEqual(len(rec), 9)
        diag = rec.where(layer=1, layer_b=1)[0]
        self.assertAlmostEqual(diag.value, 2.0 / 200, places=9)
        off = rec.where(layer=0, layer_b=2)[0]
        self.assertGreater(off.value, diag.value)

    def test_pairs_several_metrics_share_one_subsample(self):
        layers = pooled_layers(n=200, d=6, n_layers=3)
        mets = ["information_imbalance", "neighborhood_overlap", "cycle_knn", "cka"]
        p = {"neighborhood_overlap": {"k": 10}}
        rec = rq.compute_pairs(layers, metrics=mets, n_items=120, seed=5, params=p)
        self.assertEqual(len(rec), 36)
        self.assertEqual({r.n_items for r in rec}, {120})
        for m in mets:
            alone = rq.compute_pairs(layers, metrics=[m], n_items=120, seed=5, params=p)
            self.assertEqual([r.value for r in rec.where(metric=m)], [r.value for r in alone], msg=m)
        self.assertEqual(rec.where(metric="neighborhood_overlap")[0].params, {"k": 10, "l2": False, "jaccard": False})
        self.assertEqual(rec.where(metric="cycle_knn")[0].params, {"k": 10, "l2": False})
        self.assertEqual(rec.where(metric="cka")[0].params, {})
        with self.assertRaisesRegex(ValueError, "not a pair metric"):
            rq.compute_pairs(layers, metrics=["effective_rank"])
        with self.assertRaisesRegex(ValueError, "population"):
            rq.compute(layers, ["effective_rank"], n_tokens=100)

    def test_imbalance_within_one_model_matches_two_models(self):
        layers = pooled_layers(n=150, d=6, n_layers=3)
        copy = {l: x.clone() for l, x in layers.items()}  # same values, so B is treated as a second model
        p = {"information_imbalance": {"k": 3}}
        same = rq.compute_pairs(layers, metrics=["information_imbalance"], params=p)
        two = rq.compute_pairs(layers, copy, metrics=["information_imbalance"], params=p, device="cpu")
        for a, b in zip(same, two, strict=True):
            self.assertEqual((a.layer, a.layer_b), (b.layer, b.layer_b))
            self.assertAlmostEqual(a.value, b.value, places=12)
            self.assertAlmostEqual(a.extras["reverse"], b.extras["reverse"], places=12)

    def test_device_argument_for_views_and_pairs(self):
        g = torch.Generator().manual_seed(4)
        base = torch.randn(200, 8, generator=g)
        vl = {l: base.unsqueeze(0) + 0.3 * torch.randn(3, 200, 8, generator=g) for l in range(2)}
        a = rq.compute(vl, ["lidar", "infonce"])
        b = rq.compute(vl, ["lidar", "infonce"], device=torch.device("cpu"))
        self.assertEqual([r.value for r in a], [r.value for r in b])
        layers = pooled_layers(n=120, d=6, n_layers=2)
        for metric in ("cka", "neighborhood_overlap"):
            x = rq.compute_pairs(layers, metrics=[metric])
            y = rq.compute_pairs(layers, metrics=[metric], device="cpu")
            self.assertEqual([r.value for r in x], [r.value for r in y], msg=metric)


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
            rows = json.loads(j.read_text())  # development versions stored a population field
            for row, old in zip(rows, ("pooled", "frames", "tokens", "pooled"), strict=True):
                row["population"] = old
                del row["level"]
            rows[0]["metric"] = "intrinsic_dimension"
            j.write_text(json.dumps(rows))
            old_rows = rq.Records.from_json(j)
            self.assertEqual([r.level for r in old_rows], ["sequence", "sample", "population", "sequence"])
            self.assertEqual(old_rows[0].metric, "intrinsic_dimension/twonn")
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
        for kind in ("sequence", "sample", "views", "shifted"):
            for name in p.names(kind):
                rq.get_metric(name)
        self.assertEqual(p.params("sequence")["intrinsic_dimension/gride"]["scale"], 8)
        rec = rq.compute(pooled_layers(n=150, d=8), ["effective_rank", "anisotropy"], params=p.params("sequence"))
        self.assertEqual(rec[0].params, {"spectrum": "singular", "center": True, "max_eigenvalues": 2048})


if __name__ == "__main__":
    unittest.main()
