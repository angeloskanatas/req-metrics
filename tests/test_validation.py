"""Input validation, failure records and the conveniences added before 0.1.0."""

import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch

import req_metrics as rq


class ValidationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(300, 8)

    def test_non_finite_input_is_recorded_not_raised(self):
        bad = self.x.clone()
        bad[3, 4] = float("nan")
        with self.assertWarns(RuntimeWarning):
            rec = rq.compute({0: bad, 1: self.x}, ["effective_rank", "intrinsic_dimension/twonn"])
        failed = rec.where(layer=0)
        self.assertTrue(all(r.value != r.value and "non-finite" in r.extras["error"] for r in failed))
        self.assertTrue(all(r.value == r.value for r in rec.where(layer=1)))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            rq.compute_pairs({0: bad, 1: self.x}, metrics=["cka"])

    def test_silent_nan_gets_an_error_key(self):
        with self.assertWarns(RuntimeWarning):
            rec = rq.compute({0: torch.zeros(200, 8)}, ["anisotropy/spectral", "participation_ratio", "effective_rank"])
        by = {r.metric: r for r in rec}
        self.assertIn("error", by["anisotropy/spectral"].extras)
        self.assertIn("zero covariance", by["participation_ratio"].extras["error"])
        self.assertEqual(by["effective_rank"].value, 0.0)  # documented convention, not a failure
        with self.assertRaisesRegex(ValueError, "zero matrix"):
            rq.svcca(torch.zeros(100, 8), torch.randn(100, 8))

    def test_partial_sample_failures_keep_a_finite_mean(self):
        toks = {0: [torch.randn(40, 16) for _ in range(5)] + [torch.randn(80, 16) for _ in range(5)]}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rec = rq.compute(toks, ["intrinsic_dimension/mlid"], level="sample")
        self.assertTrue(rec[0].value == rec[0].value)
        self.assertEqual(rec[0].extras["n_failed"], 5)
        self.assertIn("first_error", rec[0].extras)
        self.assertNotIn("error", rec[0].extras)
        self.assertTrue(any("failed samples" in str(m.message) for m in w))

    def test_compute_pairs_records_a_failing_metric_and_keeps_the_others(self):
        small = {0: torch.randn(20, 8), 1: torch.randn(20, 8)}
        with self.assertWarns(RuntimeWarning):
            rec = rq.compute_pairs(small, metrics=["neighborhood_overlap", "cka"])  # k = 30 > N - 1
        overlap, cka = rec.where(metric="neighborhood_overlap"), rec.where(metric="cka")
        self.assertEqual((len(overlap), len(cka)), (4, 4))
        self.assertTrue(all("k must be" in r.extras["error"] for r in overlap))
        self.assertTrue(all(r.value == r.value for r in cka))

    def test_argument_checks(self):
        with self.assertRaisesRegex(ValueError, "n_items"):
            rq.compute({0: self.x}, ["effective_rank"], n_items=1)
        with self.assertRaisesRegex(ValueError, "integers"):
            rq.compute({"a": self.x}, ["effective_rank"])
        with self.assertRaisesRegex(ValueError, "level='sample'"):
            rq.compute({0: [self.x[:5], self.x[:7]]}, ["effective_rank"])
        with self.assertRaisesRegex(ValueError, "sequence level"):
            rq.compute({0: self.x}, ["effective_rank"], level="sample")
        with self.assertRaisesRegex(ValueError, "same number of samples"):
            rq.compute({0: [self.x[:5]] * 3, 1: [self.x[:5]] * 2}, ["effective_rank"], level="sample")
        with self.assertRaisesRegex(ValueError, "views.q"):
            rq.compute({0: torch.randn(3, 100, 8)}, ["lidar"], views=rq.ViewSpec("shared", q=10))
        with self.assertRaisesRegex(ValueError, "rows"):
            rq.compute({0: (self.x, {1: self.x[:100]})}, ["pte"], params={"pte": {"epochs": 1}})
        with self.assertRaisesRegex(ValueError, "call it directly"):
            rq.compute({0: [self.x]}, ["token_gram_drift"], level="sample")
        with self.assertRaisesRegex(ValueError, "not a pair metric"):
            rq.compute_pairs({0: self.x}, metrics=["token_gram_drift"])

    def test_integer_inputs_and_numpy_convergence(self):
        ints = torch.randint(0, 5, (300, 8))
        self.assertAlmostEqual(
            rq.compute({0: ints}, ["effective_rank"])[0].value, rq.effective_rank(ints.double()).value, places=9
        )
        conv = rq.convergence(np.random.default_rng(0).standard_normal((300, 16)), "effective_rank", repeats=2)
        self.assertEqual(len(conv.rows), 4)

    def test_pairs_label(self):
        y = self.x + 0.1
        self.assertEqual(rq.compute_pairs({0: self.x}, {0: y}, metrics=["cka"], model_b="text")[0].model, "A->text")
        self.assertEqual(rq.compute_pairs({0: self.x}, {0: y}, metrics=["cka"], model="audio")[0].model, "audio->B")
        self.assertEqual(rq.compute_pairs({0: self.x}, metrics=["cka"], model="m")[0].model, "m")


class ConvenienceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.rec = rq.compute(
            {0: torch.randn(300, 8), 2: torch.randn(300, 8)}, ["effective_rank", "anisotropy/spectral"]
        )

    def test_records_repr_slice_where_markdown(self):
        self.assertEqual(repr(self.rec), "Records(4 rows, 2 metrics, layers 0-2, models ['None'])")
        self.assertIsInstance(self.rec[0:2], rq.Records)
        self.assertEqual(len(self.rec[0:2]), 2)
        with self.assertRaisesRegex(ValueError, "unknown record fields"):
            self.rec.where(metrik="effective_rank")
        table = self.rec.to_markdown().splitlines()
        self.assertEqual(table[0], "| layer | anisotropy/spectral | effective_rank |")
        self.assertEqual(len(table), 4)
        self.assertEqual(
            rq.top_layers(self.rec, "effective_rank", k=1, model=None), rq.top_layers(self.rec, "effective_rank", k=1)
        )

    def test_json_writes_null_for_nan_and_reads_it_back(self):
        rec = rq.compute({0: torch.zeros(100, 8)}, ["anisotropy/spectral"])
        with tempfile.TemporaryDirectory() as d:
            p = rec.to_json(Path(d) / "r.json")
            text = p.read_text()
            self.assertNotIn("NaN", text)
            json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
            back = rq.Records.from_json(p)
            self.assertTrue(back[0].value != back[0].value)
            rec.to_csv(Path(d) / "r.csv")
            self.assertNotIn("NaN", (Path(d) / "r.csv").read_text())

    def test_describe_and_lookup_hints(self):
        text = rq.describe("effective_rank")
        self.assertTrue(text.startswith("effective_rank\n"))
        self.assertIn("Roy and Vetterli", text)
        with self.assertRaisesRegex(KeyError, "did you mean.*intrinsic_dimension/twonn"):
            rq.get_metric("twonn")
        with self.assertRaisesRegex(KeyError, "did you mean.*effective_rank"):
            rq.get_metric("effective_rnk")
        with self.assertRaisesRegex(KeyError, "its estimators are"):
            rq.get_metric("anisotropy")
        self.assertIs(rq.get_metric("token_gram_drift").inputs, rq.InputKind.TOKEN_PAIR)


if __name__ == "__main__":
    unittest.main()
