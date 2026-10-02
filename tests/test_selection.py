"""Selection rules: across runs at one layer and across layers within a run."""

import math
import unittest

import torch

import req_metrics as rq


class SelectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        full = torch.randn(300, 32)
        low = torch.randn(300, 32) @ torch.diag(torch.linspace(1, 0.02, 32))
        self.runs = {
            "wide": rq.compute({0: full, 1: full}, ["effective_rank", "alpha_req"]),
            "narrow": rq.compute({0: low, 1: low}, ["effective_rank", "alpha_req"]),
        }

    def test_rank_runs_highest_first(self):
        order = rq.rank_runs(self.runs, "effective_rank", layer=1)
        self.assertEqual([n for n, _ in order], ["wide", "narrow"])
        self.assertEqual(
            [n for n, _ in rq.rank_runs(self.runs, "effective_rank", layer=1, direction="min")], ["narrow", "wide"]
        )

    def test_rank_runs_target_and_missing(self):
        a = rq.value_at(self.runs["wide"], "alpha_req", 1)
        order = rq.rank_runs(self.runs, "alpha_req", layer=1, direction="target", target=a)
        self.assertEqual(order[0][0], "wide")
        order = rq.rank_runs(self.runs, "mlid", layer=1)  # not computed: every run sorts with nan last
        self.assertTrue(all(math.isnan(v) for _, v in order))
        with self.assertRaises(ValueError):
            rq.rank_runs(self.runs, "alpha_req", layer=1, direction="target")

    def test_top_layers(self):
        layers = {i: torch.randn(200, 16) @ torch.diag(torch.linspace(1, 0.05 + 0.2 * i, 16)) for i in range(4)}
        rec = rq.compute(layers, ["effective_rank"])
        prof = dict(rec.profile("effective_rank"))
        best = max(prof, key=prof.get)
        self.assertEqual(rq.top_layers(rec, "effective_rank", k=1), [best])
        self.assertEqual(len(rq.top_layers(rec, "effective_rank", k=3)), 3)
        self.assertEqual(rq.top_layers(rec, "effective_rank", k=1, direction="min"), [min(prof, key=prof.get)])


if __name__ == "__main__":
    unittest.main()
