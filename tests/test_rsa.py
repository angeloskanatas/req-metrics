"""RSA: Spearman correlation of pairwise-distance vectors, against scipy and under isometries."""

import unittest

import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

import req_metrics as rq


class RSATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(300, 8)

    def test_matches_scipy_for_every_distance_and_method(self):
        y = self.x[:, :5] + 0.2 * torch.randn(300, 5)
        for distance in ("cosine", "euclidean", "correlation"):
            a, b = pdist(self.x.numpy(), distance), pdist(y.numpy(), distance)
            self.assertAlmostEqual(rq.rsa(self.x, y, distance=distance).value, spearmanr(a, b).statistic, places=10)
            self.assertAlmostEqual(
                rq.rsa(self.x, y, distance=distance, method="pearson", chunk=64).value,
                np.corrcoef(a, b)[0, 1],
                places=10,
            )

    def test_isometries_give_one_and_independence_zero(self):
        q = torch.linalg.qr(torch.randn(8, 8)).Q
        self.assertAlmostEqual(rq.rsa(self.x, self.x.clone()).value, 1.0, places=12)
        self.assertAlmostEqual(rq.rsa(self.x, 3.0 * self.x @ q).value, 1.0, places=8)
        self.assertAlmostEqual(rq.rsa(self.x, 3.0 * self.x @ q + 1.0, distance="euclidean").value, 1.0, places=8)
        self.assertLess(abs(rq.rsa(self.x, torch.randn(300, 8)).value), 0.03)
        self.assertEqual(rq.rsa(self.x, self.x).extras["n_pairs"], 300 * 299 / 2)

    def test_ties_share_their_mean_rank(self):
        from req_metrics.metrics.compare import _average_ranks

        self.assertEqual(_average_ranks(torch.tensor([3.0, 1.0, 3.0, 2.0])).tolist(), [3.5, 1.0, 3.5, 2.0])

    def test_compute_pairs_map_and_item_cap(self):
        layers = {0: self.x, 1: self.x + 0.5 * torch.randn(300, 8), 2: torch.randn(300, 8)}
        rec = rq.compute_pairs(layers, metrics=["rsa"])
        self.assertEqual(len(rec), 9)
        by = {(r.layer, r.layer_b): r for r in rec}
        self.assertAlmostEqual(by[(0, 0)].value, 1.0, places=6)
        self.assertAlmostEqual(by[(0, 1)].value, by[(1, 0)].value, places=12)
        self.assertAlmostEqual(by[(0, 1)].value, rq.rsa(layers[0], layers[1]).value, places=5)
        self.assertGreater(by[(0, 1)].value, by[(0, 2)].value)
        self.assertEqual(rq.get_metric("rsa").inputs, rq.InputKind.PAIR)
        big = {0: torch.randn(4500, 4), 1: torch.randn(4500, 4)}
        self.assertEqual(rq.compute_pairs(big, metrics=["rsa"])[0].n_items, 4000)
        eu = rq.compute_pairs(layers, metrics=["rsa"], params={"rsa": {"distance": "euclidean", "method": "pearson"}})
        self.assertEqual(eu[0].params, {"distance": "euclidean", "method": "pearson"})


if __name__ == "__main__":
    unittest.main()
