"""Information imbalance between two representations of the same items."""

import unittest

import torch

import req_metrics as rq


class InformationImbalanceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(1200, 6)

    def test_identical_spaces_give_two_over_n(self):
        r = rq.information_imbalance(self.x, self.x.clone())
        self.assertAlmostEqual(r.value, 2.0 / 1200, places=9)
        self.assertAlmostEqual(r.extras["reverse"], 2.0 / 1200, places=9)

    def test_independent_spaces_near_one(self):
        y = torch.randn(1200, 6)
        r = rq.information_imbalance(self.x, y)
        self.assertAlmostEqual(r.value, 1.0, delta=0.08)
        self.assertAlmostEqual(r.extras["reverse"], 1.0, delta=0.08)

    def test_feature_subset_is_contained(self):
        full, part = self.x, self.x[:, :3]
        r = rq.information_imbalance(full, part)  # full -> part: full knows part's neighbors reasonably
        self.assertLess(r.value, r.extras["reverse"])  # part -> full: part misses three coordinates
        self.assertLess(r.value, 0.5)

    def test_shared_neighbor_tables_and_k(self):
        y = self.x @ torch.randn(6, 6)  # invertible linear map, same items
        nb_a = rq.Neighbors.from_points(self.x, 5)
        nb_b = rq.Neighbors.from_points(y, 5)
        direct = rq.information_imbalance(self.x, y, k=3)
        cached = rq.information_imbalance(self.x, y, k=3, neighbors_a=nb_a, neighbors_b=nb_b)
        self.assertAlmostEqual(direct.value, cached.value, places=12)
        self.assertEqual(rq.get_metric("information_imbalance").inputs, rq.InputKind.PAIR)


class NeighborhoodOverlapTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(800, 8)

    def test_identical_and_isometric_spaces_give_one(self):
        self.assertAlmostEqual(rq.neighborhood_overlap(self.x, self.x.clone(), k=10).value, 1.0, places=12)
        q = torch.linalg.qr(torch.randn(8, 8)).Q
        self.assertAlmostEqual(rq.neighborhood_overlap(self.x, 3.0 * self.x @ q + 1.0, k=10).value, 1.0, places=6)

    def test_independent_spaces_at_chance(self):
        r = rq.neighborhood_overlap(self.x, torch.randn(800, 8), k=30)
        self.assertAlmostEqual(r.value, r.extras["chance"], delta=0.01)
        self.assertAlmostEqual(r.extras["chance"], 30 / 799)

    def test_matches_direct_set_intersection_and_shared_tables(self):
        y = self.x[:, :4] + 0.3 * torch.randn(800, 4)
        r = rq.neighborhood_overlap(self.x, y, k=15)
        ia = rq.Neighbors.from_points(self.x.double(), 15).indices[:, 1:16]
        ib = rq.Neighbors.from_points(y.double(), 15).indices[:, 1:16]
        direct = sum(len(set(ia[i].tolist()) & set(ib[i].tolist())) for i in range(800)) / (800 * 15)
        self.assertAlmostEqual(r.value, direct, places=12)
        nb_a, nb_b = rq.Neighbors.from_points(self.x.double(), 20), rq.Neighbors.from_points(y.double(), 20)
        self.assertAlmostEqual(
            rq.neighborhood_overlap(self.x, y, k=15, neighbors_a=nb_a, neighbors_b=nb_b).value, direct, places=12
        )
        self.assertEqual(rq.get_metric("neighborhood_overlap").inputs, rq.InputKind.PAIR)

    def test_compute_pairs_overlap(self):
        layers = {0: self.x, 1: self.x + 0.5 * torch.randn(800, 8), 2: torch.randn(800, 8)}
        rec = rq.compute_pairs(layers, metric="neighborhood_overlap", k=10)
        self.assertEqual(len(rec), 9)
        diag = {r.layer: r.value for r in rec if r.layer == r.layer_b}
        self.assertTrue(all(abs(v - 1.0) < 1e-12 for v in diag.values()))
        v01 = [r.value for r in rec if (r.layer, r.layer_b) == (0, 1)][0]
        v10 = [r.value for r in rec if (r.layer, r.layer_b) == (1, 0)][0]
        v02 = [r.value for r in rec if (r.layer, r.layer_b) == (0, 2)][0]
        self.assertAlmostEqual(v01, v10, places=12)
        self.assertGreater(v01, v02)
        self.assertEqual(rec[0].params["k"], 10)
        ii = rq.compute_pairs(layers)  # default metric and k unchanged
        self.assertEqual(ii[0].metric, "information_imbalance")
        self.assertEqual(ii[0].params["k"], 1)


if __name__ == "__main__":
    unittest.main()
