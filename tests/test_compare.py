"""Information imbalance between two representations of the same items."""

import unittest

import torch

import req_metrics as rq
from req_metrics.neighbors import Neighbors


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

    def test_l2_ranks_cosine_neighbors(self):
        y = self.x * torch.rand(800, 1).add(0.5)  # same directions, different norms
        self.assertLess(rq.neighborhood_overlap(self.x, y, k=10).value, 1.0)
        self.assertAlmostEqual(rq.neighborhood_overlap(self.x, y, k=10, l2=True).value, 1.0, places=12)
        self.assertAlmostEqual(rq.information_imbalance(self.x, y, l2=True).value, 2.0 / 800, places=9)
        rec = rq.compute_pairs({0: self.x}, {0: y}, metric="neighborhood_overlap", k=10, params={"l2": True})
        self.assertAlmostEqual(rec[0].value, 1.0, places=12)
        self.assertEqual((rec[0].preprocess, rec[0].params), ("l2", {"k": 10, "l2": True}))
        keyed = rq.compute_pairs(
            {0: self.x}, {0: y}, metric="neighborhood_overlap", k=10, params={"neighborhood_overlap": {"l2": True}}
        )
        self.assertEqual(keyed[0].params, rec[0].params)

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


class CycleKnnTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(800, 8)

    def test_six_point_example_of_zhang_et_al(self):
        # Zhang et al. (2026), App. A: k = 2, cycle-kNN(X -> Y) = 5/6 and cycle-kNN(Y -> X) = 1/2.
        x = torch.tensor([15.0, 26.0, 49.0, 60.0, 87.0, 90.0]).view(-1, 1)
        y = torch.tensor([34.0, 56.0, 58.0, 57.0, 63.0, 37.0]).view(-1, 1)
        res = rq.cycle_knn(x, y, k=2)
        self.assertAlmostEqual(res.value, 5 / 6, places=12)
        self.assertAlmostEqual(res.extras["reverse"], 1 / 2, places=12)
        self.assertAlmostEqual(rq.cycle_knn(y, x, k=2).value, 1 / 2, places=12)

    def test_k_one_is_symmetric_and_identical_spaces_count_reciprocal_neighbors(self):
        y = torch.randn(800, 5)
        a, b = rq.cycle_knn(self.x, y, k=1), rq.cycle_knn(y, self.x, k=1)
        self.assertAlmostEqual(a.value, b.value, places=12)  # Zhang et al. (2026), App. A, Proposition
        nn = Neighbors.from_points(self.x.double(), 10).indices[:, 1:11]
        reciprocal = torch.tensor([bool((nn[nn[i]] == i).any()) for i in range(800)]).double().mean()
        self.assertAlmostEqual(rq.cycle_knn(self.x, self.x, k=10).value, float(reciprocal), places=12)
        self.assertLess(float(reciprocal), 1.0)

    def test_matches_the_reference_implementation_on_cosine_neighbors(self):
        y = self.x @ torch.randn(8, 12) + 0.5 * torch.randn(800, 12)

        def knn(f, k):  # Huh et al. (2024), metrics.py: inner-product neighbors of unit-norm rows, self excluded
            f = torch.nn.functional.normalize(f.double(), dim=1)
            return (f @ f.T).fill_diagonal_(-1e8).argsort(dim=1, descending=True)[:, :k]

        ka, kb = knn(self.x, 10), knn(y, 10)
        ref = (ka[kb] == torch.arange(800).view(-1, 1, 1)).flatten(1).any(1).double().mean()
        self.assertAlmostEqual(rq.cycle_knn(self.x, y, k=10, l2=True).value, float(ref), places=12)

    def test_independent_spaces_stay_near_the_chance_bound(self):
        res = rq.cycle_knn(torch.randn(2000, 6), torch.randn(2000, 6), k=10)
        self.assertEqual(res.extras["chance_bound"], 100 / 1999)
        self.assertLess(res.value, 1.5 * res.extras["chance_bound"])
        self.assertGreater(res.value, 0.4 * res.extras["chance_bound"])

    def test_compute_pairs_cycle(self):
        layers = {0: self.x, 1: self.x + 0.5 * torch.randn(800, 8), 2: torch.randn(800, 8)}
        rec = rq.compute_pairs(layers, metric="cycle_knn")
        self.assertEqual(len(rec), 9)
        self.assertEqual(rec[0].params, {"k": 10, "l2": False})
        by = {(r.layer, r.layer_b): r for r in rec}
        for (a, b), r in by.items():
            self.assertAlmostEqual(r.extras["reverse"], by[(b, a)].value, places=12)
            self.assertAlmostEqual(r.value, rq.cycle_knn(layers[a], layers[b], k=10).value, places=12)
        self.assertGreater(by[(0, 1)].value, by[(0, 2)].value)
        self.assertEqual(by[(0, 1)].extras["chance_bound"], 100 / 799)


if __name__ == "__main__":
    unittest.main()
