"""Information imbalance between two representations of the same items."""

import math
import unittest

import numpy as np
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
        small = rq.information_imbalance(self.x, y, k=3, neighbors_a=rq.Neighbors.from_points(self.x, 3))
        self.assertAlmostEqual(direct.value, small.value, places=12)  # a table without the extra neighbor is rebuilt
        self.assertEqual(rq.get_metric("information_imbalance").inputs, rq.InputKind.PAIR)

    def test_without_ties_equals_the_strict_count(self):
        y = torch.tanh(self.x @ torch.randn(6, 4)) + 0.3 * torch.randn(1200, 4)
        for k in (1, 3):
            ours = rq.information_imbalance(self.x, y, k=k)
            self.assertAlmostEqual(ours.value, _strict_delta(_dist(self.x), _dist(y), k), places=12)
            self.assertAlmostEqual(ours.extras["reverse"], _strict_delta(_dist(y), _dist(self.x), k), places=12)

    def test_row_ranks_average_ties_like_rankdata(self):
        from scipy.stats import rankdata

        from req_metrics.metrics.compare import _row_ranks

        d = torch.tensor([[float("inf"), 1.0, 2.0, 1.0, 0.5], [1.0, float("inf"), 1.0, 1.0, 3.0]], dtype=torch.float64)
        ref = torch.tensor(rankdata(d.numpy(), method="average", axis=1))
        self.assertTrue(torch.equal(_row_ranks(d), ref))

    def test_ties_equal_the_expectation_over_random_tie_breaking(self):
        g = torch.Generator().manual_seed(1)
        a = torch.randint(0, 3, (40, 2), generator=g).double()  # a 3 x 3 grid: duplicates and equal distances
        b = (torch.rand(40, 6, generator=g) < 0.3).double()  # binary rows, as a piano roll
        d_a, d_b = _dist(a), _dist(b)
        for k in (1, 3):
            ours = rq.information_imbalance(a, b, k=k)
            fwd = torch.tensor([_strict_delta(_jitter(d_a, g), _jitter(d_b, g), k) for _ in range(4000)])
            back = torch.tensor([_strict_delta(_jitter(d_b, g), _jitter(d_a, g), k) for _ in range(4000)])
            self.assertLess(abs(ours.value - float(fwd.mean())), 4 * float(fwd.std()) / 4000**0.5 + 1e-4)
            self.assertLess(abs(ours.extras["reverse"] - float(back.mean())), 4 * float(back.std()) / 4000**0.5 + 1e-4)

    def test_compute_pairs_matches_the_direct_metric_with_ties(self):
        g = torch.Generator().manual_seed(2)
        layers = {
            0: torch.randint(0, 3, (60, 2), generator=g).double(),
            1: (torch.rand(60, 5, generator=g) < 0.4).double(),
        }
        rec = rq.compute_pairs(layers, metrics=["information_imbalance"], params={"information_imbalance": {"k": 2}})
        for r in rec.rows:
            direct = rq.information_imbalance(layers[r.layer], layers[r.layer_b], k=2)
            self.assertAlmostEqual(r.value, direct.value, places=12)
            self.assertAlmostEqual(r.extras["reverse"], direct.extras["reverse"], places=12)
            self.assertAlmostEqual(r.extras["asymmetry"], direct.extras["asymmetry"], places=12)


def _dist(x: torch.Tensor) -> torch.Tensor:
    d = torch.cdist(x.double(), x.double())
    return d.fill_diagonal_(float("inf"))


def _overlap(d_a: torch.Tensor, d_b: torch.Tensor, k: int) -> float:
    """Mean shared fraction of the k nearest neighbors on full distance tables without ties."""
    nn_a = torch.topk(d_a, k, dim=1, largest=False).indices
    nn_b = torch.topk(d_b, k, dim=1, largest=False).indices
    shared = (nn_a.unsqueeze(2) == nn_b.unsqueeze(1)).any(dim=2).sum(dim=1)
    return float(shared.double().mean()) / k


def _strict_delta(d_a: torch.Tensor, d_b: torch.Tensor, k: int) -> float:
    """Glielmo et al.'s count on full distance tables without ties: ranks are one plus the items strictly closer."""
    nn = torch.topk(d_a, k, dim=1, largest=False).indices
    target = torch.gather(d_b, 1, nn)
    ranks = (d_b.unsqueeze(1) < target.unsqueeze(2)).sum(dim=2) + 1
    return float(ranks.double().mean()) / (d_a.shape[0] / 2.0)


def _jitter(d: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Ties broken at random: noise far below the smallest gap between distinct distances of the toy data."""
    return d + 1e-7 * torch.rand(d.shape, generator=g, dtype=d.dtype)


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
        rec = rq.compute_pairs(
            {0: self.x},
            {0: y},
            metrics=["neighborhood_overlap"],
            params={"neighborhood_overlap": {"k": 10, "l2": True}},
        )
        self.assertAlmostEqual(rec[0].value, 1.0, places=12)
        self.assertEqual((rec[0].preprocess, rec[0].params), ("l2", {"k": 10, "l2": True, "jaccard": False}))

    def test_jaccard_normalization_and_its_chance_level(self):
        from scipy.stats import hypergeom

        y = self.x[:, :4] + 0.3 * torch.randn(800, 4)
        r = rq.neighborhood_overlap(self.x, y, k=10, jaccard=True)
        ia = rq.Neighbors.from_points(self.x.double(), 10).indices[:, 1:11]
        ib = rq.Neighbors.from_points(y.double(), 10).indices[:, 1:11]
        sets = [(set(ia[i].tolist()), set(ib[i].tolist())) for i in range(800)]
        direct = sum(len(a & b) / len(a | b) for a, b in sets) / 800
        self.assertAlmostEqual(r.value, direct, places=12)
        self.assertLess(r.value, rq.neighborhood_overlap(self.x, y, k=10).value)  # union >= k per point
        self.assertAlmostEqual(rq.neighborhood_overlap(self.x, self.x, k=10, jaccard=True).value, 1.0, places=12)
        m = torch.arange(11).double()
        pmf = torch.tensor(hypergeom(799, 10, 10).pmf(m.numpy()))
        self.assertAlmostEqual(r.extras["chance"], float((pmf * m / (20 - m)).sum()), places=12)
        self.assertAlmostEqual(rq.neighborhood_overlap(self.x, y, k=10).extras["chance"], 10 / 799, places=15)
        # ReSi's jaccard_similarity (repsim/measures/nearest_neighbor.py): k + 1 cosine neighbors, self dropped,
        # per-item |intersection| / |union|, mean over items.
        from scipy.spatial.distance import cdist

        def resi_sets(f):
            order = cdist(f.numpy(), f.numpy(), metric="cosine").argsort(axis=1)[:, 1:11]
            return [set(row.tolist()) for row in order]

        resi = sum(len(a & b) / len(a | b) for a, b in zip(resi_sets(self.x), resi_sets(y), strict=True)) / 800
        self.assertAlmostEqual(rq.neighborhood_overlap(self.x, y, k=10, l2=True, jaccard=True).value, resi, places=12)
        rec = rq.compute_pairs(
            {0: self.x},
            {0: y},
            metrics=["neighborhood_overlap"],
            params={"neighborhood_overlap": {"k": 10, "jaccard": True}},
        )
        self.assertAlmostEqual(rec[0].value, direct, places=12)
        self.assertEqual(rec[0].params, {"k": 10, "l2": False, "jaccard": True})
        self.assertAlmostEqual(rec[0].extras["chance"], r.extras["chance"], places=15)

    def test_compute_pairs_overlap(self):
        layers = {0: self.x, 1: self.x + 0.5 * torch.randn(800, 8), 2: torch.randn(800, 8)}
        rec = rq.compute_pairs(layers, metrics=["neighborhood_overlap"], params={"neighborhood_overlap": {"k": 10}})
        self.assertEqual(len(rec), 9)
        diag = {r.layer: r.value for r in rec if r.layer == r.layer_b}
        self.assertTrue(all(abs(v - 1.0) < 1e-12 for v in diag.values()))
        v01 = [r.value for r in rec if (r.layer, r.layer_b) == (0, 1)][0]
        v10 = [r.value for r in rec if (r.layer, r.layer_b) == (1, 0)][0]
        v02 = [r.value for r in rec if (r.layer, r.layer_b) == (0, 2)][0]
        self.assertAlmostEqual(v01, v10, places=12)
        self.assertGreater(v01, v02)
        self.assertEqual(rec[0].params["k"], 10)
        ii = rq.compute_pairs(layers, metrics=["information_imbalance"])  # k defaults to the estimator's 1
        self.assertEqual(ii[0].metric, "information_imbalance")
        self.assertEqual(ii[0].params["k"], 1)

    def test_ties_equal_the_expectation_over_random_tie_breaking(self):
        g = torch.Generator().manual_seed(1)
        a = torch.randint(0, 3, (40, 2), generator=g).double()  # a 3 x 3 grid: duplicates and equal distances
        b = (torch.rand(40, 6, generator=g) < 0.3).double()  # binary rows, as a piano roll
        d_a, d_b = _dist(a), _dist(b)
        for k in (1, 4):
            ours = rq.neighborhood_overlap(a, b, k=k).value
            draws = torch.tensor([_overlap(_jitter(d_a, g), _jitter(d_b, g), k) for _ in range(4000)])
            self.assertLess(abs(ours - float(draws.mean())), 4 * float(draws.std()) / 4000**0.5 + 1e-4)

    def test_unrelated_label_spaces_stay_at_chance(self):
        g = torch.Generator().manual_seed(3)
        a = torch.nn.functional.one_hot(torch.randint(0, 4, (400,), generator=g), 4).double()
        b = torch.nn.functional.one_hot(torch.randint(0, 4, (400,), generator=g), 4).double()
        r = rq.neighborhood_overlap(a, b, k=10)
        self.assertAlmostEqual(r.value, r.extras["chance"], delta=0.005)

    def test_compute_pairs_matches_the_direct_metric_with_ties(self):
        g = torch.Generator().manual_seed(2)
        layers = {
            0: torch.randint(0, 3, (60, 2), generator=g).double(),
            1: (torch.rand(60, 5, generator=g) < 0.4).double(),
        }
        for jaccard in (False, True):
            params = {"neighborhood_overlap": {"k": 3, "jaccard": jaccard}}
            for r in rq.compute_pairs(layers, metrics=["neighborhood_overlap"], params=params).rows:
                direct = rq.neighborhood_overlap(layers[r.layer], layers[r.layer_b], k=3, jaccard=jaccard)
                self.assertAlmostEqual(r.value, direct.value, places=12)


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
        rec = rq.compute_pairs(layers, metrics=["cycle_knn"])
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


class DeconfoundedSimilarityTests(unittest.TestCase):
    """Deconfounded CKA and RSA (Cui et al., 2022, arXiv:2202.00095)."""

    def setUp(self):
        rng = np.random.default_rng(7)  # the arrays on which the authors' similarity_metrics.py was run
        z = rng.random((60, 5))
        x = np.tanh(z @ rng.random((5, 12)) - 1.5) + 0.3 * rng.random((60, 12))
        y = (z[:, :3] - 0.5) ** 2 + 0.2 * rng.random((60, 3))
        self.x, self.y, self.z = (torch.tensor(a) for a in (x, y, z))

    def test_matches_the_authors_code(self):
        # deltaCKA, deltaRSA, CKA and RSA of the authors' released code (NeurIPS 2022 supplement), float64
        d = rq.cka(self.x, self.y, confound=self.z)
        r = rq.rsa(self.x, self.y, distance="euclidean", confound=self.z)
        self.assertAlmostEqual(d.value, 0.0591703229109458, places=12)
        self.assertAlmostEqual(r.value, 0.12020446584695443, places=12)
        self.assertAlmostEqual(d.extras["biased"], 0.051249410818273174, places=12)
        self.assertAlmostEqual(r.extras["rsa"], 0.05247049111782487, places=12)

    def test_invariances(self):
        q, _ = torch.linalg.qr(torch.randn(12, 12, dtype=torch.float64, generator=torch.Generator().manual_seed(0)))
        for f in (lambda t: t @ q, lambda t: 3.0 * t):  # orthogonal map and isotropic scaling (their Props. 3.1-3.2)
            self.assertAlmostEqual(
                rq.cka(f(self.x), self.y, confound=self.z).value,
                rq.cka(self.x, self.y, confound=self.z).value,
                places=10,
            )
            self.assertAlmostEqual(
                rq.rsa(f(self.x), self.y, distance="euclidean", confound=self.z).value,
                rq.rsa(self.x, self.y, distance="euclidean", confound=self.z).value,
                places=10,
            )

    def test_confound_of_b_and_errors(self):
        other = torch.rand(60, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
        same = rq.rsa(self.x, self.y, distance="euclidean", confound=self.z).value
        apart = rq.rsa(self.x, self.y, distance="euclidean", confound=self.z, confound_b=other).value
        self.assertNotAlmostEqual(same, apart, places=6)
        with self.assertRaises(ValueError):
            rq.cka(self.x, self.y, debiased=True, confound=self.z)
        with self.assertRaises(ValueError):
            rq.rsa(self.x, self.y, confound=self.z)  # cosine distances: not the paper's
        with self.assertRaises(ValueError):
            rq.cka(self.x, self.y, confound_b=self.z)

    def test_compute_pairs_matches_the_direct_metrics(self):
        layers = {0: self.x, 1: self.y, 2: self.x[:, :6] ** 2}
        for name, kw in (("cka", {}), ("rsa", {"distance": "euclidean"})):
            rec = rq.compute_pairs(layers, metrics=[name], params={name: {**kw, "confound": self.z}})
            fn = rq.cka if name == "cka" else rq.rsa
            for r in rec.rows:
                direct = fn(layers[r.layer], layers[r.layer_b], confound=self.z, **kw)
                self.assertAlmostEqual(r.value, direct.value, places=6)  # RSA vectors are kept in float32
                self.assertTrue(r.params["deconfounded"])
                self.assertNotIn("confound", r.params)
        bad = rq.compute_pairs(
            layers, metrics=["information_imbalance"], params={"information_imbalance": {"confound": self.z}}
        )
        self.assertTrue(all("error" in r.extras for r in bad.rows))


class RidgePredictivityTest(unittest.TestCase):
    """Tuckute et al. (2023, PLOS Biology, Methods) and Pepino et al. (2025, arXiv:2511.16849, Sec. 4.3)."""

    def setUp(self):
        g = torch.Generator().manual_seed(21)
        self.x = torch.randn(240, 12, dtype=torch.float64, generator=g)
        w = torch.randn(12, 6, dtype=torch.float64, generator=g)
        self.y = self.x @ w + 0.05 * torch.randn(240, 6, dtype=torch.float64, generator=g)
        self.z = torch.randn(240, 5, dtype=torch.float64, generator=g)  # unrelated

    def test_leave_one_out_error_matches_explicit_refits(self):
        from req_metrics.metrics.compare import _ridge_fit, _ridge_loo_error

        x, y, lam = self.x[:15, :4], self.y[:15, :2], 0.7
        mean, u, s, v = _ridge_fit(x)
        yc = y - y.mean(0, keepdim=True)
        closed = _ridge_loo_error(u, s, yc, u.T @ yc, lam)
        explicit = torch.zeros(2, dtype=torch.float64)
        for i in range(15):  # each refit centers its own 14 rows: ridge with an unpenalized intercept
            keep = [j for j in range(15) if j != i]
            mx, my = x[keep].mean(0), y[keep].mean(0)
            a, b = x[keep] - mx, y[keep] - my
            w = torch.linalg.solve(a.T @ a + lam * torch.eye(4, dtype=torch.float64), a.T @ b)
            explicit += (y[i] - my - (x[i] - mx) @ w) ** 2
        torch.testing.assert_close(closed, explicit / 15, rtol=1e-9, atol=1e-12)

    def test_folds_have_the_requested_count_and_near_equal_sizes(self):
        from req_metrics.metrics.compare import _folds

        for n, k in ((10, 4), (21, 10), (9, 4), (240, 5)):
            sizes = sorted(len(test) for _, test in _folds(n, k, seed=0))
            self.assertEqual(len(sizes), k)
            self.assertLessEqual(sizes[-1] - sizes[0], 1)
            self.assertEqual(sum(sizes), n)
        x = torch.randn(10, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
        self.assertGreater(rq.ridge_predictivity(x, x @ torch.ones(3, 2, dtype=torch.float64), n_folds=4).value, 0.99)

    def test_linear_relation_is_recovered_and_independence_scores_zero(self):
        related = rq.ridge_predictivity(self.x, self.y)
        self.assertGreater(related.value, 0.95)
        # 6 targets span a 6-dimensional subspace of the 12 predictor columns: the reverse r^2 averages near 1/2.
        self.assertTrue(0.4 < related.extras["reverse"] < 0.6, related.extras["reverse"])
        self.assertEqual(related.extras["n_folds"], 5.0)
        unrelated = rq.ridge_predictivity(self.x, self.z)
        self.assertLess(unrelated.value, 0.05)
        self.assertLess(unrelated.extras["reverse"], 0.05)

    def test_predictor_invariances_and_scaling(self):
        g = torch.Generator().manual_seed(2)
        a = torch.randn(12, 12, dtype=torch.float64, generator=g)
        base = rq.ridge_predictivity(self.x, self.y).value
        self.assertAlmostEqual(rq.ridge_predictivity(self.x @ a, self.y).value, base, places=6)
        self.assertAlmostEqual(rq.ridge_predictivity(3.0 * self.x, 0.5 * self.y).value, base, places=8)
        with self.assertRaises(ValueError):
            rq.ridge_predictivity(self.x, self.y, n_folds=1)

    def test_compute_pairs_matches_the_direct_metric(self):
        layers = {0: self.x, 1: self.y, 2: self.z}
        rec = rq.compute_pairs(layers, metrics=["ridge_predictivity"], params={"ridge_predictivity": {"n_folds": 4}})
        self.assertEqual(len(rec.rows), 9)
        for r in rec.rows:
            direct = rq.ridge_predictivity(layers[r.layer], layers[r.layer_b], n_folds=4)
            self.assertAlmostEqual(r.value, direct.value, places=10)
            self.assertAlmostEqual(r.extras["reverse"], direct.extras["reverse"], places=10)
            self.assertAlmostEqual(r.extras["median"], direct.extras["median"], places=10)
            self.assertAlmostEqual(r.extras["log10_lambda_median"], direct.extras["log10_lambda_median"], places=10)
            self.assertAlmostEqual(
                r.extras["reverse_log10_lambda_median"], direct.extras["reverse_log10_lambda_median"], places=10
            )


class ProcrustesTest(unittest.TestCase):
    """Ding, Denain and Steinhardt (2021, NeurIPS, arXiv:2108.01661, Eqs. 6-7)."""

    @staticmethod
    def reference(x, y):
        """Their Eq. 7 on centered, Frobenius-normalized matrices, the narrower one zero-padded as ReSi does."""
        x = x - x.mean(0, keepdim=True)
        y = y - y.mean(0, keepdim=True)
        x, y = x / x.norm(), y / y.norm()
        d = max(x.shape[1], y.shape[1])
        x = torch.cat([x, torch.zeros(x.shape[0], d - x.shape[1], dtype=x.dtype)], 1)
        y = torch.cat([y, torch.zeros(y.shape[0], d - y.shape[1], dtype=y.dtype)], 1)
        nuclear = torch.linalg.svdvals(x.T @ y).sum()
        return float(x.square().sum() + y.square().sum() - 2 * nuclear)

    def setUp(self):
        g = torch.Generator().manual_seed(5)
        self.x = torch.randn(60, 8, dtype=torch.float64, generator=g)
        self.y = torch.randn(60, 5, dtype=torch.float64, generator=g)

    def test_matches_the_closed_form_with_unequal_widths(self):
        r = rq.procrustes(self.x, self.y)
        self.assertAlmostEqual(r.value, self.reference(self.x, self.y), places=10)
        self.assertAlmostEqual(r.extras["alignment"], 1.0 - r.value / 2, places=12)
        self.assertAlmostEqual(r.extras["distance"], math.sqrt(r.value), places=12)
        self.assertEqual((r.extras["rank_a"], r.extras["rank_b"]), (8.0, 5.0))
        self.assertAlmostEqual(rq.procrustes(self.y, self.x).value, r.value, places=12)

    def test_orthogonal_map_and_scale_give_zero(self):
        q, _ = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(6)))
        r = rq.procrustes(self.x, 3.0 * self.x @ q + 2.0)
        self.assertAlmostEqual(r.value, 0.0, places=10)
        self.assertAlmostEqual(r.extras["alignment"], 1.0, places=10)
        with self.assertRaises(ValueError):
            rq.procrustes(self.x, torch.ones(60, 3, dtype=torch.float64))

    def test_shared_singular_vectors_reduce_to_the_singular_value_distance(self):
        # their Discussion: A = U L1 V^T and B = U L2 V^T give ||l1 - l2||^2 for unit-norm singular value vectors
        g = torch.Generator().manual_seed(7)
        u, _ = torch.linalg.qr(torch.randn(60, 4, dtype=torch.float64, generator=g) - 0.0)
        u = u - u.mean(0, keepdim=True)  # centered columns, re-orthonormalized
        u, _ = torch.linalg.qr(u)
        v, _ = torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64, generator=g))
        l1 = torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float64)
        l2 = torch.tensor([4.0, 1.0, 1.0, 0.5], dtype=torch.float64)
        value = rq.procrustes(u @ torch.diag(l1) @ v.T, u @ torch.diag(l2) @ v.T).value
        self.assertAlmostEqual(value, float(((l1 / l1.norm()) - (l2 / l2.norm())).square().sum()), places=10)

    def test_compute_pairs_matches_the_direct_metric_and_is_symmetric(self):
        layers = {0: self.x, 1: self.y, 2: self.x[:, :3] ** 2}
        rec = rq.compute_pairs(layers, metrics=["procrustes"])
        self.assertEqual(len(rec.rows), 9)
        rows = {(r.layer, r.layer_b): r for r in rec.rows}
        for (a, b), r in rows.items():
            direct = rq.procrustes(layers[a], layers[b])
            self.assertAlmostEqual(r.value, direct.value, places=10)
            self.assertAlmostEqual(r.extras["alignment"], direct.extras["alignment"], places=10)
            self.assertEqual(
                (r.extras["rank_a"], r.extras["rank_b"]), (direct.extras["rank_a"], direct.extras["rank_b"])
            )
        self.assertAlmostEqual(rows[(0, 1)].value, rows[(1, 0)].value, places=12)
