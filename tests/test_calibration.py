"""Permutation-null calibration and half-sample error bars (req_metrics.calibration)."""

import functools
import unittest

import torch

import req_metrics as rq
from req_metrics.calibration import (
    _fast_imbalance,
    _fast_neighbors,
    calibrate,
    calibrate_grid,
    calibrate_layers,
    group_permutation,
    jackknife,
)


class CalibrateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(300, 40)

    def test_independent_spaces_calibrate_to_zero(self):
        y = torch.randn(300, 40)
        r = calibrate(rq.cka, self.x, y, k=100, seed=1)
        self.assertGreater(r.score, 0.05)  # raw linear CKA of independent data is well above zero at d/n = 0.13
        self.assertEqual(r.calibrated, 0.0)
        self.assertGreater(r.p_value, 0.05)
        self.assertAlmostEqual(r.null_mean, r.score, delta=0.05)

    def test_shared_signal_calibrates_high(self):
        y = self.x @ torch.randn(40, 30) + 0.1 * torch.randn(300, 30)
        r = calibrate(rq.cka, self.x, y, k=100, seed=1)
        self.assertGreater(r.calibrated, 0.5)
        self.assertAlmostEqual(r.p_value, 1.0 / 101, places=12)

    def test_threshold_is_the_order_statistic(self):
        # k = 19, alpha = 0.05: tau is the ceil(0.95 * 20) = 19th of 20 values, the largest null score here
        y = self.x @ torch.randn(40, 30)
        r = calibrate(rq.cka, self.x, y, k=19, alpha=0.05, seed=3)
        self.assertLess(r.tau, r.score)
        self.assertAlmostEqual(r.calibrated, (r.score - r.tau) / (1.0 - r.tau), places=12)

    def test_imbalance_fast_path_equals_the_general_path(self):
        g = torch.Generator().manual_seed(4)
        cases = [  # continuous rows, tied rows (a grid against binary rows), and cosine neighbors
            (self.x[:200, :6], torch.tanh(self.x[:200, :6] @ torch.randn(6, 3, generator=g)), {"k": 3}),
            (
                torch.randint(0, 3, (120, 2), generator=g).double(),
                (torch.rand(120, 6, generator=g) < 0.3).double(),
                {"k": 2},
            ),
            (self.x[:150], self.x[:150, :8], {"l2": True}),
        ]

        def general(a, b, **kw):  # the same metric through a wrapper, so calibrate takes its general path
            return rq.information_imbalance(a, b, **kw)

        for x, y, kw in cases:
            # calibrate's own k counts permutations, so the metric's k goes in through functools.partial
            fast = calibrate(
                functools.partial(rq.information_imbalance, **kw), x, y, k=30, best=0.0, lower_is_better=True, seed=2
            )
            slow = calibrate(functools.partial(general, **kw), x, y, k=30, best=0.0, lower_is_better=True, seed=2)
            for field in ("score", "calibrated", "p_value", "tau", "null_mean", "null_std"):
                self.assertAlmostEqual(getattr(fast, field), getattr(slow, field), places=12)
            fast_j = jackknife(rq.information_imbalance, x, y, seed=3, **kw)[2]
            slow_j = jackknife(general, x, y, seed=3, **kw)[2]
            for u, v in zip(fast_j, slow_j, strict=True):
                self.assertAlmostEqual(u, v, places=12)

    def test_information_imbalance_lower_is_better(self):
        same = calibrate(rq.information_imbalance, self.x, self.x.clone(), k=50, best=0.0, lower_is_better=True)
        self.assertGreater(same.calibrated, 0.95)
        self.assertGreater(same.tau, 0.8)  # the shuffled-pair null sits near 1
        other = calibrate(rq.information_imbalance, self.x, torch.randn(300, 40), k=50, best=0.0, lower_is_better=True)
        self.assertEqual(other.calibrated, 0.0)

    def test_groups_keep_frames_of_a_clip_together(self):
        # 30 clips of 10 frames; each clip has its own offset in A and an unrelated one in B. Single-frame
        # permutations call the shared grouping "similarity"; permuting whole clips does not.
        gen = torch.Generator().manual_seed(0)
        groups = torch.arange(30).repeat_interleave(10)
        a = torch.randn(30, 20, generator=gen)[groups] + 0.3 * torch.randn(300, 20, generator=gen)
        b = torch.randn(30, 20, generator=gen)[groups] + 0.3 * torch.randn(300, 20, generator=gen)
        rows = calibrate(rq.neighborhood_overlap, a, b, k=50, seed=2)
        clips = calibrate(rq.neighborhood_overlap, a, b, k=50, groups=groups, seed=2)
        self.assertGreater(rows.calibrated, 0.1)
        self.assertEqual(clips.calibrated, 0.0)

    def test_group_permutation_moves_whole_groups(self):
        groups = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 3, 3])
        perm = group_permutation(groups, torch.Generator().manual_seed(4))
        self.assertEqual(sorted(perm.tolist()), list(range(10)))
        for rows in ([0, 1, 2], [3, 4, 5], [6, 7], [8, 9]):
            target = groups[perm[rows]]
            self.assertEqual(len(set(target.tolist())), 1)  # a group lands on one group
            self.assertEqual(perm[rows].tolist(), sorted(perm[rows].tolist()))  # in its own row order

    def test_group_permutation_refuses_groups_that_cannot_move(self):
        unequal = torch.cat([torch.full((n,), i) for i, n in enumerate([5, 6, 7, 8])])
        with self.assertRaises(ValueError):
            group_permutation(unequal, torch.Generator().manual_seed(0))
        mostly_unequal = torch.cat([torch.full((n,), i) for i, n in enumerate([2, 2, 6, 7, 8])])
        with self.assertWarns(RuntimeWarning):
            group_permutation(mostly_unequal, torch.Generator().manual_seed(0))

    def test_bad_settings(self):
        with self.assertRaises(ValueError):
            calibrate(rq.cka, self.x, self.x, k=0)
        with self.assertRaises(ValueError):
            calibrate(rq.cka, self.x, self.x[:10])


class CalibrateLayersTests(unittest.TestCase):
    def test_max_over_independent_layers_calibrates_to_zero(self):
        gen = torch.Generator().manual_seed(0)
        layers_a = [torch.randn(200, 30, generator=gen) for _ in range(4)]
        layers_b = [torch.randn(200, 30, generator=gen) for _ in range(5)]
        r, grid = calibrate_layers(rq.cka, layers_a, layers_b, k=40, seed=1)
        self.assertEqual(tuple(grid.shape), (4, 5))
        self.assertAlmostEqual(r.score, float(grid.max()), places=12)
        self.assertEqual(r.calibrated, 0.0)

    def test_one_shared_pair_is_found(self):
        gen = torch.Generator().manual_seed(0)
        layers_a = [torch.randn(200, 30, generator=gen) for _ in range(3)]
        layers_b = [torch.randn(200, 30, generator=gen) for _ in range(3)]
        layers_b[1] = layers_a[2] @ torch.randn(30, 30, generator=gen)
        r, grid = calibrate_layers(rq.cka, layers_a, layers_b, k=40, seed=1)
        self.assertEqual(divmod(int(grid.argmax()), 3), (2, 1))
        self.assertGreater(r.calibrated, 0.5)

    def test_imbalance_fast_path_equals_the_general_path(self):
        gen = torch.Generator().manual_seed(5)
        base = torch.randn(150, 6, generator=gen)
        layers_a = [  # continuous rows, and a coarse grid with many ties
            base,
            torch.tanh(base @ torch.randn(6, 6, generator=gen)),
            torch.randint(0, 3, (150, 2), generator=gen).double(),
        ]
        noise = torch.randn(150, 3, generator=gen)
        layers_b = [(torch.rand(150, 5, generator=gen) < 0.3).double(), base[:, :3] + 0.1 * noise]

        def general(a, b, **kw):  # the same metric through a wrapper, so calibrate_layers takes its general path
            return rq.information_imbalance(a, b, **kw)

        self.assertEqual(
            _fast_imbalance(functools.partial(rq.information_imbalance, k=2), {}, layers_a, 150), (2, False)
        )
        self.assertIsNone(_fast_imbalance(functools.partial(general, k=2), {}, layers_a, 150))
        # discrete rows at unit length tie only up to rounding, which a recomputation on permuted rows breaks
        # differently, so the l2 case uses the continuous layers
        for kw, la, lb in (({"k": 2}, layers_a, layers_b), ({"l2": True}, layers_a[:2], [layers_b[1], noise])):
            settings = {"aggregate": "min", "k": 20, "best": 0.0, "lower_is_better": True, "seed": 3}
            fast, fast_grid = calibrate_layers(functools.partial(rq.information_imbalance, **kw), la, lb, **settings)
            slow, slow_grid = calibrate_layers(functools.partial(general, **kw), la, lb, **settings)
            self.assertLess(float((fast_grid - slow_grid).abs().max()), 1e-12)
            for field in ("score", "calibrated", "p_value", "tau", "null_mean", "null_std"):
                self.assertAlmostEqual(getattr(fast, field), getattr(slow, field), places=12)


class CalibrateGridTests(unittest.TestCase):
    def setUp(self):
        gen = torch.Generator().manual_seed(0)
        self.layers_a = [torch.randn(200, 30, generator=gen) for _ in range(3)]
        self.layers_b = [torch.randn(200, 30, generator=gen) for _ in range(2)]
        self.layers_b[1] = self.layers_a[2] @ torch.randn(30, 30, generator=gen)

    def test_cells_equal_single_pair_calibration(self):
        g = calibrate_grid(rq.cka, self.layers_a, self.layers_b, k=30, seed=2)
        self.assertEqual(tuple(g.score.shape), (3, 2))
        for i in range(3):
            for j in range(2):
                single = calibrate(rq.cka, self.layers_a[i], self.layers_b[j], k=30, seed=2)
                for field in ("score", "calibrated", "p_value", "tau", "null_mean", "null_std"):
                    self.assertAlmostEqual(getattr(g.cell(i, j), field), getattr(single, field), places=12)
        self.assertGreater(g.calibrated[2, 1], 0.5)
        self.assertAlmostEqual(float(g.p_value[2, 1]), 1.0 / 31, places=12)
        self.assertEqual(float(g.calibrated[0, 0]), 0.0)

    def test_score_grid_equals_calibrate_layers_grid(self):
        g = calibrate_grid(rq.cka, self.layers_a, self.layers_b, k=10, seed=1)
        _, grid = calibrate_layers(rq.cka, self.layers_a, self.layers_b, k=10, seed=1)
        self.assertLess(float((g.score - grid).abs().max()), 1e-12)

    def test_neighbor_fast_path_equals_the_general_path(self):
        gen = torch.Generator().manual_seed(7)
        layers_a = [torch.randn(120, 5, generator=gen), torch.randint(0, 3, (120, 2), generator=gen).double()]
        layers_b = [torch.randn(120, 4, generator=gen), layers_a[0][:, :3] + 0.2 * torch.randn(120, 3, generator=gen)]

        def general(fn):  # the same metric through a wrapper, so the calibration takes its general path
            return lambda a, b, **kw: fn(a, b, **kw)

        self.assertEqual(
            _fast_neighbors(functools.partial(rq.neighborhood_overlap, k=5), {}, layers_a, 120), (6, False)
        )
        self.assertEqual(_fast_neighbors(rq.cycle_knn, {"k": 4, "l2": True}, layers_a, 120), (5, True))
        self.assertIsNone(_fast_neighbors(general(rq.cycle_knn), {"k": 4}, layers_a, 120))
        self.assertIsNone(_fast_neighbors(rq.neighborhood_overlap, {"k": 5, "neighbors_a": None}, layers_a, 120))
        cases = [
            (rq.neighborhood_overlap, {"k": 5}),
            (rq.neighborhood_overlap, {"k": 5, "jaccard": True, "l2": True}),
            (rq.cycle_knn, {"k": 4}),
            (rq.cycle_knn, {"k": 4, "l2": True}),
        ]
        for fn, kw in cases:  # the metric's k through functools.partial: the calibration's own k counts permutations
            metric = functools.partial(fn, **kw)
            fast = calibrate_grid(metric, layers_a, layers_b, k=20, seed=3)
            slow = calibrate_grid(general(metric), layers_a, layers_b, k=20, seed=3)
            for field in ("score", "calibrated", "p_value", "tau", "null_mean", "null_std"):
                self.assertLess(
                    float((getattr(fast, field) - getattr(slow, field)).abs().max()), 1e-12, (fn.__name__, kw, field)
                )
            self.assertGreater(float(fast.calibrated[0, 1]), 0.0)  # the related pair is found at k = 20
            single_fast = calibrate(metric, layers_a[0], layers_b[1], k=20, seed=3)
            single_slow = calibrate(general(metric), layers_a[0], layers_b[1], k=20, seed=3)
            self.assertAlmostEqual(single_fast.calibrated, single_slow.calibrated, places=12)
            self.assertAlmostEqual(single_fast.null_mean, single_slow.null_mean, places=12)

    def test_grouped_permutations_and_lower_is_better(self):
        groups = torch.arange(200) // 4
        # k = 19 is the smallest number of permutations at alpha 0.05 whose threshold can sit below the observed value
        g = calibrate_grid(
            functools.partial(rq.information_imbalance, k=2),
            self.layers_a,
            self.layers_b,
            k=19,
            best=0.0,
            lower_is_better=True,
            groups=groups,
            seed=5,
        )
        self.assertTrue(bool((g.calibrated >= 0).all()) and bool((g.calibrated <= 1).all()))
        self.assertGreater(float(g.calibrated[2, 1]), 0.5)
        self.assertLess(float(g.tau[2, 1]), float(g.null_mean[2, 1]))  # lower is better: tau below the null mean


class JackknifeTests(unittest.TestCase):
    def test_five_half_samples(self):
        torch.manual_seed(0)
        x = torch.randn(400, 8)
        mean, std, values = jackknife(rq.information_imbalance, x, x[:, :4])
        self.assertEqual(len(values), 5)
        self.assertGreater(std, 0.0)
        self.assertLess(abs(mean - rq.information_imbalance(x, x[:, :4]).value), 0.1)


class InformationImbalanceAsymmetryTests(unittest.TestCase):
    def test_asymmetry_is_forward_minus_reverse(self):
        torch.manual_seed(0)
        x = torch.randn(500, 6)
        r = rq.information_imbalance(x, x[:, :3])
        self.assertAlmostEqual(r.extras["asymmetry"], r.value - r.extras["reverse"], places=12)
        self.assertLess(r.extras["asymmetry"], 0.0)  # the full space predicts its subset better than the reverse


if __name__ == "__main__":
    unittest.main()
