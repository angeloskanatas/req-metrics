"""Degenerate inputs: collapse, ties, duplicates, non-finite values."""

import unittest

import torch

import req_metrics as rq


class DegenerateInputTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_collapsed_cloud_is_rank_zero(self):
        collapsed = torch.ones(300, 16, dtype=torch.float64) * torch.randn(1, 16, dtype=torch.float64)
        self.assertEqual(rq.effective_rank(collapsed).value, 0.0)
        self.assertEqual(rq.effective_rank(collapsed.float()).value, 0.0)
        with self.assertRaisesRegex(ValueError, "zero covariance"):
            rq.participation_ratio(collapsed)
        with self.assertWarns(RuntimeWarning):
            rec = rq.compute({0: collapsed}, ["anisotropy/spectral", "participation_ratio", "effective_rank"])
        by = {r.metric: r for r in rec}
        self.assertIn("error", by["anisotropy/spectral"].extras)
        self.assertIn("zero covariance", by["participation_ratio"].extras["error"])
        self.assertEqual(by["effective_rank"].value, 0.0)
        nearly = torch.randn(300, 16, dtype=torch.float64)  # an ordinary cloud is untouched by the collapse test
        self.assertGreater(rq.effective_rank(nearly).value, 14.0)

    def test_imbalance_pipeline_matches_the_estimator_on_ties(self):
        a = torch.randn(600, 20)
        b = torch.randn(600, 6)
        b[:61] = b[0]
        rec = rq.compute_pairs({0: a}, {0: b}, metrics=["information_imbalance"])[0]
        direct = rq.information_imbalance(a, b)
        self.assertAlmostEqual(rec.value, direct.value, places=10)
        self.assertAlmostEqual(rec.extras["reverse"], direct.extras["reverse"], places=10)

    def test_duplicates_and_small_clouds_in_neighbor_estimators(self):
        base = torch.randn(30, 4, dtype=torch.float64)
        dup = torch.cat([base] * 4)
        self.assertAlmostEqual(rq.gride(dup).value, rq.gride(base).value, places=12)
        with self.assertRaisesRegex(ValueError, "distinct points"):
            rq.mle(torch.randn(15, 4))
        with self.assertRaisesRegex(ValueError, "distinct points"):
            rq.mlid(torch.randn(20, 4), k=64)

    def test_non_finite_inputs_are_refused(self):
        x = torch.randn(200, 8)
        bad = x.clone()
        bad[0, 0] = float("nan")
        for fn in (rq.twonn, rq.effective_rank, rq.mle):
            with self.assertRaisesRegex(ValueError, "non-finite"):
                fn(bad)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            rq.cka(bad, x)

    def test_lidar_counts_only_signal_eigenvalues(self):
        views = torch.randn(3, 20, 50).double()
        views = views + 0.1 * torch.randn(3, 20, 50).double()
        res = rq.lidar(views)
        self.assertLessEqual(res.extras["n_positive_eigenvalues"], 19)  # rank of the between-sample scatter


class ExactNullTests(unittest.TestCase):
    """Closed-form values the audit verified, pinned so they stay."""

    def test_eigenvalue_early_enrichment_of_a_single_direction(self):
        x = torch.randn(500, 1, dtype=torch.float64) @ torch.randn(1, 8, dtype=torch.float64)
        self.assertAlmostEqual(rq.eigenvalue_early_enrichment(x).value, 7 / 8, places=9)

    def test_lidar_of_identical_views_is_the_variance_effective_rank_of_the_class_means(self):
        torch.manual_seed(0)
        z = torch.randn(300, 12, dtype=torch.float64)
        views = z.unsqueeze(0).repeat(4, 1, 1)
        self.assertAlmostEqual(
            rq.lidar(views, delta=1e-6).value, rq.effective_rank(z, spectrum="variance").value, places=4
        )

    def test_local_rectifiability_of_an_exact_plane_is_zero(self):
        torch.manual_seed(0)
        plane = torch.randn(600, 2, dtype=torch.float64) @ torch.randn(2, 10, dtype=torch.float64)
        self.assertLess(rq.local_rectifiability(plane, n=2, n_anchors=64, n_scales=4).value, 1e-6)


if __name__ == "__main__":
    unittest.main()
