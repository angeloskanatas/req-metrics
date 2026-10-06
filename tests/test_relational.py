"""Relational metrics and collapse indicators: analytic values on uniform, collapsed and clustered clouds."""

import unittest

import torch

import req_metrics as rq


def unit_gaussian(n, d, seed=0):
    torch.manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(n, d), dim=1)


class SelfClusteringTests(unittest.TestCase):
    def test_uniform_sphere_near_zero_and_collapse_is_one(self):
        self.assertLess(abs(rq.self_clustering(unit_gaussian(4000, 64)).value), 0.01)
        collapsed = torch.ones(300, 16) * torch.randn(1, 16)
        self.assertAlmostEqual(rq.self_clustering(collapsed).value, 1.0, places=9)

    def test_two_clusters_are_high_and_extras_consistent(self):
        torch.manual_seed(2)
        centers = torch.nn.functional.normalize(torch.randn(2, 32), dim=1)
        x = centers.repeat_interleave(500, dim=0) + 0.05 * torch.randn(1000, 32)
        r = rq.self_clustering(x)
        self.assertGreater(r.value, 0.4)
        self.assertAlmostEqual(r.extras["uniform_mean_squared_cosine"], 1 / 32)
        u = torch.nn.functional.normalize(x.double(), dim=1)
        cos = u @ u.T
        off = (cos.square().sum() - 1000) / (1000 * 999)
        self.assertAlmostEqual(r.extras["mean_squared_cosine"], float(off), places=9)


class UniformityTests(unittest.TestCase):
    def test_uniform_cloud_reaches_corollary_bound_and_collapse_is_zero(self):
        d = 256
        r = rq.uniformity(unit_gaussian(3000, d))
        self.assertAlmostEqual(r.extras["lower_bound_large_d"], -4.0)
        self.assertLess(r.extras["lower_bound"], 0.0)
        self.assertGreater(r.extras["lower_bound"], -4.0)
        self.assertLess(abs(r.value - r.extras["lower_bound"]), 0.02)  # uniform on the sphere is the minimizer
        self.assertAlmostEqual(rq.uniformity(torch.ones(50, 8)).value, 0.0, places=9)

    def test_chunking_is_exact(self):
        x = torch.randn(700, 24)
        self.assertAlmostEqual(rq.uniformity(x, chunk=64).value, rq.uniformity(x, chunk=5000).value, places=10)
        u = torch.nn.functional.normalize(x.double(), dim=1)
        ref = torch.log(torch.exp(-2.0 * torch.pdist(u).square()).mean())
        self.assertAlmostEqual(rq.uniformity(x).value, float(ref), places=10)  # the reference two-liner


class NormalizedStdTests(unittest.TestCase):
    def test_isotropic_reference_and_collapse(self):
        r = rq.normalized_std(torch.randn(6000, 64))
        self.assertAlmostEqual(r.value, 1 / 8, delta=0.004)
        self.assertAlmostEqual(r.extras["ratio_to_isotropic"], 1.0, delta=0.03)
        self.assertAlmostEqual(rq.normalized_std(torch.ones(100, 16) * torch.randn(1, 16)).value, 0.0, places=9)


class AlignmentTests(unittest.TestCase):
    def test_identical_views_zero_and_unrelated_near_two(self):
        v = torch.randn(1, 500, 64).repeat(3, 1, 1)
        self.assertAlmostEqual(rq.alignment(v).value, 0.0, places=9)
        torch.manual_seed(0)
        r = rq.alignment(torch.randn(2, 4000, 256))
        self.assertAlmostEqual(r.value, 2.0, delta=0.03)
        self.assertEqual(r.extras["n_pairs"], 1.0)
        self.assertEqual(rq.alignment(torch.randn(4, 50, 8)).extras["n_pairs"], 6.0)

    def test_through_compute_with_view_spec(self):
        rec = rq.compute(
            {0: torch.randn(2, 300, 16)},
            ["alignment", "uniformity"] if False else ["alignment"],
            views=rq.ViewSpec(source="objective", q=2),
        )
        self.assertEqual(rec[0].n_views, 2)


class CorrectedParticipationRatioTests(unittest.TestCase):
    def test_row_correction_recovers_known_dimensionality(self):
        torch.manual_seed(0)
        d, q, p = 20, 64, 300
        w = torch.linalg.qr(torch.randn(q, d)).Q.T  # d orthonormal directions in R^q
        x = torch.randn(p, d) @ w  # population PR over all q units is exactly d
        naive = rq.participation_ratio(x, normalized=False).value
        row = rq.participation_ratio(x, normalized=False, correction="row")
        self.assertLess(naive, d - 0.5)  # plug-in bias about PR/N = 7%
        self.assertAlmostEqual(row.value, d, delta=0.6)
        self.assertAlmostEqual(row.extras["naive"], naive, places=6)

    def test_one_pass_estimates_match_the_spectrum_path_and_the_pipeline(self):
        x = torch.randn(400, 30) @ torch.randn(30, 30)
        plug_in = rq.participation_ratio(rq.Spectrum.from_points(x), normalized=False).value
        r = rq.participation_ratio(x, normalized=False, correction="both")
        self.assertAlmostEqual(r.extras["naive"], plug_in, places=6)
        self.assertEqual(r.value, r.extras["both"])
        rec = rq.compute({0: x}, ["participation_ratio"], params={"participation_ratio": {"correction": "row"}})
        self.assertAlmostEqual(rec[0].value * 30, r.extras["row"], places=9)
        self.assertAlmostEqual(rec[0].extras["naive"], plug_in, places=6)
        with self.assertRaises(ValueError):
            rq.participation_ratio(rq.Spectrum.from_points(x), correction="row")


class ConvergenceTests(unittest.TestCase):
    def test_effective_rank_rises_with_n_and_anisotropy_is_flat(self):
        torch.manual_seed(0)
        x = torch.randn(4000, 128) @ torch.diag(torch.linspace(1.0, 0.05, 128))
        c = rq.convergence(x, "effective_rank", fractions=(0.1, 0.5, 1.0), repeats=3)
        self.assertEqual([r.n_items for r in c.rows], [400, 2000, 4000])
        self.assertTrue(c.rows[0].mean < c.rows[1].mean < c.rows[2].mean)  # entropic rank grows with N
        self.assertTrue(c.rows[0].rel_change != c.rows[0].rel_change)  # nan for the first row
        a = rq.convergence(x, "anisotropy/spectral", fractions=(0.1, 1.0), repeats=3)
        self.assertGreater(abs(a.rows[1].rel_change), 0.1)  # D/N = 0.32 at the small fraction: top eigenvalue inflated
        y = torch.randn(4000, 32) * torch.cat([torch.tensor([10.0]), torch.ones(31)])
        b = rq.convergence(y, "anisotropy/spectral", fractions=(0.1, 1.0), repeats=3)
        self.assertLess(abs(b.rows[1].rel_change), 0.02)  # N >> D and a dominant direction: flat
        self.assertIn("| fraction |", b.to_markdown())


if __name__ == "__main__":
    unittest.main()
