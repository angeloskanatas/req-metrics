"""Spectral group: analytic nulls and internal consistency. Run: python -m unittest discover tests."""

import math
import unittest

import torch

import req_metrics as rq


def gaussian(n=4000, d=32, seed=0):
    torch.manual_seed(seed)
    return torch.randn(n, d)


class EffectiveRankTests(unittest.TestCase):
    def test_isotropic_gaussian_near_full_rank(self):
        r = rq.effective_rank(gaussian())
        self.assertGreater(r.value, 0.97 * 32)
        self.assertAlmostEqual(r.extras["normalized_entropy"], 1.0, places=1)

    def test_rank_k_data(self):
        torch.manual_seed(1)
        basis = torch.linalg.qr(torch.randn(16, 3)).Q.T  # three orthonormal directions
        x = torch.randn(3000, 3) @ basis  # equal variance on each -> PR = 3
        self.assertLessEqual(rq.effective_rank(x).value, 3.0 + 1e-6)
        self.assertAlmostEqual(rq.participation_ratio(x, normalized=False).value, 3.0, delta=0.05)

    def test_variance_convention_is_log_of_matrix_entropy(self):
        x = gaussian(500, 8)
        er = rq.effective_rank(x, spectrum="variance", center=False)
        me = rq.matrix_entropy(x, alpha=1.0, normalization="raw", center=False)
        self.assertAlmostEqual(math.log(er.value), me.value, places=9)

    def test_spectrum_object_is_accepted(self):
        x = gaussian(500, 8)
        s = rq.Spectrum.from_points(x, center=True)
        self.assertAlmostEqual(rq.effective_rank(s).value, rq.effective_rank(x).value, places=9)


class EntropyTests(unittest.TestCase):
    def test_renyi_two_matches_closed_form(self):
        x = gaussian(600, 10)
        p = rq.Spectrum.from_points(x, center=False).normalized("variance")
        h2 = -math.log(float((p**2).sum()))
        self.assertAlmostEqual(rq.matrix_entropy(x, alpha=2.0, normalization="raw").value, h2, places=9)

    def test_spectral_entropy_normalized_is_near_one_for_gaussian(self):
        self.assertGreater(rq.spectral_entropy(gaussian()).value, 0.98)


class AlphaReqTests(unittest.TestCase):
    def test_recovers_known_power_law(self):
        torch.manual_seed(0)
        n, d, alpha = 40000, 256, 1.0
        scales = torch.arange(1, d + 1, dtype=torch.float64) ** (-alpha / 2)
        x = torch.randn(n, d, dtype=torch.float64) * scales
        r = rq.alpha_req(x)
        self.assertAlmostEqual(r.value, alpha, delta=0.08)
        self.assertGreater(r.extras["r2"], 0.95)

    def test_rank_deficient_window_raises(self):
        x = torch.randn(200, 3) @ torch.randn(3, 64)
        with self.assertRaises(ValueError):
            rq.alpha_req(x)


class AnisotropyTests(unittest.TestCase):
    def test_isotropic_gaussian_near_one_over_d(self):
        a = rq.anisotropy_spectral(gaussian())
        self.assertLess(a.value, 2.0 / 32)
        self.assertAlmostEqual(a.value + a.extras["isotropy_score"], 1.0, places=12)

    def test_single_direction_gives_one(self):
        x = torch.randn(300, 1) @ torch.randn(1, 8)
        self.assertAlmostEqual(rq.anisotropy_spectral(x, l2=False).value, 1.0, places=9)

    def test_cosine_extremes(self):
        same = torch.ones(50, 6) * torch.randn(1, 6)
        self.assertAlmostEqual(rq.anisotropy_cosine(same).value, 1.0, places=9)
        self.assertLess(abs(rq.anisotropy_cosine(gaussian(2000, 64)).value), 0.02)


class ShapeTests(unittest.TestCase):
    def test_participation_ratio_and_eee_on_gaussian(self):
        x = gaussian()
        self.assertGreater(rq.participation_ratio(x).value, 0.9)
        self.assertLess(rq.eigenvalue_early_enrichment(x).value, 0.15)

    def test_eee_one_direction(self):
        x = torch.randn(300, 1) @ torch.randn(1, 8)
        self.assertGreater(rq.eigenvalue_early_enrichment(x).value, 0.8)


class GaussianityTests(unittest.TestCase):
    def test_gaussian_scores_below_heavy_tailed(self):
        torch.manual_seed(0)
        g = torch.randn(3000, 16)
        heavy = torch.distributions.StudentT(2.0).sample((3000, 16))
        for method in ("epps_pulley", "ks", "swd"):
            self.assertLess(rq.gaussianity(g, method=method).value, rq.gaussianity(heavy, method=method).value, method)

    def test_swd_terms_near_zero_for_standard_normal(self):
        r = rq.gaussianity(gaussian(5000, 16), method="swd")
        self.assertLess(r.value, 0.01)
        self.assertLess(r.extras["center"], 0.01)
        self.assertLess(r.extras["scale"], 0.01)

    def test_seed_reproducible(self):
        x = gaussian(500, 8)
        self.assertEqual(rq.gaussianity(x, seed=3).value, rq.gaussianity(x, seed=3).value)


class SparsityTests(unittest.TestCase):
    def test_dense_gaussian(self):
        r = rq.sparsity(gaussian(2000, 64))
        self.assertAlmostEqual(r.extras["m_l0"], 1.0)
        self.assertAlmostEqual(r.value, 2.0 / math.pi, delta=0.03)

    def test_one_hot_rows(self):
        x = torch.eye(8).repeat(10, 1)
        r = rq.sparsity(x)
        self.assertAlmostEqual(r.value, 1.0 / 8, places=9)
        self.assertAlmostEqual(r.extras["m_l0"], 1.0 / 8, places=9)


class RegistryTests(unittest.TestCase):
    def test_spectral_group_registered(self):
        names = rq.list_metrics()
        for n in (
            "effective_rank",
            "effective_rank/variance",
            "anisotropy",
            "anisotropy/cosine",
            "alpha_req",
            "gaussianity",
            "gaussianity/swd",
            "matrix_entropy",
            "participation_ratio",
            "eigenvalue_early_enrichment",
            "sparsity",
            "spectral_entropy",
        ):
            self.assertIn(n, names)
        spec = rq.get_metric("anisotropy")
        self.assertEqual(spec.preprocess.describe(), "center+l2")
        self.assertIn("paper-canonical", spec.tags)
        self.assertAlmostEqual(spec.fn(gaussian(300, 8)).value, rq.anisotropy_spectral(gaussian(300, 8)).value)


if __name__ == "__main__":
    unittest.main()
