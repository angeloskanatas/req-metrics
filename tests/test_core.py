"""Core tests: preprocessing, registry, spectrum. Run: python -m unittest discover tests."""

import unittest

import torch

from req_metrics import (
    InputKind,
    MetricResult,
    Preprocess,
    Spectrum,
    apply_preprocess,
    get_metric,
    list_metrics,
    register_metric,
)


class PreprocessTests(unittest.TestCase):
    def test_center_and_l2_on_points_and_views(self):
        torch.manual_seed(0)
        x = torch.randn(100, 8) + 3.0
        y = apply_preprocess(x, Preprocess(center=True, l2=True))
        self.assertTrue(torch.allclose(y.norm(dim=-1), torch.ones(100), atol=1e-5))
        v = torch.randn(3, 100, 8) + torch.tensor([1.0, 2.0, 3.0]).view(3, 1, 1)
        c = apply_preprocess(v, Preprocess(center=True))
        self.assertTrue(torch.allclose(c.mean(dim=-2), torch.zeros(3, 8), atol=1e-5))

    def test_standardize_unit_variance(self):
        x = torch.randn(1000, 4) * torch.tensor([1.0, 5.0, 0.1, 2.0])
        z = apply_preprocess(x, Preprocess(standardize=True))
        self.assertTrue(torch.allclose(z.std(dim=0, unbiased=False), torch.ones(4), atol=1e-4))

    def test_describe(self):
        self.assertEqual(Preprocess().describe(), "none")
        self.assertEqual(Preprocess(center=True, l2=True).describe(), "center+l2")


class RegistryTests(unittest.TestCase):
    def test_register_list_get(self):
        @register_metric("_test/dummy", inputs=InputKind.POINTS, citation=("x",), tags=("test",))
        def dummy(x):
            """Mean of x."""
            return MetricResult(float(x.mean()))

        self.assertIn("_test/dummy", list_metrics("_test/*"))
        spec = get_metric("_test/dummy")
        self.assertEqual(spec.description, "Mean of x.")
        self.assertEqual(spec.fn(torch.ones(4, 2)).value, 1.0)
        with self.assertRaises(KeyError):
            register_metric("_test/dummy", inputs=InputKind.POINTS)(dummy)
        with self.assertRaises(KeyError):
            get_metric("_test/missing")


class SpectrumTests(unittest.TestCase):
    def test_rank_k_data(self):
        torch.manual_seed(0)
        x = torch.randn(500, 3) @ torch.randn(3, 16)
        s = Spectrum.from_points(x)
        # float32 inputs leave roundoff singular values around 1e-7 of the leading one
        self.assertEqual((s.singular_values > 1e-5 * s.singular_values[0]).sum().item(), 3)
        self.assertEqual(s.padded_eigenvalues().numel(), 16)
        self.assertAlmostEqual(s.normalized("variance").sum().item(), 1.0, places=9)

    def test_centering_removes_mean_direction(self):
        x = torch.randn(400, 8) + 10.0
        self.assertLess(
            Spectrum.from_points(x).singular_values[0], Spectrum.from_points(x, center=False).singular_values[0]
        )


if __name__ == "__main__":
    unittest.main()
