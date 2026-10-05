"""CKA and SVCCA: parity with the authors' reference code, invariances, and the pair pipeline."""

import unittest

import numpy as np
import torch

import req_metrics as rq


def _notebook_inputs():
    # The demo inputs of the CKA reference notebook (google-research/representation_similarity).
    np.random.seed(1337)
    x = np.random.randn(100, 10)
    y = np.random.randn(100, 10) + x
    return torch.tensor(x), torch.tensor(y)


def _svcca_inputs():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((2000, 8))
    a = z @ rng.standard_normal((8, 40)) + 0.5 * rng.standard_normal((2000, 40))
    b = np.tanh(z @ rng.standard_normal((8, 30))) + 0.5 * rng.standard_normal((2000, 30))
    return torch.tensor(a), torch.tensor(b)


class CKATests(unittest.TestCase):
    def test_reference_notebook_values(self):
        x, y = _notebook_inputs()
        self.assertAlmostEqual(rq.cka(x, y).value, 0.557613256612, places=11)  # recorded notebook output
        self.assertAlmostEqual(rq.cka(x, y, debiased=True).value, 0.513456489506, places=11)

    def test_wider_than_n_matches_gram_form(self):
        # Gram-form values of the reference notebook's cka(gram_linear(a), gram_linear(b), debiased).
        rng = np.random.default_rng(0)
        a = rng.standard_normal((30, 80))
        b = a[:, :50] @ rng.standard_normal((50, 60)) + rng.standard_normal((30, 60))
        r = rq.cka(torch.tensor(a), torch.tensor(b))
        self.assertAlmostEqual(r.value, 0.792213603996487, places=12)
        self.assertAlmostEqual(r.extras["debiased"], 0.5664140411433203, places=12)

    def test_orthogonal_and_scale_invariance_but_not_linear(self):
        torch.manual_seed(0)
        x = torch.randn(500, 12, dtype=torch.float64)
        q, _ = torch.linalg.qr(torch.randn(12, 12, dtype=torch.float64))
        self.assertAlmostEqual(rq.cka(x, 3.0 * x @ q + 1.0).value, 1.0, places=12)
        stretch = torch.diag(torch.logspace(0, 2, 12, dtype=torch.float64))
        self.assertLess(rq.cka(x, x @ stretch).value, 0.9)

    def test_debiased_removes_small_sample_inflation(self):
        torch.manual_seed(0)
        x, y = torch.randn(20, 50), torch.randn(20, 50)
        r = rq.cka(x, y)
        self.assertGreater(r.extras["biased"], 0.5)
        self.assertLess(abs(r.extras["debiased"]), 0.2)

    def test_argument_checks(self):
        with self.assertRaises(ValueError):
            rq.cka(torch.randn(10, 3), torch.randn(9, 3))
        with self.assertRaises(ValueError):
            rq.cka(torch.randn(3, 2), torch.randn(3, 2), debiased=True)


class SVCCATests(unittest.TestCase):
    def test_matches_cca_core_at_zero_epsilon(self):
        # Reference: google/svcca cca_core.get_cca_similarity(epsilon=0) on the SVD-reduced
        # representations, mean of cca_coef1, with the App. A rule on singular values.
        a, b = _svcca_inputs()
        for threshold, value, k_a, k_b in (
            (0.99, 0.3207378427636194, 39, 30),
            (0.9, 0.33582969852558503, 27, 26),
            (0.5, 0.8869856607168971, 5, 9),
        ):
            r = rq.svcca(a, b, threshold=threshold)
            self.assertAlmostEqual(r.value, value, places=12)
            self.assertEqual((r.extras["k_a"], r.extras["k_b"]), (k_a, k_b))

    def test_invariant_to_invertible_linear_maps(self):
        torch.manual_seed(0)
        x = torch.randn(800, 10, dtype=torch.float64)
        m = torch.randn(10, 10, dtype=torch.float64) + 3 * torch.eye(10, dtype=torch.float64)
        self.assertAlmostEqual(rq.svcca(x, x @ m, threshold=1.0).value, 1.0, places=10)
        self.assertLess(rq.svcca(x, torch.randn(800, 10, dtype=torch.float64)).value, 0.2)

    def test_threshold_check(self):
        with self.assertRaises(ValueError):
            rq.svcca(torch.randn(50, 3), torch.randn(50, 3), threshold=0.0)


class PairPipelineTests(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        base = torch.randn(400, 6, generator=g)
        self.layers = {
            i: base @ torch.randn(6, 10 + i, generator=g) + 0.2 * torch.randn(400, 10 + i, generator=g)
            for i in range(3)
        }

    def test_matches_direct_calls_and_is_symmetric(self):
        for metric, params, fn in (("cka", {"debiased": True}, rq.cka), ("svcca", {"threshold": 0.9}, rq.svcca)):
            recs = rq.compute_pairs(self.layers, metrics=[metric], params={metric: params})
            table = {(r.layer, r.layer_b): r for r in recs.rows}
            self.assertEqual(len(table), 9)
            self.assertAlmostEqual(table[(0, 2)].value, fn(self.layers[0], self.layers[2], **params).value, places=12)
            self.assertAlmostEqual(table[(0, 2)].value, table[(2, 0)].value, places=12)
            self.assertAlmostEqual(table[(1, 1)].value, 1.0, places=10)
            self.assertEqual(table[(0, 2)].params, params)

    def test_two_checkpoints(self):
        later = {i: x + 0.05 * torch.randn_like(x) for i, x in self.layers.items()}
        recs = rq.compute_pairs(self.layers, later, metrics=["cka"], model="step1000", model_b="step2000")
        self.assertEqual(recs.rows[0].model, "step1000->step2000")
        self.assertGreater(min(r.value for r in recs.rows if r.layer == r.layer_b), 0.95)


if __name__ == "__main__":
    unittest.main()
