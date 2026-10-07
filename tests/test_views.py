"""View metrics: LiDAR, InfoNCE, DiME on controlled view constructions."""

import math
import unittest
import warnings

import torch

import req_metrics as rq


def views_with_noise(n=2000, d=16, q=4, noise_dims=8, noise=3.0, seed=0):
    """Class means spanning d dims; augmentation noise only on the first noise_dims, and large."""
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(n, d, generator=g)
    eps = torch.zeros(q, n, d)
    eps[:, :, :noise_dims] = noise * torch.randn(q, n, noise_dims, generator=g)
    return means.unsqueeze(0) + eps


class LidarTests(unittest.TestCase):
    def test_discounts_augmentation_directions(self):
        v = views_with_noise()
        r = rq.lidar(v)
        self.assertGreater(r.value, 7.0)  # the 8 clean directions survive the whitening
        self.assertLess(r.value, 10.5)
        plain = rq.effective_rank(v.mean(dim=0)).value
        self.assertGreater(plain, 15.0)  # the class means alone are full rank

    def test_identical_views_reduce_to_class_mean_rank(self):
        g = torch.Generator().manual_seed(1)
        x = torch.randn(1500, 12, generator=g)
        r = rq.lidar(torch.stack([x, x, x]), delta=1e-6)
        self.assertGreater(r.value, 11.5)

    def test_shape_checks(self):
        with self.assertRaises(ValueError):
            rq.lidar(torch.randn(1, 50, 4))
        with self.assertRaises(ValueError):
            rq.lidar(torch.randn(50, 4))


class InfoNCETests(unittest.TestCase):
    def test_identical_views_near_zero_independent_near_log_n(self):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(1000, 32, generator=g)
        same = rq.infonce(torch.stack([x, x]))
        self.assertLess(same.value, 0.3)  # 999 negatives at cos ~ N(0, 1/sqrt(32)) leave log(1 + ~0.2)
        self.assertGreater(same.extras["log_n_minus_loss"], 0.95 * math.log(1000))
        self.assertEqual(same.extras["contrastive_accuracy"], 1.0)
        indep = rq.infonce(torch.stack([x, torch.randn(1000, 32, generator=g)]))
        # unrelated views: loss ~ log N + var(cos / tau) / 2 = 6.9 + 1.6, so the MI bound is at or below zero
        self.assertGreater(indep.value, math.log(1000))
        self.assertLess(indep.extras["log_n_minus_loss"], 0.0)
        self.assertGreater(indep.value, same.value)

    def test_matches_previous_formula(self):
        g = torch.Generator().manual_seed(2)
        a = torch.randn(300, 8, generator=g)
        b = a + 0.5 * torch.randn(300, 8, generator=g)

        def old(u, w, t=0.1):  # earlier implementation: center, l2, cosine logits over t, cross-entropy
            u = u - u.mean(0)
            w = w - w.mean(0)
            u = u / u.norm(dim=1, keepdim=True)
            w = w / w.norm(dim=1, keepdim=True)
            return float(torch.nn.functional.cross_entropy((u @ w.T) / t, torch.arange(300)))

        self.assertAlmostEqual(rq.infonce(torch.stack([a, b])).value, old(a.double(), b.double()), places=9)

    def test_view_pairs_full_graph_core_view_and_symmetric(self):
        g = torch.Generator().manual_seed(3)
        x = torch.randn(400, 16, generator=g)
        v = torch.stack([x + 0.3 * torch.randn(400, 16, generator=g) for _ in range(4)])
        pair = {(a, b): rq.infonce(v[[a, b]]).value for a in range(4) for b in range(4) if a != b}
        full = rq.infonce(v)
        self.assertAlmostEqual(full.value, sum(pair[(a, b)] for a in range(4) for b in range(a + 1, 4)) / 6, places=12)
        self.assertEqual(full.extras["n_pairs"], 6)
        core = rq.infonce(v, anchor=2)
        self.assertAlmostEqual(core.value, (pair[(2, 0)] + pair[(2, 1)] + pair[(2, 3)]) / 3, places=12)
        sym = rq.infonce(v[:2], symmetric=True)
        self.assertAlmostEqual(sym.value, (pair[(0, 1)] + pair[(1, 0)]) / 2, places=12)
        with self.assertRaises(ValueError):
            rq.infonce(v, anchor=4)


class DimeTests(unittest.TestCase):
    def test_paired_views_above_independent(self):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(300, 16, generator=g)
        paired = rq.dime(torch.stack([x, x + 0.3 * torch.randn(300, 16, generator=g)]), n_perm=3)
        indep = rq.dime(torch.stack([x, torch.randn(300, 16, generator=g)]), n_perm=3)
        self.assertGreater(paired.value, 0.2)
        self.assertLess(abs(indep.value), 0.05)
        self.assertLess(paired.extras["joint_entropy"], paired.extras["permuted_joint_entropy"])

    def test_dxd_shortcut_is_not_the_definition(self):
        """The old N > D shortcut (Hadamard product of D x D covariances) differs from Eq. 2.2."""
        g = torch.Generator().manual_seed(3)
        x = torch.randn(200, 10, generator=g).double()
        y = (x + 0.5 * torch.randn(200, 10, generator=g)).double()
        kx, ky = x @ x.T, y @ y.T  # N x N Grams (unnormalized rows, as the old code)
        cx, cy = x.T @ x, y.T @ y  # D x D covariances, the shortcut

        def vn(k):
            lam = torch.linalg.eigvalsh(k).clamp_min(0)
            p = lam / lam.sum()
            p = p[p > 0]
            return float(-(p * p.log()).sum())

        self.assertAlmostEqual(vn(kx), vn(cx), places=9)  # single matrices agree (same spectrum)
        self.assertNotAlmostEqual(vn(kx * ky), vn(cx * cy), places=2)  # Hadamard products do not

    def test_rbf_and_seed(self):
        g = torch.Generator().manual_seed(4)
        v = torch.stack([torch.randn(200, 6, generator=g)] * 2)
        self.assertGreater(rq.dime(v, kernel="rbf", n_perm=3).value, 0.5)
        self.assertEqual(rq.dime(v, seed=7, n_perm=2).value, rq.dime(v, seed=7, n_perm=2).value)


class ViewConstructionTests(unittest.TestCase):
    def test_make_views_shapes_seeds_and_order(self):
        torch.manual_seed(0)
        clips = torch.randn(50, 20)  # 50 "waveforms" of 20 samples
        proj = torch.randn(20, 6)
        encode = lambda batch: batch @ proj  # any user readout: (B, 20) -> (B, 6)
        augment = lambda batch, gen: batch + 0.1 * torch.randn(batch.shape, generator=gen)
        v = rq.make_views(encode, clips, augment, q=3, seed=1, batch_size=16)
        self.assertEqual(tuple(v.shape), (3, 50, 6))
        v2 = rq.make_views(encode, clips, augment, q=3, seed=1, batch_size=8)
        self.assertTrue(torch.allclose(v, v2))  # the seeded noise stream does not depend on the batch size
        self.assertFalse(torch.allclose(v[0], v[1]))  # passes differ
        self.assertTrue(torch.allclose(rq.make_views(encode, clips, augment, q=3, seed=1, batch_size=16), v))
        spec = rq.ViewSpec(
            source="shared", augmentations=("Gain(-9..6 dB, p=0.6)",), excluded=("PitchShift",), q=3, seed=1
        )
        self.assertIn("excluded: PitchShift", spec.describe())
        self.assertEqual(tuple(rq.stack_views(v[0], v[1]).shape), (2, 50, 6))
        with self.assertRaises(ValueError):
            rq.stack_views(v[0])


if __name__ == "__main__":
    unittest.main()


class IdenticalViewsTests(unittest.TestCase):
    def test_identical_views_warn_once(self):
        x = torch.randn(200, 16)
        stack = torch.stack([x, x, x], dim=0)  # q = 3 identical passes
        with self.assertWarnsRegex(RuntimeWarning, "identical"):
            rq.compute({0: stack}, ["lidar"])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            rq.compute({0: torch.stack([x, x + 0.1 * torch.randn_like(x)], dim=0)}, ["lidar"])
