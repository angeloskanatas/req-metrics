"""PTE on a synthetic representation with known transposition structure."""

import unittest

import torch

import req_metrics as rq


def tonal_embeddings(n=700, d=24, noise=0.05, seed=0):
    """Clips with a latent key in Z12 and an embedding linear in its one-hot; shifting adds k to the key."""
    g = torch.Generator().manual_seed(seed)
    keys = torch.randint(0, 12, (n,), generator=g)
    w = torch.randn(12, d, generator=g)

    def embed(kk):
        return torch.nn.functional.one_hot(kk % 12, 12).float() @ w + noise * torch.randn(n, d, generator=g)

    z = embed(keys)
    shifted = {k: embed(keys + k) for k in (1, 3, 5, 7)}
    return z, shifted, keys, w


class PTETests(unittest.TestCase):
    def test_equivariant_representation_scores_high(self):
        z, shifted, _, _ = tonal_embeddings()
        r = rq.pte(z, shifted, epochs=60, batch_size=128, patience=10, seed=1)
        self.assertGreater(r.value, 0.9)
        self.assertGreater(r.extras["mean_abs_cpsd"], 0.3)  # the trainedness gate
        self.assertLess(r.extras["phase_rmse"], 0.3)
        self.assertIn("rmse_k7", r.extras)

    def test_unrelated_shifts_score_near_random_phase(self):
        z, shifted, _, _ = tonal_embeddings()
        g = torch.Generator().manual_seed(9)
        scrambled = {k: v[torch.randperm(v.shape[0], generator=g)] for k, v in shifted.items()}  # breaks the pairing
        r = rq.pte(z, scrambled, epochs=40, batch_size=128, patience=8, seed=1)
        self.assertLess(r.value, 0.6)

    def test_deterministic_and_variants(self):
        z, shifted, _, _ = tonal_embeddings(n=500, d=12)
        a = rq.pte(z, shifted, epochs=15, batch_size=128, seed=3)
        b = rq.pte(z, shifted, epochs=15, batch_size=128, seed=3)
        self.assertEqual(a.value, b.value)
        c = rq.pte(z, shifted, epochs=15, batch_size=128, seed=3, score="cpsd")
        self.assertAlmostEqual(c.value, 1 - c.extras["cpsd_rmse"] / 2, places=12)
        m = rq.pte(z, shifted, epochs=10, batch_size=128, seed=3, probe="mlp", hidden_units=(32,))
        self.assertTrue(0.0 <= m.value <= 1.0)

    def test_input_checks(self):
        z, shifted, _, _ = tonal_embeddings(n=400, d=8)
        with self.assertRaises(ValueError):
            rq.pte(z, {0: z}, epochs=1, batch_size=64)
        with self.assertRaises(ValueError):
            rq.pte(z, {1: z[:10]}, epochs=1, batch_size=64)
        with self.assertRaises(ValueError):
            rq.pte(z, shifted, epochs=1, batch_size=10000)

    def test_make_shifted_and_spec(self):
        clips = torch.randn(30, 16)
        proj = torch.randn(16, 5)
        out = rq.make_shifted(lambda b: b @ proj, clips, lambda b, k: b * (1 + 0.01 * k), (1, -2), batch_size=7)
        self.assertEqual(set(out), {1, -2})
        self.assertEqual(tuple(out[1].shape), (30, 5))
        spec = rq.ShiftSpec(method="waveform pitch shift", semitones=(1, -2))
        self.assertIn("up for positive k", spec.describe())
        self.assertIn("pte", rq.list_metrics())
        self.assertEqual(rq.get_metric("pte").inputs, rq.InputKind.SHIFTED)


if __name__ == "__main__":
    unittest.main()
