"""Token-field health metrics on constructed token fields."""

import unittest

import torch

import req_metrics as rq


class TokenHealthTests(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.tokens = torch.randn(200, 32, generator=g)
        self.outliers = self.tokens.clone()
        self.outliers[:10, :2] *= 40.0  # 5 percent of tokens carry huge values in two channels

    def test_norm_outliers_relative_and_absolute(self):
        clean = rq.token_norm_outliers(self.tokens)
        self.assertEqual(clean.value, 0.0)
        bad = rq.token_norm_outliers(self.outliers)
        self.assertAlmostEqual(bad.value, 0.05, places=9)
        self.assertGreater(bad.extras["max_over_median"], 5.0)
        self.assertLess(bad.extras["channel_participation_ratio"], 3.0)  # energy sits in two channels
        self.assertGreater(bad.extras["top1_channel_mass"], 0.4)
        self.assertAlmostEqual(rq.token_norm_outliers(self.outliers, cutoff=1e9).value, 0.0)

    def test_token_cosine_extremes(self):
        same = torch.ones(50, 1) * torch.randn(1, 16)
        self.assertAlmostEqual(rq.token_cosine(same).value, 1.0, places=9)
        self.assertLess(abs(rq.token_cosine(self.tokens, n_tokens=64, seed=1).value), 0.1)
        self.assertEqual(rq.token_cosine(self.tokens, n_tokens=64).extras["n_tokens"], 64.0)

    def test_cls_patch_cosine(self):
        patches = self.tokens
        cls = patches.mean(0, keepdim=True) * 5 + 0.1 * torch.randn(1, 32)
        aligned = rq.cls_patch_cosine(torch.cat([cls, patches]))
        orthogonal = rq.cls_patch_cosine(torch.cat([torch.randn(1, 32), patches]))
        self.assertGreater(aligned.value, orthogonal.value)

    def test_gram_drift(self):
        self.assertAlmostEqual(rq.token_gram_drift(self.tokens, self.tokens.clone()).value, 0.0, places=12)
        self.assertAlmostEqual(
            rq.token_gram_drift(self.tokens, self.tokens[torch.randperm(200)]).value, 2.0 / 32, delta=0.02
        )  # independent cosines ~ N(0, 1/D)
        with self.assertRaises(ValueError):
            rq.token_gram_drift(self.tokens, self.tokens[:100])

    def test_pipeline_frames_and_tokens_populations(self):
        clips = {0: [self.outliers, self.tokens, self.tokens], 1: [self.tokens] * 3}
        rec = rq.compute(clips, ["token_norm_outliers", "token_cosine"], population="frames")
        self.assertAlmostEqual(rec.where(metric="token_norm_outliers", layer=0)[0].value, 0.05 / 3, places=9)
        pooled = rq.compute(clips, ["token_norm_outliers"], population="tokens")
        self.assertAlmostEqual(pooled.where(layer=0)[0].value, 10 / 600, places=9)  # against the pooled median
        with self.assertRaises(ValueError):
            rq.compute({0: self.tokens}, ["token_cosine"], population="pooled")
        for n in ("token_norm_outliers", "token_cosine", "cls_patch_cosine", "token_gram_drift"):
            self.assertIn(n, rq.list_metrics())

    def test_token_fields_run_per_clip_under_the_tokens_population(self):
        g = torch.Generator().manual_seed(3)
        clips = []
        for _ in range(5):
            cls = torch.randn(1, 32, generator=g)  # each clip's patches follow its own class token
            clips.append(torch.cat([cls, cls + 0.3 * torch.randn(16, 32, generator=g)]))
        want = sum(rq.cls_patch_cosine(c).value for c in clips) / len(clips)
        for population in ("frames", "tokens"):
            rec = rq.compute({0: clips}, ["cls_patch_cosine"], population=population)
            self.assertAlmostEqual(rec[0].value, want, places=9)


if __name__ == "__main__":
    unittest.main()


class EmbeddingNormAndMixedKindsTests(unittest.TestCase):
    def test_embedding_norm(self):
        g = torch.Generator().manual_seed(0)
        x = 3.0 * torch.randn(4000, 64, generator=g)
        r = rq.embedding_norm(x)
        self.assertAlmostEqual(r.value, 3.0 * 8.0, delta=0.2)  # E||x|| ~ sigma sqrt(D)
        self.assertLess(r.extras["cv"], 0.15)
        self.assertIn("embedding_norm", rq.list_metrics())

    def test_mixed_per_clip_kinds_in_one_frames_call(self):
        g = torch.Generator().manual_seed(1)
        clips = {0: [torch.randn(80, 12, generator=g).cumsum(0) for _ in range(6)]}
        rec = rq.compute(
            clips,
            ["effective_rank", "trajectory_curvature", "token_norm_outliers", "embedding_norm"],
            population="frames",
        )
        self.assertEqual(
            sorted(r.metric for r in rec),
            ["effective_rank", "embedding_norm", "token_norm_outliers", "trajectory_curvature"],
        )
        self.assertEqual(rec.where(metric="trajectory_curvature")[0].extras["n_items"], 6)
        with self.assertRaises(ValueError):
            rq.compute({0: torch.randn(50, 4)}, ["effective_rank", "lidar"])
