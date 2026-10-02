"""Token layouts for frame-sequence and spectrogram-patch encoders, and the pairs algorithm."""

import unittest

import torch

import req_metrics as rq


class LayoutTests(unittest.TestCase):
    def test_grid_to_trajectory_concat_and_mean(self):
        f, t, d = 4, 10, 6
        grid = torch.arange(f * t * d, dtype=torch.float32).reshape(f, t, d)  # (F, T, D), freq-major
        traj = rq.grid_to_trajectory(grid)
        self.assertEqual(tuple(traj.shape), (t, f * d))
        self.assertTrue(torch.equal(traj[3, :d], grid[0, 3]))  # band 0 first, then band 1 ...
        self.assertTrue(torch.equal(traj[3, d : 2 * d], grid[1, 3]))
        self.assertEqual(tuple(rq.grid_to_trajectory(grid, mode="freq_mean").shape), (t, d))
        self.assertTrue(torch.allclose(rq.grid_to_trajectory(grid.transpose(0, 1), time_axis=0), traj))
        self.assertEqual(tuple(rq.grid_to_tokens(grid).shape), (f * t, d))
        self.assertEqual(tuple(rq.grid_to_pooled(grid).shape), (d,))
        freq_concat_mean = rq.grid_to_pooled(grid, mode="freq_concat_mean")
        self.assertEqual(tuple(freq_concat_mean.shape), (f * d,))
        self.assertTrue(
            torch.allclose(freq_concat_mean, traj.mean(0))
        )  # freq_concat_mean = time mean of the concat trajectory
        part = rq.grid_to_pooled(grid, mode="partitioned", freq_chunks=2, time_chunks=5)
        self.assertEqual(tuple(part.shape), (2 * 5 * d,))
        self.assertTrue(
            torch.allclose(rq.grid_to_pooled(grid, mode="partitioned", freq_chunks=f, time_chunks=1), freq_concat_mean)
        )
        self.assertTrue(torch.allclose(rq.grid_to_pooled(grid, mode="partitioned"), rq.grid_to_pooled(grid)))

    def test_frame_pooling_and_prefix(self):
        frames = torch.randn(20, 5)
        self.assertTrue(torch.allclose(rq.frame_tokens_to_pooled(frames), frames.mean(0)))
        self.assertTrue(torch.equal(rq.frame_tokens_to_pooled(frames, mode="last"), frames[-1]))
        self.assertTrue(torch.equal(rq.frame_tokens_to_pooled(frames, mode="max"), frames.max(0).values))
        tokens = torch.randn(3, 11, 5)  # (N, 1 + 10 patches, D)
        self.assertEqual(tuple(rq.strip_prefix_tokens(tokens).shape), (3, 10, 5))
        with self.assertRaises(ValueError):
            rq.stack_clips([torch.randn(4, 3), torch.randn(5, 3)])
        self.assertEqual(tuple(rq.stack_clips([torch.randn(4, 3), torch.randn(4, 3)]).shape), (2, 4, 3))

    def test_grid_trajectory_feeds_frames_population(self):
        g = torch.Generator().manual_seed(0)
        clips = {0: [rq.grid_to_trajectory(torch.randn(4, 30, 6, generator=g).cumsum(1)) for _ in range(8)]}
        rec = rq.compute(clips, ["trajectory_curvature"], population="frames", pooling="freq-concat")
        self.assertEqual(rec[0].dim, 24)
        self.assertEqual(rec[0].pooling, "freq-concat")


class PairsAlgorithmTests(unittest.TestCase):
    def test_pairs_equal_direct_metric(self):
        g = torch.Generator().manual_seed(1)
        base = torch.randn(150, 5, generator=g)
        layers = {0: base, 1: base + 0.5 * torch.randn(150, 5, generator=g), 2: torch.randn(150, 5, generator=g)}
        for k in (1, 2):
            rec = rq.compute_pairs(layers, k=k, chunk=64)
            for la in layers:
                for lb in layers:
                    direct = rq.information_imbalance(layers[la], layers[lb], k=k)
                    row = rec.where(layer=la, layer_b=lb)[0]
                    self.assertAlmostEqual(row.value, direct.value, places=10)
                    self.assertAlmostEqual(row.extras["reverse"], direct.extras["reverse"], places=10)


if __name__ == "__main__":
    unittest.main()
