"""Trajectory curvature: analytic nulls, conventions, and the RECURVE normalization."""

import math
import unittest

import torch

import req_metrics as rq


class TrajectoryCurvatureTests(unittest.TestCase):
    def test_straight_line_is_zero(self):
        z = torch.arange(50.0).unsqueeze(1) * torch.tensor([[1.0, 2.0, -0.5]])
        for conv in ("signed", "abs"):
            self.assertAlmostEqual(rq.trajectory_curvature(z, convention=conv).value, 0.0, places=6)
        self.assertAlmostEqual(rq.trajectory_curvature(z, normalize="path_length").value, 0.0, places=6)

    def test_independent_frames_give_120_and_60_degrees(self):
        torch.manual_seed(0)
        z = torch.randn(20000, 64)
        signed = rq.trajectory_curvature(z)
        self.assertAlmostEqual(signed.value, 2 * math.pi / 3, delta=0.02)
        self.assertAlmostEqual(signed.extras["degrees"], 120.0, delta=1.5)
        self.assertAlmostEqual(signed.extras["mean_cos"], -0.5, delta=0.01)
        self.assertAlmostEqual(rq.trajectory_curvature(z, convention="abs").value, math.pi / 3, delta=0.02)
        self.assertAlmostEqual(rq.trajectory_curvature(z, k=2).value, 2 * math.pi / 3, delta=0.02)

    def test_random_walk_gives_90_degrees(self):
        torch.manual_seed(1)
        z = torch.randn(20000, 64).cumsum(dim=0)
        self.assertAlmostEqual(rq.trajectory_curvature(z).value, math.pi / 2, delta=0.02)
        self.assertLess(rq.trajectory_curvature(z, convention="abs").value, math.pi / 2)

    def test_recurve_on_regular_polygon(self):
        n, radius = 36, 3.0  # turning angle 2pi/n, step length 2 r sin(pi/n)
        t = torch.arange(n + 3, dtype=torch.float64) * (2 * math.pi / n)
        z = radius * torch.stack([t.cos(), t.sin()], dim=1)
        theta, step = 2 * math.pi / n, 2 * radius * math.sin(math.pi / n)
        self.assertAlmostEqual(rq.trajectory_curvature(z).value, theta, places=9)
        self.assertAlmostEqual(rq.trajectory_curvature(z, normalize="path_length").value, theta / (2 * step), places=9)

    def test_zero_steps_are_excluded_and_counted(self):
        z = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 1.0]])
        r = rq.trajectory_curvature(z)
        self.assertEqual(r.extras["n_zero_steps"], 2.0)  # the repeated frame kills two consecutive pairs
        self.assertEqual(r.extras["n_angles"], 1.0)
        with self.assertRaises(ValueError):
            rq.trajectory_curvature(torch.zeros(5, 2))

    def test_registry_variants(self):
        torch.manual_seed(2)
        z = torch.randn(500, 8)
        self.assertAlmostEqual(
            rq.get_metric("trajectory_curvature/abs").fn(z).value,
            rq.trajectory_curvature(z, convention="abs").value,
            places=12,
        )
        self.assertEqual(rq.get_metric("trajectory_curvature").inputs, rq.InputKind.TRAJECTORY)


if __name__ == "__main__":
    unittest.main()
