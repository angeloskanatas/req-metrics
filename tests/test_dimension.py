"""Neighbor-based metrics on manifolds of known dimension."""

import math
import unittest

import torch

import req_metrics as rq


def ball(n, d, ambient, seed=0):
    """Uniform points in a d-ball, isometrically embedded in `ambient` dimensions."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, d, generator=g)
    v = v / v.norm(dim=1, keepdim=True) * torch.rand(n, 1, generator=g) ** (1.0 / d)
    basis = torch.linalg.qr(torch.randn(ambient, d, generator=g)).Q.T  # (d, ambient), orthonormal rows
    return v @ basis


class DimensionTests(unittest.TestCase):
    def test_five_ball_in_twenty_dims(self):
        x = ball(4000, 5, 20)
        self.assertAlmostEqual(rq.twonn(x).value, 5.0, delta=0.6)
        self.assertAlmostEqual(rq.twonn(x, algorithm="ml").value, 5.0, delta=0.6)
        self.assertAlmostEqual(rq.mle(x).value, 5.0, delta=0.6)
        self.assertAlmostEqual(rq.mlid(x, k=64).value, 5.0, delta=1.0)
        g = rq.gride(x, scale=8, range_max=64)
        self.assertAlmostEqual(g.value, 5.0, delta=0.7)
        self.assertIn("id_rank64", g.extras)
        self.assertAlmostEqual(rq.mst_dimension(x[:2000]).value, 5.0, delta=1.5)

    def test_plane_in_ten_dims(self):
        x = ball(3000, 2, 10, seed=1)
        self.assertAlmostEqual(rq.twonn(x).value, 2.0, delta=0.3)
        self.assertAlmostEqual(rq.gride(x, scale=4).value, 2.0, delta=0.3)
        self.assertAlmostEqual(rq.mle(x).value, 2.0, delta=0.3)
        self.assertAlmostEqual(rq.mst_dimension(x[:1500]).value, 2.0, delta=0.5)

    def test_shared_neighbors_table(self):
        x = ball(1500, 3, 8, seed=2)
        nb = rq.Neighbors.from_points(x, 64)
        self.assertAlmostEqual(rq.mlid(nb, k=64).value, rq.mlid(x, k=64).value, places=9)
        self.assertAlmostEqual(rq.gride(nb, scale=8).value, rq.gride(x, scale=8).value, places=9)
        self.assertAlmostEqual(rq.mle(nb).value, rq.mle(x).value, places=9)

    def test_duplicates_do_not_break_twonn(self):
        x = ball(1000, 3, 6, seed=3)
        x = torch.cat([x, x[:50]])
        self.assertAlmostEqual(rq.twonn(x).value, 3.0, delta=0.5)

    def test_neighbor_table_conventions(self):
        x = torch.randn(200, 4)
        nb = rq.Neighbors.from_points(x, 5)
        self.assertEqual(tuple(nb.distances.shape), (200, 6))
        self.assertTrue(torch.all(nb.distances[:, 0] == 0))
        self.assertTrue(torch.all(nb.indices[:, 0] == torch.arange(200)))
        self.assertTrue(torch.all(nb.distances[:, 1:] >= 0))
        self.assertTrue(torch.all(nb.distances.diff(dim=1) >= 0))


class LocalGeometryTests(unittest.TestCase):
    def test_curvature_analytic_lattice_and_curve(self):
        # square lattice: the 4 axis neighbors give mean pairwise cosine -1/3, the 8 nearest give -1/7
        ij = torch.cartesian_prod(torch.arange(40.0), torch.arange(40.0))
        self.assertAlmostEqual(rq.neighborhood_curvature(ij, k=4).value, -1.0 / 3.0, delta=0.05)
        self.assertAlmostEqual(rq.neighborhood_curvature(ij, k=8).value, -1.0 / 7.0, delta=0.05)
        t = torch.linspace(0, 2 * math.pi, 2000)[:-1]
        circle = torch.stack([t.cos(), t.sin()], dim=1)
        self.assertLess(rq.neighborhood_curvature(circle, k=2).value, -0.9)  # one neighbor each side
        # a Gaussian cloud is not an isotropy null: outer points see neighbors biased toward the center
        torch.manual_seed(0)
        self.assertGreater(rq.neighborhood_curvature(torch.randn(3000, 8), k=32).value, 0.0)

    def test_rectifiability_prefers_true_tangent_dimension(self):
        x = ball(1500, 2, 10, seed=4)
        on_plane = rq.local_rectifiability(x, n=2, n_anchors=64, n_scales=4)
        off_plane = rq.local_rectifiability(x, n=1, n_anchors=64, n_scales=4)
        self.assertLess(on_plane.value, 0.1 * off_plane.value)
        self.assertIn("local_id_scale0", on_plane.extras)


class RegistryTests(unittest.TestCase):
    def test_registered(self):
        for name in (
            "intrinsic_dimension",
            "intrinsic_dimension/gride",
            "intrinsic_dimension/mle",
            "mlid",
            "mst_dimension",
            "neighborhood_curvature",
            "local_rectifiability",
        ):
            self.assertIn(name, rq.list_metrics())
        self.assertIn("paper-canonical", rq.get_metric("intrinsic_dimension/gride").tags)


if __name__ == "__main__":
    unittest.main()
