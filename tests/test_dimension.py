"""Neighbor-based metrics on manifolds of known dimension."""

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


class RegistryTests(unittest.TestCase):
    def test_registered(self):
        for name in (
            "intrinsic_dimension/twonn",
            "intrinsic_dimension/gride",
            "intrinsic_dimension/mle",
            "intrinsic_dimension/mlid",
            "intrinsic_dimension/mst",
        ):
            self.assertIn(name, rq.list_metrics())
        self.assertIn("paper-canonical", rq.get_metric("intrinsic_dimension/twonn").tags)


if __name__ == "__main__":
    unittest.main()
