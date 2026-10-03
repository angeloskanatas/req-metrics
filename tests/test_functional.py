"""Jacobian effective rank: exact on linear maps, power iterations, the Jacobian spectrum of every layer."""

import unittest

import torch
import torch.nn as nn
from torch.func import jacrev

import req_metrics as rq


class _Net(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d, 2, 32, batch_first=True) for _ in range(3)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class JacobianTests(unittest.TestCase):
    def test_exact_for_a_linear_map_with_a_full_probe_basis(self):
        torch.manual_seed(0)
        u, _ = torch.linalg.qr(torch.randn(12, 8))
        v, _ = torch.linalg.qr(torch.randn(8, 8))
        s = torch.tensor([5.0, 4, 3, 2, 1, 0.5, 0.25, 0.1])
        lin = nn.Linear(8, 12, bias=False)
        lin.weight.data = u @ torch.diag(s) @ v.T
        products = rq.jacobian_products(lin, torch.randn(4, 8), [lin], lambda o: o, num_probes=8)
        r = rq.jacobian_effective_rank(products[0])
        self.assertAlmostEqual(r.value, float(s.sum() ** 2 / s.square().sum()), places=4)
        self.assertEqual(r.extras["n_probes"], 8)

    def test_power_iterations_recover_the_leading_singular_values(self):
        torch.manual_seed(0)
        u, _ = torch.linalg.qr(torch.randn(64, 64, dtype=torch.float64))
        v, _ = torch.linalg.qr(torch.randn(64, 64, dtype=torch.float64))
        s = 0.7 ** torch.arange(64, dtype=torch.float64)
        lin = nn.Linear(64, 64, bias=False).double()
        lin.weight.data = u @ torch.diag(s) @ v.T
        x = torch.randn(3, 64, dtype=torch.float64)
        err = {}
        for p in (0, 5):
            sketch = rq.jacobian_products(lin, x, [lin], lambda o: o, num_probes=8, power_iters=p)[0]
            err[p] = float((torch.linalg.svdvals(sketch) - s[:8]).abs().max())
        self.assertLess(err[5], 1e-4)  # error ratio (s_9 / s_8)^11 per Halko et al., 2011
        self.assertLess(err[5], err[0] / 100)

    def test_sketch_spectrum_equals_the_jacobian_spectrum_of_every_layer(self):
        torch.manual_seed(0)
        net, x, pool = _Net().eval(), torch.randn(2, 5, 16), rq.make_pooler("mean")
        fast = torch.backends.mha.get_fastpath_enabled()
        with torch.no_grad():
            sketches = rq.jacobian_products(net, x, list(net.blocks), pool, num_probes=16, power_iters=1, seed=1)
        self.assertEqual(torch.backends.mha.get_fastpath_enabled(), fast)
        self.assertEqual({i: tuple(t.shape) for i, t in sketches.items()}, {i: (2, 16, 80) for i in range(3)})
        from torch.nn.attention import SDPBackend, sdpa_kernel

        torch.backends.mha.set_fastpath_enabled(False)
        try:
            with sdpa_kernel(SDPBackend.MATH):
                jac = jacrev(lambda z: pool(net.blocks[1](net.blocks[0](z.unsqueeze(0))))[0])(x[0])
        finally:
            torch.backends.mha.set_fastpath_enabled(fast)
        exact = torch.linalg.svdvals(jac.reshape(16, 80).double())  # 16 readout dims: k = 16 spans the range
        self.assertTrue(torch.allclose(torch.linalg.svdvals(sketches[1][0].double()), exact, rtol=1e-4, atol=1e-6))

    def test_registered_and_computed_through_the_pipeline(self):
        rec = rq.compute({0: torch.randn(3, 4, 10), 1: torch.randn(3, 4, 10)}, ["jacobian_effective_rank"])
        self.assertEqual([r.layer for r in rec], [0, 1])
        self.assertTrue(all(1.0 <= r.value <= 4.0 for r in rec))
        self.assertEqual(rq.get_metric("jacobian_effective_rank").inputs, rq.InputKind.JACOBIAN)


if __name__ == "__main__":
    unittest.main()
