"""Jacobian effective rank: exact on linear maps, products equal to J v, every layer in one pass."""

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

    def test_products_equal_jacobian_vector_products_for_every_layer(self):
        torch.manual_seed(0)
        net, x, pool = _Net().eval(), torch.randn(2, 5, 16), rq.make_pooler("mean")
        fast = torch.backends.mha.get_fastpath_enabled()
        with torch.no_grad():
            products = rq.jacobian_products(net, x, list(net.blocks), pool, num_probes=4, seed=1)
        self.assertEqual(torch.backends.mha.get_fastpath_enabled(), fast)
        self.assertEqual({i: tuple(t.shape) for i, t in products.items()}, {i: (2, 4, 16) for i in range(3)})
        gen = torch.Generator().manual_seed(1)
        probe = torch.linalg.qr(torch.randn(80, 4, generator=gen, dtype=torch.float64))[0].T.float()[0].view(5, 16)
        from torch.nn.attention import SDPBackend, sdpa_kernel

        torch.backends.mha.set_fastpath_enabled(False)
        try:
            with sdpa_kernel(SDPBackend.MATH):
                jac = jacrev(lambda z: pool(net.blocks[1](net.blocks[0](z.unsqueeze(0))))[0])(x[0])
        finally:
            torch.backends.mha.set_fastpath_enabled(fast)
        self.assertTrue(torch.allclose(products[1][0, 0], torch.einsum("dtk,tk->d", jac, probe), atol=1e-5))

    def test_registered_and_computed_through_the_pipeline(self):
        rec = rq.compute({0: torch.randn(3, 4, 10), 1: torch.randn(3, 4, 10)}, ["jacobian_effective_rank"])
        self.assertEqual([r.layer for r in rec], [0, 1])
        self.assertTrue(all(1.0 <= r.value <= 4.0 for r in rec))
        self.assertEqual(rq.get_metric("jacobian_effective_rank").inputs, rq.InputKind.JACOBIAN)


if __name__ == "__main__":
    unittest.main()
