"""The same seed gives the same value on CPU and GPU tensors (run where a GPU is available)."""

import unittest

import torch

import req_metrics as rq


@unittest.skipUnless(torch.cuda.is_available(), "no GPU")
class DeviceAgreementTests(unittest.TestCase):
    def test_point_and_view_metrics_match_across_devices(self):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(300, 6, generator=g) @ torch.randn(6, 16, generator=g) + 0.3 * torch.randn(300, 16, generator=g)
        views = torch.stack([x + 0.2 * torch.randn(300, 16, generator=g) for _ in range(2)])
        tokens = torch.randn(40, 12, generator=g)
        cases = [
            (rq.twonn, (x,), {}),
            (rq.gaussianity, (x,), {}),
            (rq.local_rectifiability, (x,), {"n": 3, "n_anchors": 64}),
            (rq.dime, (views,), {"n_perm": 3}),
            (rq.token_cosine, (tokens,), {"n_tokens": 20}),
        ]
        for fn, args, kw in cases:
            cpu = fn(*args, **kw).value
            gpu = fn(*(a.cuda() for a in args), **kw).value
            self.assertAlmostEqual(cpu, gpu, delta=1e-6 * max(1.0, abs(cpu)), msg=fn.__name__)


if __name__ == "__main__":
    unittest.main()
