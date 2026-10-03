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

    def test_compute_device_argument_matches_the_cpu(self):
        g = torch.Generator().manual_seed(1)
        clips = {l: [torch.randn(50, 8, generator=g).cumsum(0) for _ in range(30)] for l in range(2)}
        for population, n in (("frames", 10), ("tokens", 600)):
            cpu = rq.compute(clips, ["effective_rank", "intrinsic_dimension"], population=population, n=n)
            gpu = rq.compute(
                clips, ["effective_rank", "intrinsic_dimension"], population=population, n=n, device="cuda"
            )
            for a, b in zip(cpu, gpu, strict=True):
                self.assertAlmostEqual(a.value, b.value, delta=1e-6 * max(1.0, abs(a.value)), msg=population)

    def test_monitor_view_passes_run_per_layer_on_the_model_device(self):
        model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)).cuda()
        data = [torch.randn(16, 8, generator=torch.Generator().manual_seed(i)) for i in range(4)]
        mon = rq.LayerMonitor(list(model), pool=lambda o: o, metrics=["effective_rank"], n_items=64,
                              view_metrics=["lidar"], augment=lambda x, g: x + 0.1 * torch.randn(x.shape, generator=g), q=3)  # fmt: skip
        rec = mon.sweep(lambda x: model(x.cuda()), data, step=0)
        self.assertEqual(len(rec.where(metric="lidar").rows), 2)


if __name__ == "__main__":
    unittest.main()
