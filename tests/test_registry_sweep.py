"""Every registered metric runs through the pipeline on synthetic inputs without recording an error."""

import unittest

import torch

import req_metrics as rq

SMALL = {"cluster_quality": {"k": 8}}  # the default k = 1024 needs more points than these fixtures


class RegistrySweepTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _assert_clean(self, rec):
        bad = [(r.metric, r.extras.get("error")) for r in rec if "error" in r.extras or r.value != r.value]
        self.assertEqual(bad, [], f"metrics failed through the pipeline: {bad}")

    def test_point_metrics(self):
        names = [m for m in rq.list_metrics() if rq.get_metric(m).inputs == rq.InputKind.POINTS]
        x = torch.randn(600, 24) @ torch.diag(torch.linspace(1, 0.1, 24))
        self._assert_clean(rq.compute({0: x, 1: 2 * x + 1}, names, n=600, params=SMALL))

    def test_point_metrics_on_frames_and_tokens(self):
        names = [
            m
            for m in rq.list_metrics()
            if rq.get_metric(m).inputs == rq.InputKind.POINTS
            and m not in ("intrinsic_dimension/mst", "local_rectifiability")
        ]  # slow per clip
        clips = [torch.randn(80, 16) for _ in range(6)]
        self._assert_clean(rq.compute({0: clips}, names, population="frames", n=6, params=SMALL))
        self._assert_clean(rq.compute({0: clips}, names, population="tokens", n=400, params=SMALL))

    def test_trajectory_and_token_metrics(self):
        traj = [torch.cumsum(torch.randn(60, 16), dim=0) for _ in range(5)]
        names = [m for m in rq.list_metrics() if rq.get_metric(m).inputs == rq.InputKind.TRAJECTORY]
        self._assert_clean(rq.compute({0: traj}, names, population="frames"))
        toks = [torch.randn(40, 16) for _ in range(5)]
        names = [
            m for m in rq.list_metrics() if rq.get_metric(m).inputs == rq.InputKind.TOKENS and m != "cls_patch_cosine"
        ]
        self._assert_clean(rq.compute({0: toks}, names, population="tokens"))
        self._assert_clean(
            rq.compute(
                {0: toks}, ["cls_patch_cosine"], population="tokens", params={"cls_patch_cosine": {"n_prefix": 1}}
            )
        )

    def test_view_and_pair_metrics(self):
        base = torch.randn(300, 16)
        views = torch.stack([base + 0.3 * torch.randn(300, 16) for _ in range(3)])
        names = [
            m
            for m in rq.list_metrics()
            if rq.get_metric(m).inputs == rq.InputKind.VIEWS and m not in ("infonce", "dime")
        ]
        self._assert_clean(rq.compute({0: views}, names, views=rq.ViewSpec(source="objective", q=3)))
        self._assert_clean(
            rq.compute(
                {0: views[:2]},
                ["infonce", "dime"],
                params={"dime": {"n_perm": 2}},
                views=rq.ViewSpec(source="objective", q=2),
            )
        )
        layers = {0: base, 1: base[:, :8] + 0.2 * torch.randn(300, 8)}
        self._assert_clean(rq.compute_pairs(layers))
        self._assert_clean(rq.compute_pairs(layers, metric="neighborhood_overlap", k=10))


if __name__ == "__main__":
    unittest.main()
