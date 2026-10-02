"""Time every registered point metric on one random (N, D) cloud and print a Markdown table.

Usage: python scripts/benchmark.py --n 10000 --d 768 [--device cuda] [--skip pte,dime]
Costs are for one layer; a sweep multiplies by the number of layers. Metrics that
share a Spectrum or Neighbors table are timed once with the shared object so the
numbers add up the way compute() spends them.
"""

from __future__ import annotations

import argparse
import time

import torch

import req_metrics as rq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--skip", default="")
    a = ap.parse_args()
    skip = set(filter(None, a.skip.split(",")))
    torch.manual_seed(0)
    x = torch.randn(a.n, a.d, device=a.device)
    rows = []

    def timed(name, fn):
        t = time.perf_counter()
        fn()
        rows.append((name, time.perf_counter() - t))

    t = time.perf_counter()
    spec = rq.Spectrum.from_points(x)
    rows.append(("Spectrum (shared SVD)", time.perf_counter() - t))
    for name in (
        "effective_rank",
        "spectral_entropy",
        "matrix_entropy",
        "alpha_req",
        "participation_ratio",
        "eigenvalue_early_enrichment",
    ):
        timed(name + " (from Spectrum)", lambda name=name: rq.get_metric(name).fn(spec))
    timed("anisotropy (own spectrum, center+l2)", lambda: rq.anisotropy_spectral(x))
    t = time.perf_counter()
    nb = rq.Neighbors.from_points(x, 64)
    rows.append(("Neighbors k=64 (shared kNN)", time.perf_counter() - t))
    timed("mlid (from Neighbors)", lambda: rq.mlid(nb, k=64))
    timed("intrinsic_dimension/gride (from Neighbors, range 64)", lambda: rq.gride(nb, scale=8, range_max=64))
    timed("intrinsic_dimension/mle (from Neighbors)", lambda: rq.mle(nb))
    timed("intrinsic_dimension (twonn, own kNN)", lambda: rq.twonn(x))
    timed("neighborhood_curvature (from Neighbors)", lambda: rq.neighborhood_curvature(x, k=64, neighbors=nb))
    timed("gaussianity (256 directions)", lambda: rq.gaussianity(x))
    timed("gaussianity/swd", lambda: rq.gaussianity(x, method="swd"))
    timed("sparsity", lambda: rq.sparsity(x))
    timed("embedding_norm", lambda: rq.embedding_norm(x))
    timed("mst_dimension (first 1024 points)", lambda: rq.mst_dimension(x[:1024]))
    timed(
        "local_rectifiability (2048 points, 128 anchors)",
        lambda: rq.local_rectifiability(x[:2048], n=16, n_anchors=128, n_scales=4),
    )
    timed("trajectory_curvature (one 2000-frame clip)", lambda: rq.trajectory_curvature(x[:2000]))
    views = x.unsqueeze(0) + 0.1 * torch.randn(a.views, a.n, a.d, device=a.device)
    timed(f"lidar ({a.views} views)", lambda: rq.lidar(views))
    timed("infonce (2 views)", lambda: rq.infonce(views[:2]))
    if "dime" not in skip:
        timed("dime (2 views, first 2000 clips, 10 permutations)", lambda: rq.dime(views[:2, :2000]))
    timed(
        "information_imbalance (two layers)",
        lambda: rq.information_imbalance(x, x @ torch.randn(a.d, a.d, device=a.device)),
    )
    print(f"\nN={a.n}, D={a.d}, device={a.device}, torch {torch.__version__}\n")
    print("| metric | seconds |\n|---|---|")
    for name, sec in rows:
        print(f"| {name} | {sec:.2f} |")


if __name__ == "__main__":
    main()
