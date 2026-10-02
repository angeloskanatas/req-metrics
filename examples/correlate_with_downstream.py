"""Within-model Spearman correlation between metric layer profiles and downstream layer profiles.

Downstream file: {"downstream": {"<task>": [score per layer, ...], ...}}, layers in the same
order as the metric records (the earliest extracted representation is layer 0). Records come
from rq.compute / Records.to_json. For every (metric, task) pair the script reports Spearman
rho over layers and, following the depth-controlled analysis of Kanatas et al. (2026), the partial
Spearman controlling for layer index.

Usage:
    python examples/correlate_with_downstream.py records.json downstream.json [--tasks genre key ...]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import spearmanr

import req_metrics as rq


def partial_spearman(x, y, z):
    """Partial rank correlation of x and y controlling for z (first-order formula on Spearman rhos)."""
    rxy, rxz, ryz = spearmanr(x, y)[0], spearmanr(x, z)[0], spearmanr(y, z)[0]
    den = np.sqrt((1 - rxz**2) * (1 - ryz**2))
    return float("nan") if den == 0 else (rxy - rxz * ryz) / den


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("records")
    p.add_argument("downstream")
    p.add_argument("--tasks", nargs="*", default=None)
    a = p.parse_args()
    rec = rq.Records.from_json(a.records)
    down = json.load(open(a.downstream))["downstream"]
    tasks = a.tasks or sorted(down)
    metrics = sorted({r.metric for r in rec if r.layer_b is None})
    print(f"{'metric':32s} {'task':22s} {'rho':>7s} {'partial':>8s} {'layers':>7s}")
    for m in metrics:
        prof = dict(rec.profile(m))
        for t in tasks:
            y = down[t]
            layers = [l for l in sorted(prof) if l < len(y) and y[l] is not None and prof[l] == prof[l]]
            if len(layers) < 4:
                continue
            x = np.array([prof[l] for l in layers])
            yy = np.array([y[l] for l in layers], dtype=float)
            print(
                f"{m:32s} {t:22s} {spearmanr(x, yy)[0]:7.3f} {partial_spearman(x, yy, np.array(layers)):8.3f} {len(layers):7d}"
            )


if __name__ == "__main__":
    main()
