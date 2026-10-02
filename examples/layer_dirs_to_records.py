"""Compute registry metrics on stored layer directories and write records.

Input layout (one directory per model, one subdirectory per layer):

    <root>/layer<idx>/sequence-level/embeddings.dat   float memmap, (N, D) rows = items
    <root>/layer<idx>/sequence-level/metadata.json    {"shape": [N, D], "dtype": "float32", ...}

Usage:
    python examples/layer_dirs_to_records.py <root> --metrics effective_rank self_clustering uniformity \
        --n 10000 --model my-encoder --corpus "my corpus, 15 s clips" --out records.json

Any extractor that writes this layout works; nothing here depends on how the
embeddings were produced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import req_metrics as rq


def load_layer_dirs(root: str | Path, dtype: torch.dtype = torch.float32) -> dict[int, torch.Tensor]:
    """Layer index -> (N, D) tensor from <root>/layer<idx>/sequence-level/{embeddings.dat, metadata.json}."""
    root = Path(root)
    layers: dict[int, torch.Tensor] = {}
    for d in sorted(root.glob("layer*")):
        meta_path = d / "sequence-level" / "metadata.json"
        dat_path = d / "sequence-level" / "embeddings.dat"
        if not (meta_path.exists() and dat_path.exists()):
            continue
        meta = json.loads(meta_path.read_text())
        shape = tuple(meta["shape"]) if "shape" in meta else tuple(meta["allocated_shape"])
        actual = tuple(meta.get("actual_shape", shape))
        mm = np.memmap(dat_path, dtype=np.dtype(meta.get("dtype", "float32")), mode="r", shape=shape)
        layers[int(d.name[len("layer") :])] = torch.from_numpy(np.array(mm[: actual[0]], copy=True)).to(dtype)
    if not layers:
        raise FileNotFoundError(f"no layer*/sequence-level/embeddings.dat under {root}")
    return layers


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root")
    p.add_argument("--metrics", nargs="+", required=True)
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default=None)
    p.add_argument("--pooling", default=None)
    p.add_argument("--corpus", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    layers = load_layer_dirs(a.root)
    rec = rq.compute(layers, a.metrics, n=a.n, seed=a.seed, model=a.model, pooling=a.pooling, corpus=a.corpus)
    rec.to_json(a.out)
    for m in a.metrics:
        print(m, [round(v, 4) for _, v in rec.profile(m)])


if __name__ == "__main__":
    main()
