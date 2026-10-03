"""Result records: one row per (model, layer, metric) with every protocol field, and writers."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Record:
    """One metric value with everything needed to reproduce and compare it.

    Attributes:
        metric: Registry name, variants included ("intrinsic_dimension/gride").
        value: The headline number; nan when the estimator failed (see extras["error"]).
        layer: Layer index; layer_b is set only for two-layer comparisons.
        depth: Layer index over the largest provided index, in [0, 1].
        model: Label of the representation source.
        population: "pooled", "frames" or "tokens".
        pooling: Label of how pooled vectors were formed ("time-mean", "final-token").
        corpus: Label of the input clips.
        n_items: Clips (or tokens) the estimator saw after subsetting.
        dim: Representation width.
        n_views: Views per clip for view metrics; shifts for PTE.
        preprocess: The preprocessing applied, as text.
        params: Estimator parameters that differ from or pin the defaults.
        views: ViewSpec description, if any. shifts: ShiftSpec description, if any.
        seed: Subsampling seed.
        tags: Registry tags of the metric.
        extras: The estimator's secondary quantities, plus aggregation fields for frames.
    """

    metric: str
    value: float
    layer: int | None = None
    layer_b: int | None = None
    depth: float | None = None
    model: str | None = None
    population: str = "pooled"
    pooling: str | None = None
    corpus: str | None = None
    n_items: int | None = None
    dim: int | None = None
    n_views: int | None = None
    preprocess: str = "none"
    params: dict[str, Any] = field(default_factory=dict)
    views: str | None = None
    shifts: str | None = None
    seed: int | None = None
    tags: tuple[str, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)
    version: str = ""
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, Any]:
        """Plain dictionary with JSON-serializable values."""
        d = asdict(self)
        d["tags"] = list(self.tags)
        return d


_CSV_FIELDS = [
    "model",
    "layer",
    "layer_b",
    "depth",
    "metric",
    "value",
    "population",
    "pooling",
    "corpus",
    "n_items",
    "dim",
    "n_views",
    "preprocess",
    "seed",
    "tags",
    "params",
    "views",
    "shifts",
    "extras",
    "version",
    "created",
]


_ATLAS_NAMES = {
    "effective_rank": ("effective_rank", "default"),
    "effective_rank/variance": ("effective_rank", "variance"),
    "anisotropy": ("anisotropy", "spectral"),
    "cosine_anisotropy": ("anisotropy", "cosine"),
    "intrinsic_dimension": ("id", "twonn"),
    "intrinsic_dimension/mle": ("id", "mle"),
    "intrinsic_dimension/mlid": ("id", "mlid"),
    "alpha_req": ("alpha", "default"),
    "infonce": ("infonce", "default"),
    "lidar": ("lidar", "default"),
    "trajectory_curvature": ("curvature", "signed"),
    "pte": ("pte", "lin_phase"),
}


def atlas_rows(r: Record) -> list[tuple[str, str, float]]:
    """(metric, variant, value) triples in the companion site's vocabulary for one record.

    Conventions the site stores as separate variants are read from the extras of
    one record: the folded curvature next to the signed one, and the cpsd distance
    of a PTE probe next to its phase distance.
    """
    if r.metric == "intrinsic_dimension/gride":
        return [("id", f"gride_k{int(r.params.get('scale', 8))}", r.value)]
    if r.metric == "trajectory_curvature" and "signed" in r.extras and "abs" in r.extras:
        return [("curvature", "signed", float(r.extras["signed"])), ("curvature", "default", float(r.extras["abs"]))]
    if r.metric == "pte" and "phase_rmse" in r.extras and "cpsd_rmse" in r.extras:
        probe = "mlp" if r.params.get("probe") == "mlp" else "lin"
        return [
            ("pte", f"{probe}_phase", 1.0 - float(r.extras["phase_rmse"]) / 2.0),
            ("pte", f"{probe}_cpsd", 1.0 - float(r.extras["cpsd_rmse"]) / 2.0),
        ]
    metric, variant = _ATLAS_NAMES.get(r.metric, (r.metric.replace("/", "_"), "default"))
    return [(metric, variant, r.value)]


def atlas_name(r: Record) -> tuple[str, str]:
    """(metric, variant) of the first site row of a record."""
    metric, variant, _ = atlas_rows(r)[0]
    return metric, variant


class Records:
    """A list of Record with filters and writers; iterable and indexable."""

    def __init__(self, rows: Iterable[Record] = ()):
        self.rows: list[Record] = list(rows)

    def __iter__(self) -> Iterator[Record]:
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Record:
        return self.rows[i]

    def extend(self, other: Iterable[Record]) -> Records:
        """Append records in place and return self."""
        self.rows.extend(other)
        return self

    def where(self, **conditions: Any) -> Records:
        """Rows whose fields equal the given values, e.g. where(metric="effective_rank")."""
        return Records(r for r in self.rows if all(getattr(r, k) == v for k, v in conditions.items()))

    def profile(self, metric: str, model: str | None = None) -> list[tuple[int, float]]:
        """(layer, value) pairs for one metric, sorted by layer."""
        rows = [
            r for r in self.rows if r.metric == metric and (model is None or r.model == model) and r.layer_b is None
        ]
        return sorted((r.layer, r.value) for r in rows if r.layer is not None)

    def to_json(self, path: str | Path) -> Path:
        """Write all records as a JSON list; returns the path."""
        path = Path(path)
        path.write_text(json.dumps([r.to_dict() for r in self.rows], indent=1, default=float))
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> Records:
        """Read records written by to_json."""
        rows = json.loads(Path(path).read_text())
        return cls(Record(**{**r, "tags": tuple(r.get("tags", ()))}) for r in rows)

    def to_csv(self, path: str | Path) -> Path:
        """Flat CSV with params, views, shifts and extras serialized as JSON strings."""
        path = Path(path)
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            w.writeheader()
            for r in self.rows:
                d = r.to_dict()
                d["tags"] = ";".join(r.tags)
                for k in ("params", "extras"):
                    d[k] = json.dumps(d[k], default=float)
                w.writerow({k: d[k] for k in _CSV_FIELDS})
        return path

    def to_atlas_json(
        self,
        path: str | Path,
        *,
        model: str,
        space: str = "pooled embeddings",
        note: str = "",
        corpus: str | None = None,
    ) -> Path:
        """Write the companion-site record format: one entry per metric variant with its layer profile.

        The site (music-fms-layer-eval) stores {schema_version, model, space, note,
        exported, records:[{metric, variant, layers, n_layers, tags, corpus, source}]}
        with short metric names ("id", "alpha", "curvature", "pte") and variant
        labels ("gride_k8", "twonn", "spectral", "lin_phase"). Library names are mapped
        onto that vocabulary; unmapped metrics keep their registry name with "/"
        replaced by "_" and variant "default".
        """
        groups: dict[tuple[str, str], list[tuple[Record, float]]] = {}
        for r in self.rows:
            if r.layer_b is not None:
                continue
            for metric, variant, value in atlas_rows(r):
                groups.setdefault((metric, variant), []).append((r, value))
        records = []
        for (metric, variant), rows in sorted(groups.items()):
            rows = sorted(rows, key=lambda rv: rv[0].layer if rv[0].layer is not None else -1)
            records.append(
                {
                    "metric": metric,
                    "variant": variant,
                    "layers": [value for _, value in rows],
                    "n_layers": len(rows),
                    "tags": sorted({t for r, _ in rows for t in r.tags}),
                    "corpus": corpus if corpus is not None else rows[0][0].corpus,
                    "source": f"req-metrics {rows[0][0].version}",
                }
            )
        payload = {
            "schema_version": 1,
            "model": model,
            "space": space,
            "note": note,
            "exported": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "records": records,
        }
        path = Path(path)
        path.write_text(json.dumps(payload, indent=1, default=float))
        return path

    def to_pandas(self):
        """DataFrame with extras expanded into columns (requires pandas)."""
        import pandas as pd

        rows = []
        for r in self.rows:
            d = r.to_dict()
            extras = d.pop("extras")
            d["tags"] = ";".join(r.tags)
            d["params"] = json.dumps(d["params"], default=float)
            rows.append({**d, **{f"extra_{k}": v for k, v in extras.items() if not isinstance(v, (list, dict))}})
        return pd.DataFrame(rows)
