"""Metric registry: a decorator that records each estimator's contract and provenance.

Plain functions in a module-level dict with glob-style listing; no stateful
metric objects.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Callable
from dataclasses import dataclass

from req_metrics._types import InputKind, MetricResult, Preprocess

Estimator = Callable[..., MetricResult]


@dataclass(frozen=True)
class MetricSpec:
    """Registry entry for one metric.

    Attributes:
        name: Registry key, lowercase with underscores; another estimator of the
            same quantity takes a slash suffix, e.g. "intrinsic_dimension/gride".
        fn: The estimator.
        inputs: Tensor contract the estimator expects.
        preprocess: fn's default preprocessing. For cache="spectrum" the pipeline
            applies it, with fn's own center, standardize and l2 arguments, before
            building the shared Spectrum; other estimators preprocess internally.
            Records store the preprocessing a call applied.
        citation: BibTeX keys in docs/references.bib, origin first.
        arxiv: arXiv identifier of the origin paper, if any.
        description: One sentence.
        tags: Free-form markers, e.g. "paper-canonical", "relational".
        cache: Shared intermediate the estimator can consume instead of the raw
            tensor: "spectrum" (accepts a Spectrum as its first argument),
            "neighbors" (accepts a Neighbors table as its first argument),
            "neighbors_kw" (takes the tensor plus a neighbors= keyword), or None.
        max_items: Default cap on the items the estimator sees; the pipeline draws a
            seeded subsample above it and records the count. For estimators whose
            cost grows faster than N log N.
        per_clip: The definition pairs tokens within one clip (a class token with its
            patches, token pairs of one clip), so the estimator runs per clip under
            the "tokens" population too, instead of on the pooled token cloud.
    """

    name: str
    fn: Estimator
    inputs: InputKind
    preprocess: Preprocess
    citation: tuple[str, ...]
    arxiv: str | None
    description: str
    tags: tuple[str, ...] = ()
    cache: str | None = None
    max_items: int | None = None
    per_clip: bool = False


_REGISTRY: dict[str, MetricSpec] = {}


def register_metric(
    name: str,
    *,
    inputs: InputKind,
    preprocess: Preprocess = Preprocess(),
    citation: tuple[str, ...] = (),
    arxiv: str | None = None,
    description: str = "",
    tags: tuple[str, ...] = (),
    cache: str | None = None,
    max_items: int | None = None,
    per_clip: bool = False,
) -> Callable[[Estimator], Estimator]:
    """Register an estimator under name. Re-registering a name is an error."""

    def decorator(fn: Estimator) -> Estimator:
        if name in _REGISTRY:
            raise KeyError(f"metric already registered: {name}")
        _REGISTRY[name] = MetricSpec(
            name=name,
            fn=fn,
            inputs=inputs,
            preprocess=preprocess,
            citation=tuple(citation),
            arxiv=arxiv,
            description=description or (fn.__doc__ or "").strip().splitlines()[0],
            tags=tuple(tags),
            cache=cache,
            max_items=max_items,
            per_clip=per_clip,
        )
        return fn

    return decorator


def list_metrics(pattern: str = "*") -> list[str]:
    """Registered names matching a glob pattern, sorted."""
    return sorted(n for n in _REGISTRY if fnmatch.fnmatch(n, pattern))


def get_metric(name: str) -> MetricSpec:
    """The registered specification of a metric; KeyError for unknown names."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown metric {name!r}; see list_metrics()") from None
