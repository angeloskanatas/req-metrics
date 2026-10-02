# Contributing

## Adding a metric

1. It must have a published definition. Read the paper in full and, where one
   exists, the reference implementation; note any place where the two disagree.
2. Implement it as a pure function on tensors in the matching module under
   `src/req_metrics/metrics/`, returning `MetricResult(value, extras)`. Apply the
   canonical preprocessing inside the estimator only when the definition requires
   it, and declare it in the registry entry otherwise.
3. Register it with `register_metric(name, inputs=..., preprocess=..., citation=...,
   arxiv=..., tags=..., cache=..., max_items=...)`. Citation keys point into
   `docs/references.bib`; add the entry there with venue, year and arXiv id.
4. Docstring, Google style: one line of purpose, then the definition with the
   source (author, year, venue, equation), the preprocessing, known caveats and
   deviations of reference implementations, then Args and Returns. No emphasis in
   capitals, no figure references, no project-internal names.
5. Tests in `tests/`: at least one analytic case (a null or a closed-form value)
   and, where reference code exists, a parity check. The registry smoke test
   (`tests/test_registry_sweep.py`) runs every metric through the pipeline.
6. Run `scripts/build_metric_cards.py` to regenerate `docs/metrics/`, add a line to
   `CHANGELOG.md`, and extend `docs/METRICS.md` when the metric adds a family or a
   verification note.

## Style

`ruff check` and `ruff format` (configured in `pyproject.toml`) run in CI and via
`pre-commit`. Public functions and classes carry docstrings. Code paths that may
run on GPUs keep float64 for spectra, kNN and LDA unless a dtype argument says
otherwise. Commit messages are plain, imperative and carry no trailers.
