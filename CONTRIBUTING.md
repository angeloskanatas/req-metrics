# Contributing

## Adding a metric

1. The metric must have a published definition. Where a reference implementation
   exists and deviates from the definition, the docstring states the deviation.
2. Implement it as a function on tensors in the matching module under
   `src/req_metrics/metrics/`, returning `MetricResult(value, extras)`.
3. Register it with `register_metric`, giving its input kind, canonical
   preprocessing and citation keys, and add the reference to `docs/references.bib`.
4. Document the definition and its source in the docstring, with `Args:` and
   `Returns:` sections.
5. Add tests in `tests/`: an analytic case (a null or a closed-form value) and,
   where reference code exists, a parity check against it.
6. Regenerate the metric cards with `python scripts/build_metric_cards.py` and add
   an entry to `CHANGELOG.md`.

## Checks

`ruff check`, `ruff format`, `mypy` and the test suite run in CI; `pre-commit install`
runs the first two locally.
