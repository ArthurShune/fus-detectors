# Contributing

Thanks for contributing to `fus-detectors`.

## Development Setup
- `conda env create -f environment.yml`
- `conda activate fus-detectors`
- `pre-commit install`

## Before Opening a PR
- Keep changes scoped and behavior-preserving unless the PR explicitly changes behavior.
- Run the quick verification gate:
  - `make refactor-quick`
- If your change affects refactor boundaries, run:
  - `make refactor-phase`

## Style and Quality
- Python formatting/linting is configured via `black` and `ruff` (`pyproject.toml`).
- Avoid adding large generated artifacts to git.
- Prefer small, reviewable commits with clear messages.

## PR Expectations
- Describe:
  - what changed,
  - why it changed,
  - how it was validated.
- Link any relevant issue/report and include exact commands used for validation.

## Reproducibility Notes
- For paper-critical paths, keep scripts/arguments explicit and deterministic.
- If you modify reproducibility surfaces, update `repro_manifest.json` and related docs.
