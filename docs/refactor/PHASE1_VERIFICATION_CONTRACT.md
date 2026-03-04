# Phase 1 Verification Contract

## Goal
Define a minimal but meaningful regression gate so refactor velocity stays high without sacrificing paper reproducibility.

## Gate Levels

### `quick` (default refactor guard)
Scope:
- Fast STAP core unit/regression tests.
- One-window Brain-OpenSkull replay (`MC-SVD` vs `STAP`) with fixed seed/profile.
- Threshold checks from `configs/refactor_verify_thresholds.json`.

Runtime target:
- ~10-20 minutes on a CUDA-capable dev machine (depends on cache/cold start).

Command:
```bash
PYTHONPATH=. python scripts/verify_refactor.py --mode quick --execute
```
Make target:
```bash
make refactor-quick
```

### `phase` (phase close gate)
Scope:
- Includes `quick`.
- Cross-window threshold transfer audit refresh.
- OpenSkull steady-state latency replay (windows 2..N emphasized).
- Optional steps auto-skip when required data roots are missing.

Command:
```bash
PYTHONPATH=. python scripts/verify_refactor.py --mode phase --execute
```
Make target:
```bash
make refactor-phase
```

### `full` (milestone/release gate)
Scope:
- Includes `phase`.
- Full Table 5 reproduction script.
- Repro manifest refresh.

Command:
```bash
PYTHONPATH=. python scripts/verify_refactor.py --mode full --execute
```
Make target:
```bash
make refactor-full
```

## Pass/Fail Rules (current)
From `configs/refactor_verify_thresholds.json`:
- STAP minima in quick gate (`open`, offset 0):
  - `tpr@0.0001 >= 0.04`
  - `tpr@0.0003 >= 0.045`
  - `tpr@0.001 >= 0.065`
- Baseline cap:
  - `MC-SVD tpr@0.001 <= 0.10`
- STAP realized-FPR drift tolerance at each target:
  - relative drift <= `25%`

These are guardrails, not publication numbers. Tighten after Phase 0 baseline freeze if needed.

## Design Constraints
- Refactor changes should not alter algorithmic behavior unless explicitly declared.
- If behavior changes are intentional, update thresholds and log rationale in the phase-close commit.
- Quick gate is required before merging major structural changes.
- Phase gate is required before closing a phase.

## Dry-Run Planning
To print steps without executing:
```bash
PYTHONPATH=. python scripts/verify_refactor.py --mode phase
```

## CI Integration
- GitHub Actions workflow: `.github/workflows/refactor_quick_gate.yml`
- CI executes `quick` mode in a data-safe configuration:
  - `--python-runner local`
  - `--allow-missing-data-gates`
- Local data-backed gate remains authoritative for threshold checks.

CI command:
```bash
PYTHONPATH=. python scripts/verify_refactor.py \
  --mode quick \
  --execute \
  --python-runner local \
  --allow-missing-data-gates
```
