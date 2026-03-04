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

### `full` (milestone/release gate)
Scope:
- Includes `phase`.
- Full Table 5 reproduction script.
- Repro manifest refresh.

Command:
```bash
PYTHONPATH=. python scripts/verify_refactor.py --mode full --execute
```

## Pass/Fail Rules (current)
From `configs/refactor_verify_thresholds.json`:
- STAP minima in quick gate (`open`, offset 0):
  - `tpr@0.0001 >= 0.30`
  - `tpr@0.0003 >= 0.50`
  - `tpr@0.001 >= 0.70`
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
