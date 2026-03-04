# Phase 2 Kickoff Checklist (Structural Cleanup)

## Objective
Execute behavior-preserving structural cleanup while keeping paper-critical reproducibility stable.

## Scope For Phase 2
- Reorganize script surface into clearer ownership groups (repro/runtime vs analysis/figure utilities).
- Remove/archive approved legacy candidates.
- Eliminate obvious duplication in orchestration entrypoints (without changing algorithm behavior).

## Non-Goals
- No detector math changes.
- No KA/STAP algorithm redesign.
- No performance tuning beyond accidental regressions fixes.

## Chunking Plan
1. **Legacy cleanup chunk**
   - Remove/archive `foo.txt`, `foo_conda.txt`, `tmp_ratio_current.py`, `tmp_ratio_mix.py`.
   - Run `make refactor-quick`.
2. **Script layout chunk(s)**
   - Group reproducibility scripts under a stable namespace/path convention.
   - Keep backward-compatible CLI shims for old entrypoints where needed.
   - Run `make refactor-quick` after each chunk.
3. **Orchestration dedupe chunk**
   - Consolidate duplicated wrapper logic into shared utilities.
   - Keep CLI outputs/arguments stable unless explicitly versioned.
   - Run `make refactor-quick`.
4. **Phase close chunk**
   - Run `make refactor-phase`.
   - Record results with:
     - `make phase-close-report PHASE=2 VERIFY_MODE=phase`

## Verification Cadence (Minimal Sweep)
- **Every structural chunk/PR:** `make refactor-quick`
- **Every 3-5 merged chunks or before large moves:** `make refactor-phase`
- **Phase close:** `make refactor-phase` (required)

## Acceptance Criteria
- `make refactor-phase` passes.
- No threshold-contract violations in quick gate.
- No parity break in phase latency replay.
- Updated inventory reflects new structure:
  - `make refactor-inventory`

