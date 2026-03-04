# Phase 3 Kickoff Checklist (Core Runtime Refactor)

## Objective
Improve runtime maintainability by separating algorithm, orchestration, and I/O concerns without changing detector behavior.

## Scope For Phase 3
- Isolate orchestration/runtime plumbing from core STAP math paths.
- Introduce clearer module boundaries for replay/verification execution.
- Add lightweight deterministic hooks to simplify debugging and regression analysis.

## Non-Goals
- No STAP/KA scoring equation changes.
- No publication-number retuning.
- No major performance optimization work (reserved for Phase 4).

## Chunking Plan
1. **Runtime boundary chunk(s)**
   - Extract command execution/runtime helpers from orchestration scripts into dedicated modules.
   - Keep existing CLI interfaces stable.
   - Run `make refactor-quick`.
2. **Replay boundary chunk(s)**
   - Separate reusable data-loading / output-writing utilities from replay control flow.
   - Keep output bundle schema unchanged.
   - Run `make refactor-quick`.
3. **Telemetry boundary chunk(s)**
   - Isolate telemetry shaping/serialization from compute code paths.
   - Preserve telemetry field names and semantics.
   - Run `make refactor-quick`.
4. **Phase close chunk**
   - Run `make refactor-phase`.
   - Record closeout with:
     - `make phase-close-report PHASE=3 VERIFY_MODE=phase`

## Verification Cadence (Minimal Sweep)
- **Every structural chunk/PR:** `make refactor-quick`
- **Before/after larger replay boundary moves:** `make refactor-phase`
- **Phase close:** `make refactor-phase` (required)

## Acceptance Criteria
- `make refactor-phase` passes.
- Quick-gate threshold contract remains satisfied.
- No replay parity regressions in phase latency gate.
- Inventory and phase reports updated for new module boundaries.

