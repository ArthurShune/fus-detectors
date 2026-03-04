# Phase 4 Kickoff Checklist (Performance Refactor)

## Objective
Execute profile-driven runtime optimization while preserving detector behavior and paper-critical reproducibility.

## Scope For Phase 4
- Reduce end-to-end STAP latency in the Brain-* replay path with hotspot-driven changes.
- Improve CUDA/Triton efficiency in top kernels (`tyler:update`, `tyler:solve`, launch overhead).
- Keep optimizations rollback-safe via explicit toggles and parity checks.

## Non-Goals
- No STAP/KA scoring equation changes.
- No threshold-contract retuning for paper claims.
- No broad API redesign unrelated to measured hotspots.

## Chunking Plan
1. **Profiling baseline chunk**
   - Capture baseline hotspots for `core_pd` and end-to-end `_stap_pd`.
   - Record stage timing and top CUDA kernels in a phase report.
2. **Launch-overhead chunk(s)**
   - Target fixed-shape launch overhead reductions (e.g., CUDA graph / kernel fusion opportunities).
   - Preserve existing fallbacks for unsupported shapes/devices.
3. **Tyler kernel chunk(s)**
   - Optimize `tyler:update`/`tyler:solve` path based on measured kernel-level evidence.
   - Keep deterministic behavior and numerical guards intact.
4. **Memory-traffic chunk(s)**
   - Reduce avoidable data movement and synchronization in hot paths.
   - Validate no parity regressions in replay outputs.
5. **Phase close chunk**
   - Run `make refactor-phase`.
   - Record closeout with:
     - `make phase-close-report PHASE=4 VERIFY_MODE=phase`

## Verification Cadence (Minimal Sweep)
- **Every optimization chunk/PR:** `make refactor-quick`
- **Before/after major runtime path changes:** `make refactor-phase`
- **Phase close:** `make refactor-phase` (required)

## Acceptance Criteria
- `make refactor-phase` passes at phase close.
- Quick-gate threshold contract remains satisfied.
- Latency phase gate remains parity-safe and within agreed drift.
- Each optimization chunk has before/after timing evidence in `reports/refactor/`.
