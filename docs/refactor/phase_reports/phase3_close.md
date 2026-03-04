# Phase 3 Close Report

- Generated: 2026-03-04 14:04:20Z (UTC)
- Branch: `main`
- Commit: `563ace3` (`563ace382935484973f49b38d5642eb377e79d25`)
- Working tree: `dirty`

## Summary

- Outcome: PASS. Phase 3 core runtime refactor objectives are met with stable detector behavior and passing phase gate.
- Scope closed:
  - Isolated replay runtime boundaries from orchestration control flow in `scripts/replay_stap_from_run.py`.
  - Extracted reusable replay helpers for source/window I/O, telemetry shaping, input prep, write-arg construction, warmup, and profile preset application under `scripts/refactor/`.
  - Kept CLI behavior and output bundle schema stable while reducing monolithic replay script surface area.
  - Extended quick-gate lint coverage to include all newly introduced replay boundary modules.
  - Refreshed refactor inventory outputs for updated runtime module boundaries.
- Not in scope / deferred:
  - STAP/KA detector math changes and publication-number retuning.
  - Performance-optimization sprint work (reserved for Phase 4).
  - Cleanup of unrelated pre-existing dirty-tree files outside refactor scope.

## Verification

- Gate mode: `phase`
- Command: `PYTHONPATH=. python scripts/verify_refactor.py --mode phase --execute`
- Result: PASS
- Duration: ~2–3 minutes on local CUDA workstation (includes latency replay over windows 0/64/128).

## Metrics And Drift

- TPR/FPR drift vs baseline:
  - Quick-gate threshold contract remained satisfied.
  - OpenSkull quick matrix remained stable: STAP `tpr@1e-4=0.0577`, `tpr@3e-4=0.0596`, `tpr@1e-3=0.0834`; MC--SVD remained `tpr=0` at these strict-tail points (`reports/refactor/refactor_quick_fair_matrix.json`).
  - Cross-window threshold audit artifacts were refreshed (`reports/refactor/refactor_phase_crosswindow.{csv,json}` and `.tex` table export).
- Latency drift vs baseline:
  - Phase latency parity gate passed (`runs/latency_phase_gate_open`).
  - Legacy steady (avg windows 2..N) `stap_total_ms ≈ 1734.70`; optimized steady `stap_total_ms ≈ 1122.29`.
  - Optimized latency path remained parity-safe: `[parity] OK`.
- Threshold contract updates:
  - None in Phase 3.

## Reproducibility

- Manifest files updated:
  - `docs/refactor/REPO_CLASSIFICATION_PHASE0.csv`
  - `docs/refactor/REPO_CLASSIFICATION_PHASE0.md`
- New/retired scripts:
  - added: `scripts/refactor/verify_runtime.py`
  - added: `scripts/refactor/replay_bundle_io.py`
  - added: `scripts/refactor/replay_telemetry.py`
  - added: `scripts/refactor/replay_inputs.py`
  - added: `scripts/refactor/replay_write_args.py`
  - added: `scripts/refactor/replay_warmup.py`
  - added: `scripts/refactor/replay_profiles.py`
  - updated boundary consumer: `scripts/replay_stap_from_run.py`
- Data roots required:
  - `runs/pilot/r4c_kwave_seed1`
  - `runs/pilot/fair_filter_matrix_pd_r3_localbaselines`
  - `runs/latency_pilot_open`

## Risks

- Remaining technical risks:
  - `scripts/replay_stap_from_run.py` still has a large CLI parser surface; additional parser decomposition may be needed in Phase 4/5.
  - Runtime behavior depends on environment defaults (`STAP_*` vars); drift risk remains if external wrappers override unset defaults.
- Operational risks:
  - Phase/latency gate timing remains sensitive to GPU cold/warm state and background system load.
  - Existing duplicate `Makefile` target warning (`r4c-skull-generate`) remains outside this phase scope.

## Phase Handoff

- Ready for next phase: yes
- Next phase first tasks:
  - start Phase 4 hotspot work with profiler-driven priorities (`tyler:update`, `tyler:solve`, kernel launch overhead).
  - preserve parity harness and run `refactor-quick` per optimization chunk.
  - keep latency reporting focused on steady windows (2..N) with capture/cold effects reported separately.

## Commit Record

- Phase close commit: (this commit)
- Additional commits in phase:
  - `d33d6d5` phase3-chunk1: kickoff runtime boundary cleanup
  - `f415d92` phase3-chunk2: extract replay source/window IO helpers
  - `98f9500` phase3-chunk3: extract replay telemetry shaping helpers
  - `bddaf44` phase3-chunk4: extract replay input preparation helpers
  - `ff8c5e5` phase3-chunk5: extract replay write-args builder
  - `a3d0a6c` phase3-chunk6: extract replay warmup helpers
  - `563ace3` phase3-chunk7: extract replay profile preset helpers
