# Phase 0 Baseline Freeze Checklist

## Purpose
Create a stable, auditable baseline before broad refactor work so we can distinguish intentional improvements from regressions.

## Baseline Sources (already centralized)
- `repro_manifest.json` (artifact + dataset command index)
- `appendix_repro_manifest.tex` (paper-visible manifest)
- `reports/brain_kwave_vnext_baselines_table.tex`
- `reports/brain_crosswindow_calibration_table.tex`
- `reports/brain_baseline_sanity_relaxed_table.tex`
- `reports/brain_detector_ablation_table.tex`
- `reports/brain_detector_swap_table.tex`
- `reports/brain_cov_train_ablation_table.tex`

## Checklist
- [ ] Create a clean commit point for freeze (`git status` clean).
- [ ] Refresh manifest from the canonical environment.
- [ ] Run Phase 1 quick gate once and store outputs under `reports/refactor/`.
- [ ] Snapshot baseline metrics (quick-gate JSON + thresholds + environment info).
- [ ] Tag baseline (`phase0-baseline-freeze`) and record tag in release notes.

## Commands
```bash
# 1) Refresh central manifest
PYTHONPATH=. conda run -n stap-fus python scripts/generate_repro_manifest.py

# 2) Generate phase-0 inventory map
PYTHONPATH=. python scripts/refactor_inventory.py

# 3) Run quick regression gate (writes reports/refactor/refactor_quick_fair_matrix.json)
PYTHONPATH=. python scripts/verify_refactor.py --mode quick --execute
```

## Exit Criteria
1. `scripts/verify_refactor.py --mode quick --execute` passes.
2. `repro_manifest.json` and `appendix_repro_manifest.tex` are refreshed from the same commit.
3. Phase 0 baseline tag exists and points to a clean tree.

## Commit Policy
- One commit at the end of each phase, plus optional fixup commits while in-progress.
- Phase-close commit message format:
  - `phaseN: <short outcome>`
- Required in commit body:
  - verification mode run (`quick` / `phase` / `full`)
  - pass/fail + key metric deltas
  - manifest hash or file updates
