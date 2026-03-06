# PyMUST/SIMUS Clin Plan

This plan is derived from [pymust_specv2.md](/home/arthu/stap-for-fus/pymust_specv2.md), but it is grounded in the code that already exists in this repo.

## Repo reality

Already implemented:

- `sim/simus/pymust_smoke.py`
  - deterministic PyMUST pulse loop (`simus -> rf2iq -> dasmtx`)
  - canonical `Icube` output
  - legacy masks: `mask_flow`, `mask_bg`, `mask_alias_expected`
- `sim/simus/pilot_pymust_simus.py`
  - dataset writer with hashes and provenance
  - optional acceptance-bundle derivation through `write_acceptance_bundle_from_icube(...)`
- `sim/simus/cli_generate.py`
  - multi-clip canonical run generation for named clinical profiles
- `sim/simus/cli_bundle.py`
  - frozen bundle derivation for canonical SIMUS runs
- `scripts/simus_eval_structural.py`
  - structural benchmark evaluation against explicit `H1/H0` masks
- downstream reuse already works:
  - `scripts/icube_make_bundle.py`
  - `scripts/physical_doppler_sanity_link.py`
  - `scripts/icube_baseline_compare.py`
  - `scripts/hab_contract_check.py`
- smoke coverage already exists in [tests/test_pymust_simus_smoke.py](/home/arthu/stap-for-fus/tests/test_pymust_simus_smoke.py)
- structural-eval smoke coverage exists in [tests/test_simus_eval_structural_smoke.py](/home/arthu/stap-for-fus/tests/test_simus_eval_structural_smoke.py)

Not implemented yet:

- functional extension

## Phases

### Phase 0: Existing baseline

Goal:
- keep the current smoke and paper-scale PyMUST path working
- do not change detector math

Status:
- done

### Phase 1: Clinical structural dataset contract

Goal:
- keep the existing PyMUST backend, but upgrade the dataset contract so the simulator can support the benchmark in `pymust_specv2.md`

Scope:
- add named profiles:
  - `ClinIntraOp-Pf-v1`
  - `ClinMobile-Pf-v1`
  - `ClinIntraOp-Pf-Struct-v2`
- support multiple vessel roles in one clip:
  - `microvascular`
  - `nuisance_pa`
- emit explicit structural masks:
  - `mask_h1_pf_main.npy`
  - `mask_h1_alias_qc.npy`
  - `mask_h0_bg.npy`
  - `mask_h0_nuisance_pa.npy`
- emit expected Doppler maps:
  - `expected_fd_true_hz.npy`
  - `expected_fd_sampled_hz.npy`
- keep legacy masks for downstream compatibility
- keep bundle generation unchanged

Implementation files:
- `sim/simus/config.py`
- `sim/simus/labels.py`
- `sim/simus/pymust_smoke.py`
- `sim/simus/pilot_pymust_simus.py`
- `tests/test_pymust_simus_smoke.py`

Deliverable:
- one clinically aligned single-clip structural generator with reproducible labels and hashes

Notes:
- `ClinIntraOp-Pf-v1` remains the motion-stress profile used during diagnosis.
- `ClinIntraOp-Pf-Struct-v2` is the lighter structural benchmark profile with motion/phase disabled and stronger microvascular contrast.

Status:
- done

### Phase 2: Motion and residual-mismatch layer

Goal:
- add nuisance conditions that are actually relevant for STAP:
  - rigid motion
  - residual smooth mismatch after nominal registration
  - optional channel phase screen

Scope:
- add motion parameter blocks to profile configs
- move tissue and blood together with shared rigid motion
- add residual smooth deformation after the rigid component
- write motion telemetry into `meta.json` and debug artifacts

Recommended files:
- `sim/simus/motion.py`
- `sim/simus/pymust_smoke.py`
- `tests/test_pymust_motion_smoke.py`

Deliverable:
- motion-enabled `ClinMobile-Pf-v1` clips that remain deterministic

Status:
- done

### Phase 3: Structural evaluation and bundle-facing CLIs

Goal:
- evaluate the clinical structural task using the new `H1/H0` contract instead of legacy `flow vs bg`

Scope:
- add generation CLI wrappers that can create profile-named run families
- add structural evaluation script that reports:
  - `TPR@1e-3`
  - `TPR@3e-4`
  - nuisance-region FPR
  - alias QC separately

Recommended files:
- `sim/simus/cli_generate.py`
- `sim/simus/cli_bundle.py`
- `scripts/simus_eval_structural.py`

Deliverable:
- reproducible structural benchmark reports under `reports/`

Status:
- done

### Phase 4: Motion benchmark and sanity-link

Goal:
- test whether STAP helps when nuisance structure matches the stated paper motivation

Scope:
- add motion ladder evaluation
- add SIMUS-specific sanity-link summary that reuses the existing telemetry style:
  - PSD band energies
  - `fpeak`
  - low-rank proxies
  - lag-1 coherence

Recommended files:
- `scripts/simus_eval_motion.py`
- `scripts/simus_sanity_link.py`
- `scripts/simus_motion_calibration.py`
- `scripts/simus_failure_decomposition.py`
- `scripts/simus_pd_readout_audit.py`

Deliverable:
- motion benchmark report plus real-IQ sanity-link report

Outputs:
- `reports/simus_motion/simus_motion_ladder_intraop_paper_seed21.{csv,json}`
- `reports/simus_motion/simus_motion_ladder_mobile_paper_seed21.{csv,json}`
- `reports/simus_motion/simus_phase4_motion_summary.{csv,json}`
- `reports/simus_motion/simus_phase4_calibration_summary.{csv,json}`
- `reports/simus_motion/simus_phase4_failure_decomposition_seed21.{csv,json}`
- `reports/simus_motion/simus_phase4_pd_readout_audit_seed21.{csv,json}`
- `reports/simus_sanity_link/phase4_motion_ladders_seed21_{summary,table,deltas}.{json,csv}`

Notes:
- structural reporting now labels the chained pipeline explicitly as `MC-SVD -> STAP`
- the paper-tier motion ladders show that the frozen STAP chain only helps at the zero-motion anchor; once clinically meaningful motion is introduced, the advantage disappears quickly
- calibration against the current Shin/Gammex telemetry indicates the zero-motion anchor is still the closest match to Shin, while nonzero-motion clips only move toward the Gammex phantom regime
- failure decomposition indicates the dominant collapse is in `pd_stap` rather than `score_stap_preka`; registration and the MC-SVD upstream stage are secondary in the first nonzero-motion regime
- the readout audit makes the mechanism explicit: `pd_stap` is constructed as `pd_base * band_fraction`, with background pixels forced back to `pd_base`; on the audited SIMUS runs `mask_h0_bg` lies entirely inside that invariant background mask, so `pd_stap` cannot improve H0-bg tail behavior and becomes nearly decorrelated from `score_stap_preka` once motion is introduced
- candidate monotone transforms of the band-fraction suppression (`-log band_fraction`, `1 / band_fraction`) recover some nuisance separation, but the strongest audited right-tail score remains `score_stap_preka`; any PD-style replacement should be introduced as a new named readout/profile rather than silently changing the existing paper path

Status:
- done

### Phase 5: Functional ensemble extension

Goal:
- support block-design ensembles for functional-map hygiene claims

Scope:
- multi-ensemble generation
- activation ROI modulation
- activation-map evaluation and false-cluster reporting

Recommended files:
- `scripts/simus_eval_functional.py`

Deliverable:
- optional functional appendix benchmark

## Phase ordering rationale

`pymust_specv2.md` is aiming at a stronger clinical benchmark, but the repo already has a usable PyMUST core and bundle pipeline. The lowest-risk way forward is:

1. upgrade the dataset contract first
2. add motion second
3. only then add new evaluation CLIs

That sequence preserves reproducibility and avoids inventing benchmark scripts before the labels are correct.
