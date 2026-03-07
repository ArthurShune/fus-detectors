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
- `reports/simus_motion/simus_motion_ladder_intraop_paper_seed22.{csv,json}`
- `reports/simus_motion/simus_motion_ladder_mobile_paper_seed22.{csv,json}`
- `reports/simus_motion/simus_phase4_motion_summary.{csv,json}`
- `reports/simus_motion/simus_phase4_calibration_summary.{csv,json}`
- `reports/simus_motion/simus_phase4_failure_decomposition_seed21.{csv,json}`
- `reports/simus_motion/simus_phase4_pd_readout_audit_seed21.{csv,json}`
- `reports/simus_motion/simus_fusion_readout_bench_seed21.{csv,json}`
- `reports/simus_motion/simus_fusion_readout_summary_seed21.{csv,json}`
- `reports/simus_motion/simus_stap_profile_sweep_seed21.{csv,json}`
- `reports/simus_motion/simus_stap_profile_sweep_summary_seed21.{csv,json}`
- `reports/simus_motion/simus_stap_compromise_search_seed2122_partial.{csv,json}`
- `reports/simus_motion/simus_stap_rule_eval_seed2122_partial.{csv,json}`
- `reports/simus_motion/simus_stap_compromise_search_seed2122_full.{csv,json}`
- `reports/simus_motion/simus_stap_rule_eval_seed2122_full.{csv,json}`
- `reports/simus_motion/simus_motion_policy_headline_seed2122_full.{csv,json}`
- `reports/simus_motion/simus_motion_policy_headline_regshift_seed2122_full.{csv,json}`
- `reports/simus_motion/simus_motion_ladder_intraop_paper_seed21_regshift_policy.{csv,json}`
- `reports/simus_motion/simus_motion_ladder_mobile_paper_seed21_regshift_policy.{csv,json}`
- `reports/simus_motion/simus_real_envelope_check_seed2122.{csv,json}`
- `reports/simus_sanity_link/phase4_motion_ladders_seed21_{summary,table,deltas}.{json,csv}`
- `reports/simus_sanity_link/simus_motion_policy_bucket_check_seed21.{csv,json}`
- `reports/simus_sanity_link/simus_motion_policy_bucket_check_regshift_seed21.{csv,json}`
- `reports/simus_sanity_link/real_motion_proxy_telemetry.{csv,json}`

Notes:
- structural reporting now labels the chained pipeline explicitly as `MC-SVD -> STAP`
- the earlier motion-collapse story was driven in large part by a SIMUS random-walk scaling bug; after fixing that bug, corrected motion telemetry falls back inside the audited real-data proxy envelope and the full corrected paper-tier runs no longer show STAP collapse under motion
- failure decomposition and the readout audit still matter: `pd_stap` is not the right headline detector score for SIMUS motion benchmarking; `score_stap_preka` / `eval_score=vnext` remains the correct STAP detection score, while `pd_stap` should be treated as a post-filter PD product
- candidate monotone transforms of the band-fraction suppression (`-log band_fraction`, `1 / band_fraction`) recover some nuisance separation, but the strongest audited right-tail score remains `score_stap_preka`; any PD-style replacement should be introduced as a new named readout/profile rather than silently changing the existing paper path
- structural and motion report rows now carry explicit score semantics (`PD-after-STAP` vs `STAP detector`), and the evaluation CLIs can rescore existing bundles via `--reuse-bundles` so changing `eval_score` does not trigger unnecessary recomputation
- readout-only fusion benchmarks still do not produce a clean Pareto improvement over the raw STAP detector on motion cases; the best audited STAP score remains the detector output itself, not a simple baseline+STAP fusion map
- named SIMUS STAP profile sweeps still indicate that temporal aperture `Lt` is the dominant lever, but after the corrected motion reruns the profile dependence is much narrower than first thought
- the corrected actual paper-tier endpoint runs (`seed21` and `seed22`) show a single fixed STAP profile is plausible again: `Brain-SIMUS-Clin-MotionRobust-v0` beats the original profile on average across the eight corrected actual cases
- the strongest fairness result comes from the explicit frozen-profile search:
  - tune one frozen profile per method family on corrected `seed21`
  - hold those profiles fixed on corrected `seed22`
  - selected frozen profiles are:
    - `STAP`: `Brain-SIMUS-Clin-MotionMidRobust-v0`
    - `MC-SVD`: `ef95`
    - `Adaptive Local SVD`: `conservative_r8`
    - `Local SVD`: `tile12_s4_ef95`
    - `RPCA`: `lam0p5_it250`
    - `HOSVD`: `ef99_ds2_t32`
  - on held-out corrected `seed22`, the single frozen STAP profile leads all frozen baselines:
    - `STAP`: mean `auc_main_vs_bg = 0.9265`, mean `auc_main_vs_nuisance = 0.9986`, mean nuisance `FPR@TPR0.5 = 0.0`
    - next-best baseline on `auc_main_vs_bg`: `HOSVD = 0.8388`
    - next-best baseline on `auc_main_vs_nuisance`: `Local SVD = 0.2900`
    - all frozen baselines remain at nuisance `FPR@TPR0.5 >= 0.930`
- direct real-IQ proxy telemetry still matters for realism checks:
  - Shin Fig3 (`frames 0:128`) remains effectively motion-free on this proxy: `reg_shift_p90 = 0.0038 px`
  - Gammex along-linear17 (`frames 0:5`) reaches `reg_shift_p90 = 1.33-1.93 px`
  - Gammex across-linear17 (`frames 0:5`) remains lower: `reg_shift_p90 = 0.30-0.44 px`
  - the corrected actual SIMUS endpoints used in the fair search now lie inside that measured real-data motion envelope, which is why they are the current benchmark anchor rather than the earlier over-aggressive motion ladder
  - this means further profile or policy tuning on the current motion ladder would still be tuning against an over-aggressive simulator regime rather than a clinically anchored one
- a simulator correctness bug explained a large part of that mismatch: the slow random-walk motion term was being RMS-normalized inside `_normalized_random_walk`, so any nonzero `random_walk_sigma_px` produced nearly the same displacement magnitude regardless of the configured scale
  - this is now fixed in `sim/simus/motion.py`, and `tests/test_pymust_motion_random_walk.py` guards that larger `random_walk_sigma_px` values produce proportionally larger rigid-motion telemetry
  - a cheap post-fix warp audit on existing no-motion beamformed cubes (`reports/simus_motion/simus_motion_proxy_warp_check_seed21.{csv,json}`) now puts the corrected motion ladder back inside the measured real-data `reg_shift_p90` envelope at the audited scales (`ClinIntraOp: 0.009-0.101 px`, `ClinMobile: 0.099-0.854 px`)
  - that warp audit is only a calibration bridge because it warps beamformed IQ rather than regenerating RF, but it is strong enough to justify rerunning the key paper-tier motion endpoints on fresh post-fix datasets before considering any detector-level changes
- the real-data bucket check is a limitation, not a confirmation: all nonzero-motion seed21 SIMUS policy cases still land nearest the Gammex phantom telemetry bucket, so the policy is currently validated as a better SIMUS regime split, not yet as a clinically grounded Shin-vs-Gammex separator
- current evidence therefore still does not justify a detector-level algorithmic redesign; the next non-algorithmic move is to recalibrate the SIMUS motion ladder downward so that the audited cases actually populate the measured real-data envelope, and only then revisit fixed-profile vs proxy-policy comparisons:
  - SIMUS analysis policy: `motion_disp_rms_px` threshold for near-oracle benchmarking
  - deployable/real-data proxy policy: `reg_shift_p90` threshold as the measurable approximation to that benchmark rule

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
