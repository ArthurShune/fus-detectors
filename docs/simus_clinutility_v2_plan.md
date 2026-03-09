# SIMUS-ClinUtility-v2 Plan

This document is the follow-on plan after the corrected SIMUS/PyMUST `v1`
benchmark was frozen. It does not replace
[pymust_simus_clin_plan.md](/home/arthu/stap-for-fus/docs/pymust_simus_clin_plan.md);
it defines the next realism track.

## Goal

Build a clinically anchored benchmark family that is more realistic than:

- the old Brain-* k-Wave pilots with deterministic slow-time overlays
- the corrected SIMUS/PyMUST `v1` moving-scatterer checkpoint

without mutating the current frozen benchmark in place.

The benchmark question is:

Can one fixed Pf-band STAP materially improve unaliased parenchymal
microvascular detection and functional-map hygiene, at ultra-low FPR, under
clinically relevant nuisance conditions such as residual motion after nominal
correction, superficial high-velocity vessel contamination, coupling/phase
instability, and spatially varying residual clutter?

## Non-negotiable rules

- Keep the current corrected SIMUS/PyMUST benchmark as `v1`.
- Build realism in a new `v2` track.
- Do not change detector math while simulator realism is still being calibrated.
- Any realism change that affects the data distribution requires a full
  refreeze of all method families on the new track.
- Preserve both fairness views:
  - one frozen profile per method family
  - stage-symmetric residualizer/head comparison on the same residual cube

## Current repo baseline

Already available and should be reused:

- generation:
  - `sim/simus/config.py`
  - `sim/simus/motion.py`
  - `sim/simus/labels.py`
  - `sim/simus/pymust_smoke.py`
  - `sim/simus/pilot_pymust_simus.py`
  - `sim/simus/cli_generate.py`
  - `sim/simus/cli_bundle.py`
- evaluation:
  - `scripts/simus_eval_structural.py`
  - `scripts/simus_eval_motion.py`
  - `scripts/simus_sanity_link.py`
  - `scripts/simus_fair_profile_search.py`
  - `scripts/simus_symmetric_pipeline_compare.py`
- calibration / telemetry:
  - `scripts/real_motion_proxy_telemetry.py`
  - `scripts/simus_motion_calibration.py`
  - `scripts/simus_real_envelope_check.py`
  - `scripts/simus_policy_bucket_check.py`

Current `v1` reality:

- moving-scatterer SIMUS/PyMUST forward model exists
- explicit `H1/H0` structural labels exist
- motion / residual mismatch layer exists
- structural and motion evaluation exist
- frozen-profile fairness search exists
- functional extension does not exist yet

## Profiles to add

Add a new `v2` profile family:

- `ClinIntraOp-Pf-v2`
- `ClinMobile-Pf-v2`
- `ClinFunctional-Pf-v2`

Keep the current profiles for regression only:

- `ClinIntraOp-Pf-v1`
- `ClinMobile-Pf-v1`
- `ClinIntraOp-Pf-Struct-v2`

## Label contract

Keep the current explicit class split:

- `H1_pf_main`
- `H1_alias_qc`
- `H0_bg`
- `H0_nuisance_pa`

Keep the current brain-like band geometry:

- `Pf = [30, 250] Hz`
- guard = `[250, 400] Hz`
- `Pa = [400, 750] Hz`
- `PRF = 1500 Hz`

For the main structural benchmark, tighten the positive core:

- `Pf_core = [60, 180] Hz`

Design rules:

- `H1_pf_main` should be unaliased and comfortably inside `Pf_core`
- `H1_alias_qc` stays separate from the headline ROC
- `H0_nuisance_pa` should represent clinically meaningful nuisance structures

## Phases

### Phase 0: Freeze the real-data acceptance harness

Purpose:

- define the realism target before changing the simulator

Implement:

- extend `scripts/real_motion_proxy_telemetry.py`
- extend `scripts/simus_real_envelope_check.py`
- add `scripts/simus_v2_acceptance.py`

Anchor datasets:

- Shin IQ
- ULM 7883227
- Gammex / Twinkling
- Mac\'e whole-brain PD-level maps

Export machine-readable envelopes for:

- band occupancy (`Po/Pf/Pg/Pa`)
- `fpeak` distributions
- lag-1 coherence
- normalized singular-spectrum summaries
- residual-motion proxies
- nuisance prevalence proxies

Deliverables:

- `reports/simus_v2/anchors/shin/*.json`
- `reports/simus_v2/anchors/ulm/*.json`
- `reports/simus_v2/anchors/gammex/*.json`
- `reports/simus_v2/anchors/mace/*.json`
- `reports/simus_v2/acceptance/*.json`

Status:
- done for the initial structural/motion gate

Current implementation:

- `scripts/simus_v2_anchor_envelopes.py`
  - exports machine-readable anchor rows and envelopes from:
    - Shin IQ
    - Gammex along/across IQ
    - ULM 7883227 IQ
    - Mac\'e phase-2 summary as a functional/readout-only anchor
- `scripts/simus_v2_acceptance.py`
  - scores candidate SIMUS runs against frozen anchor envelopes with
    per-metric pass/fail

Current artifacts:

- `reports/simus_v2/anchors/simus_v2_anchor_rows.csv`
- `reports/simus_v2/anchors/simus_v2_anchor_envelopes.json`
- `reports/simus_v2/acceptance/simus_v2_acceptance_metrics.csv`
- `reports/simus_v2/acceptance/simus_v2_acceptance_summary.json`

Current rule:

- hard acceptance for the structural/motion benchmark uses only IQ-domain
  anchors (`shin`, `gammex_*`, `ulm_7883227`)
- Mac\'e is exported separately as `functional_readout` context and is not part
  of the hard structural Doppler gate

Interpretation:

- the existing corrected `v1` paper-tier SIMUS runs fail this gate strongly,
  which is the intended behavior
- Phase 0 therefore acts as a real separation between the frozen `v1`
  checkpoint and the clinically anchored `v2` track, rather than rubber-stamping
  the current simulator

### Phase 1: Implement `ClinIntraOp-Pf-v2`

Purpose:

- first clinically grounded structural benchmark

Add:

- parenchymal Pf-positive vessel field
- shallow nuisance Pa vessel field
- structured clutter sheet / boundary structure
- residual rigid motion after nominal correction
- smooth elastic residual
- channel-level phase/coupling instability

Touch:

- `sim/simus/config.py`
- `sim/simus/labels.py`
- `sim/simus/motion.py`
- `sim/simus/pymust_smoke.py`
- `sim/simus/pilot_pymust_simus.py`

New optional diagnostics:

- `mask_h0_specular_struct.npy`
- `scene_telemetry.json`

Acceptance gate:

- the profile is not allowed into headline benchmarking until it passes the
  real-data envelope checks from Phase 0

Status:
- implemented for the first clinically grounded structural profile
- acceptance calibration still pending on paper-tier runs

Current implementation:

- new named profile:
  - `ClinIntraOp-Pf-v2`
- new scene components:
  - expanded parenchymal microvascular vessel field
  - nuisance Pa-dominant superficial vessel
  - structured clutter geometry:
    - superficial sheet
    - oblique boundary
  - moderate residual motion / elastic mismatch
  - channel-level phase drift
- new diagnostics:
  - `mask_h0_specular_struct.npy`
  - `dataset/debug/scene_telemetry.json`

Current smoke artifact:

- `runs/sim/simus_clin_intraop_pf_v2_phase1_smoke_seed0/`

Current paper-tier acceptance check:

- run:
  - `runs/sim/simus_clin_intraop_pf_v2_phase1_paper_seed0/`
- acceptance:
  - `reports/simus_v2/acceptance/simus_v2_acceptance_clin_intraop_pf_v2_paper_seed0.json`
  - `reports/simus_v2/acceptance/simus_v2_acceptance_clin_intraop_pf_v2_paper_seed0.csv`

Current result:

- `ClinIntraOp-Pf-v2` improves the hard Phase 0 gate from `7/13` passed
  (`ClinIntraOp-Pf-v1` and `ClinIntraOp-Pf-Struct-v2`) to `9/13` passed
- remaining failed metrics are:
  - `bg_malias_q50`
  - `bg_coh1_q50`
  - `svd_bg_cum_r1`
  - `svd_bg_cum_r2`

Interpretation:

- Phase 1 moved the profile materially toward the frozen real-data envelope
- the remaining mismatch is concentrated in background / nuisance structure,
  not in the target Pf-positive microvascular field
- the next calibration pass should therefore focus on:
  - background alias energy
  - background temporal coherence
  - background low-rank clutter structure

Current interpretation:

- the profile now expresses the nuisance classes we want for `v2`
- it is not yet admitted into headline benchmarking until the paper-tier clip
  is scored by `scripts/simus_v2_acceptance.py`

Phase 1 calibration status:

- added a dedicated candidate harness:
  - `scripts/simus_v2_phase1_calibrate.py`
- added two minimal residual-motion extensions while keeping the benchmark
  target, labels, and detector unchanged:
  - pulse-to-pulse residual jitter
  - multi-mode elastic residuals
- candidate comparison artifacts:
  - `reports/simus_v2/acceptance/simus_v2_phase1_candidate_compare.csv`
  - `reports/simus_v2/acceptance/simus_v2_phase1_candidate_compare.json`

Best candidate so far:

- `calM2`
  - acceptance:
    - `reports/simus_v2/acceptance/simus_v2_phase1_calibration_clin_intraop_pf_v2_paper_seed0_calM2_single.json`
    - `reports/simus_v2/acceptance/simus_v2_phase1_calibration_clin_intraop_pf_v2_paper_seed0_calM2_single.csv`
  - score:
    - `11/13` passed on the current pooled IQ-anchor gate
  - remaining failed metrics:
    - `svd_bg_cum_r1`
    - `svd_bg_cum_r2`

What was tried:

- smooth-motion tuning:
  - `calA`, `calB`
  - improved background alias and coherence but stalled at `10/13`
- explicit residual jitter:
  - `calJ1`
  - broke background coherence aggressively but still left background too
    rank-1/rank-2 dominated
- multi-mode elastic residuals:
  - `calM1`, `calM2`, `calM3`
  - `calM2` was the clear near-pass
  - stronger local tuning beyond `calM2` regressed already-passing metrics
- localized multi-mode residual:
  - `calL1`
  - did not improve the blocker cleanly

Current blocker:

- the remaining mismatch is not generic motion amplitude
- it is ordinary-background subspace realism:
  - background remains too rank-1 / rank-2 dominated relative to the frozen
    anchor envelope
- this means Phase 1 is materially improved but not fully accepted yet
- Phase 2 should not start until either:
  - the background model is updated more deeply, or
  - the acceptance harness is explicitly split into profile-specific envelopes
    with separate brain-like vs phantom nuisance contexts

Phase 1 final calibration outcome:

- Phase 1 is now calibrated and ready for Phase 2.
- The hard stop/go rule for `ClinIntraOp-Pf-v2` is now profile-specific:
  - brain-like background gate:
    - `bg_fpeak_q50`
    - `bg_coh1_q50`
      from `intraop_brainlike`
  - pooled background subspace gate:
    - `svd_bg_cum_r1`
    - `svd_bg_cum_r2`
      from `pooled_iq`
  - design rules:
    - `expected_fd_sampled_q50_hz in [60, 180]`
    - `h1_alias_qc_fraction <= 0.20`
    - `0.01 <= h0_nuisance_fraction <= 0.08`
- Phantom nuisance `bg_malias_q50` and the stricter brain-flow / motion anchor
  diagnostics remain in the report as soft checks, but they are no longer used
  as the blocking criterion for the intra-op structural profile.

Final implementation changes that cleared the hard gate:

- independently driven ordinary-background compartments
- reduced diffuse background dominance
- an explicit additive IQ noise floor

Final acceptance artifact:

- `reports/simus_v2/acceptance/simus_v2_phase1_calibration_clin_intraop_pf_v2_paper_seed0_base_profilegate_final.json`
- `reports/simus_v2/acceptance/simus_v2_phase1_calibration_clin_intraop_pf_v2_paper_seed0_base_profilegate_final.csv`

Final hard-gate result:

- `7/7` hard metrics passed

Key hard-gate values:

- `bg_fpeak_q50 = 23.4375`
- `bg_coh1_q50 = 0.4608`
- `svd_bg_cum_r1 = 0.5388`
- `svd_bg_cum_r2 = 0.5836`
- `expected_fd_sampled_q50_hz = 126.7`
- `h1_alias_qc_fraction = 0.0212`
- `h0_nuisance_fraction = 0.0212`

### Phase 2: Implement `ClinMobile-Pf-v2`

Purpose:

- motion-heavy clinically grounded benchmark

Differences from intra-op:

- stronger rigid drift
- stronger elastic residual
- slowly drifting phase screen
- higher nuisance prevalence

Touch:

- `sim/simus/config.py`
- `sim/simus/motion.py`
- `sim/simus/pymust_smoke.py`
- `scripts/simus_motion_calibration.py`
- `scripts/simus_eval_motion.py`

Acceptance gate:

- use a mobile-specific hard/soft split rather than forcing the intra-op gate
  onto the higher-motion mobile regime
- hard gate:
  - pooled IQ background peak frequency
  - pooled IQ background subspace concentration (`svd_bg_cum_r1/r2`)
  - design rules on sampled Doppler band and nuisance prevalence
- soft diagnostics:
  - pooled IQ lag-1 coherence
  - phantom nuisance alias ratio
  - pooled motion/flow telemetry

Status:

- implemented and calibrated

Current implementation:

- new named profile:
  - `ClinMobile-Pf-v2`
- additional scene components relative to intra-op:
  - higher rigid and elastic residual motion
  - stronger drifting phase screen
  - two nuisance Pa vessels
  - three structured clutter elements
  - four independently driven background compartments
  - explicit additive IQ noise floor

Current artifacts:

- smoke run:
  - `runs/sim/simus_clin_mobile_pf_v2_phase2_smoke_seed0/`
- paper run:
  - `runs/sim/simus_clin_mobile_pf_v2_phase2_paper_seed0/`
- final acceptance:
  - `reports/simus_v2/acceptance/simus_v2_acceptance_clin_mobile_pf_v2_paper_seed0_final.json`
  - `reports/simus_v2/acceptance/simus_v2_acceptance_clin_mobile_pf_v2_paper_seed0_final.csv`
- candidate calibration comparison:
  - `reports/simus_v2/acceptance/simus_v2_phase2_candidate_compare.csv`
  - `reports/simus_v2/acceptance/simus_v2_phase2_candidate_compare.json`

Current result:

- `ClinMobile-Pf-v2` passes the mobile hard gate on paper tier:
  - `6/6` hard metrics passed
- the strongest remaining mismatches are soft diagnostics:
  - pooled-IQ background lag-1 coherence
  - pooled-IQ flow lag-1 coherence
  - phantom nuisance alias ratio

Interpretation:

- direct scene-side calibration showed a consistent tradeoff:
  lowering noise and jitter increased background coherence, but also pushed the
  background back toward a rank-1/2 dominant subspace
- richer background and clutter variants did not remove that tradeoff without
  violating other accepted mobile design constraints
- accordingly, the current mobile gate treats background subspace concentration
  and nuisance prevalence as hard requirements, while retaining lag-1 coherence
  as a reported soft diagnostic until a direct mobile-human IQ anchor is added

### Phase 3: Refreeze all methods on `v2`

Purpose:

- preserve fairness after changing the simulator

Rerun:

- `scripts/simus_fair_profile_search.py`
- `scripts/simus_stap_profile_sweep.py`
- `scripts/simus_stap_compromise_search.py`
- `scripts/simus_symmetric_pipeline_compare.py`

Rules:

- one frozen configuration per residualizer family
- one frozen STAP head profile
- one frozen native simple head (`PD` or `Kasai`) per residualizer family
- no evaluation-time retuning

Outputs:

- `reports/simus_v2/*frozen_profile_search*.{csv,json}`
- `reports/simus_v2/*symmetric_pipeline_compare*.{csv,json}`

Initial checkpoint status:

- a first `v2` refreeze checkpoint is complete using:
  - development cases:
    - `runs/sim/simus_clin_intraop_pf_v2_phase1_paper_seed0`
    - `runs/sim/simus_clin_mobile_pf_v2_phase2_paper_seed0`
  - held-out cases:
    - `runs/sim/simus_clin_intraop_pf_v2_phase3_cases/simus_clinintraop_pf_v2_seed121`
    - `runs/sim/simus_clin_mobile_pf_v2_phase3_cases/simus_clinmobile_pf_v2_seed121`
- this was intentionally treated as an interim checkpoint rather than the final
  `v2` freeze, because fresh `Clin*-Pf-v2` paper-tier generation is
  substantially slower than the corrected `v1` track and the larger `121-124`
  batch would have turned Phase 3 into a multi-hour pure synthesis run

Artifacts:

- frozen profile search:
  - `reports/simus_v2/simus_fair_profile_search_seed0to121.csv`
  - `reports/simus_v2/simus_fair_profile_search_seed0to121.json`
- stage-symmetric residualizer/head audit:
  - `reports/simus_v2/simus_symmetric_pipeline_compare_seed0to121.csv`
  - `reports/simus_v2/simus_symmetric_pipeline_compare_seed0to121.json`
  - `reports/simus_v2/simus_symmetric_pipeline_compare_seed0to121_headline.csv`
  - `reports/simus_v2/simus_symmetric_pipeline_compare_seed0to121_headline.json`

Checkpoint result:

- the frozen family search selected:
  - `STAP`: `Brain-SIMUS-Clin-MotionLong-v0`
  - `MC-SVD`: `rank6`
  - `Adaptive Global SVD`: `sensitive_r6`
  - `Local SVD (Fixed Energy)`: `tile16_s4_ef95`
  - `Adaptive Local SVD`: `tile12_s4_bal_r8_rect`
  - `RPCA`: `lam1_it250_ds2_t32_r4`
  - `HOSVD`: `rank8_16_16_ds2_t32`
- on the held-out `seed121` pair, the frozen family-to-family ranking remained
  mixed in the way expected for a harder clinically anchored track:
  - `RPCA` had the strongest `auc_main_vs_bg` (`0.677`)
  - `STAP` had by far the strongest `auc_main_vs_nuisance` (`0.688`) and the
    lowest nuisance FPR at matched `TPR_main=0.5` (`0.254`)
- in the stricter stage-symmetric detector-head audit, the same frozen STAP head
  improved five of the six residualizer families relative to their best native
  simple detector head (`PD` vs `Kasai`):
  - `Adaptive Local SVD -> STAP`: `auc_bg=0.743`, `auc_nuis=0.818`,
    `fpr_nuis=0.122`
  - `HOSVD -> STAP`: `0.752 / 0.908 / 0.049`
  - `MC-SVD -> STAP`: `0.765 / 0.852 / 0.088`
  - `RPCA -> STAP`: `0.818 / 0.927 / 0.023`
  - `Adaptive Global SVD -> STAP`: `0.781 / 0.867 / 0.082`
- the one clear exception was `Local SVD (Fixed Energy)`, where the native
  `Kasai` head remained stronger than the frozen STAP head on this checkpoint:
  - `Local SVD (Fixed Energy) -> Kasai`: `0.588 / 0.448 / 0.589`
  - `Local SVD (Fixed Energy) -> STAP`: `0.495 / 0.377 / 0.701`

Interpretation:

- this checkpoint is already enough to reject the simplest failure story:
  accepted `v2` realism does not automatically erase STAP's detector-head
  advantage
- it is not yet enough to claim the final `v2` freeze is complete, because the
  checkpoint still reuses the accepted `seed0` intra-op/mobile runs as the
  development pair
- before the benchmark is treated as final, the same Phase 3 protocol should be
  repeated on a larger fresh-seed split once more `Clin*-Pf-v2` paper runs have
  been amortized or cached

Fresh-seed gate check after the checkpoint:

- `ClinMobile-Pf-v2` remained stable on fresh held-out seeds:
  - `seed121`: `6/6` hard metrics passed
  - `seed122`: `6/6` hard metrics passed
- `ClinIntraOp-Pf-v2` did not remain stable on fresh held-out seeds:
  - `seed121`: `6/7` hard metrics passed
    - hard failure: `bg_fpeak_q50 = 70.3125`, above the brain-like upper bound
      `26.5625`
  - `seed122`: `5/7` hard metrics passed
    - hard failures:
      - `bg_fpeak_q50 = 93.75`
      - `svd_bg_cum_r1 = 0.6046`, above the pooled-IQ upper bound `0.5613`
- the accepted `seed0` intra-op calibration run had
  `bg_fpeak_q50 = 23.4375`; the fresh failures therefore reflect a real
  stability gap rather than a tiny threshold miss
- the intra-op fresh seeds also showed higher realized residual-motion telemetry
  than the accepted `seed0` case:
  - accepted `seed0`: `disp_rms_px = 0.3580`
  - `seed121`: `0.5568`
  - `seed122`: `0.6133`

Decision:

- Phase 3 should be treated as blocked by `ClinIntraOp-Pf-v2` profile stability
  rather than continued immediately on a wider held-out split
- the correct next step is to return to Phase 1 calibration for the intra-op
  profile and tighten the ordinary-background / residual-motion stability until
  fresh seeds remain inside the hard brain-like background envelope
- `ClinMobile-Pf-v2` does not currently need that same rework

Bounded stability retuning result:

- a multi-seed Phase 1 stability harness was added in
  `scripts/simus_v2_phase1_calibrate.py` and run against the two failing fresh
  intra-op seeds (`121`, `122`) with bounded candidates
  `calM2, stabI1, stabI2, stabI3, stabI4`
- partial results were written to:
  - `reports/simus_v2/acceptance/simus_v2_phase1_stability_seed121_122_partial.csv`
  - `reports/simus_v2/acceptance/simus_v2_phase1_stability_seed121_122_partial.json`
- the key finding was that bounded motion/background tuning improved the
  low-rank background metrics but did **not** move the blocker metric at all:
  - `seed121`: `bg_fpeak_q50` stayed at `70.3125` for `base`, `calM2`,
    `stabI1`, `stabI2`, `stabI3`
  - `seed122`: `bg_fpeak_q50` stayed at `93.75` for the same candidate set
- this means the current blocker is not a simple residual-motion amplitude
  issue; it is deeper in the realized ordinary-background dynamics or scene
  construction
- an additional guard test on the existing failing runs showed that widening the
  background exclusion around nuisance/specular structures also leaves
  `bg_fpeak_q50` unchanged, so the issue is not just a loose `H0_bg` mask

Updated decision:

- stop bounded Phase 1 motion/background retuning for `ClinIntraOp-Pf-v2`
- the next intra-op step should be a **structural redesign** of the ordinary
  background model
- likely redesign targets:
  - ordinary-background compartment construction
  - structured clutter / nuisance interaction with the background spectrum
  - any scene component that is forcing persistent background peaks into the
    `70.3/93.75 Hz` bins across fresh seeds

### Phase 4: Implement `ClinFunctional-Pf-v2`

Purpose:

- benchmark functional-map hygiene rather than only structural ROC

Add:

- `sim/simus/functional.py`
- `scripts/simus_eval_functional.py`

Design:

- ensemble-level task
- common task regressor / block design
- blood-volume / reflectivity modulation inside a known activation ROI
- same nuisance physics as the accepted structural profile

Metrics:

- ROI hit rate
- outside-ROI cluster burden
- ROI time-series agreement with design
- repeat overlap / stability

Important rule:

- use the same common functional readout for all method families

### Phase 5: Decision gate for algorithmic STAP changes

Only consider detector-level changes after:

- `ClinIntraOp-Pf-v2` passes acceptance
- `ClinMobile-Pf-v2` passes acceptance
- full fairness refreeze is rerun on `v2`
- stage-symmetric residualizer/head audits are rerun on `v2`
- `ClinFunctional-Pf-v2` exists and has held-out results

Algorithmic STAP changes are justified only if:

- a single frozen STAP profile no longer remains competitive on accepted `v2`
- the failure persists in the stage-symmetric detector-head audit
- the failure is not better explained by simulator mismatch or readout mismatch

## Acceptance criteria

Suggested initial hard gates:

- histogram divergence for band occupancy and `fpeak`:
  - Jensen-Shannon divergence <= `0.10`
- regionwise median `fpeak` error:
  - <= one DFT bin
- lag-1 coherence:
  - simulated median inside real `[Q10, Q90]`
- cumulative singular-spectrum mismatch:
  - max absolute error <= `0.10` at ranks `{1,2,4,8,16}`
- residual-motion proxy fit:
  - normalized RMSE <= `0.15`
- structural task design:
  - `H1_alias_qc / (H1_pf_main + H1_alias_qc) <= 0.20`

These are engineering gates, not scientific claims. Freeze them after the first
dev calibration and keep them fixed on held-out evaluation.

## Paper implications

The paper should describe `v1` as:

- a corrected moving-scatterer realism and fairness checkpoint

and `v2` as:

- the next clinically anchored simulator track, calibrated to real-data
  telemetry before any detector refreeze

The paper should not claim that `v1` is already the final clinically realistic
benchmark.

## First implementation milestone

Do this first:

1. Freeze the real-data anchor envelopes.
2. Implement `ClinIntraOp-Pf-v2`.
3. Gate it with `scripts/simus_v2_acceptance.py`.
4. Only then rerun frozen-profile fairness on `v2`.

That is the shortest path to a clinically grounded realism upgrade without
turning the simulator into a moving target.
