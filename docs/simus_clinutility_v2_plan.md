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

- same as intra-op, but calibrated to the mobile anchor envelope rather than
  forced to match intra-op telemetry

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
