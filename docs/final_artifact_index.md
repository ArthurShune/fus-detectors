# Final Artifact Index

This index points to the final artifacts used for the accepted `SIMUS-ClinUtility-v2` narrative, the accepted structural/functional benchmark checkpoints, the latency frontier, and the clinical-translation packet.

## Structural accepted-v2
- Frozen family search: `reports/simus_v2/simus_fair_profile_search_seed125_126_to_127_128.json`
- Best STAP stack search: `reports/simus_v2/simus_stap_stack_search_seed125_126_to_127_128_fixed.json`
- Stage-symmetric detector-head audit: `reports/simus_v2/simus_symmetric_pipeline_compare_seed125_126_to_127_128_headline.json`
- Latency frontier: `reports/simus_v2/simus_v2_latency_frontier.csv`
- Latency Pareto subset: `reports/simus_v2/simus_v2_latency_frontier_pareto.csv`

## Functional accepted-v2
- Functional held-out checkpoint: `reports/simus_v2/simus_eval_functional_seed221_222_to_223_224_ec6_bgcdf_outside_headline.json`
- Broad same-residual STAP-head audit: `reports/simus_v2/simus_functional_stap_head_search_seed221_222_to_223_224_ec6_bgcdf_outside_headline.json`
- Final targeted pair check: `reports/simus_v2/simus_functional_family_search_targeted_221_222_to_223_224_bgcdf_outside_headline.json`

## Accepted profile gates
- Intra-op parenchymal accepted profile summary: `reports/simus_v2/acceptance/simus_v2_acceptance_clin_intraop_parenchyma_pf_v3_fix3_summary.json`
- Mobile accepted profile example: `reports/simus_v2/acceptance/simus_v2_acceptance_clin_mobile_pf_v2_paper_seed0_final.json`
- Anchor envelopes: `reports/simus_v2/anchors/simus_v2_anchor_envelopes.json`

## Real-data clinical translation
- Translation packet: `reports/clinical_translation/clinical_translation_packet.md`
- Uncertainty/consistency summary: `reports/clinical_translation/clinical_translation_uncertainty.md`
- Same-residual real-data detector-head audit: `reports/clinical_translation/realdata_detector_head_audit.md`
- Macé hold-out retrospective check: `reports/mace_alias_gate_holdout.json`

## Real-data and historical latency
- 4080 SUPER replay summary: `reports/latency/4080super_latency_summary.csv`
- Accepted-v2 latency summary: `reports/simus_v2/simus_v2_latency_summary.csv`

## Manuscript outputs
- Main methodology/paper text: `stap_fus_methodology.tex`
- Supplement results appendix: `appendix_supp_results.tex`
- Built supplement PDF: `stap_fus_supplement.pdf`

## Notes
- The reproducibility manifest is regenerated locally to describe the current commit truthfully and is intentionally not committed again, because committing the regenerated manifest would immediately stale its recorded commit hash.
- The development-only surface-dominated intra-op profile remains telemetry-only and is not part of the accepted competitive benchmark.
