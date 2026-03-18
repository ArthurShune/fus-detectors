.ONESHELL:
SHELL := /bin/bash

ENV_NAME := fus-detectors

# Monte-Carlo sweep defaults (override via `make mc_sweep MC_SEEDS=8 ...`)
MC_METHOD ?= stap
MC_FPRS ?= 1e-3 3e-4 1e-4
MC_SEEDS ?= 50
MC_SEED_OFFSET ?= 0
MC_NPOS ?= 100000
MC_NNEG ?= 300000
MC_HEIGHT ?= 128
MC_WIDTH ?= 128
MC_ROC_THRESH ?= 4096
MC_DEVICE ?= cuda
MC_CONFIRM2 ?= 1
MC_ALPHA2 ?= 1e-3 3e-4 1e-4
MC_CONFIRM2_PAIRS ?= 200000
MC_CONFIRM2_RHO ?= 0.5
MC_CONFIRM2_CI_ALPHA ?= 0.05
MC_OUT ?= results/mc_latest.csv
MC_JSON_DIR ?= runs/mc_latest

# Stress sweep defaults
STRESS_GRID ?= motion_um=10,30,50,80,120,150 motion_freq_hz=0.2,0.5,1.0 heterogeneity=low,medium,high prf=2000,3000,4500 prf_jitter_pct=0,5 sensor_dropout=0,0.05,0.1 scatter_density=0.8,1.0,1.4 skull_db=0,3 K=5,7,9,11,13
STRESS_TILES ?= 8x8:4,12x12:6
STRESS_SEEDS ?= 20
STRESS_SEED_OFFSET ?= 0
STRESS_NPOS ?= 100000
STRESS_NNEG ?= 300000
STRESS_HEIGHT ?= 128
STRESS_WIDTH ?= 128
STRESS_FPR_TARGET ?= 1e-3
STRESS_ROC_THRESH ?= 4096
STRESS_DEVICE ?= cuda
STRESS_CONFIRM2 ?= 1
STRESS_ALPHA2 ?= 1e-4
STRESS_CONFIRM2_RHO ?= 0.5
STRESS_CONFIRM2_PAIRS ?= 200000
STRESS_CONFIRM2_CI_ALPHA ?= 0.05
STRESS_RHO_INFLATE ?= 0.0
STRESS_ROBUST_COV ?= scm
STRESS_HUBER_C ?= 5.0
STRESS_STEER_GRID ?=
STRESS_STEER_FUSE ?= max
STRESS_MOTION_COMP ?= 0
STRESS_GROUPING ?= none
STRESS_CONFIGS ?= configs/sims_r1.yaml configs/sims_r2.yaml
STRESS_OUTDIR ?= runs/stress_latest

# Aggregation defaults
AGG_CSV ?= results/mc_latest.csv
AGG_GROUP_COLS ?= fpr_target
AGG_OUT_CSV ?= results/mc_summary.csv
AGG_OUT_MD ?= results/mc_summary.md

.PHONY: env activate fmt lint tests figs precommit acceptance report mc_sweep stress_sweep aggregate_runs motion-figs motion-metrics r4c_brw_replay r4c_brw_roc r4c_brw_confirm2 r4c_brw_all
.PHONY: refactor-inventory refactor-quick refactor-quick-ci refactor-phase refactor-full phase-close-report

env:
	conda env create -f environment.yml || conda env update -f environment.yml
	@echo "Run: conda activate $(ENV_NAME)"

fmt:
	ruff check --fix .
	black .

lint:
	ruff check .
	black --check .

tests:
	pytest -q

refactor-inventory:
	PYTHONPATH=. python scripts/refactor_inventory.py

refactor-quick:
	PYTHONPATH=. python scripts/verify_refactor.py --mode quick --execute

refactor-quick-ci:
	PYTHONPATH=. python scripts/verify_refactor.py --mode quick --execute --python-runner local --allow-missing-data-gates

refactor-phase:
	PYTHONPATH=. python scripts/verify_refactor.py --mode phase --execute

refactor-full:
	PYTHONPATH=. python scripts/verify_refactor.py --mode full --execute

PHASE ?= 1
VERIFY_MODE ?= phase
PHASE_REPORT_OUT ?=

phase-close-report:
	PYTHONPATH=. python scripts/phase_close_report.py --phase $(PHASE) --verification-mode $(VERIFY_MODE) $(if $(PHASE_REPORT_OUT),--out $(PHASE_REPORT_OUT),)

figs:
	python scripts/legacy_figs/fig1_before_after.py
	python scripts/legacy_figs/fig2_roc_calibrated.py
	python scripts/legacy_figs/fig3_confirm2_curve.py
	python scripts/legacy_figs/fig4_latency_angle_trade.py
	python scripts/legacy_figs/fig5_ablation_bars.py
	python scripts/legacy_figs/fig6_telemetry_rank.py

motion-figs:
	PYTHONPATH=. python scripts/pilot_visualize.py

.PHONY: motion-metrics

MOTION_RUNS ?= alias alias_pftrace alias_aliascap alias_stride3_lt4_shrink alias_pftrace_stride3_lt4_shrink
MOTION_DATASET ?= pw_7.5MHz_5ang_2ens_128T_seed1
MOTION_REPORT_DIR ?= reports/motion_figs
MOTION_COVERAGE_THRESH ?= 0.2 0.5 0.8
MOTION_COVERAGE_JSON ?= $(MOTION_REPORT_DIR)/coverage_summary.json
MOTION_CONFIRM2_JSON ?= reports/motion_confirm2_summary.json
MOTION_CONFIRM2_ALPHA2 ?= 1e-5
MOTION_CONFIRM2_CAL ?= 5000
MOTION_CONFIRM2_TEST ?= 600
MOTION_CONFIRM2_SCORE ?= pd
MOTION_CONFIRM2_MIN_EXC ?= 50
MOTION_CONFIRM2_Q0 ?= 0.9
MOTION_CONFIRM2_BUNDLES ?= \
	alias=runs/motion/alias_final/$(MOTION_DATASET) \
	alias=runs/motion/alias_final_c0/$(MOTION_DATASET) \
	pftrace=runs/motion/pftrace_final/$(MOTION_DATASET) \
	pftrace=runs/motion/pftrace_final_c0/$(MOTION_DATASET)

motion-metrics:
	mkdir -p $(MOTION_REPORT_DIR)
	PYTHONPATH=. python scripts/analyze_coverage_roc.py \
		$(foreach run,$(MOTION_RUNS),--bundle runs/motion/$(run)/$(MOTION_DATASET)) \
		--thresholds $(MOTION_COVERAGE_THRESH) \
		--fpr-target 1e-5 \
		--json $(MOTION_COVERAGE_JSON)
	PYTHONPATH=. python scripts/confirm2_eval.py \
		$(foreach bundle,$(MOTION_CONFIRM2_BUNDLES),--bundle $(bundle)) \
		--alpha2 $(MOTION_CONFIRM2_ALPHA2) \
		--cal-pairs $(MOTION_CONFIRM2_CAL) \
		--test-pairs $(MOTION_CONFIRM2_TEST) \
		--ci-alpha 0.05 \
		--score-mode $(MOTION_CONFIRM2_SCORE) \
		--min-exceedances $(MOTION_CONFIRM2_MIN_EXC) \
		--q0 $(MOTION_CONFIRM2_Q0) \
		--output $(MOTION_CONFIRM2_JSON)

.PHONY: motion-guardrails
motion-guardrails:
	@PYTHONPATH=. python - <<'PY'
	import json, sys
	from pathlib import Path

	def load_meta(path):
	    return json.loads(Path(path).read_text())

	def check_ratio(meta_path, thresh, label):
	    tele = load_meta(meta_path).get("stap_fallback_telemetry", {})
	    val = tele.get("flow_pdmask_ratio_median")
	    if val is None:
	        sys.exit(f"[guard] missing flow_pdmask_ratio_median for {label}")
	    if float(val) < thresh:
	        sys.exit(f"[guard] PD-mask ratio {val:.3f} below {thresh:.3f} for {label}")
	    bg = tele.get("bg_var_ratio_actual")
	    if bg is not None and abs(float(bg) - 1.0) > 0.15:
	        sys.exit(f"[guard] BG variance ratio {bg:.3f} off target for {label}")

	# Amplitude guards (contrast-off and production contrast)
	check_ratio("runs/motion/alias_final_c0/pw_7.5MHz_5ang_3ens_192T_seed1/meta.json", 0.98, "alias_final_c0")
	check_ratio("runs/motion/pftrace_final_c0/pw_7.5MHz_5ang_3ens_192T_seed1/meta.json", 0.98, "pftrace_final_c0")
	check_ratio("runs/motion/alias_final/pw_7.5MHz_5ang_3ens_192T_seed1/meta.json", 0.90, "alias_final")
	check_ratio("runs/motion/pftrace_final/pw_7.5MHz_5ang_3ens_192T_seed1/meta.json", 0.90, "pftrace_final")

	# ROC guard (TPR@FPR=1e-5 >= 0.33 for ≥50% coverage slice)
	cov_summary = json.loads(Path("reports/motion_final/coverage_summary.json").read_text())
	for entry in cov_summary:
	    path = entry["path"]
	    for res in entry["coverage_results"]:
	        if abs(res["threshold"] - 0.5) < 1e-6:
	            if float(res["tpr_stap"]) < 0.33:
	                sys.exit(f"[guard] TPR {res['tpr_stap']:.3f} below 0.33 for {path}")

	# Confirm-2 guard
	conf = json.loads(Path("reports/motion_final/confirm2_summary.json").read_text())
	for record in conf:
	    label = record["label"]
	    pred = float(record.get("predicted_pair_pfa", 1.0))
	    ci_hi = float(record.get("pair_ci_hi", 1.0))
	    if pred > 2e-5:
	        sys.exit(f"[guard] predicted pair-Pfa {pred:.2e} too high for {label}")
	    if ci_hi > 6.2e-3:
	        sys.exit(f"[guard] empirical CI upper bound {ci_hi:.2e} too high for {label}")

	print("[guard] motion guardrails passed")
	PY

MCSV_SRC ?= runs/pilot/r3_kwave
MCSV_DATASET ?= pw_7.5MHz_5ang_3ens_192T_seed1
MCSV_OUT_ROOT ?= runs/motion
MCSV_K_VALUES ?= 3 5 8
MCSV_ENERGY_FRACS ?= 0.85:p85 0.90:p90
MCSV_REG_SUBPIXEL ?= 4
MCSV_MSD_CONTRAST_ALPHA ?= 0.8
MCSV_ROC_JSON ?= reports/mcsvd_grid_roc.json
MCSV_THRESH ?= 0.0 0.2 0.5 0.8
MCSV_EXTRA_FPRS ?= 1e-4 1e-3

.PHONY: mcsvd-sweep
mcsvd-sweep:
	set -euo pipefail
	for k in $(MCSV_K_VALUES); do \
		PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/replay_stap_from_run.py \
			--src $(MCSV_SRC) \
			--out $(MCSV_OUT_ROOT)/mcsvd_r3_k$${k}_reg$(MCSV_REG_SUBPIXEL) \
			--baseline mc_svd \
			--reg-enable --reg-method phasecorr --reg-subpixel $(MCSV_REG_SUBPIXEL) --reg-reference median \
			--svd-rank $${k} \
			--msd-contrast-alpha $(MCSV_MSD_CONTRAST_ALPHA) \
			--stap-device cuda; \
	done
	for pair in $(MCSV_ENERGY_FRACS); do \
		IFS=: read frac tag <<< $$pair; \
		PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/replay_stap_from_run.py \
			--src $(MCSV_SRC) \
			--out $(MCSV_OUT_ROOT)/mcsvd_r3_$${tag}_reg$(MCSV_REG_SUBPIXEL) \
			--baseline mc_svd \
			--reg-enable --reg-method phasecorr --reg-subpixel $(MCSV_REG_SUBPIXEL) --reg-reference median \
			--svd-energy-frac $${frac} \
			--msd-contrast-alpha $(MCSV_MSD_CONTRAST_ALPHA) \
			--stap-device cuda; \
	done
	PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/replay_stap_from_run.py \
		--src $(MCSV_SRC) \
		--out $(MCSV_OUT_ROOT)/mcsvd_r3_k3_regoff \
		--baseline mc_svd \
		--reg-disable \
		--svd-rank 3 \
		--msd-contrast-alpha $(MCSV_MSD_CONTRAST_ALPHA) \
		--stap-device cuda
	PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/analyze_coverage_roc.py \
		$(foreach k,$(MCSV_K_VALUES),--bundle $(MCSV_OUT_ROOT)/mcsvd_r3_k$(k)_reg$(MCSV_REG_SUBPIXEL)/$(MCSV_DATASET)) \
		$(foreach pair,$(MCSV_ENERGY_FRACS),--bundle $(MCSV_OUT_ROOT)/mcsvd_r3_$(word 2,$(subst :, ,$(pair)))_reg$(MCSV_REG_SUBPIXEL)/$(MCSV_DATASET)) \
		--bundle $(MCSV_OUT_ROOT)/mcsvd_r3_k3_regoff/$(MCSV_DATASET) \
		--thresholds $(MCSV_THRESH) \
		--fpr-target 1e-5 \
		--extra-fpr-targets $(MCSV_EXTRA_FPRS) \
		--pauc-max 1e-3 \
		--json $(MCSV_ROC_JSON)

R4_SEEDS ?= 1 2 3 4 5
R4_ANGLES ?= -12,-6,0,6,12
R4_JITTER_UM ?= 40
R4_PULSES ?= 64
R4_PRF ?= 1500
R4_PILOT_ROOT ?= runs/pilot
R4_RUN_PREFIX ?= r4_kwave_seed
R4_DATASET_PREFIX ?= pw_7.5MHz_5ang_5ens_320T_seed
R4_TILE ?= 10
R4_TILE_STRIDE ?= 5
R4_LT ?= 4
R4_MCSV_OUT ?= runs/motion/r4
R4_MCSV_REG ?= 4
R4_MCSV_K_VALUES ?= 3 5 8
R4_ANALYSIS_MCSV ?= mcsvd_k8_reg4

# -------------------------------------------------------------------
# Latency suite: run baseline + clinical STAP PD (PD-fast path + conditional
# gating) on the three key Brain-* profiles used in the paper: Brain-OpenSkull,
# Brain-AliasContract (alias-augmented contract brain), and Brain-SkullOR
# (skull/OR-inspired brain realism). Produces bundles with saved telemetry and
# runs hab_contract_check on each.
# -------------------------------------------------------------------
.PHONY: run-stap-suite latency-suite brain-suite
LAT_BRAIN_SRC ?= $(R4_PILOT_ROOT)/$(R4_RUN_PREFIX)1
LAT_BRAIN_OUT ?= $(LAT_BRAIN_SRC)_svdlit_stap_pd_clinical_fastpdonly_batched_flowgate
LAT_BRAIN_DATASET ?= $(R4_DATASET_PREFIX)1

LAT_HAB_SRC ?= $(R4_PILOT_ROOT)/$(R4C_HAB_RUN_PREFIX)2
LAT_HAB_OUT ?= $(LAT_HAB_SRC)_svdlit_stap_pd_clinical_fastpdonly_batched_flowgate
LAT_HAB_DATASET ?= $(R4C_HAB_DATASET_PREFIX)2

LAT_SKULL_SRC ?= $(R4_PILOT_ROOT)/$(R4C_SKULL_RUN_PREFIX)2
LAT_SKULL_OUT ?= $(LAT_SKULL_SRC)_latency_mcsvd_stap_pd_clinical_gpu_fast_pdonly_batched_flowgate
LAT_SKULL_DATASET ?= $(R4C_SKULL_DATASET_PREFIX)2
LAT_FIG_DIR ?= figs/latency

run-stap-suite latency-suite:
	# 0) Generate pilots and replays for all Brain-* profiles and baselines.
	$(MAKE) r4-generate
	$(MAKE) r4c-generate
	$(MAKE) r4c-hab-generate
	$(MAKE) r4c-skull-generate
	$(MAKE) r4c-pial-generate
	$(MAKE) r4c-replay
	$(MAKE) r4c-replay-rpca
	$(MAKE) r4c-replay-hosvd
	$(MAKE) r4c-replay-ka
	$(MAKE) r4c-pial-replay
	$(MAKE) r4c-pial-replay-ka
	$(MAKE) r4c-pial-analyze
	$(MAKE) r4c-analyze
	$(MAKE) r4c-analyze-baselines
	# Open-skull brain (seed 1)
	PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/replay_stap_from_run.py \
		--baseline mc_svd --svd-profile literature --reg-disable \
		--stap-profile clinical --stap-device cuda \
		--score-mode pd --time-window-length 64 \
		--src $(LAT_BRAIN_SRC) \
		--out $(LAT_BRAIN_OUT)
	PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/hab_contract_check.py \
		$(LAT_BRAIN_OUT)/$(LAT_BRAIN_DATASET) --score-mode pd
	# Brain-AliasContract profile (alias-augmented contract brain, seed 2)
	PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/replay_stap_from_run.py \
		--baseline mc_svd --svd-profile literature --reg-disable \
		--stap-profile clinical --stap-device cuda \
		--score-mode pd --time-window-length 64 \
		--src $(LAT_HAB_SRC) \
		--out $(LAT_HAB_OUT)
	PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/hab_contract_check.py \
		$(LAT_HAB_OUT)/$(LAT_HAB_DATASET) --score-mode pd
	# Brain-SkullOR profile (skull/OR-inspired brain realism, seed 2)
	PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/replay_stap_from_run.py \
		--baseline mc_svd --svd-profile literature --reg-disable \
		--stap-profile clinical --stap-device cuda \
		--score-mode pd --time-window-length 64 \
		--src $(LAT_SKULL_SRC) \
		--out $(LAT_SKULL_OUT)
	PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/hab_contract_check.py \
		$(LAT_SKULL_OUT)/$(LAT_SKULL_DATASET) --score-mode pd
	# Generate latency/ROC figures for the three regimes
	PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/latency_figs.py \
		--brain $(LAT_BRAIN_OUT)/$(LAT_BRAIN_DATASET) \
		--hab $(LAT_HAB_OUT)/$(LAT_HAB_DATASET) \
		--skull $(LAT_SKULL_OUT)/$(LAT_SKULL_DATASET) \
		--out-dir $(LAT_FIG_DIR)
R4_ROC_JSON ?= reports/r4_msd_roc.json
R4_ROC_AGG_JSON ?= reports/r4_msd_roc_aggregate.json
R4_ROC_THRESH ?= 0.0 0.2 0.5 0.8
R4_EXTRA_FPRS ?= 1e-4 1e-3
R4_ROC_SCORE_MODE ?= msd
R4_FLOW_MASK_KIND ?= pd
R4_CONFIRM2_JSON ?= reports/r4_confirm2.json
R4_CONFIRM2_SCORE ?= msd

.PHONY: r4-motion r4-generate r4-replay r4-analyze

r4-motion: r4-generate r4-replay r4-analyze

r4-generate:
	set -euo pipefail
	mkdir -p $(R4_MCSV_OUT)
	for s in $(R4_SEEDS); do \
		out_dir=$(R4_PILOT_ROOT)/$(R4_RUN_PREFIX)$${s}; \
		echo "[r4] generating pilot for seed $${s} -> $${out_dir}"; \
		PYTHONPATH=. conda run -n $(ENV_NAME) python sim/kwave/pilot_motion.py \
			--out $${out_dir} \
			--angles=$(R4_ANGLES) \
			--ensembles 5 \
			--jitter_um $(R4_JITTER_UM) \
			--pulses $(R4_PULSES) \
			--prf $(R4_PRF) \
			--seed $${s} \
			--tile-h $(R4B_TILE) --tile-w $(R4B_TILE) --tile-stride $(R4B_TILE_STRIDE) \
			--lt $(R4B_LT) \
			--diag-load 1e-2 --cov-estimator tyler_pca --huber-c 5.0 \
			--fd-span-mode psd --fd-span-rel "0.30,1.10" \
			--msd-lambda 5e-2 --msd-ridge 0.06 --msd-agg median --msd-ratio-rho 0.05 \
			--motion-half-span-rel 0.25 --msd-contrast-alpha 0.8 \
			--ka-mode library \
			--ka-prior-path $(R4C_KA_PRIOR_PATH) \
			--ka-directional-beta \
			--ka-kappa 30 \
			--ka-beta-bounds "0.05,0.50" \
			--ka-target-shrink-perp 0.95 \
			--stap-device cuda; \
	done

r4-replay:
	set -euo pipefail
	for s in $(R4_SEEDS); do \
		src_dir=$(R4_PILOT_ROOT)/$(R4_RUN_PREFIX)$${s}; \
		for k in $(R4_MCSV_K_VALUES); do \
			out_dir=$(R4_MCSV_OUT)/mcsvd_k$${k}_reg$(R4_MCSV_REG)_seed$${s}; \
			echo "[r4] MC-SVD reg-on K=$${k} seed $${s}"; \
			PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/replay_stap_from_run.py \
				--src $${src_dir} \
				--out $${out_dir} \
				--baseline mc_svd \
				--reg-enable --reg-method phasecorr --reg-subpixel $(R4_MCSV_REG) --reg-reference median \
				--svd-rank $${k} \
				--msd-contrast-alpha 0.8 \
				--stap-device cuda; \
		done; \
		echo "[r4] MC-SVD reg-off K=3 seed $${s}"; \
		PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/replay_stap_from_run.py \
			--src $${src_dir} \
			--out $(R4_MCSV_OUT)/mcsvd_k3_regoff_seed$${s} \
			--baseline mc_svd \
			--reg-disable \
			--svd-rank 3 \
			--msd-contrast-alpha 0.8 \
			--stap-device cuda; \
	done

r4-analyze:
	set -euo pipefail
	ROC_BUNDLES=""
	for s in $(R4_SEEDS); do \
		ds=$(R4_DATASET_PREFIX)$${s}; \
		ROC_BUNDLES="$$ROC_BUNDLES --bundle $(R4_MCSV_OUT)/$(R4_ANALYSIS_MCSV)_seed$${s}/$${ds}"; \
		ROC_BUNDLES="$$ROC_BUNDLES --bundle $(R4_PILOT_ROOT)/$(R4_RUN_PREFIX)$${s}/$${ds}"; \
	done; \
	echo "[r4] analyzing ROC -> $(R4_ROC_JSON)"; \
	PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/analyze_coverage_roc.py \
		$$ROC_BUNDLES \
		--thresholds $(R4_ROC_THRESH) \
		--fpr-target 1e-5 \
		--extra-fpr-targets $(R4_EXTRA_FPRS) \
		--pauc-max 1e-3 \
		--flow-mask-kind $(R4_FLOW_MASK_KIND) \
		--score-mode $(R4_ROC_SCORE_MODE) \
		--json $(R4_ROC_JSON)
	CONF_ARGS=""
	for s in $(R4_SEEDS); do \
		ds=$(R4_DATASET_PREFIX)$${s}; \
		CONF_ARGS="$$CONF_ARGS --bundle ka=$(R4_PILOT_ROOT)/$(R4_RUN_PREFIX)$${s}/$${ds}"; \
	done; \
	for s in $(R4_SEEDS); do \
		ds=$(R4_DATASET_PREFIX)$${s}; \
		CONF_ARGS="$$CONF_ARGS --bundle mcsvd=$(R4_MCSV_OUT)/$(R4_ANALYSIS_MCSV)_seed$${s}/$${ds}"; \
	done; \
	echo "[r4] Confirm-2 KA (pd) @1e-4 -> reports/r4_confirm2_ka_1e4.json"; \
	PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/confirm2_eval.py \
		$(foreach s,$(R4_SEEDS),--bundle ka=$(R4_PILOT_ROOT)/$(R4_RUN_PREFIX)$(s)/$(R4_DATASET_PREFIX)$(s)) \
		--alpha2 1e-4 \
		--cal-pairs 2000 \
		--test-pairs 500 \
		--ci-alpha 0.05 \
		--score-mode pd \
		--min-exceedances 10 \
		--q0 0.75 \
		--output reports/r4_confirm2_ka_1e4.json; \
	echo "[r4] Confirm-2 MC-SVD (pd) @1e-4 -> reports/r4_confirm2_mcsvd_1e4.json"; \
	PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/confirm2_eval.py \
		$(foreach s,$(R4_SEEDS),--bundle mcsvd=$(R4_MCSV_OUT)/mcsvd_k8_reg4_seed$(s)/$(R4_DATASET_PREFIX)$(s)) \
		--alpha2 1e-4 \
		--cal-pairs 2000 \
		--test-pairs 500 \
		--ci-alpha 0.05 \
		--score-mode pd \
		--min-exceedances 10 \
		--q0 0.75 \
		--output reports/r4_confirm2_mcsvd_1e4.json
	AGG_ARGS=""
	for s in $(R4_SEEDS); do \
		ds=$(R4_DATASET_PREFIX)$${s}; \
		AGG_ARGS="$$AGG_ARGS --bundle mcsvd=$(R4_MCSV_OUT)/mcsvd_k8_reg4_seed$${s}/$${ds}"; \
	done; \
	for s in $(R4_SEEDS); do \
		ds=$(R4_DATASET_PREFIX)$${s}; \
		AGG_ARGS="$$AGG_ARGS --bundle ka=$(R4_PILOT_ROOT)/$(R4_RUN_PREFIX)$${s}/$${ds}"; \
	done; \
	echo "[r4] aggregate ROC -> $(R4_ROC_AGG_JSON)"; \
	PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/analyze_coverage_roc.py \
		--aggregate $$AGG_ARGS \
		--thresholds $(R4_ROC_THRESH) \
		--fpr-target 1e-5 \
		--extra-fpr-targets $(R4_EXTRA_FPRS) \
		--pauc-max 1e-3 \
		--flow-mask-kind $(R4_FLOW_MASK_KIND) \
		--score-mode $(R4_ROC_SCORE_MODE) \
		--json $(R4_ROC_AGG_JSON)

R4B_SEEDS ?= 1 2 3
R4B_RUN_PREFIX ?= r4b_kwave_seed
R4B_DATASET_PREFIX ?= pw_7.5MHz_5ang_5ens_320T_seed
R4B_TILE ?= 8
R4B_TILE_STRIDE ?= 3
R4B_LT ?= 4
R4B_MCSV_OUT ?= runs/motion/r4b
R4B_MCSV_REG ?= 4
R4B_MCSV_K_VALUES ?= 3 5 8
R4B_ANALYSIS_MCSV ?= mcsvd_k8_reg4
R4B_ROC_JSON ?= reports/r4b_msd_roc.json
R4B_ROC_AGG_JSON ?= reports/r4b_msd_roc_aggregate.json
R4B_FPR_TARGET ?= 1e-4
R4B_EXTRA_FPRS ?= 1e-5 1e-3
R4B_PRF ?= 1500
R4B_FLOW_ALIAS_HZ ?= 900
R4B_FLOW_ALIAS_FRAC ?= 0.65
R4B_FLOW_ALIAS_DEPTH ?= 0.60

.PHONY: r4b-motion r4b-generate r4b-replay r4b-analyze
r4b-motion: r4b-generate r4b-replay r4b-analyze

r4b-generate:
	set -euo pipefail
	mkdir -p $(R4B_MCSV_OUT)
	for s in $(R4B_SEEDS); do \
		out_dir=$(R4_PILOT_ROOT)/$(R4B_RUN_PREFIX)$${s}; \
		echo "[r4b] generating alias-stress pilot seed $${s} -> $${out_dir}"; \
		PYTHONPATH=. conda run -n $(ENV_NAME) python sim/kwave/pilot_motion.py \
			--out $${out_dir} \
			--angles=$(R4_ANGLES) \
			--ensembles 5 \
			--jitter_um $(R4_JITTER_UM) \
			--pulses $(R4_PULSES) \
			--prf $(R4B_PRF) \
				--seed $${s} \
				--tile-h $(R4B_TILE) --tile-w $(R4B_TILE) --tile-stride $(R4B_TILE_STRIDE) \
				--lt $(R4B_LT) \
				--diag-load 1e-2 --cov-estimator tyler_pca --huber-c 5.0 \
				--fd-span-mode psd --fd-span-rel "0.30,1.10" \
				--msd-lambda 5e-2 --msd-ridge 0.06 --msd-agg median --msd-ratio-rho 0.05 \
			--motion-half-span-rel 0.25 --msd-contrast-alpha 0.8 \
			--ka-mode library \
			--ka-prior-path runs/motion/priors/ka_prior_lt4_prf3k.npy \
			--ka-directional-beta \
			--ka-kappa 30 \
			--ka-beta-bounds "0.05,0.50" \
			--ka-target-shrink-perp 0.95 \
			--flow-alias-hz $(R4B_FLOW_ALIAS_HZ) \
			--flow-alias-fraction $(R4B_FLOW_ALIAS_FRAC) \
			--flow-alias-depth-max-frac $(R4B_FLOW_ALIAS_DEPTH) \
			--stap-device cuda; \
	done

r4b-replay:
	set -euo pipefail
	for s in $(R4B_SEEDS); do \
		src_dir=$(R4_PILOT_ROOT)/$(R4B_RUN_PREFIX)$${s}; \
		for k in $(R4B_MCSV_K_VALUES); do \
			out_dir=$(R4B_MCSV_OUT)/mcsvd_k$${k}_reg$(R4B_MCSV_REG)_seed$${s}; \
			echo "[r4b] MC-SVD reg-on K=$${k} seed $${s}"; \
			PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/replay_stap_from_run.py \
				--src $${src_dir} \
				--out $${out_dir} \
				--baseline mc_svd \
				--tile-h $(R4B_TILE) --tile-w $(R4B_TILE) --tile-stride $(R4B_TILE_STRIDE) \
				--lt $(R4B_LT) \
					--reg-enable --reg-method phasecorr --reg-subpixel $(R4B_MCSV_REG) --reg-reference median \
					--svd-rank $${k} \
					--msd-contrast-alpha 0.8 \
					--flow-mask-mode default --flow-mask-pd-quantile 0.995 \
					--flow-mask-depth-min-frac 0.20 --flow-mask-depth-max-frac 0.95 \
					--flow-mask-dilate-iters 2 \
					--stap-device cuda; \
		done; \
		echo "[r4b] MC-SVD reg-off K=3 seed $${s}"; \
		PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/replay_stap_from_run.py \
			--src $${src_dir} \
			--out $(R4B_MCSV_OUT)/mcsvd_k3_regoff_seed$${s} \
			--baseline mc_svd \
			--tile-h $(R4B_TILE) --tile-w $(R4B_TILE) --tile-stride $(R4B_TILE_STRIDE) \
			--lt $(R4B_LT) \
				--reg-disable \
				--svd-rank 3 \
				--msd-contrast-alpha 0.8 \
				--flow-mask-mode default --flow-mask-pd-quantile 0.995 \
				--flow-mask-depth-min-frac 0.20 --flow-mask-depth-max-frac 0.95 \
				--flow-mask-dilate-iters 2 \
				--stap-device cuda; \
	done

r4b-analyze:
	set -euo pipefail
	AGG_ARGS=""
	for s in $(R4B_SEEDS); do \
		ds=$(R4B_DATASET_PREFIX)$${s}; \
		AGG_ARGS="$$AGG_ARGS --bundle mcsvd=$(R4B_MCSV_OUT)/mcsvd_k8_reg4_seed$${s}/$${ds}"; \
	done; \
	for s in $(R4B_SEEDS); do \
		ds=$(R4B_DATASET_PREFIX)$${s}; \
		AGG_ARGS="$$AGG_ARGS --bundle ka=$(R4_PILOT_ROOT)/$(R4B_RUN_PREFIX)$${s}/$${ds}"; \
	done; \
	echo "[r4b] aggregate ROC -> $(R4B_ROC_AGG_JSON)"; \
	PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/analyze_coverage_roc.py \
		--aggregate $$AGG_ARGS \
		--thresholds $(R4_ROC_THRESH) \
		--fpr-target $(R4B_FPR_TARGET) \
		--extra-fpr-targets $(R4B_EXTRA_FPRS) \
		--pauc-max 1e-3 \
		--flow-mask-kind $(R4_FLOW_MASK_KIND) \
		--score-mode $(R4_ROC_SCORE_MODE) \
		--json $(R4B_ROC_AGG_JSON)

R4C_SEEDS ?= 1 2 3 4 5
R4C_RUN_PREFIX ?= r4c_kwave_seed
R4C_DATASET_PREFIX ?= pw_7.5MHz_5ang_5ens_320T_seed
R4C_TILE ?= 8
R4C_TILE_STRIDE ?= 3
R4C_LT ?= 8
R4C_MCSV_OUT ?= runs/motion/r4c
R4C_MCSV_REG ?= 4
# Default MC-SVD rank sweep for the k-Wave brain fUS scenario; use K=8 only for analysis to save runtime.
R4C_MCSV_K_VALUES ?= 8
R4C_ANALYSIS_MCSV ?= mcsvd_k8_reg4
R4C_ROC_JSON ?= reports/r4c_msd_roc.json
R4C_ROC_AGG_JSON ?= reports/r4c_msd_roc_aggregate.json
R4C_BAND_JSON ?= reports/r4c_bandratio_roc_aggregate.json
R4C_PRF ?= 1500
R4C_BAND_FLOW_LOW ?= 30
R4C_BAND_FLOW_HIGH ?= 250
R4C_BAND_ALIAS_CENTER ?= 575
R4C_BAND_ALIAS_WIDTH ?= 175
R4C_FLOW_DOPPLER_MIN ?= 60
R4C_FLOW_DOPPLER_MAX ?= 180
R4C_BG_ALIAS_HZ ?= 650
R4C_BG_ALIAS_FRAC ?= 0.3
R4C_BG_ALIAS_DEPTH_MIN ?= 0.30
R4C_BG_ALIAS_DEPTH_MAX ?= 0.70
R4C_BG_ALIAS_JITTER ?= 50.0
R4C_PSD_TAPERS ?= 3
R4C_PSD_BANDWIDTH ?= 2.0
R4C_PHASE_STD ?= 0.8
R4C_PHASE_CORR_LEN ?= 14
R4C_PHASE_SEED ?= 111
R4C_CLUTTER_BETA ?= 1.0
R4C_CLUTTER_SNR_DB ?= 20.0
R4C_PML_SIZE ?= 16
R4C_GRID_STEP_REL ?= 0.10
R4C_MAX_PTS ?= 3
# Spatial grid (set to FFT-friendly dims to avoid prime-factor warnings)
R4C_NX ?= 240
R4C_NY ?= 240
ifeq ($(R4C_LT),8)
R4C_KA_PRIOR_PATH ?= runs/motion/priors/ka_prior_lt8_prf3k.npy
else
R4C_KA_PRIOR_PATH ?= runs/motion/priors/ka_prior_lt4_prf3k.npy
endif
R4C_CLUTTER_DEPTH_MIN ?= 0.20
R4C_CLUTTER_DEPTH_MAX ?= 0.95
R4C_FLOW_MASK_PD_Q ?= 0.995
R4C_FLOW_MASK_DEPTH_MIN ?= 0.25
R4C_FLOW_MASK_DEPTH_MAX ?= 0.85
R4C_FPR_TARGET ?= 1e-4
R4C_PAUC_MAX ?= 1e-3
R4C_EXTRA_FPRS ?= 1e-5 1e-3
R4C_ANALYSIS_NOTE ?= reports/r4c_analysis_note.txt
R4C_RECOMMEND_JSON ?= reports/r4c_effective_targets.json
R4C_RECOMMEND_ENV ?= reports/r4c_effective_targets.env

# Full-burst whitened replay (band-ratio) defaults for the k-Wave brain fUS scenario
R4C_BRW_SEEDS ?= $(R4C_SEEDS)
R4C_BRW_OUT ?= runs/motion/r4c_brw
R4C_BRW_BASELINE_PREFIX ?= mcsvd_fullburst_seed
R4C_BRW_KA_PREFIX ?= ka_fullburst_psdselect_seed
R4C_BRW_SCORE_MODE ?= band_ratio_whitened
R4C_BRW_PSD_TAPERS ?= 3
R4C_BRW_PSD_BANDWIDTH ?= 2.0
R4C_BRW_FLOW_LOW ?= 120
R4C_BRW_FLOW_HIGH ?= 400
R4C_BRW_ALIAS_CENTER ?= 900
R4C_BRW_ALIAS_WIDTH ?= 15.625
R4C_BRW_ALIAS_RATIO ?= 1.15
R4C_BRW_ROC_THRESH ?= 0.0 0.2 0.5 0.8
R4C_BRW_FPR_TARGET ?= 1e-3
R4C_BRW_PAUC_MAX ?= 1e-3
R4C_BRW_EXTRA_FPRS ?= 1e-4
R4C_BRW_ROC_JSON ?= reports/r4c_brw_bandratio_roc.json
R4C_BRW_CONFIRM_JSON ?= reports/r4c_brw_confirm2.json
R4C_BRW_CONFIRM_ALPHA2 ?= 1e-3
R4C_BRW_CONFIRM_U ?= 0.995

R4C_FEAS_SEEDS ?= 1 2 3 4 5 6 7 8 9 10 11 12
R4C_FEAS_SEEDS_COUNT := $(words $(R4C_FEAS_SEEDS))

# Short-burst pial-alias brain fUS scenario (T≈128, KA-friendly regime)
R4C_PIAL_SEEDS ?= 1 2 3 4
R4C_PIAL_RUN_PREFIX ?= r4c_kwave_pial_seed
R4C_PIAL_DATASET_PREFIX ?= pw_7.5MHz_5ang_4ens_256T_seed
R4C_PIAL_ENSEMBLES ?= 4
R4C_PIAL_PULSES ?= 32
# Pial-v3 alias / vibration configuration.
R4C_PIAL_BG_ALIAS_HZ ?= 650
R4C_PIAL_BG_ALIAS_FRAC ?= 0.7
R4C_PIAL_BG_ALIAS_DEPTH_MIN ?= 0.15
R4C_PIAL_BG_ALIAS_DEPTH_MAX ?= 0.30
# Slightly larger jitter to mimic slow alias wobble.
R4C_PIAL_BG_ALIAS_JITTER ?= 35.0
# Global vibration stronger and with slower decay in depth.
R4C_PIAL_VIBRATION_HZ ?= 450.0
R4C_PIAL_VIBRATION_AMP ?= 0.35
R4C_PIAL_VIBRATION_DEPTH_MIN ?= 0.15
R4C_PIAL_VIBRATION_DEPTH_DECAY ?= 0.30
# Shallow-deep alias split: a small deep alias fraction to keep realism
# while preserving Pf-dominant parenchyma.
R4C_PIAL_FLOW_ALIAS_HZ ?= 650
R4C_PIAL_FLOW_ALIAS_FRAC ?= 0.05
R4C_PIAL_FLOW_ALIAS_DEPTH_MIN ?= 0.35
R4C_PIAL_FLOW_ALIAS_DEPTH_MAX ?= 0.75
R4C_PIAL_FLOW_ALIAS_JITTER ?= 25.0
R4C_PIAL_ROC_AGG_JSON ?= reports/r4c_pial_msd_roc_aggregate.json

# Alias-augmented contract brain profile (Brain-AliasContract; simulator preset hab_contract, 320T, KA-friendly alias + headroom)
R4C_HAB_RUN_PREFIX ?= r4c_kwave_hab_seed
R4C_HAB_DATASET_PREFIX ?= pw_7.5MHz_5ang_5ens_320T_seed
R4C_HAB_SEEDS ?= $(R4C_SEEDS)
# Pial-like shallow alias layer (Zone A).
R4C_HAB_PIAL_DEPTH_MIN ?= 0.12
R4C_HAB_PIAL_DEPTH_MAX ?= 0.28
R4C_HAB_BG_ALIAS_HZ ?= 650
R4C_HAB_BG_ALIAS_FRAC ?= 0.75
R4C_HAB_BG_ALIAS_JITTER ?= 50.0
# Sparse deep alias component (Zones B/C) injected on flow mask for now.
R4C_HAB_DEEP_ALIAS_DEPTH_MIN ?= 0.28
R4C_HAB_DEEP_ALIAS_DEPTH_MAX ?= 0.85
R4C_HAB_FLOW_ALIAS_HZ ?= 650
R4C_HAB_FLOW_ALIAS_FRAC ?= 0.15
R4C_HAB_FLOW_ALIAS_JITTER ?= 35.0
# Shallow vibration mode in Pa-adjacent band.
R4C_HAB_VIBRATION_HZ ?= 450.0
R4C_HAB_VIBRATION_AMP ?= 0.30
R4C_HAB_VIBRATION_DEPTH_MIN ?= 0.12
R4C_HAB_VIBRATION_DEPTH_DECAY ?= 0.30
R4C_HAB_FLOW_AMP_SCALE ?= 1.0
R4C_HAB_ALIAS_AMP_SCALE ?= 1.0

# Skull/OR-inspired brain realism profile (Brain-SkullOR; simulator preset hab_v3_skull, 320T, alias + skull/OR realism patches)
R4C_SKULL_RUN_PREFIX ?= r4c_kwave_hab_v3_skull_seed
R4C_SKULL_DATASET_PREFIX ?= $(R4C_HAB_DATASET_PREFIX)
R4C_SKULL_SEEDS ?= $(R4C_SEEDS)

.PHONY: r4c-motion r4c-generate r4c-replay r4c-replay-ka r4c-analyze
r4c-motion: r4c-generate r4c-replay r4c-replay-ka r4c-analyze

.PHONY: r4c-feasibility
r4c-feasibility:
	$(MAKE) r4c-generate R4C_SEEDS="$(R4C_FEAS_SEEDS)"

.PHONY: r4c-feas-check
r4c-feas-check:
	@set -euo pipefail; \
	BUNDLES=""; \
	for s in $(R4C_SEEDS); do \
		BUNDLES="$$BUNDLES $(R4C_MCSV_OUT)/ka_seed$${s}/$(R4C_DATASET_PREFIX)$${s}"; \
	done; \
	echo "[r4c] checking feasibility telemetry across $$BUNDLES"; \
	PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/feasibility_check.py \
		--bundles $$BUNDLES \
		--min-seeds $(R4C_FEAS_SEEDS_COUNT)

r4c-generate:
	set -euo pipefail
	mkdir -p $(R4C_MCSV_OUT)
	for s in $(R4C_SEEDS); do \
		out_dir=$(R4_PILOT_ROOT)/$(R4C_RUN_PREFIX)$${s}; \
		echo "[r4c] generating pilot seed $${s} -> $${out_dir}"; \
		PYTHONPATH=. conda run -n $(ENV_NAME) python sim/kwave/pilot_motion.py \
			--out $${out_dir} \
			--profile Brain-OpenSkull \
			--angles=$(R4_ANGLES) \
			--ensembles 5 \
			--jitter_um $(R4_JITTER_UM) \
			--pulses $(R4_PULSES) \
			--prf $(R4C_PRF) \
			--seed $${s} \
			--Nx $(R4C_NX) --Ny $(R4C_NY) \
			--tile-h $(R4C_TILE) --tile-w $(R4C_TILE) --tile-stride $(R4C_TILE_STRIDE) \
			--lt $(R4C_LT) \
			--pml-size $(R4C_PML_SIZE) \
			--band-ratio-flow-low-hz $(R4C_BAND_FLOW_LOW) \
			--band-ratio-flow-high-hz $(R4C_BAND_FLOW_HIGH) \
			--band-ratio-alias-center-hz $(R4C_BAND_ALIAS_CENTER) \
			--band-ratio-alias-width-hz $(R4C_BAND_ALIAS_WIDTH) \
			--diag-load 1e-2 --cov-estimator tyler_pca --huber-c 5.0 \
			--fd-span-mode psd --fd-span-rel "0.30,1.10" \
			--msd-lambda 5e-2 --msd-ridge 0.06 --msd-agg median --msd-ratio-rho 0.05 \
			--motion-half-span-rel 0.25 --msd-contrast-alpha 0.8 \
			--flow-doppler-min-hz $(R4C_FLOW_DOPPLER_MIN) --flow-doppler-max-hz $(R4C_FLOW_DOPPLER_MAX) \
			--bg-alias-hz $(R4C_BG_ALIAS_HZ) --bg-alias-fraction $(R4C_BG_ALIAS_FRAC) \
			--bg-alias-depth-min-frac $(R4C_BG_ALIAS_DEPTH_MIN) --bg-alias-depth-max-frac $(R4C_BG_ALIAS_DEPTH_MAX) \
			--bg-alias-jitter-hz $(R4C_BG_ALIAS_JITTER) \
			--ka-mode library \
			--ka-prior-path runs/motion/priors/ka_prior_lt4_prf3k.npy \
			--ka-directional-beta \
			--ka-kappa 30 \
			--ka-beta-bounds "0.05,0.50" \
			--ka-target-shrink-perp 0.95 \
			--ka-gate-enable \
			--feasibility-mode updated \
			--aperture-phase-std $(R4C_PHASE_STD) \
			--aperture-phase-corr-len $(R4C_PHASE_CORR_LEN) \
			--aperture-phase-seed $(R4C_PHASE_SEED) \
			--clutter-beta $(R4C_CLUTTER_BETA) \
			--clutter-snr-db $(R4C_CLUTTER_SNR_DB) \
			--clutter-depth-min-frac $(R4C_CLUTTER_DEPTH_MIN) \
			--clutter-depth-max-frac $(R4C_CLUTTER_DEPTH_MAX) \
			--stap-device cuda; \
	done

.PHONY: r4c-hab-generate
r4c-hab-generate:
	set -euo pipefail
	mkdir -p $(R4C_MCSV_OUT)
	for s in $(R4C_HAB_SEEDS); do \
		out_dir=$(R4_PILOT_ROOT)/$(R4C_HAB_RUN_PREFIX)$${s}; \
		echo "[r4c-hab] generating HAB pilot seed $${s} -> $${out_dir}"; \
		PYTHONPATH=. conda run -n $(ENV_NAME) python sim/kwave/pilot_motion.py \
			--out $${out_dir} \
			--profile Brain-AliasContract \
			--angles=$(R4_ANGLES) \
			--ensembles 5 \
			--jitter_um $(R4_JITTER_UM) \
			--pulses $(R4_PULSES) \
			--prf $(R4C_PRF) \
			--seed $${s} \
			--Nx $(R4C_NX) --Ny $(R4C_NY) \
			--tile-h $(R4C_TILE) --tile-w $(R4C_TILE) --tile-stride $(R4C_TILE_STRIDE) \
			--lt $(R4C_LT) \
			--pml-size $(R4C_PML_SIZE) \
			--band-ratio-flow-low-hz $(R4C_BAND_FLOW_LOW) \
			--band-ratio-flow-high-hz $(R4C_BAND_FLOW_HIGH) \
			--band-ratio-alias-center-hz $(R4C_BAND_ALIAS_CENTER) \
			--band-ratio-alias-width-hz $(R4C_BAND_ALIAS_WIDTH) \
			--diag-load 1e-2 --cov-estimator tyler_pca --huber-c 5.0 \
			--fd-span-mode psd --fd-span-rel "0.30,1.10" \
			--msd-lambda 5e-2 --msd-ridge 0.06 --msd-agg median --msd-ratio-rho 0.05 \
			--motion-half-span-rel 0.25 --msd-contrast-alpha 0.8 \
			--flow-doppler-min-hz $(R4C_FLOW_DOPPLER_MIN) --flow-doppler-max-hz $(R4C_FLOW_DOPPLER_MAX) \
			--bg-alias-hz $(R4C_HAB_BG_ALIAS_HZ) \
			--bg-alias-fraction $(R4C_HAB_BG_ALIAS_FRAC) \
			--bg-alias-depth-min-frac $(R4C_HAB_PIAL_DEPTH_MIN) \
			--bg-alias-depth-max-frac $(R4C_HAB_PIAL_DEPTH_MAX) \
			--bg-alias-jitter-hz $(R4C_HAB_BG_ALIAS_JITTER) \
			--flow-alias-hz $(R4C_HAB_FLOW_ALIAS_HZ) \
			--flow-alias-fraction $(R4C_HAB_FLOW_ALIAS_FRAC) \
			--flow-alias-depth-min-frac $(R4C_HAB_DEEP_ALIAS_DEPTH_MIN) \
			--flow-alias-depth-max-frac $(R4C_HAB_DEEP_ALIAS_DEPTH_MAX) \
			--flow-alias-jitter-hz $(R4C_HAB_FLOW_ALIAS_JITTER) \
			--flow-amp-scale $(R4C_HAB_FLOW_AMP_SCALE) \
			--alias-amp-scale $(R4C_HAB_ALIAS_AMP_SCALE) \
			--vibration-hz $(R4C_HAB_VIBRATION_HZ) \
			--vibration-amp $(R4C_HAB_VIBRATION_AMP) \
			--vibration-depth-min-frac $(R4C_HAB_VIBRATION_DEPTH_MIN) \
			--vibration-depth-decay-frac $(R4C_HAB_VIBRATION_DEPTH_DECAY) \
			--ka-mode library \
			--ka-prior-path $(R4C_KA_PRIOR_PATH) \
			--ka-directional-beta \
			--ka-kappa 30 \
			--ka-beta-bounds "0.05,0.50" \
			--ka-target-shrink-perp 0.95 \
			--ka-gate-enable \
			--feasibility-mode updated \
			--aperture-phase-std $(R4C_PHASE_STD) \
			--aperture-phase-corr-len $(R4C_PHASE_CORR_LEN) \
			--aperture-phase-seed $(R4C_PHASE_SEED) \
			--clutter-beta $(R4C_CLUTTER_BETA) \
			--clutter-snr-db $(R4C_CLUTTER_SNR_DB) \
			--clutter-depth-min-frac $(R4C_CLUTTER_DEPTH_MIN) \
			--clutter-depth-max-frac $(R4C_CLUTTER_DEPTH_MAX) \
			--stap-device cuda; \
	done

.PHONY: r4c-skull-generate
r4c-skull-generate:
	set -euo pipefail
	mkdir -p $(R4C_MCSV_OUT)
	for s in $(R4C_SKULL_SEEDS); do \
		out_dir=$(R4_PILOT_ROOT)/$(R4C_SKULL_RUN_PREFIX)$${s}; \
		echo "[r4c-skull] generating Skull/OR HAB pilot seed $${s} -> $${out_dir}"; \
		PYTHONPATH=. conda run -n $(ENV_NAME) python sim/kwave/pilot_motion.py \
			--out $${out_dir} \
			--profile Brain-SkullOR \
			--angles=$(R4_ANGLES) \
			--ensembles 5 \
			--jitter_um $(R4_JITTER_UM) \
			--pulses $(R4_PULSES) \
			--prf $(R4C_PRF) \
			--seed $${s} \
			--Nx $(R4C_NX) --Ny $(R4C_NY) \
			--tile-h $(R4C_TILE) --tile-w $(R4C_TILE) --tile-stride $(R4C_TILE_STRIDE) \
			--lt $(R4C_LT) \
			--pml-size $(R4C_PML_SIZE) \
			--band-ratio-flow-low-hz $(R4C_BAND_FLOW_LOW) \
			--band-ratio-flow-high-hz $(R4C_BAND_FLOW_HIGH) \
			--band-ratio-alias-center-hz $(R4C_BAND_ALIAS_CENTER) \
			--band-ratio-alias-width-hz $(R4C_BAND_ALIAS_WIDTH) \
			--diag-load 1e-2 --cov-estimator tyler_pca --huber-c 5.0 \
			--fd-span-mode psd --fd-span-rel "0.30,1.10" \
			--msd-lambda 5e-2 --msd-ridge 0.06 --msd-agg median --msd-ratio-rho 0.05 \
			--motion-half-span-rel 0.25 --msd-contrast-alpha 0.8 \
			--flow-doppler-min-hz $(R4C_FLOW_DOPPLER_MIN) --flow-doppler-max-hz $(R4C_FLOW_DOPPLER_MAX) \
			--bg-alias-hz $(R4C_BG_ALIAS_HZ) --bg-alias-fraction $(R4C_BG_ALIAS_FRAC) \
			--bg-alias-depth-min-frac $(R4C_BG_ALIAS_DEPTH_MIN) --bg-alias-depth-max-frac $(R4C_BG_ALIAS_DEPTH_MAX) \
			--bg-alias-jitter-hz $(R4C_BG_ALIAS_JITTER) \
			--ka-mode library \
			--ka-prior-path $(R4C_KA_PRIOR_PATH) \
			--ka-directional-beta \
			--ka-kappa 30 \
			--ka-beta-bounds "0.05,0.50" \
			--ka-target-shrink-perp 0.95 \
			--ka-gate-enable \
			--feasibility-mode updated \
			--aperture-phase-std $(R4C_PHASE_STD) \
			--aperture-phase-corr-len $(R4C_PHASE_CORR_LEN) \
			--aperture-phase-seed $(R4C_PHASE_SEED) \
			--clutter-beta $(R4C_CLUTTER_BETA) \
			--clutter-snr-db $(R4C_CLUTTER_SNR_DB) \
			--clutter-depth-min-frac $(R4C_CLUTTER_DEPTH_MIN) \
			--clutter-depth-max-frac $(R4C_CLUTTER_DEPTH_MAX) \
			--stap-device cuda; \
	done

.PHONY: r4c-skull-generate
r4c-skull-generate:
	set -euo pipefail
	mkdir -p $(R4C_MCSV_OUT)
	for s in $(R4C_SKULL_SEEDS); do \
		out_dir=$(R4_PILOT_ROOT)/$(R4C_SKULL_RUN_PREFIX)$${s}; \
		echo "[r4c-skull] generating Skull/OR HAB pilot seed $${s} -> $${out_dir}"; \
		PYTHONPATH=. conda run -n $(ENV_NAME) python sim/kwave/pilot_motion.py \
			--out $${out_dir} \
			--profile Brain-SkullOR \
			--angles=$(R4_ANGLES) \
			--ensembles 5 \
			--jitter_um $(R4_JITTER_UM) \
			--pulses $(R4_PULSES) \
			--prf $(R4C_PRF) \
			--seed $${s} \
			--Nx $(R4C_NX) --Ny $(R4C_NY) \
			--tile-h $(R4C_TILE) --tile-w $(R4C_TILE) --tile-stride $(R4C_TILE_STRIDE) \
			--lt $(R4C_LT) \
			--pml-size $(R4C_PML_SIZE) \
			--band-ratio-flow-low-hz $(R4C_BAND_FLOW_LOW) \
			--band-ratio-flow-high-hz $(R4C_BAND_FLOW_HIGH) \
			--band-ratio-alias-center-hz $(R4C_BAND_ALIAS_CENTER) \
			--band-ratio-alias-width-hz $(R4C_BAND_ALIAS_WIDTH) \
			--diag-load 1e-2 --cov-estimator tyler_pca --huber-c 5.0 \
			--fd-span-mode psd --fd-span-rel "0.30,1.10" \
			--msd-lambda 5e-2 --msd-ridge 0.06 --msd-agg median --msd-ratio-rho 0.05 \
			--motion-half-span-rel 0.25 --msd-contrast-alpha 0.8 \
			--flow-doppler-min-hz $(R4C_FLOW_DOPPLER_MIN) --flow-doppler-max-hz $(R4C_FLOW_DOPPLER_MAX) \
			--bg-alias-hz $(R4C_BG_ALIAS_HZ) --bg-alias-fraction $(R4C_BG_ALIAS_FRAC) \
			--bg-alias-depth-min-frac $(R4C_BG_ALIAS_DEPTH_MIN) --bg-alias-depth-max-frac $(R4C_BG_ALIAS_DEPTH_MAX) \
			--bg-alias-jitter-hz $(R4C_BG_ALIAS_JITTER) \
			--ka-mode library \
			--ka-prior-path $(R4C_KA_PRIOR_PATH) \
			--ka-directional-beta \
			--ka-kappa 30 \
			--ka-beta-bounds "0.05,0.50" \
			--ka-target-shrink-perp 0.95 \
			--ka-gate-enable \
			--feasibility-mode updated \
			--aperture-phase-std $(R4C_PHASE_STD) \
			--aperture-phase-corr-len $(R4C_PHASE_CORR_LEN) \
			--aperture-phase-seed $(R4C_PHASE_SEED) \
			--clutter-beta $(R4C_CLUTTER_BETA) \
			--clutter-snr-db $(R4C_CLUTTER_SNR_DB) \
			--clutter-depth-min-frac $(R4C_CLUTTER_DEPTH_MIN) \
			--clutter-depth-max-frac $(R4C_CLUTTER_DEPTH_MAX) \
			--stap-device cuda; \
	done

.PHONY: r4c-hab-realistic-replay
r4c-hab-realistic-replay:
	set -euo pipefail
	for s in $(R4C_HAB_SEEDS); do \
		src_dir=$(R4_PILOT_ROOT)/$(R4C_HAB_RUN_PREFIX)$${s}; \
		out_stap=$(R4_PILOT_ROOT)/$(R4C_HAB_RUN_PREFIX)$${s}_replay_stap_msd_realistic; \
		out_ka=$(R4_PILOT_ROOT)/$(R4C_HAB_RUN_PREFIX)$${s}_replay_stap_ka_msd_realistic; \
		echo "[r4c-hab-realistic-replay] STAP-only MSD realistic seed $${s} -> $${out_stap}"; \
		PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/replay_stap_from_run.py \
			--src $${src_dir} \
			--out $${out_stap} \
			--baseline mc_svd --reg-disable --svd-rank 3 \
			--tile-h $(R4C_TILE) --tile-w $(R4C_TILE) --tile-stride $(R4C_TILE_STRIDE) \
			--lt $(R4C_LT) \
			--diag-load 0.03 \
			--cov-estimator tyler_pca --huber-c 5.0 \
			--fd-span-mode psd --fd-span-rel "0.30,1.10" \
			--grid-step-rel 0.18 --max-pts 4 --fd-min-pts 3 \
			--msd-lambda 0.05 --msd-ridge 0.08 --msd-agg median \
			--msd-ratio-rho 0.05 \
			--motion-half-span-rel 0.25 --msd-contrast-alpha 0.7 \
			--constraint-mode exp+deriv --constraint-ridge 0.15 \
			--mvdr-load-mode auto --mvdr-auto-kappa 100.0 \
			--score-mode msd \
			--feasibility-mode updated \
			--flow-mask-mode default \
			--band-ratio-mode whitened \
			--psd-br-flow-low 60.0 --psd-br-flow-high 200.0 \
			--psd-br-alias-center 650.0 --psd-br-alias-width 140.0 \
			--bg-guard-enabled --bg-guard-metric tile_p90 \
			--bg-guard-target-p90 1.15 --bg-guard-min-alpha 0.6 \
			--flow-alias-hz $(R4C_HAB_FLOW_ALIAS_HZ) \
			--flow-alias-fraction $(R4C_HAB_FLOW_ALIAS_FRAC) \
			--flow-alias-depth-min-frac $(R4C_HAB_DEEP_ALIAS_DEPTH_MIN) \
			--flow-alias-depth-max-frac $(R4C_HAB_DEEP_ALIAS_DEPTH_MAX) \
			--flow-alias-jitter-hz 80.0 \
			--bg-alias-hz $(R4C_HAB_BG_ALIAS_HZ) \
			--bg-alias-fraction $(R4C_HAB_BG_ALIAS_FRAC) \
			--bg-alias-depth-min-frac $(R4C_HAB_PIAL_DEPTH_MIN) \
			--bg-alias-depth-max-frac $(R4C_HAB_PIAL_DEPTH_MAX) \
			--bg-alias-jitter-hz 120.0 \
			--flow-doppler-min-hz 40.0 --flow-doppler-max-hz 220.0 \
			--vibration-hz $(R4C_HAB_VIBRATION_HZ) \
			--vibration-amp $(R4C_HAB_VIBRATION_AMP) \
			--vibration-depth-min-frac $(R4C_HAB_VIBRATION_DEPTH_MIN) \
			--vibration-depth-decay-frac $(R4C_HAB_VIBRATION_DEPTH_DECAY) \
			--clutter-beta $(R4C_CLUTTER_BETA) \
			--clutter-snr-db 25.0 \
			--clutter-depth-min-frac $(R4C_CLUTTER_DEPTH_MIN) \
			--clutter-depth-max-frac $(R4C_CLUTTER_DEPTH_MAX) \
			--aperture-phase-std 1.2 \
			--aperture-phase-corr-len 10.0 \
			--aperture-phase-seed $(R4C_PHASE_SEED); \
		echo "[r4c-hab-realistic-replay] STAP+KA MSD realistic seed $${s} -> $${out_ka}"; \
		PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/replay_stap_from_run.py \
			--src $${src_dir} \
			--out $${out_ka} \
			--baseline mc_svd --reg-disable --svd-rank 3 \
			--tile-h $(R4C_TILE) --tile-w $(R4C_TILE) --tile-stride $(R4C_TILE_STRIDE) \
			--lt $(R4C_LT) \
			--diag-load 0.03 \
			--cov-estimator tyler_pca --huber-c 5.0 \
			--fd-span-mode psd --fd-span-rel "0.30,1.10" \
			--grid-step-rel 0.18 --max-pts 4 --fd-min-pts 3 \
			--msd-lambda 0.05 --msd-ridge 0.08 --msd-agg median \
			--msd-ratio-rho 0.05 \
			--motion-half-span-rel 0.25 --msd-contrast-alpha 0.7 \
			--constraint-mode exp+deriv --constraint-ridge 0.15 \
			--mvdr-load-mode auto --mvdr-auto-kappa 100.0 \
			--score-mode msd \
			--feasibility-mode updated \
			--flow-mask-mode default \
			--band-ratio-mode whitened \
			--psd-br-flow-low 60.0 --psd-br-flow-high 200.0 \
			--psd-br-alias-center 650.0 --psd-br-alias-width 140.0 \
			--bg-guard-enabled --bg-guard-metric tile_p90 \
			--bg-guard-target-p90 1.15 --bg-guard-min-alpha 0.6 \
			--flow-alias-hz $(R4C_HAB_FLOW_ALIAS_HZ) \
			--flow-alias-fraction $(R4C_HAB_FLOW_ALIAS_FRAC) \
			--flow-alias-depth-min-frac $(R4C_HAB_DEEP_ALIAS_DEPTH_MIN) \
			--flow-alias-depth-max-frac $(R4C_HAB_DEEP_ALIAS_DEPTH_MAX) \
			--flow-alias-jitter-hz 80.0 \
			--bg-alias-hz $(R4C_HAB_BG_ALIAS_HZ) \
			--bg-alias-fraction $(R4C_HAB_BG_ALIAS_FRAC) \
			--bg-alias-depth-min-frac $(R4C_HAB_PIAL_DEPTH_MIN) \
			--bg-alias-depth-max-frac $(R4C_HAB_PIAL_DEPTH_MAX) \
			--bg-alias-jitter-hz 120.0 \
			--flow-doppler-min-hz 40.0 --flow-doppler-max-hz 220.0 \
			--vibration-hz $(R4C_HAB_VIBRATION_HZ) \
			--vibration-amp $(R4C_HAB_VIBRATION_AMP) \
			--vibration-depth-min-frac $(R4C_HAB_VIBRATION_DEPTH_MIN) \
			--vibration-depth-decay-frac $(R4C_HAB_VIBRATION_DEPTH_DECAY) \
			--clutter-beta $(R4C_CLUTTER_BETA) \
			--clutter-snr-db 25.0 \
			--clutter-depth-min-frac $(R4C_CLUTTER_DEPTH_MIN) \
			--clutter-depth-max-frac $(R4C_CLUTTER_DEPTH_MAX) \
			--aperture-phase-std 1.2 \
			--aperture-phase-corr-len 10.0 \
			--aperture-phase-seed $(R4C_PHASE_SEED) \
			--ka-mode library \
			--ka-prior-path runs/motion/priors/ka_prior_lt8_prf3k.npy \
			--ka-directional-beta \
			--ka-kappa 30.0 \
			--ka-beta-bounds 0.05,0.50 \
			--ka-target-shrink-perp 0.95 \
			--ka-gate-enable; \
	done

.PHONY: r4c-pial-generate
r4c-pial-generate:
	set -euo pipefail
	mkdir -p $(R4C_MCSV_OUT)
	for s in $(R4C_PIAL_SEEDS); do \
		out_dir=$(R4_PILOT_ROOT)/$(R4C_PIAL_RUN_PREFIX)$${s}; \
		echo "[r4c-pial] generating pilot seed $${s} -> $${out_dir}"; \
		PYTHONPATH=. conda run -n $(ENV_NAME) python sim/kwave/pilot_motion.py \
			--out $${out_dir} \
			--profile Brain-Pial128 \
			--angles=$(R4_ANGLES) \
			--ensembles $(R4C_PIAL_ENSEMBLES) \
			--jitter_um $(R4_JITTER_UM) \
			--pulses $(R4C_PIAL_PULSES) \
			--prf $(R4C_PRF) \
			--seed $${s} \
			--Nx $(R4C_NX) --Ny $(R4C_NY) \
			--tile-h $(R4C_TILE) --tile-w $(R4C_TILE) --tile-stride $(R4C_TILE_STRIDE) \
			--lt $(R4C_LT) \
			--pml-size $(R4C_PML_SIZE) \
			--band-ratio-flow-low-hz $(R4C_BAND_FLOW_LOW) \
			--band-ratio-flow-high-hz $(R4C_BAND_FLOW_HIGH) \
			--band-ratio-alias-center-hz $(R4C_BAND_ALIAS_CENTER) \
			--band-ratio-alias-width-hz $(R4C_BAND_ALIAS_WIDTH) \
			--diag-load 1e-2 --cov-estimator tyler_pca --huber-c 5.0 \
			--fd-span-mode psd --fd-span-rel "0.30,1.10" \
			--msd-lambda 5e-2 --msd-ridge 0.06 --msd-agg median --msd-ratio-rho 0.05 \
			--motion-half-span-rel 0.25 --msd-contrast-alpha 0.8 \
			--flow-alias-hz $(R4C_PIAL_FLOW_ALIAS_HZ) \
			--flow-alias-fraction $(R4C_PIAL_FLOW_ALIAS_FRAC) \
			--flow-alias-depth-min-frac $(R4C_PIAL_FLOW_ALIAS_DEPTH_MIN) \
			--flow-alias-depth-max-frac $(R4C_PIAL_FLOW_ALIAS_DEPTH_MAX) \
			--flow-alias-jitter-hz $(R4C_PIAL_FLOW_ALIAS_JITTER) \
			--flow-doppler-min-hz $(R4C_FLOW_DOPPLER_MIN) --flow-doppler-max-hz $(R4C_FLOW_DOPPLER_MAX) \
			--bg-alias-hz $(R4C_PIAL_BG_ALIAS_HZ) \
			--bg-alias-fraction $(R4C_PIAL_BG_ALIAS_FRAC) \
			--bg-alias-depth-min-frac $(R4C_PIAL_BG_ALIAS_DEPTH_MIN) \
			--bg-alias-depth-max-frac $(R4C_PIAL_BG_ALIAS_DEPTH_MAX) \
			--bg-alias-jitter-hz $(R4C_PIAL_BG_ALIAS_JITTER) \
			--vibration-hz $(R4C_PIAL_VIBRATION_HZ) \
			--vibration-amp $(R4C_PIAL_VIBRATION_AMP) \
			--vibration-depth-min-frac $(R4C_PIAL_VIBRATION_DEPTH_MIN) \
			--vibration-depth-decay-frac $(R4C_PIAL_VIBRATION_DEPTH_DECAY) \
			--ka-mode library \
			--ka-prior-path $(R4C_KA_PRIOR_PATH) \
			--ka-directional-beta \
			--ka-kappa 30 \
			--ka-beta-bounds "0.05,0.50" \
			--ka-target-shrink-perp 0.95 \
			--ka-gate-enable \
			--feasibility-mode updated \
			--aperture-phase-std $(R4C_PHASE_STD) \
			--aperture-phase-corr-len $(R4C_PHASE_CORR_LEN) \
			--aperture-phase-seed $(R4C_PHASE_SEED) \
			--clutter-beta $(R4C_CLUTTER_BETA) \
			--clutter-snr-db $(R4C_CLUTTER_SNR_DB) \
			--clutter-depth-min-frac $(R4C_CLUTTER_DEPTH_MIN) \
			--clutter-depth-max-frac $(R4C_CLUTTER_DEPTH_MAX) \
			--stap-device cuda; \
	done

.PHONY: brain-suite
brain-suite: run-stap-suite

.PHONY: r4c-replay
r4c-replay:
	set -euo pipefail
	for s in $(R4C_SEEDS); do \
		src_dir=$(R4_PILOT_ROOT)/$(R4C_RUN_PREFIX)$${s}; \
		for k in $(R4C_MCSV_K_VALUES); do \
			out_dir=$(R4C_MCSV_OUT)/mcsvd_k$${k}_reg$(R4C_MCSV_REG)_seed$${s}; \
			echo "[r4c] MC-SVD reg-on K=$${k} seed $${s}"; \
            PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/replay_stap_from_run.py \
                --src $${src_dir} \
                --out $${out_dir} \
                --baseline mc_svd \
                --reg-enable --reg-method phasecorr --reg-subpixel $(R4C_MCSV_REG) --reg-reference median \
                --svd-rank $${k} \
                --msd-contrast-alpha 0.8 \
                --tile-h $(R4C_TILE) --tile-w $(R4C_TILE) --tile-stride $(R4C_TILE_STRIDE) \
                --lt $(R4C_LT) \
				--psd-br-flow-low $(R4C_BAND_FLOW_LOW) \
				--psd-br-flow-high $(R4C_BAND_FLOW_HIGH) \
				--psd-br-alias-center $(R4C_BAND_ALIAS_CENTER) \
				--psd-br-alias-width $(R4C_BAND_ALIAS_WIDTH) \
				--flow-doppler-min-hz $(R4C_FLOW_DOPPLER_MIN) --flow-doppler-max-hz $(R4C_FLOW_DOPPLER_MAX) \
					--bg-alias-hz $(R4C_BG_ALIAS_HZ) --bg-alias-fraction $(R4C_BG_ALIAS_FRAC) \
					--bg-alias-depth-min-frac $(R4C_BG_ALIAS_DEPTH_MIN) --bg-alias-depth-max-frac $(R4C_BG_ALIAS_DEPTH_MAX) \
					--bg-alias-jitter-hz $(R4C_BG_ALIAS_JITTER) \
					--flow-mask-mode default \
					--flow-mask-pd-quantile $(R4C_FLOW_MASK_PD_Q) \
					--flow-mask-depth-min-frac $(R4C_FLOW_MASK_DEPTH_MIN) \
					--flow-mask-depth-max-frac $(R4C_FLOW_MASK_DEPTH_MAX) \
				--flow-mask-dilate-iters 2 \
				--aperture-phase-std $(R4C_PHASE_STD) \
				--aperture-phase-corr-len $(R4C_PHASE_CORR_LEN) \
				--aperture-phase-seed $(R4C_PHASE_SEED) \
				--clutter-beta $(R4C_CLUTTER_BETA) \
				--clutter-snr-db $(R4C_CLUTTER_SNR_DB) \
				--clutter-depth-min-frac $(R4C_CLUTTER_DEPTH_MIN) \
				--clutter-depth-max-frac $(R4C_CLUTTER_DEPTH_MAX) \
				--grid-step-rel $(R4C_GRID_STEP_REL) \
				--max-pts $(R4C_MAX_PTS) \
				--stap-device cuda; \
		done; \
	done

.PHONY: r4c-pial-replay
r4c-pial-replay:
	set -euo pipefail
	for s in $(R4C_PIAL_SEEDS); do \
		src_dir=$(R4_PILOT_ROOT)/$(R4C_PIAL_RUN_PREFIX)$${s}; \
		out_dir=$(R4C_MCSV_OUT)/pial_mcsvd_k8_reg$(R4C_MCSV_REG)_seed$${s}; \
		echo "[r4c-pial] MC-SVD reg-on K=8 seed $${s} -> $${out_dir}"; \
		PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/replay_stap_from_run.py \
			--src $${src_dir} \
			--out $${out_dir} \
			--baseline mc_svd \
			--reg-enable --reg-method phasecorr --reg-subpixel $(R4C_MCSV_REG) --reg-reference median \
			--svd-rank 8 \
			--msd-contrast-alpha 0.8 \
			--tile-h $(R4C_TILE) --tile-w $(R4C_TILE) --tile-stride $(R4C_TILE_STRIDE) \
			--lt $(R4C_LT) \
			--psd-br-flow-low $(R4C_BAND_FLOW_LOW) \
			--psd-br-flow-high $(R4C_BAND_FLOW_HIGH) \
			--psd-br-alias-center $(R4C_BAND_ALIAS_CENTER) \
			--psd-br-alias-width $(R4C_BAND_ALIAS_WIDTH) \
			--flow-doppler-min-hz $(R4C_FLOW_DOPPLER_MIN) --flow-doppler-max-hz $(R4C_FLOW_DOPPLER_MAX) \
			--bg-alias-hz $(R4C_PIAL_BG_ALIAS_HZ) --bg-alias-fraction $(R4C_PIAL_BG_ALIAS_FRAC) \
				--bg-alias-depth-min-frac $(R4C_PIAL_BG_ALIAS_DEPTH_MIN) --bg-alias-depth-max-frac $(R4C_PIAL_BG_ALIAS_DEPTH_MAX) \
				--bg-alias-jitter-hz $(R4C_PIAL_BG_ALIAS_JITTER) \
				--vibration-hz $(R4C_PIAL_VIBRATION_HZ) \
				--vibration-amp $(R4C_PIAL_VIBRATION_AMP) \
				--vibration-depth-min-frac $(R4C_PIAL_VIBRATION_DEPTH_MIN) \
				--vibration-depth-decay-frac $(R4C_PIAL_VIBRATION_DEPTH_DECAY) \
				--flow-mask-mode default \
				--flow-mask-pd-quantile $(R4C_FLOW_MASK_PD_Q) \
				--flow-mask-depth-min-frac $(R4C_FLOW_MASK_DEPTH_MIN) \
				--flow-mask-depth-max-frac $(R4C_FLOW_MASK_DEPTH_MAX) \
				--flow-mask-suppress-alias-depth \
			--flow-mask-dilate-iters 2 \
			--aperture-phase-std $(R4C_PHASE_STD) \
			--aperture-phase-corr-len $(R4C_PHASE_CORR_LEN) \
			--aperture-phase-seed $(R4C_PHASE_SEED) \
			--clutter-beta $(R4C_CLUTTER_BETA) \
			--clutter-snr-db $(R4C_CLUTTER_SNR_DB) \
			--clutter-depth-min-frac $(R4C_CLUTTER_DEPTH_MIN) \
			--clutter-depth-max-frac $(R4C_CLUTTER_DEPTH_MAX) \
			--grid-step-rel $(R4C_GRID_STEP_REL) \
			--max-pts $(R4C_MAX_PTS) \
			--stap-device cuda; \
	done

.PHONY: r4c-replay-ka
r4c-replay-ka:
	set -euo pipefail
	for s in $(R4C_SEEDS); do \
		src_dir=$(R4_PILOT_ROOT)/$(R4C_RUN_PREFIX)$${s}; \
		base_out=$(R4C_MCSV_OUT)/ka_seed$${s}; \
		diag_out=$(R4C_MCSV_OUT)/ka_alias_forced_seed$${s}; \
		echo "[r4c] KA acceptance seed $${s} -> $${base_out}"; \
        PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/replay_stap_from_run.py \
            --src $${src_dir} \
            --out $${base_out} \
            --baseline svd \
            --tile-h $(R4C_TILE) --tile-w $(R4C_TILE) --tile-stride $(R4C_TILE_STRIDE) \
            --lt $(R4C_LT) \
			--psd-br-flow-low $(R4C_BAND_FLOW_LOW) \
			--psd-br-flow-high $(R4C_BAND_FLOW_HIGH) \
			--psd-br-alias-center $(R4C_BAND_ALIAS_CENTER) \
			--psd-br-alias-width $(R4C_BAND_ALIAS_WIDTH) \
				--flow-doppler-min-hz $(R4C_FLOW_DOPPLER_MIN) --flow-doppler-max-hz $(R4C_FLOW_DOPPLER_MAX) \
				--bg-alias-hz $(R4C_BG_ALIAS_HZ) --bg-alias-fraction $(R4C_BG_ALIAS_FRAC) \
				--bg-alias-depth-min-frac $(R4C_BG_ALIAS_DEPTH_MIN) --bg-alias-depth-max-frac $(R4C_BG_ALIAS_DEPTH_MAX) \
				--bg-alias-jitter-hz $(R4C_BG_ALIAS_JITTER) \
				--flow-mask-mode default \
				--flow-mask-pd-quantile $(R4C_FLOW_MASK_PD_Q) \
				--flow-mask-depth-min-frac $(R4C_FLOW_MASK_DEPTH_MIN) \
				--flow-mask-depth-max-frac $(R4C_FLOW_MASK_DEPTH_MAX) \
			--flow-mask-dilate-iters 2 \
			--aperture-phase-std $(R4C_PHASE_STD) \
			--aperture-phase-corr-len $(R4C_PHASE_CORR_LEN) \
			--aperture-phase-seed $(R4C_PHASE_SEED) \
			--clutter-beta $(R4C_CLUTTER_BETA) \
			--clutter-snr-db $(R4C_CLUTTER_SNR_DB) \
			--clutter-depth-min-frac $(R4C_CLUTTER_DEPTH_MIN) \
			--clutter-depth-max-frac $(R4C_CLUTTER_DEPTH_MAX) \
			--grid-step-rel $(R4C_GRID_STEP_REL) \
			--max-pts $(R4C_MAX_PTS) \
			--psd-telemetry --psd-tapers $(R4C_PSD_TAPERS) --psd-bandwidth $(R4C_PSD_BANDWIDTH) \
			--ka-mode library \
			--ka-prior-path $(R4C_KA_PRIOR_PATH) \
			--ka-directional-beta \
			--ka-kappa 30 \
			--ka-beta-bounds "0.05,0.50" \
			--ka-target-retain-f 1.05 \
			--ka-target-shrink-perp 0.95 \
			--ka-equalize-pf-trace \
			--feasibility-mode updated \
			--stap-device cuda; \
	done

.PHONY: r4c-pial-replay-ka
r4c-pial-replay-ka:
	set -euo pipefail
	for s in $(R4C_PIAL_SEEDS); do \
		src_dir=$(R4_PILOT_ROOT)/$(R4C_PIAL_RUN_PREFIX)$${s}; \
		base_out=$(R4C_MCSV_OUT)/pial_ka_seed$${s}; \
		echo "[r4c-pial] KA STAP seed $${s} -> $${base_out}"; \
		PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/replay_stap_from_run.py \
			--src $${src_dir} \
			--out $${base_out} \
			--baseline svd \
			--tile-h $(R4C_TILE) --tile-w $(R4C_TILE) --tile-stride $(R4C_TILE_STRIDE) \
			--lt $(R4C_LT) \
			--psd-br-flow-low $(R4C_BAND_FLOW_LOW) \
			--psd-br-flow-high $(R4C_BAND_FLOW_HIGH) \
			--psd-br-alias-center $(R4C_BAND_ALIAS_CENTER) \
			--psd-br-alias-width $(R4C_BAND_ALIAS_WIDTH) \
			--flow-doppler-min-hz $(R4C_FLOW_DOPPLER_MIN) --flow-doppler-max-hz $(R4C_FLOW_DOPPLER_MAX) \
			--bg-alias-hz $(R4C_PIAL_BG_ALIAS_HZ) --bg-alias-fraction $(R4C_PIAL_BG_ALIAS_FRAC) \
			--bg-alias-depth-min-frac $(R4C_PIAL_BG_ALIAS_DEPTH_MIN) --bg-alias-depth-max-frac $(R4C_PIAL_BG_ALIAS_DEPTH_MAX) \
			--bg-alias-jitter-hz $(R4C_PIAL_BG_ALIAS_JITTER) \
				--vibration-hz $(R4C_PIAL_VIBRATION_HZ) \
				--vibration-amp $(R4C_PIAL_VIBRATION_AMP) \
				--vibration-depth-min-frac $(R4C_PIAL_VIBRATION_DEPTH_MIN) \
				--vibration-depth-decay-frac $(R4C_PIAL_VIBRATION_DEPTH_DECAY) \
				--flow-mask-mode default \
				--flow-mask-pd-quantile $(R4C_FLOW_MASK_PD_Q) \
				--flow-mask-depth-min-frac $(R4C_FLOW_MASK_DEPTH_MIN) \
				--flow-mask-depth-max-frac $(R4C_FLOW_MASK_DEPTH_MAX) \
			--flow-mask-suppress-alias-depth \
			--flow-mask-dilate-iters 2 \
			--aperture-phase-std $(R4C_PHASE_STD) \
			--aperture-phase-corr-len $(R4C_PHASE_CORR_LEN) \
			--aperture-phase-seed $(R4C_PHASE_SEED) \
			--clutter-beta $(R4C_CLUTTER_BETA) \
			--clutter-snr-db $(R4C_CLUTTER_SNR_DB) \
			--clutter-depth-min-frac $(R4C_CLUTTER_DEPTH_MIN) \
			--clutter-depth-max-frac $(R4C_CLUTTER_DEPTH_MAX) \
			--grid-step-rel $(R4C_GRID_STEP_REL) \
			--max-pts $(R4C_MAX_PTS) \
			--psd-telemetry --psd-tapers $(R4C_PSD_TAPERS) --psd-bandwidth $(R4C_PSD_BANDWIDTH) \
			--ka-mode library \
			--ka-prior-path $(R4C_KA_PRIOR_PATH) \
			--ka-directional-beta \
			--ka-kappa 30 \
			--ka-beta-bounds "0.05,0.50" \
			--ka-target-retain-f 1.05 \
			--ka-target-shrink-perp 0.95 \
			--ka-gate-enable \
			--ka-gate-alias-rmin 0.0 \
			--ka-gate-flow-cov-min 0.06 \
			--ka-gate-depth-min-frac 0.15 \
			--ka-gate-depth-max-frac 0.40 \
			--feasibility-mode updated \
			--stap-device cuda; \
	done

.PHONY: r4c-replay-rpca
r4c-replay-rpca:
	set -euo pipefail
	for s in $(R4C_SEEDS); do \
		src_dir=$(R4_PILOT_ROOT)/$(R4C_RUN_PREFIX)$${s}; \
		out_dir=$(R4C_MCSV_OUT)/rpca_stride3_v2_seed$${s}; \
		echo "[r4c] RPCA baseline seed $${s} -> $${out_dir}"; \
		PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/replay_stap_from_run.py \
			--src $${src_dir} \
			--out $${out_dir} \
			--baseline rpca \
			--rpca-enable --rpca-max-iters 250 \
			--msd-contrast-alpha 0.8 \
			--tile-h $(R4C_TILE) --tile-w $(R4C_TILE) --tile-stride $(R4C_TILE_STRIDE) \
			--lt $(R4C_LT) \
			--psd-br-flow-low $(R4C_BAND_FLOW_LOW) \
			--psd-br-flow-high $(R4C_BAND_FLOW_HIGH) \
			--psd-br-alias-center $(R4C_BAND_ALIAS_CENTER) \
			--psd-br-alias-width $(R4C_BAND_ALIAS_WIDTH) \
				--flow-doppler-min-hz $(R4C_FLOW_DOPPLER_MIN) --flow-doppler-max-hz $(R4C_FLOW_DOPPLER_MAX) \
				--bg-alias-hz $(R4C_BG_ALIAS_HZ) --bg-alias-fraction $(R4C_BG_ALIAS_FRAC) \
				--bg-alias-depth-min-frac $(R4C_BG_ALIAS_DEPTH_MIN) --bg-alias-depth-max-frac $(R4C_BG_ALIAS_DEPTH_MAX) \
				--bg-alias-jitter-hz $(R4C_BG_ALIAS_JITTER) \
				--flow-mask-mode default \
				--flow-mask-pd-quantile $(R4C_FLOW_MASK_PD_Q) \
				--flow-mask-depth-min-frac $(R4C_FLOW_MASK_DEPTH_MIN) \
				--flow-mask-depth-max-frac $(R4C_FLOW_MASK_DEPTH_MAX) \
			--flow-mask-dilate-iters 2 \
			--aperture-phase-std $(R4C_PHASE_STD) \
			--aperture-phase-corr-len $(R4C_PHASE_CORR_LEN) \
			--aperture-phase-seed $(R4C_PHASE_SEED) \
			--clutter-beta $(R4C_CLUTTER_BETA) \
			--clutter-snr-db $(R4C_CLUTTER_SNR_DB) \
			--clutter-depth-min-frac $(R4C_CLUTTER_DEPTH_MIN) \
			--clutter-depth-max-frac $(R4C_CLUTTER_DEPTH_MAX) \
			--fd-span-mode psd --fd-span-rel "$(R4C_BAND_FLOW_LOW),$(R4C_BAND_FLOW_HIGH)" \
			--grid-step-rel $(R4C_GRID_STEP_REL) \
			--max-pts $(R4C_MAX_PTS) \
			--fd-min-pts 3 --fd-min-abs-hz 0.0 \
			--stap-device cuda; \
	done

.PHONY: r4c-replay-hosvd
r4c-replay-hosvd:
	set -euo pipefail
	for s in $(R4C_SEEDS); do \
		src_dir=$(R4_PILOT_ROOT)/$(R4C_RUN_PREFIX)$${s}; \
		out_dir=$(R4C_MCSV_OUT)/hosvd_stride3_seed$${s}; \
		echo "[r4c] HOSVD baseline seed $${s} -> $${out_dir}"; \
		PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/replay_stap_from_run.py \
			--src $${src_dir} \
			--out $${out_dir} \
			--baseline hosvd \
			--hosvd-spatial-downsample 2 \
			--hosvd-energy-fracs 0.99,0.99,0.99 \
			--msd-contrast-alpha 0.8 \
			--tile-h $(R4C_TILE) --tile-w $(R4C_TILE) --tile-stride $(R4C_TILE_STRIDE) \
			--lt $(R4C_LT) \
			--psd-br-flow-low $(R4C_BAND_FLOW_LOW) \
			--psd-br-flow-high $(R4C_BAND_FLOW_HIGH) \
			--psd-br-alias-center $(R4C_BAND_ALIAS_CENTER) \
			--psd-br-alias-width $(R4C_BAND_ALIAS_WIDTH) \
				--flow-doppler-min-hz $(R4C_FLOW_DOPPLER_MIN) --flow-doppler-max-hz $(R4C_FLOW_DOPPLER_MAX) \
				--bg-alias-hz $(R4C_BG_ALIAS_HZ) --bg-alias-fraction $(R4C_BG_ALIAS_FRAC) \
				--bg-alias-depth-min-frac $(R4C_BG_ALIAS_DEPTH_MIN) --bg-alias-depth-max-frac $(R4C_BG_ALIAS_DEPTH_MAX) \
				--bg-alias-jitter-hz $(R4C_BG_ALIAS_JITTER) \
				--flow-mask-mode default \
				--flow-mask-pd-quantile $(R4C_FLOW_MASK_PD_Q) \
				--flow-mask-depth-min-frac $(R4C_FLOW_MASK_DEPTH_MIN) \
				--flow-mask-depth-max-frac $(R4C_FLOW_MASK_DEPTH_MAX) \
			--flow-mask-dilate-iters 2 \
			--aperture-phase-std $(R4C_PHASE_STD) \
			--aperture-phase-corr-len $(R4C_PHASE_CORR_LEN) \
			--aperture-phase-seed $(R4C_PHASE_SEED) \
			--clutter-beta $(R4C_CLUTTER_BETA) \
			--clutter-snr-db $(R4C_CLUTTER_SNR_DB) \
			--clutter-depth-min-frac $(R4C_CLUTTER_DEPTH_MIN) \
			--clutter-depth-max-frac $(R4C_CLUTTER_DEPTH_MAX) \
			--fd-span-mode psd --fd-span-rel "$(R4C_BAND_FLOW_LOW),$(R4C_BAND_FLOW_HIGH)" \
			--grid-step-rel $(R4C_GRID_STEP_REL) \
			--max-pts $(R4C_MAX_PTS) \
			--fd-min-pts 3 --fd-min-abs-hz 0.0 \
			--stap-device cuda; \
	done
.PHONY: r4c-analyze
r4c-analyze:
	set -euo pipefail
	run_msd() {
		local fpr="$$1"
		local pauc="$$2"
		local roc_args=""
		for s in $(R4C_SEEDS); do
			local ds="$(R4C_DATASET_PREFIX)$${s}"
			roc_args="$$roc_args --bundle mcsvd=$(R4C_MCSV_OUT)/$(R4C_ANALYSIS_MCSV)_seed$${s}/$${ds}"
		done
		for s in $(R4C_SEEDS); do
			local ds="$(R4C_DATASET_PREFIX)$${s}"
			roc_args="$$roc_args --bundle ka=$(R4C_MCSV_OUT)/ka_seed$${s}/$${ds}"
		done
		echo "[r4c] aggregate MSD ROC (FPR=$${fpr}, pAUC<= $${pauc}) -> $(R4C_ROC_AGG_JSON)"
		PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/analyze_coverage_roc.py \
			--aggregate $$roc_args \
			--thresholds $(R4_ROC_THRESH) \
			--fpr-target $${fpr} \
			--extra-fpr-targets $(R4C_EXTRA_FPRS) \
			--pauc-max $${pauc} \
			--flow-mask-kind pd \
			--score-mode msd \
			--json $(R4C_ROC_AGG_JSON)
	}
	run_band() {
		local fpr="$$1"
		local pauc="$$2"
		local band_args=""
		for s in $(R4C_SEEDS); do
			local ds="$(R4C_DATASET_PREFIX)$${s}"; \
			band_args="$$band_args --bundle mcsvd=$(R4C_MCSV_OUT)/$(R4C_ANALYSIS_MCSV)_seed$${s}/$${ds}"
		done
		for s in $(R4C_SEEDS); do
			local ds="$(R4C_DATASET_PREFIX)$${s}"; \
			local forced_dir="$(R4C_MCSV_OUT)/ka_alias_forced_seed$${s}/$${ds}"; \
			if [ -f "$${forced_dir}/meta.json" ]; then \
				band_args="$$band_args --bundle ka=$${forced_dir}"; \
			else \
				band_args="$$band_args --bundle ka=$(R4C_MCSV_OUT)/ka_seed$${s}/$${ds}"; \
			fi
		done
		echo "[r4c] aggregate band-ratio ROC (FPR=$${fpr}, pAUC<= $${pauc}) -> $(R4C_BAND_JSON)"
		PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/analyze_coverage_roc.py \
			--aggregate $$band_args \
			--thresholds $(R4_ROC_THRESH) \
			--fpr-target $${fpr} \
			--extra-fpr-targets $(R4C_EXTRA_FPRS) \
			--pauc-max $${pauc} \
			--flow-mask-kind pd \
			--score-mode band_ratio \
			--json $(R4C_BAND_JSON)
	}
	effective_fpr="$(R4C_FPR_TARGET)"
	effective_pauc="$(R4C_PAUC_MAX)"
	run_msd "$$effective_fpr" "$$effective_pauc"
	PYTHONPATH=. python scripts/r4c_guards.py recommend \
		--roc-json $(R4C_ROC_AGG_JSON) \
		--desired-fpr $$effective_fpr \
		--desired-pauc $$effective_pauc \
		--note-path $(R4C_ANALYSIS_NOTE) \
		--out-json $(R4C_RECOMMEND_JSON) >/dev/null
	python -c "import json, pathlib; data=json.loads(pathlib.Path('$(R4C_RECOMMEND_JSON)').read_text()); \
print('needs_rerun={}'.format(int(data.get('rerun', False)))); \
print('recommended_fpr={}'.format(data.get('effective_fpr'))); \
print('recommended_pauc={}'.format(data.get('effective_pauc')))" > $(R4C_RECOMMEND_ENV)
	. $(R4C_RECOMMEND_ENV)
	if [ "$$needs_rerun" -eq 1 ]; then \
		effective_fpr="$$recommended_fpr"; \
		effective_pauc="$$recommended_pauc"; \
		echo "[r4c] insufficient negatives for $(R4C_FPR_TARGET); re-running at FPR=$${effective_fpr}, pAUC<=$${effective_pauc}"; \
		run_msd "$$effective_fpr" "$$effective_pauc"; \
	fi
	if [ -f "$(R4C_ANALYSIS_NOTE)" ]; then \
		echo "[r4c] note: $$(cat $(R4C_ANALYSIS_NOTE))"; \
	fi
	run_band "$$effective_fpr" "$$effective_pauc"
	PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/r4c_guards.py \
		amplitude --seeds "$(R4C_SEEDS)" --base-dir $(R4C_MCSV_OUT) --dataset-prefix $(R4C_DATASET_PREFIX)

.PHONY: r4c-pial-analyze
r4c-pial-analyze:
	set -euo pipefail
	roc_args=""
	for s in $(R4C_PIAL_SEEDS); do \
		ds="$(R4C_PIAL_DATASET_PREFIX)$${s}"; \
		roc_args="$$roc_args --bundle mcsvd=$(R4C_MCSV_OUT)/pial_mcsvd_k8_reg$(R4C_MCSV_REG)_seed$${s}/$${ds}"; \
	done
	for s in $(R4C_PIAL_SEEDS); do \
		ds="$(R4C_PIAL_DATASET_PREFIX)$${s}"; \
		roc_args="$$roc_args --bundle ka=$(R4C_MCSV_OUT)/pial_ka_seed$${s}/$${ds}"; \
	done
	echo "[r4c-pial] aggregate MSD ROC (FPR=$(R4C_FPR_TARGET), pAUC<= $(R4C_PAUC_MAX)) -> $(R4C_PIAL_ROC_AGG_JSON)"
	PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/analyze_coverage_roc.py \
		--aggregate $$roc_args \
		--thresholds $(R4_ROC_THRESH) \
		--fpr-target $(R4C_FPR_TARGET) \
		--extra-fpr-targets $(R4C_EXTRA_FPRS) \
		--pauc-max $(R4C_PAUC_MAX) \
		--flow-mask-kind pd \
		--score-mode msd \
		--json $(R4C_PIAL_ROC_AGG_JSON)

.PHONY: r4c-analyze-baselines
r4c-analyze-baselines:
	set -euo pipefail; \
	effective_fpr="$(R4C_FPR_TARGET)"; \
	effective_pauc="$(R4C_PAUC_MAX)"; \
	roc_args=""; \
	for s in $(R4C_SEEDS); do \
		ds="$(R4C_DATASET_PREFIX)$${s}"; \
		roc_args="$$roc_args --bundle mcsvd=$(R4C_MCSV_OUT)/$(R4C_ANALYSIS_MCSV)_seed$${s}/$${ds}"; \
	done; \
	for s in $(R4C_SEEDS); do \
		ds="$(R4C_DATASET_PREFIX)$${s}"; \
		roc_args="$$roc_args --bundle rpca=$(R4C_MCSV_OUT)/rpca_stride3_v2_seed$${s}/$${ds}"; \
	done; \
	for s in $(R4C_SEEDS); do \
		ds="$(R4C_DATASET_PREFIX)$${s}"; \
		roc_args="$$roc_args --bundle hosvd=$(R4C_MCSV_OUT)/hosvd_stride3_seed$${s}/$${ds}"; \
	done; \
	echo "[r4c] aggregate MSD ROC across MC-SVD, RPCA, HOSVD (FPR=$${effective_fpr}, pAUC<=$${effective_pauc})"; \
	PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/analyze_coverage_roc.py \
		--aggregate $$roc_args \
		--thresholds $(R4_ROC_THRESH) \
		--fpr-target $${effective_fpr} \
		--extra-fpr-targets $(R4C_EXTRA_FPRS) \
		--pauc-max $${effective_pauc} \
		--flow-mask-kind pd \
		--score-mode msd \
		--json reports/r4c_all_baselines_msd_roc_aggregate.json; \
	PYTHONPATH=. python scripts/r4c_guards.py recommend \
		--roc-json reports/r4c_all_baselines_msd_roc_aggregate.json \
		--desired-fpr $${effective_fpr} \
		--desired-pauc $${effective_pauc} \
		--note-path reports/r4c_all_baselines_note.txt \
		--out-json reports/r4c_all_baselines_effective_targets.json >/dev/null; \
	python -c "import json, pathlib; data=json.loads(pathlib.Path('reports/r4c_all_baselines_effective_targets.json').read_text()); \
print('needs_rerun={}'.format(int(data.get('rerun', False)))); \
print('recommended_fpr={}'.format(data.get('effective_fpr'))); \
print('recommended_pauc={}'.format(data.get('effective_pauc')))" > reports/r4c_all_baselines_effective_targets.env; \
	. reports/r4c_all_baselines_effective_targets.env; \
	if [ "$$needs_rerun" -eq 1 ]; then \
		effective_fpr="$$recommended_fpr"; \
		effective_pauc="$$recommended_pauc"; \
		echo "[r4c] [baselines] insufficient negatives for $(R4C_FPR_TARGET); re-running at FPR=$${effective_fpr}, pAUC<=$${effective_pauc}"; \
		PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/analyze_coverage_roc.py \
			--aggregate $$roc_args \
			--thresholds $(R4_ROC_THRESH) \
			--fpr-target $${effective_fpr} \
			--extra-fpr-targets $(R4C_EXTRA_FPRS) \
			--pauc-max $${effective_pauc} \
			--flow-mask-kind pd \
			--score-mode msd \
			--json reports/r4c_all_baselines_msd_roc_aggregate.json; \
	fi

.PHONY: r4c_brw_replay
r4c_brw_replay:
	mkdir -p $(R4C_BRW_OUT)
	for s in $(R4C_BRW_SEEDS); do \
		src_dir=$(R4_PILOT_ROOT)/$(R4C_RUN_PREFIX)$${s}; \
		base_out=$(R4C_BRW_OUT)/$(R4C_BRW_BASELINE_PREFIX)$${s}; \
		ka_out=$(R4C_BRW_OUT)/$(R4C_BRW_KA_PREFIX)$${s}; \
		echo "[r4c-brw] seed $${s}: MC-SVD -> $$base_out"; \
		PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/replay_stap_from_run.py \
			--src $$src_dir \
			--out $$base_out \
				--baseline mc_svd \
				--tile-h $(R4C_TILE) --tile-w $(R4C_TILE) --tile-stride 1 \
				--lt $(R4C_LT) \
				--flow-mask-mode default \
				--flow-mask-pd-quantile $(R4C_FLOW_MASK_PD_Q) \
				--flow-mask-depth-min-frac $(R4C_FLOW_MASK_DEPTH_MIN) \
				--flow-mask-depth-max-frac $(R4C_FLOW_MASK_DEPTH_MAX) \
			--score-mode $(R4C_BRW_SCORE_MODE) \
			--psd-telemetry --psd-tapers $(R4C_BRW_PSD_TAPERS) --psd-bandwidth $(R4C_BRW_PSD_BANDWIDTH) \
			--psd-br-flow-low $(R4C_BRW_FLOW_LOW) \
			--psd-br-flow-high $(R4C_BRW_FLOW_HIGH) \
			--psd-br-alias-center $(R4C_BRW_ALIAS_CENTER) \
			--psd-br-alias-width $(R4C_BRW_ALIAS_WIDTH) \
			--flow-alias-hz $(R4C_FLOW_ALIAS_HZ) \
			--flow-alias-fraction $(R4C_FLOW_ALIAS_FRAC) \
			--flow-alias-depth-min-frac $(R4C_FLOW_ALIAS_DEPTH_MIN) \
			--flow-alias-depth-max-frac $(R4C_FLOW_ALIAS_DEPTH_MAX) \
			--flow-alias-jitter-hz $(R4C_FLOW_ALIAS_JITTER) \
			--aperture-phase-std $(R4C_PHASE_STD) \
			--aperture-phase-corr-len $(R4C_PHASE_CORR_LEN) \
			--aperture-phase-seed $(R4C_PHASE_SEED) \
			--clutter-beta $(R4C_CLUTTER_BETA) \
			--clutter-snr-db $(R4C_CLUTTER_SNR_DB) \
			--clutter-depth-min-frac $(R4C_CLUTTER_DEPTH_MIN) \
			--clutter-depth-max-frac $(R4C_CLUTTER_DEPTH_MAX) \
			--stap-device cuda; \
		echo "[r4c-brw] seed $${s}: KA -> $$ka_out"; \
		PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/replay_stap_from_run.py \
			--src $$src_dir \
			--out $$ka_out \
				--baseline mc_svd \
				--tile-h $(R4C_TILE) --tile-w $(R4C_TILE) --tile-stride 1 \
				--lt $(R4C_LT) \
				--flow-mask-mode default \
				--flow-mask-pd-quantile $(R4C_FLOW_MASK_PD_Q) \
				--flow-mask-depth-min-frac $(R4C_FLOW_MASK_DEPTH_MIN) \
				--flow-mask-depth-max-frac $(R4C_FLOW_MASK_DEPTH_MAX) \
			--score-mode $(R4C_BRW_SCORE_MODE) \
			--psd-telemetry --psd-tapers $(R4C_BRW_PSD_TAPERS) --psd-bandwidth $(R4C_BRW_PSD_BANDWIDTH) \
			--psd-br-flow-low $(R4C_BRW_FLOW_LOW) \
			--psd-br-flow-high $(R4C_BRW_FLOW_HIGH) \
			--psd-br-alias-center $(R4C_BRW_ALIAS_CENTER) \
			--psd-br-alias-width $(R4C_BRW_ALIAS_WIDTH) \
			--alias-psd-select --alias-psd-select-bins 1 --alias-psd-select-ratio $(R4C_BRW_ALIAS_RATIO) \
			--flow-alias-hz $(R4C_FLOW_ALIAS_HZ) \
			--flow-alias-fraction $(R4C_FLOW_ALIAS_FRAC) \
			--flow-alias-depth-min-frac $(R4C_FLOW_ALIAS_DEPTH_MIN) \
			--flow-alias-depth-max-frac $(R4C_FLOW_ALIAS_DEPTH_MAX) \
			--flow-alias-jitter-hz $(R4C_FLOW_ALIAS_JITTER) \
			--aperture-phase-std $(R4C_PHASE_STD) \
			--aperture-phase-corr-len $(R4C_PHASE_CORR_LEN) \
			--aperture-phase-seed $(R4C_PHASE_SEED) \
			--clutter-beta $(R4C_CLUTTER_BETA) \
			--clutter-snr-db $(R4C_CLUTTER_SNR_DB) \
			--clutter-depth-min-frac $(R4C_CLUTTER_DEPTH_MIN) \
			--clutter-depth-max-frac $(R4C_CLUTTER_DEPTH_MAX) \
			--stap-device cuda; \
	done

.PHONY: r4c_brw_roc
r4c_brw_roc:
	{ \
		bundles=""; \
		for s in $(R4C_BRW_SEEDS); do \
			ds="$(R4C_DATASET_PREFIX)$${s}"; \
			bundles="$$bundles --bundle mcsvd=$(R4C_BRW_OUT)/$(R4C_BRW_BASELINE_PREFIX)$${s}/$${ds}"; \
		done; \
		for s in $(R4C_BRW_SEEDS); do \
			ds="$(R4C_DATASET_PREFIX)$${s}"; \
			bundles="$$bundles --bundle ka=$(R4C_BRW_OUT)/$(R4C_BRW_KA_PREFIX)$${s}/$${ds}"; \
		done; \
		extra_targets=""; \
		if [ -n "$(strip $(R4C_BRW_EXTRA_FPRS))" ]; then \
			extra_targets="--extra-fpr-targets $(R4C_BRW_EXTRA_FPRS)"; \
		fi; \
		echo "[r4c-brw] aggregate ROC -> $(R4C_BRW_ROC_JSON)"; \
		PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/analyze_coverage_roc.py \
			--aggregate $$bundles \
			--thresholds $(R4C_BRW_ROC_THRESH) \
			--fpr-target $(R4C_BRW_FPR_TARGET) \
			$$extra_targets \
			--pauc-max $(R4C_BRW_PAUC_MAX) \
			--flow-mask-kind pd \
			--score-mode band_ratio \
			--json $(R4C_BRW_ROC_JSON); \
	}

.PHONY: r4c_brw_confirm2
r4c_brw_confirm2:
	{ \
		bundles=""; \
		for s in $(R4C_BRW_SEEDS); do \
			ds="$(R4C_DATASET_PREFIX)$${s}"; \
			bundles="$$bundles --bundle ka=$(R4C_BRW_OUT)/$(R4C_BRW_KA_PREFIX)$${s}/$${ds}"; \
		done; \
		echo "[r4c-brw] Confirm2 -> $(R4C_BRW_CONFIRM_JSON)"; \
		PYTHONPATH=. conda run -n $(ENV_NAME) python scripts/confirm2_eval.py \
			$$bundles \
			--alpha2 $(R4C_BRW_CONFIRM_ALPHA2) \
			--u-quantile $(R4C_BRW_CONFIRM_U) \
			--score-mode band_ratio \
			--output $(R4C_BRW_CONFIRM_JSON); \
	}

.PHONY: r4c_brw_all
r4c_brw_all: r4c_brw_replay r4c_brw_roc r4c_brw_confirm2

precommit:
	pre-commit install

acceptance:
	python -m eval.acceptance_cli --simulate --alpha 1e-5 --fpr_target 1e-5

report:
	python -c "import glob, os; from eval.summary_pdf import save_summary_pdf; \
runs = sorted(glob.glob('runs/acceptance_*.json')); \
assert runs, 'No runs/acceptance_*.json found'; \
latest = runs[-1]; os.makedirs('reports', exist_ok=True); \
save_summary_pdf('reports/acceptance_summary_latest.pdf', latest, thumbs_dir='figs/outputs'); \
print('Wrote reports/acceptance_summary_latest.pdf')"

mc_sweep:
	mkdir -p $(dir $(MC_OUT)) $(MC_JSON_DIR)
	python -m eval.sweep_mc \
		--method $(MC_METHOD) \
		--fprs $(MC_FPRS) \
		--seeds $(MC_SEEDS) \
		--seed-offset $(MC_SEED_OFFSET) \
		--npos $(MC_NPOS) \
		--nneg $(MC_NNEG) \
		--height $(MC_HEIGHT) \
		--width $(MC_WIDTH) \
		--roc-thresholds $(MC_ROC_THRESH) \
		--device $(MC_DEVICE) \
		$(if $(filter 1 yes true,$(MC_CONFIRM2)),--confirm2,) \
		$(if $(filter 1 yes true,$(MC_CONFIRM2)),--alpha2 $(MC_ALPHA2),) \
		$(if $(filter 1 yes true,$(MC_CONFIRM2)),--confirm2-pairs $(MC_CONFIRM2_PAIRS),) \
		$(if $(filter 1 yes true,$(MC_CONFIRM2)),--confirm2-rho $(MC_CONFIRM2_RHO),) \
		$(if $(filter 1 yes true,$(MC_CONFIRM2)),--confirm2-ci-alpha $(MC_CONFIRM2_CI_ALPHA),) \
	--out $(MC_OUT) \
		--json-dir $(MC_JSON_DIR)

stress_sweep:
	mkdir -p $(STRESS_OUTDIR)
	python -m eval.stress_suite \
		--grid $(STRESS_GRID) \
		--tiles $(STRESS_TILES) \
		--seeds $(STRESS_SEEDS) \
		--seed-offset $(STRESS_SEED_OFFSET) \
		--npos $(STRESS_NPOS) \
		--nneg $(STRESS_NNEG) \
		--height $(STRESS_HEIGHT) \
		--width $(STRESS_WIDTH) \
		--fpr-target $(STRESS_FPR_TARGET) \
		--roc-thresholds $(STRESS_ROC_THRESH) \
		--device $(STRESS_DEVICE) \
		$(if $(filter 1 yes true,$(STRESS_CONFIRM2)),--confirm2,) \
		$(if $(filter 1 yes true,$(STRESS_CONFIRM2)),--alpha2 $(STRESS_ALPHA2),) \
		$(if $(filter 1 yes true,$(STRESS_CONFIRM2)),--confirm2-rho $(STRESS_CONFIRM2_RHO),) \
		$(if $(filter 1 yes true,$(STRESS_CONFIRM2)),--confirm2-pairs $(STRESS_CONFIRM2_PAIRS),) \
		$(if $(filter 1 yes true,$(STRESS_CONFIRM2)),--confirm2-ci-alpha $(STRESS_CONFIRM2_CI_ALPHA),) \
		$(if $(STRESS_RHO_INFLATE),--rho-inflate $(STRESS_RHO_INFLATE),) \
		--robust-cov $(STRESS_ROBUST_COV) \
		--huber-c $(STRESS_HUBER_C) \
		$(if $(filter 1 yes true,$(STRESS_MOTION_COMP)),--motion-comp,) \
		$(if $(STRESS_STEER_GRID),--steer-grid $(STRESS_STEER_GRID),) \
		$(if $(STRESS_STEER_FUSE),--steer-fuse $(STRESS_STEER_FUSE),) \
		$(if $(STRESS_GROUPING),--grouping $(STRESS_GROUPING),) \
		--configs $(STRESS_CONFIGS) \
		--outdir $(STRESS_OUTDIR)

aggregate_runs:
	mkdir -p $(dir $(AGG_OUT_CSV))
	python -m eval.aggregate \
		--csv $(AGG_CSV) \
		--group-cols $(AGG_GROUP_COLS) \
		--out-csv $(AGG_OUT_CSV) \
		$(if $(AGG_OUT_MD),--out-md $(AGG_OUT_MD),)
