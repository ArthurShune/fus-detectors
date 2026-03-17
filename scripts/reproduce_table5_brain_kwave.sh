#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${FUS_DETECTORS_CONDA_ENV:-${STAP_FUS_CONDA_ENV:-fus-detectors}}"
STAP_DEVICE="${STAP_DEVICE:-cuda}"

run_py() {
  if command -v conda >/dev/null 2>&1; then
    PYTHONPATH=. conda run -n "${ENV_NAME}" python "$@"
    return
  fi
  if command -v micromamba >/dev/null 2>&1; then
    PYTHONPATH=. micromamba run -n "${ENV_NAME}" python "$@"
    return
  fi
  PYTHONPATH=. python "$@"
}

looks_like_source_dir() {
  local d="$1"
  [[ -d "${d}" ]] || return 1
  compgen -G "${d}/angle_*" >/dev/null && return 0
  compgen -G "${d}/ens*_angle_*" >/dev/null && return 0
  return 1
}

ensure_pilot() {
  local out_dir="$1"
  shift
  if looks_like_source_dir "${out_dir}"; then
    echo "[repro] found pilot source: ${out_dir}"
    return
  fi
  echo "[repro] generating pilot source: ${out_dir}"
  run_py sim/kwave/pilot_motion.py --out "${out_dir}" "$@"
}

# ---------------------------------------------------------------------
# 0) Generate the three fixed-profile pilot sources used in Table 5.
# ---------------------------------------------------------------------
ensure_pilot "runs/pilot/r4c_kwave_seed1" \
  --profile Brain-OpenSkull \
  --angles=-12,-6,0,6,12 \
  --ensembles 5 \
  --jitter_um 40 \
  --pulses 64 \
  --prf 1500 \
  --seed 1 \
  --Nx 240 --Ny 240 \
  --tile-h 8 --tile-w 8 --tile-stride 3 \
  --lt 8 \
  --pml-size 16 \
  --band-ratio-flow-low-hz 30 \
  --band-ratio-flow-high-hz 250 \
  --band-ratio-alias-center-hz 575 \
  --band-ratio-alias-width-hz 175 \
  --diag-load 1e-2 --cov-estimator tyler_pca --huber-c 5.0 \
  --fd-span-mode psd --fd-span-rel "0.30,1.10" \
  --msd-lambda 5e-2 --msd-ridge 0.06 --msd-agg median --msd-ratio-rho 0.05 \
  --motion-half-span-rel 0.25 --msd-contrast-alpha 0.8 \
  --flow-doppler-min-hz 60 --flow-doppler-max-hz 180 \
  --bg-alias-hz 650 --bg-alias-fraction 0.3 \
  --bg-alias-depth-min-frac 0.30 --bg-alias-depth-max-frac 0.70 \
  --bg-alias-jitter-hz 50.0 \
  --feasibility-mode updated \
  --aperture-phase-std 0.8 \
  --aperture-phase-corr-len 14 \
  --aperture-phase-seed 111 \
  --clutter-beta 1.0 \
  --clutter-snr-db 20.0 \
  --clutter-depth-min-frac 0.20 \
  --clutter-depth-max-frac 0.95 \
  --stap-device "${STAP_DEVICE}"

ensure_pilot "runs/pilot/r4c_kwave_hab_seed2" \
  --profile Brain-AliasContract \
  --angles=-12,-6,0,6,12 \
  --ensembles 5 \
  --jitter_um 40 \
  --pulses 64 \
  --prf 1500 \
  --seed 2 \
  --Nx 240 --Ny 240 \
  --tile-h 8 --tile-w 8 --tile-stride 3 \
  --lt 8 \
  --pml-size 16 \
  --band-ratio-flow-low-hz 30 \
  --band-ratio-flow-high-hz 250 \
  --band-ratio-alias-center-hz 575 \
  --band-ratio-alias-width-hz 175 \
  --diag-load 1e-2 --cov-estimator tyler_pca --huber-c 5.0 \
  --fd-span-mode psd --fd-span-rel "0.30,1.10" \
  --msd-lambda 5e-2 --msd-ridge 0.06 --msd-agg median --msd-ratio-rho 0.05 \
  --motion-half-span-rel 0.25 --msd-contrast-alpha 0.8 \
  --flow-doppler-min-hz 60 --flow-doppler-max-hz 180 \
  --bg-alias-hz 650 \
  --bg-alias-fraction 0.75 \
  --bg-alias-depth-min-frac 0.12 \
  --bg-alias-depth-max-frac 0.28 \
  --bg-alias-jitter-hz 50.0 \
  --flow-alias-hz 650 \
  --flow-alias-fraction 0.15 \
  --flow-alias-depth-min-frac 0.28 \
  --flow-alias-depth-max-frac 0.85 \
  --flow-alias-jitter-hz 35.0 \
  --flow-amp-scale 1.0 \
  --alias-amp-scale 1.0 \
  --vibration-hz 450.0 \
  --vibration-amp 0.30 \
  --vibration-depth-min-frac 0.12 \
  --vibration-depth-decay-frac 0.30 \
  --feasibility-mode updated \
  --aperture-phase-std 0.8 \
  --aperture-phase-corr-len 14 \
  --aperture-phase-seed 111 \
  --clutter-beta 1.0 \
  --clutter-snr-db 20.0 \
  --clutter-depth-min-frac 0.20 \
  --clutter-depth-max-frac 0.95 \
  --stap-device "${STAP_DEVICE}"

ensure_pilot "runs/pilot/r4c_kwave_hab_v3_skull_seed2" \
  --profile Brain-SkullOR \
  --angles=-12,-6,0,6,12 \
  --ensembles 5 \
  --jitter_um 40 \
  --pulses 64 \
  --prf 1500 \
  --seed 2 \
  --Nx 240 --Ny 240 \
  --tile-h 8 --tile-w 8 --tile-stride 3 \
  --lt 8 \
  --pml-size 16 \
  --band-ratio-flow-low-hz 30 \
  --band-ratio-flow-high-hz 250 \
  --band-ratio-alias-center-hz 575 \
  --band-ratio-alias-width-hz 175 \
  --diag-load 1e-2 --cov-estimator tyler_pca --huber-c 5.0 \
  --fd-span-mode psd --fd-span-rel "0.30,1.10" \
  --msd-lambda 5e-2 --msd-ridge 0.06 --msd-agg median --msd-ratio-rho 0.05 \
  --motion-half-span-rel 0.25 --msd-contrast-alpha 0.8 \
  --flow-doppler-min-hz 60 --flow-doppler-max-hz 180 \
  --bg-alias-hz 650 --bg-alias-fraction 0.3 \
  --bg-alias-depth-min-frac 0.30 --bg-alias-depth-max-frac 0.70 \
  --bg-alias-jitter-hz 50.0 \
  --feasibility-mode updated \
  --aperture-phase-std 0.8 \
  --aperture-phase-corr-len 14 \
  --aperture-phase-seed 111 \
  --clutter-beta 1.0 \
  --clutter-snr-db 20.0 \
  --clutter-depth-min-frac 0.20 \
  --clutter-depth-max-frac 0.95 \
  --stap-device "${STAP_DEVICE}"

# ---------------------------------------------------------------------
# 1) Generate acceptance bundles + strict-tail report + LaTeX table.
# ---------------------------------------------------------------------
echo "[repro] running Table 5 baseline matrix + table render (device=${STAP_DEVICE})"
run_py scripts/fair_filter_comparison.py \
  --mode matrix --eval-score vnext \
  --matrix-regimes open,aliascontract,skullor \
  --matrix-seeds-open 1 --matrix-seeds-aliascontract 2 --matrix-seeds-skullor 2 \
  --window-length 64 --window-offsets 0,64,128,192,256 \
  --matrix-use-profile \
  --matrix-mcsvd-energy-frac 0.90 --matrix-mcsvd-baseline-support window \
  --methods mcsvd,svd_similarity,local_svd,rpca,hosvd,stap_full \
  --generated-root runs/pilot/fair_filter_matrix_pd_r3_localbaselines \
  --autogen-missing --stap-device "${STAP_DEVICE}" \
  --out-csv reports/fair_matrix_vnext_r3_localbaselines.csv \
  --out-json reports/fair_matrix_vnext_r3_localbaselines.json

run_py scripts/brain_kwave_vnext_baselines_table.py \
  --fair-matrix-json reports/fair_matrix_vnext_r3_localbaselines.json \
  --out-tex reports/brain_kwave_vnext_baselines_table.tex

echo "[repro] wrote reports/brain_kwave_vnext_baselines_table.tex"
echo "[repro] wrote reports/fair_matrix_vnext_r3_localbaselines.json"
