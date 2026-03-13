#!/usr/bin/env python3
"""
Fair, 1:1 low-FPR comparison across MC-SVD / RPCA / HOSVD and STAP.

Two modes:
  - static: compare a small set of pre-existing hand-picked bundles.
  - matrix: build a seed/window matrix, optionally regenerate missing replay
    bundles so all methods are evaluated on identical windows.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


REPO = Path(__file__).resolve().parents[1]

# NOTE: Prefer the r4c Brain-* pilots by default since they include the
# canonical synthetic Doppler/alias/clutter overlays used throughout the
# methodology (see Makefile r4c-* targets). The older r4_* pilots are kept as
# a fallback but typically do not include those overlays.
DEFAULT_SRC_TEMPLATE_OPEN = (
    "runs/pilot/r4c_kwave_seed{seed},runs/pilot/r4c_kwave_hab_seed{seed},runs/pilot/r4_kwave_seed{seed}"
)
DEFAULT_SRC_TEMPLATE_ALIAS = (
    # Historical naming: older repos used r4c_kwave_hab_contract_seed{seed}[_v2],
    # while current Makefile targets emit r4c_kwave_hab_seed{seed}.
    "runs/pilot/r4c_kwave_hab_contract_seed{seed}_v2,"
    "runs/pilot/r4c_kwave_hab_contract_seed{seed},"
    "runs/pilot/r4c_kwave_hab_seed{seed}"
)
DEFAULT_SRC_TEMPLATE_SKULL = (
    "runs/pilot/r4c_kwave_hab_v3_skull_seed{seed}_v2,runs/pilot/r4c_kwave_hab_v3_skull_seed{seed}"
)

_INJECT_META_KEY_TO_FLAG: Dict[str, str] = {
    # Synthetic Doppler (flow) overlays
    "flow_doppler_min_hz": "--flow-doppler-min-hz",
    "flow_doppler_max_hz": "--flow-doppler-max-hz",
    "flow_doppler_tone_amp": "--flow-doppler-tone-amp",
    "flow_doppler_noise_amp": "--flow-doppler-noise-amp",
    "flow_doppler_noise_rho": "--flow-doppler-noise-rho",
    "flow_doppler_noise_mode": "--flow-doppler-noise-mode",
    # Alias overlays
    "flow_alias_hz": "--flow-alias-hz",
    "flow_alias_fraction": "--flow-alias-fraction",
    "flow_alias_depth_min_frac": "--flow-alias-depth-min-frac",
    "flow_alias_depth_max_frac": "--flow-alias-depth-max-frac",
    "flow_alias_jitter_hz": "--flow-alias-jitter-hz",
    "bg_alias_hz": "--bg-alias-hz",
    "bg_alias_fraction": "--bg-alias-fraction",
    "bg_alias_depth_min_frac": "--bg-alias-depth-min-frac",
    "bg_alias_depth_max_frac": "--bg-alias-depth-max-frac",
    "bg_alias_jitter_hz": "--bg-alias-jitter-hz",
    # High-rank bg alias overlays (optional)
    "bg_alias_highrank_mode": "--bg-alias-highrank-mode",
    "bg_alias_highrank_deep_patch_coverage": "--bg-alias-highrank-coverage",
    "bg_alias_highrank_shallow_patch_coverage": "--bg-alias-highrank-shallow-coverage",
    "bg_alias_highrank_margin_px": "--bg-alias-highrank-margin-px",
    "bg_alias_highrank_freq_jitter_hz": "--bg-alias-highrank-freq-jitter-hz",
    "bg_alias_highrank_drift_step_hz": "--bg-alias-highrank-drift-step-hz",
    "bg_alias_highrank_drift_block_len": "--bg-alias-highrank-drift-block-len",
    "bg_alias_highrank_pf_leak_eta": "--bg-alias-highrank-pf-leak-eta",
    "bg_alias_highrank_amp": "--bg-alias-highrank-amp",
    # Vibration overlay
    "vibration_hz": "--vibration-hz",
    "vibration_amp": "--vibration-amp",
    "vibration_depth_min_frac": "--vibration-depth-min-frac",
    "vibration_depth_decay_frac": "--vibration-depth-decay-frac",
    # Aperture phase screen overlay
    "aperture_phase_std": "--aperture-phase-std",
    "aperture_phase_corr_len": "--aperture-phase-corr-len",
    "aperture_phase_seed": "--aperture-phase-seed",
    # Temporal clutter overlay
    "clutter_beta": "--clutter-beta",
    "clutter_snr_db": "--clutter-snr-db",
    "clutter_mode": "--clutter-mode",
    "clutter_rank": "--clutter-rank",
    "clutter_depth_min_frac": "--clutter-depth-min-frac",
    "clutter_depth_max_frac": "--clutter-depth-max-frac",
    # Amplitude scaling knobs
    "flow_amp_scale": "--flow-amp-scale",
    "alias_amp_scale": "--alias-amp-scale",
}


def _find_best_source_bundle_meta(source_dir: Path) -> Dict[str, Any] | None:
    candidates: List[Tuple[Path, Dict[str, Any]]] = []
    for bundle in sorted(source_dir.glob("pw_*")):
        meta_path = bundle / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = _load_json(meta_path)
        except Exception:
            continue
        candidates.append((bundle, meta))

    if not candidates:
        return None

    def _score(candidate: Tuple[Path, Dict[str, Any]]) -> Tuple[int, float]:
        _bundle, meta = candidate
        total_frames = meta.get("total_frames")
        try:
            frames = int(total_frames) if total_frames is not None else 0
        except (TypeError, ValueError):
            frames = 0
        # Prefer bundles that clearly encode the canonical overlays.
        overlays = 0.0
        for k in ("bg_alias_hz", "flow_doppler_min_hz", "clutter_beta", "vibration_hz"):
            v = meta.get(k)
            if v is None:
                continue
            try:
                overlays += 1.0 if float(v) != 0.0 else 0.0
            except (TypeError, ValueError):
                overlays += 1.0
        return frames, overlays

    # Choose the "best" meta by (frames, overlays) then by newest mtime.
    candidates.sort(key=_score, reverse=True)
    return candidates[0][1]


def _meta_get_in(meta: Mapping[str, Any], key: str) -> Any:
    if key in meta:
        return meta.get(key)
    # Fallbacks for older bundle schemas where some fields were nested.
    if key == "aperture_phase_seed":
        phase = meta.get("phase_screen")
        if isinstance(phase, dict):
            return phase.get("phase_seed")
    if key in {
        "clutter_beta",
        "clutter_snr_db",
        "clutter_mode",
        "clutter_rank",
        "clutter_depth_min_frac",
        "clutter_depth_max_frac",
    }:
        clutter = meta.get("temporal_clutter")
        if isinstance(clutter, dict):
            mapping = {
                "clutter_beta": "beta",
                "clutter_snr_db": "snr_db",
                "clutter_mode": "mode",
                "clutter_rank": "rank",
                "clutter_depth_min_frac": "depth_min_frac",
                "clutter_depth_max_frac": "depth_max_frac",
            }
            return clutter.get(mapping[key])
    return None


def _injection_cli_args_from_source(
    source_dir: Path,
    *,
    clutter_mode_override: str = "source",
    clutter_rank_override: int = 3,
) -> List[str]:
    meta = _find_best_source_bundle_meta(source_dir)
    if meta is None:
        return []

    vals: Dict[str, Any] = {k: _meta_get_in(meta, k) for k in _INJECT_META_KEY_TO_FLAG.keys()}

    def _f(key: str) -> float | None:
        v = vals.get(key)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _s(key: str) -> str | None:
        v = vals.get(key)
        if v is None:
            return None
        s = str(v).strip()
        return None if s.lower() in {"", "none"} else s

    bg_alias_enabled = (_f("bg_alias_hz") or 0.0) != 0.0
    flow_doppler_enabled = vals.get("flow_doppler_min_hz") is not None and vals.get(
        "flow_doppler_max_hz"
    ) is not None
    flow_alias_enabled = (_f("flow_alias_hz") or 0.0) != 0.0 and (_f("flow_alias_fraction") or 0.0) > 0.0
    vibration_enabled = (_f("vibration_hz") or 0.0) != 0.0 and (_f("vibration_amp") or 0.0) > 0.0
    aperture_phase_enabled = (_f("aperture_phase_std") or 0.0) > 0.0
    clutter_enabled = (_f("clutter_beta") or 0.0) > 0.0

    clutter_mode_override_norm = str(clutter_mode_override or "source").strip().lower()
    if clutter_mode_override_norm not in {"source", "fullrank", "lowrank"}:
        clutter_mode_override_norm = "source"
    if clutter_enabled and clutter_mode_override_norm != "source":
        vals["clutter_mode"] = clutter_mode_override_norm
        if clutter_mode_override_norm == "lowrank":
            vals["clutter_rank"] = int(clutter_rank_override)

    hr_mode = _s("bg_alias_highrank_mode")
    hr_cov = _f("bg_alias_highrank_deep_patch_coverage") or 0.0
    hr_amp = _f("bg_alias_highrank_amp") or 0.0
    bg_alias_highrank_enabled = hr_mode is not None and hr_mode.lower() != "none" and hr_cov > 0.0 and hr_amp > 0.0

    any_injection = (
        bg_alias_enabled
        or flow_doppler_enabled
        or flow_alias_enabled
        or vibration_enabled
        or aperture_phase_enabled
        or clutter_enabled
        or bg_alias_highrank_enabled
    )
    if not any_injection:
        return []

    def _add(key: str) -> None:
        flag = _INJECT_META_KEY_TO_FLAG[key]
        v = vals.get(key)
        if v is None:
            return
        if isinstance(v, str) and v.strip().lower() in {"none", ""}:
            return
        args.extend([flag, str(v)])

    args: List[str] = []

    if flow_doppler_enabled:
        _add("flow_doppler_min_hz")
        _add("flow_doppler_max_hz")
        if (_f("flow_doppler_tone_amp") or 0.0) > 0.0:
            _add("flow_doppler_tone_amp")
        if (_f("flow_doppler_noise_amp") or 0.0) > 0.0:
            _add("flow_doppler_noise_amp")
            _add("flow_doppler_noise_rho")
            _add("flow_doppler_noise_mode")

    if flow_alias_enabled:
        _add("flow_alias_hz")
        _add("flow_alias_fraction")
        _add("flow_alias_depth_min_frac")
        _add("flow_alias_depth_max_frac")
        _add("flow_alias_jitter_hz")
        alias_amp = _f("alias_amp_scale")
        if alias_amp is not None and abs(alias_amp - 1.0) > 1e-9:
            _add("alias_amp_scale")

    if bg_alias_enabled:
        _add("bg_alias_hz")
        _add("bg_alias_fraction")
        _add("bg_alias_depth_min_frac")
        _add("bg_alias_depth_max_frac")
        _add("bg_alias_jitter_hz")

    if bg_alias_highrank_enabled:
        _add("bg_alias_highrank_mode")
        _add("bg_alias_highrank_deep_patch_coverage")
        _add("bg_alias_highrank_shallow_patch_coverage")
        _add("bg_alias_highrank_margin_px")
        _add("bg_alias_highrank_freq_jitter_hz")
        _add("bg_alias_highrank_drift_step_hz")
        _add("bg_alias_highrank_drift_block_len")
        _add("bg_alias_highrank_pf_leak_eta")
        _add("bg_alias_highrank_amp")

    if vibration_enabled:
        _add("vibration_hz")
        _add("vibration_amp")
        _add("vibration_depth_min_frac")
        _add("vibration_depth_decay_frac")

    if aperture_phase_enabled:
        _add("aperture_phase_std")
        _add("aperture_phase_corr_len")
        _add("aperture_phase_seed")

    if clutter_enabled:
        _add("clutter_beta")
        _add("clutter_snr_db")
        _add("clutter_mode")
        if str(vals.get("clutter_mode") or "").strip().lower() == "lowrank":
            _add("clutter_rank")
        _add("clutter_depth_min_frac")
        _add("clutter_depth_max_frac")

    flow_amp = _f("flow_amp_scale")
    if flow_amp is not None and abs(flow_amp - 1.0) > 1e-9:
        _add("flow_amp_scale")

    return args


def _find_single_bundle(run_dir: Path) -> Path:
    bundles = sorted(run_dir.glob("pw_*"))
    if not bundles:
        raise FileNotFoundError(f"No pw_* bundle found under {run_dir}")
    if len(bundles) != 1:
        raise RuntimeError(f"Expected 1 pw_* bundle under {run_dir}, found {len(bundles)}")
    return bundles[0]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _load_mask(bundle_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    mask_flow = np.load(bundle_dir / "mask_flow.npy").astype(bool, copy=False)
    mask_bg = np.load(bundle_dir / "mask_bg.npy").astype(bool, copy=False)
    if mask_flow.shape != mask_bg.shape:
        raise RuntimeError(f"Mask shape mismatch: {mask_flow.shape} vs {mask_bg.shape} in {bundle_dir}")
    if np.any(mask_flow & mask_bg):
        raise RuntimeError(f"Masks overlap in {bundle_dir}")
    return mask_flow, mask_bg


def _score_from_pd_with_convention(
    pd_map: np.ndarray,
    *,
    roc_convention: str | None,
    legacy_default: str,
) -> Tuple[np.ndarray, str]:
    roc = (roc_convention or "").strip().lower()
    if "lower_tail_on_pd" in roc:
        return (-pd_map), "score=-pd (meta pd_mode: lower_tail_on_pd)"
    if "right_tail_on_pd" in roc:
        return pd_map, "score=pd (meta pd_mode: right_tail_on_pd)"
    if legacy_default == "lower":
        return (-pd_map), "score=-pd (legacy default; missing score_pd + pd_mode)"
    if legacy_default == "upper":
        return pd_map, "score=pd (legacy default; missing score_pd + pd_mode)"
    raise ValueError(f"Unrecognized legacy_default={legacy_default!r}")


def _load_pd_score(
    bundle_dir: Path,
    *,
    which: str,
    legacy_default: str,
) -> Tuple[np.ndarray, str, str | None]:
    if which not in {"base", "stap"}:
        raise ValueError(f"which must be 'base' or 'stap', got {which!r}")
    meta = _load_json(bundle_dir / "meta.json")
    pd_mode = meta.get("pd_mode") if isinstance(meta.get("pd_mode"), dict) else {}
    roc_convention = pd_mode.get("roc_convention") if isinstance(pd_mode, dict) else None

    score_path = bundle_dir / f"score_pd_{which}.npy"
    if score_path.exists():
        return np.load(score_path), f"file:{score_path.name}", str(roc_convention) if roc_convention else None

    pd_path = bundle_dir / f"pd_{which}.npy"
    if not pd_path.exists():
        raise FileNotFoundError(f"Missing {pd_path.name} and {score_path.name} in {bundle_dir}")
    pd_map = np.load(pd_path)
    score, desc = _score_from_pd_with_convention(
        pd_map, roc_convention=str(roc_convention) if roc_convention else None, legacy_default=legacy_default
    )
    return score, desc, str(roc_convention) if roc_convention else None


def _load_vnext_score(bundle_dir: Path, *, which: str) -> Tuple[np.ndarray, str, str | None]:
    """
    Load score-vNext exports:
      - score_base.npy (baseline detector score; right-tail)
      - score_stap_preka.npy (STAP detector score pre-KA; right-tail)
      - score_stap.npy (STAP detector score post-KA; right-tail; fallback)
    """
    if which not in {"base", "stap"}:
        raise ValueError(f"which must be 'base' or 'stap', got {which!r}")

    score_name = None
    score_name_path = bundle_dir / "score_name.txt"
    if score_name_path.exists():
        try:
            score_name = score_name_path.read_text(encoding="utf-8").strip()
        except Exception:
            score_name = None

    if which == "base":
        primary = bundle_dir / "score_base.npy"
        candidates = [primary]
    else:
        primary = bundle_dir / "score_stap_preka.npy"
        candidates = [primary, bundle_dir / "score_stap.npy"]
        # Prefer the pre-KA detector score; if missing, fall back to the final score.
    for cand in candidates:
        if cand.exists():
            return np.load(cand), f"file:{cand.name}", score_name

    # Fallback for older bundles (pre score-vNext export block).
    legacy = bundle_dir / ("base_score_map.npy" if which == "base" else "stap_score_map.npy")
    if legacy.exists():
        return np.load(legacy), f"file:{legacy.name}", score_name

    if which == "base":
        missing = "score_base.npy"
    else:
        missing = "score_stap_preka.npy (or score_stap.npy)"
    raise FileNotFoundError(f"Missing {missing} and {legacy.name} in {bundle_dir}")


def _tpr_at_fpr(
    scores_pos: np.ndarray, scores_neg: np.ndarray, fpr: float
) -> Tuple[float, float, float]:
    if not (0.0 < fpr < 1.0):
        raise ValueError(f"fpr must be in (0,1), got {fpr}")
    scores_neg = np.asarray(scores_neg, dtype=float)
    scores_pos = np.asarray(scores_pos, dtype=float)
    if scores_neg.size == 0 or scores_pos.size == 0:
        return float("nan"), float("nan"), float("nan")
    neg_sorted = np.sort(scores_neg)
    n_neg = int(neg_sorted.size)
    idx = int(np.floor((1.0 - float(fpr)) * n_neg))
    idx = max(0, min(idx, n_neg - 1))
    thr = float(neg_sorted[idx])
    fpr_emp = float((scores_neg >= thr).mean())
    # If there is a large mass at `thr` (ties), the empirical FPR can exceed
    # the target. Promote the threshold by one ulp to enforce fpr_emp <= fpr.
    if fpr_emp > float(fpr):
        thr = float(np.nextafter(thr, np.inf))
        fpr_emp = float((scores_neg >= thr).mean())
    tpr = float((scores_pos >= thr).mean())
    return thr, tpr, fpr_emp


def _safe_get_latency_ms(meta: Mapping[str, Any]) -> Tuple[float | None, float | None]:
    tele = meta.get("stap_fallback_telemetry") if isinstance(meta.get("stap_fallback_telemetry"), dict) else {}
    baseline_ms = tele.get("baseline_ms") if isinstance(tele, dict) else None
    stap_total_ms = tele.get("stap_total_ms") if isinstance(tele, dict) else None
    try:
        baseline_ms_f = float(baseline_ms) if baseline_ms is not None else None
    except (TypeError, ValueError):
        baseline_ms_f = None
    try:
        stap_total_ms_f = float(stap_total_ms) if stap_total_ms is not None else None
    except (TypeError, ValueError):
        stap_total_ms_f = None
    return baseline_ms_f, stap_total_ms_f


def _parse_int_csv(spec: str) -> List[int]:
    return [int(float(s.strip())) for s in spec.replace(";", ",").split(",") if s.strip()]


def _parse_float_csv(spec: str) -> List[float]:
    return [float(s.strip()) for s in spec.replace(";", ",").split(",") if s.strip()]


def _parse_csv(spec: str) -> List[str]:
    return [s.strip() for s in spec.replace(";", ",").split(",") if s.strip()]


def _as_path(template_or_path: str) -> Path:
    p = Path(template_or_path)
    return p if p.is_absolute() else (REPO / p)


def _looks_like_replay_source_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if any(path.glob("angle_*")):
        return True
    if any(path.glob("ens*_angle_*")):
        return True
    return False


def _resolve_source_dir(regime: str, seed: int, template_spec: str) -> Path:
    templates = _parse_csv(template_spec)
    for template in templates:
        candidate = _as_path(template.format(seed=seed))
        if _looks_like_replay_source_dir(candidate):
            return candidate
    raise FileNotFoundError(
        f"[{regime} seed={seed}] no source dir found from templates: {templates}"
    )


def _resolve_injection_meta_dir(regime: str, seeds: Sequence[int], template_spec: str) -> Path | None:
    """
    Select a *single* directory to source overlay injection params from.

    This enforces the fixed-profile discipline: the injection configuration is
    frozen per regime and reused across seeds/windows. We choose the first seed
    whose resolved source directory contains at least one pw_* bundle meta.
    """
    for seed in seeds:
        try:
            source_dir = _resolve_source_dir(regime, int(seed), template_spec)
        except FileNotFoundError:
            continue
        if _find_best_source_bundle_meta(source_dir) is not None:
            return source_dir
    return None


def _bundle_map_by_window(run_dir: Path, window_length: int) -> Dict[int, Path]:
    offset_to_candidates: Dict[int, List[Path]] = {}
    for bundle in sorted(run_dir.glob("pw_*")):
        meta_path = bundle / "meta.json"
        if not meta_path.exists():
            continue
        meta = _load_json(meta_path)
        tw = meta.get("time_window") if isinstance(meta.get("time_window"), dict) else {}
        offset = int(tw.get("offset", 0))
        length = tw.get("length", None)
        if length is not None and int(length) != int(window_length):
            continue
        offset_to_candidates.setdefault(offset, []).append(bundle)

    offset_to_bundle: Dict[int, Path] = {}
    for offset, candidates in offset_to_candidates.items():
        chosen = max(candidates, key=lambda p: p.stat().st_mtime)
        offset_to_bundle[offset] = chosen
    return offset_to_bundle


def _run_replay_generation(
    *,
    python_exe: str,
    source_dir: Path,
    out_dir: Path,
    profile: str | None,
    baseline: str,
    stap_disable: bool,
    inject_args: Sequence[str],
    conditional: bool,
    window_length: int,
    offsets: Sequence[int],
    stap_device: str,
    stap_detector_variant: str = "msd_ratio",
    stap_whiten_gamma: float = 1.0,
    hybrid_rescue_rule: str = "guard_frac_v1",
    stap_cov_trim_q: float = 0.0,
    diag_load: float | None = None,
    cov_estimator: str | None = None,
    huber_c: float | None = None,
    mvdr_auto_kappa: float | None = None,
    constraint_ridge: float | None = None,
    synth_amp_jitter: float | None,
    synth_phase_jitter: float | None,
    synth_noise_level: float | None,
    synth_shift_max_px: int | None,
    reg_enable: bool,
    mcsvd_reg_enable: bool,
    mcsvd_energy_frac: float,
    mcsvd_baseline_support: str,
    rpca_lambda: float | None,
    rpca_max_iters: int,
    hosvd_spatial_downsample: int,
    hosvd_energy_fracs: str,
) -> None:
    cmd: List[str] = [
        python_exe,
        str(REPO / "scripts" / "replay_stap_from_run.py"),
        "--src",
        str(source_dir),
        "--out",
        str(out_dir),
        "--stap-profile",
        "clinical",
    ]
    if profile:
        cmd += ["--profile", profile]
    cmd += [
        "--baseline",
        baseline,
        "--stap-device",
        stap_device,
        "--stap-detector-variant",
        str(stap_detector_variant),
        "--stap-whiten-gamma",
        str(float(stap_whiten_gamma)),
        "--hybrid-rescue-rule",
        str(hybrid_rescue_rule),
        "--stap-cov-trim-q",
        str(float(stap_cov_trim_q or 0.0)),
        # Disable per-window debug tile capture so batched CUDA fast paths remain eligible.
        # Debug capture is useful for interactive audits but makes matrix mode prohibitively slow.
        "--stap-debug-samples",
        "0",
        "--score-mode",
        "pd",
        "--flow-mask-mode",
        "default",
        "--time-window-length",
        str(window_length),
    ]
    if any(v is not None for v in (diag_load, cov_estimator, huber_c, mvdr_auto_kappa, constraint_ridge)):
        cmd += ["--allow-custom-stap-hyperparams"]
    if diag_load is not None:
        cmd += ["--diag-load", str(float(diag_load))]
    if cov_estimator is not None:
        cmd += ["--cov-estimator", str(cov_estimator)]
    if huber_c is not None:
        cmd += ["--huber-c", str(float(huber_c))]
    if mvdr_auto_kappa is not None:
        cmd += ["--mvdr-auto-kappa", str(float(mvdr_auto_kappa))]
    if constraint_ridge is not None:
        cmd += ["--constraint-ridge", str(float(constraint_ridge))]
    if synth_amp_jitter is not None:
        cmd += ["--synth-amp-jitter", str(float(synth_amp_jitter))]
    if synth_phase_jitter is not None:
        cmd += ["--synth-phase-jitter", str(float(synth_phase_jitter))]
    if synth_noise_level is not None:
        cmd += ["--synth-noise-level", str(float(synth_noise_level))]
    if synth_shift_max_px is not None:
        cmd += ["--synth-shift-max-px", str(int(synth_shift_max_px))]
    for off in offsets:
        cmd += ["--time-window-offset", str(int(off))]

    if baseline == "mc_svd":
        support_norm = str(mcsvd_baseline_support or "full").strip().lower()
        if support_norm not in {"window", "full"}:
            support_norm = "full"
        # Match the frozen Brain-* MC--SVD baseline used throughout the
        # methodology: motion-compensated SVD with a tune-once energy-fraction
        # cutoff. For fairness, we apply the same registration +
        # energy-fraction to both the MC--SVD baseline maps and the downstream
        # STAP runs that build on the MC--SVD residual.
        cmd += [
            "--baseline-support",
            support_norm,
            "--svd-profile",
            "literature",
            "--svd-energy-frac",
            str(float(mcsvd_energy_frac)),
        ]
        if mcsvd_reg_enable:
            cmd += [
                "--reg-enable",
                "--reg-method",
                "phasecorr",
                "--reg-subpixel",
                "4",
                "--reg-reference",
                "median",
            ]
        else:
            cmd += ["--reg-disable"]
    elif baseline == "hosvd":
        cmd += [
            "--hosvd-spatial-downsample",
            str(int(hosvd_spatial_downsample)),
            "--hosvd-energy-fracs",
            str(hosvd_energy_fracs),
        ]
        if reg_enable:
            cmd += [
                "--reg-enable",
                "--reg-method",
                "phasecorr",
                "--reg-subpixel",
                "4",
                "--reg-reference",
                "median",
            ]
    elif baseline == "rpca":
        cmd += ["--rpca-enable"]
        if rpca_lambda is not None:
            cmd += ["--rpca-lambda", str(float(rpca_lambda))]
        cmd += ["--rpca-max-iters", str(int(rpca_max_iters))]
        if reg_enable:
            cmd += [
                "--reg-enable",
                "--reg-method",
                "phasecorr",
                "--reg-subpixel",
                "4",
                "--reg-reference",
                "median",
            ]
    elif baseline == "svd_similarity":
        cmd += ["--svd-rank", "32"]
        if reg_enable:
            cmd += [
                "--reg-enable",
                "--reg-method",
                "phasecorr",
                "--reg-subpixel",
                "4",
                "--reg-reference",
                "median",
            ]
    elif baseline == "local_svd":
        cmd += ["--svd-energy-frac", str(float(mcsvd_energy_frac))]
        if reg_enable:
            cmd += [
                "--reg-enable",
                "--reg-method",
                "phasecorr",
                "--reg-subpixel",
                "4",
                "--reg-reference",
                "median",
            ]
    else:
        raise ValueError(f"Unsupported baseline for generation: {baseline}")

    if stap_disable:
        cmd += ["--stap-disable"]

    cmd += list(inject_args)

    cmd.append("--stap-conditional-enable" if conditional else "--stap-conditional-disable")

    env = os.environ.copy()
    # Ensure repo modules (sim/, pipeline/, etc.) are importable in the replay subprocess.
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
    env.setdefault("STAP_FAST_PATH", "1")
    env.setdefault("STAP_FAST_PD_ONLY", "1")
    env.setdefault("STAP_TILING_UNFOLD", "1")
    env.setdefault("STAP_MAX_SNAPSHOTS", "64")
    env.setdefault("STAP_SNAPSHOT_STRIDE", "4")

    print("[generate]", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)


@dataclass(frozen=True)
class MethodSpec:
    method: str
    role: str
    bundle_dir: Path


@dataclass(frozen=True)
class MatrixMethodSpec:
    key: str
    method: str
    role: str
    run_key: str
    score_kind: str


@dataclass(frozen=True)
class RunConfig:
    run_key: str
    baseline: str
    conditional: bool
    stap_disable: bool
    detector_variant: str = "msd_ratio"
    stap_cov_trim_q: float = 0.0


RUN_CONFIGS: Dict[str, RunConfig] = {
    "mcsvd_full": RunConfig("mcsvd_full", baseline="mc_svd", conditional=False, stap_disable=False),
    "mcsvd_cond": RunConfig("mcsvd_cond", baseline="mc_svd", conditional=True, stap_disable=False),
    "rpca_full": RunConfig("rpca_full", baseline="rpca", conditional=False, stap_disable=True),
    "hosvd_full": RunConfig("hosvd_full", baseline="hosvd", conditional=False, stap_disable=True),
    "svd_similarity_full": RunConfig("svd_similarity_full", baseline="svd_similarity", conditional=False, stap_disable=True),
    "local_svd_full": RunConfig("local_svd_full", baseline="local_svd", conditional=False, stap_disable=True),
    # Detector-swap ablations: compute STAP score on top of alternative baseline residuals.
    "rpca_stap_full": RunConfig("rpca_stap_full", baseline="rpca", conditional=False, stap_disable=False),
    "hosvd_stap_full": RunConfig("hosvd_stap_full", baseline="hosvd", conditional=False, stap_disable=False),
    # Detector ablations (same MC--SVD residual; different score statistic).
    "mcsvd_det_whitened_power": RunConfig(
        "mcsvd_det_whitened_power",
        baseline="mc_svd",
        conditional=False,
        stap_disable=False,
        detector_variant="whitened_power",
    ),
    "mcsvd_det_unwhitened_ratio": RunConfig(
        "mcsvd_det_unwhitened_ratio",
        baseline="mc_svd",
        conditional=False,
        stap_disable=False,
        detector_variant="unwhitened_ratio",
    ),
    "mcsvd_covtrim_q05": RunConfig(
        "mcsvd_covtrim_q05",
        baseline="mc_svd",
        conditional=False,
        stap_disable=False,
        detector_variant="msd_ratio",
        stap_cov_trim_q=0.05,
    ),
}


MATRIX_METHODS: Dict[str, MatrixMethodSpec] = {
    "mcsvd": MatrixMethodSpec(
        key="mcsvd",
        method="MC-SVD",
        role="baseline",
        run_key="mcsvd_full",
        score_kind="base",
    ),
    "svd_similarity": MatrixMethodSpec(
        key="svd_similarity",
        method="Adaptive SVD (similarity cutoff)",
        role="baseline",
        run_key="svd_similarity_full",
        score_kind="base",
    ),
    "local_svd": MatrixMethodSpec(
        key="local_svd",
        method="Local SVD (block-wise)",
        role="baseline",
        run_key="local_svd_full",
        score_kind="base",
    ),
    "rpca": MatrixMethodSpec(
        key="rpca",
        method="RPCA",
        role="baseline",
        run_key="rpca_full",
        score_kind="base",
    ),
    "hosvd": MatrixMethodSpec(
        key="hosvd",
        method="HOSVD",
        role="baseline",
        run_key="hosvd_full",
        score_kind="base",
    ),
    "rpca_pair": MatrixMethodSpec(
        key="rpca_pair",
        method="RPCA+PD (paired)",
        role="baseline",
        run_key="rpca_stap_full",
        score_kind="base",
    ),
    "rpca_stap": MatrixMethodSpec(
        key="rpca_stap",
        method="RPCA+STAP (paired; pre-KA)",
        role="stap",
        run_key="rpca_stap_full",
        score_kind="stap",
    ),
    "hosvd_pair": MatrixMethodSpec(
        key="hosvd_pair",
        method="HOSVD+PD (paired)",
        role="baseline",
        run_key="hosvd_stap_full",
        score_kind="base",
    ),
    "hosvd_stap": MatrixMethodSpec(
        key="hosvd_stap",
        method="HOSVD+STAP (paired; pre-KA)",
        role="stap",
        run_key="hosvd_stap_full",
        score_kind="stap",
    ),
    "stap_full": MatrixMethodSpec(
        key="stap_full",
        method="STAP (MC-SVD+STAP, full)",
        role="stap",
        run_key="mcsvd_full",
        score_kind="stap",
    ),
    "stap_covtrim_q05": MatrixMethodSpec(
        key="stap_covtrim_q05",
        method="STAP (MC-SVD+STAP, cov-trim q=0.05)",
        role="stap",
        run_key="mcsvd_covtrim_q05",
        score_kind="stap",
    ),
    "stap_det_whitened_power": MatrixMethodSpec(
        key="stap_det_whitened_power",
        method="Detector ablation: whitened power (no band)",
        role="stap",
        run_key="mcsvd_det_whitened_power",
        score_kind="stap",
    ),
    "stap_det_unwhitened_ratio": MatrixMethodSpec(
        key="stap_det_unwhitened_ratio",
        method="Detector ablation: unwhitened ratio (no whitening)",
        role="stap",
        run_key="mcsvd_det_unwhitened_ratio",
        score_kind="stap",
    ),
    "stap_cond": MatrixMethodSpec(
        key="stap_cond",
        method="STAP (MC-SVD+STAP, conditional)",
        role="stap",
        run_key="mcsvd_cond",
        score_kind="stap",
    ),
}


def _scenario_specs_default() -> Dict[str, List[MethodSpec]]:
    open_stap_full = _find_single_bundle(
        REPO / "runs" / "pilot" / "r4_kwave_seed1_svdlit_stap_pd_clinical_fastfull"
    )
    open_rpca = _find_single_bundle(
        REPO / "runs" / "pilot" / "r4_kwave_seed1_rpca_stap_pd_clinical_T64"
    )
    open_hosvd = _find_single_bundle(
        REPO / "runs" / "pilot" / "r4_kwave_seed1_hosvd_stap_pd_clinical_T64"
    )

    skullor_stap_full = _find_single_bundle(
        REPO
        / "runs"
        / "pilot"
        / "r4c_kwave_hab_v3_skull_seed2_v2_latency_mcsvd_stap_pd_clinical"
    )
    skullor_stap_cond = _find_single_bundle(
        REPO
        / "runs"
        / "pilot"
        / "r4c_kwave_hab_v3_skull_seed2_v2_latency_mcsvd_stap_pd_clinical_gpu_fast_pdonly_batched_flowgate"
    )
    skullor_rpca = _find_single_bundle(
        REPO
        / "runs"
        / "pilot"
        / "r4c_kwave_hab_v3_skull_seed2_v2_latency_rpca_stap_pd_clinical_fastpdonly"
    )
    skullor_hosvd = _find_single_bundle(
        REPO
        / "runs"
        / "pilot"
        / "r4c_kwave_hab_v3_skull_seed2_v2_latency_hosvd_stap_pd_clinical_fastpdonly"
    )

    return {
        "open_seed1": [
            MethodSpec("MC-SVD", "baseline", open_stap_full),
            MethodSpec("RPCA", "baseline", open_rpca),
            MethodSpec("HOSVD", "baseline", open_hosvd),
            MethodSpec("STAP (MC-SVD+STAP, full)", "stap", open_stap_full),
        ],
        "skullor_seed2": [
            MethodSpec("MC-SVD", "baseline", skullor_stap_full),
            MethodSpec("RPCA", "baseline", skullor_rpca),
            MethodSpec("HOSVD", "baseline", skullor_hosvd),
            MethodSpec("STAP (MC-SVD+STAP, full)", "stap", skullor_stap_full),
            MethodSpec("STAP (MC-SVD+STAP, conditional)", "stap", skullor_stap_cond),
        ],
    }


def _evaluate_method_row(
    *,
    scenario: str,
    regime: str | None,
    seed: int | None,
    window_offset: int | None,
    window_length: int | None,
    method: str,
    role: str,
    bundle_dir: Path,
    score_kind: str,
    fprs: Sequence[float],
    legacy_default: str,
    eval_score: str,
) -> Dict[str, Any]:
    meta = _load_json(bundle_dir / "meta.json")
    baseline_ms, stap_total_ms = _safe_get_latency_ms(meta)
    total_ms = None
    if baseline_ms is not None and role == "baseline":
        total_ms = baseline_ms
    if baseline_ms is not None and stap_total_ms is not None and role == "stap":
        total_ms = baseline_ms + stap_total_ms

    score_name = None
    if eval_score == "pd":
        score_base, base_desc, roc_conv = _load_pd_score(
            bundle_dir, which="base", legacy_default=legacy_default
        )
        score_stap, stap_desc, _ = _load_pd_score(
            bundle_dir, which="stap", legacy_default=legacy_default
        )
    elif eval_score == "vnext":
        score_base, base_desc, score_name = _load_vnext_score(bundle_dir, which="base")
        score_stap, stap_desc, score_name_stap = _load_vnext_score(bundle_dir, which="stap")
        # Prefer the STAP score_name when it exists.
        score_name = score_name_stap or score_name
        roc_conv = None

        # `score_stap.npy` is only well-defined as a full-image detector score
        # when STAP ran on all tiles. Conditional STAP can skip tiles and export
        # a score map that is intentionally clamped on skipped tiles, which can
        # yield misleading ROC points if treated as a full detector. Some bundle
        # writers always persist the conditional mask for auditing even when
        # conditional execution is disabled; therefore we treat conditional STAP
        # as "active" only when the run actually skipped tiles.
        if score_kind == "stap":
            tele = (
                meta.get("stap_fallback_telemetry")
                if isinstance(meta.get("stap_fallback_telemetry"), dict)
                else {}
            )
            skipped = 0
            if isinstance(tele, dict):
                try:
                    skipped = int(tele.get("stap_tiles_skipped_flow0") or 0)
                except Exception:
                    skipped = 0
            if skipped > 0:
                raise RuntimeError(
                    f"[{scenario}] conditional execution skipped {skipped} tiles in {bundle_dir}; "
                    "do not evaluate vNext scores on conditional-STAP bundles. "
                    "Re-run with --stap-conditional-disable (full STAP) or evaluate PD scores instead."
                )
    else:
        raise ValueError(f"Unsupported eval_score={eval_score!r}")
    if score_kind == "base":
        score = score_base
        score_desc = f"base({base_desc})"
    else:
        score = score_stap
        score_desc = f"stap({stap_desc})"

    mask_flow, mask_bg = _load_mask(bundle_dir)
    n_pos = int(mask_flow.sum())
    n_neg = int(mask_bg.sum())
    fpr_floor = 1.0 / max(1, n_neg)

    pos = score[mask_flow]
    neg = score[mask_bg]

    out: Dict[str, Any] = {
        "scenario": scenario,
        "regime": regime,
        "seed": seed,
        "window_offset": window_offset,
        "window_length": window_length,
        "method": method,
        "role": role,
        "bundle_dir": str(bundle_dir),
        "eval_score": eval_score,
        "score": score_desc,
        "pd_mode_roc_convention": roc_conv,
        "score_name": score_name,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "fpr_floor": fpr_floor,
        "baseline_ms": baseline_ms,
        "stap_total_ms": stap_total_ms if role == "stap" else None,
        "total_ms": total_ms,
        "prf_hz": meta.get("prf_hz"),
        "Lt": meta.get("Lt"),
        "tile_hw": meta.get("tile_hw"),
        "tile_stride": meta.get("tile_stride"),
        "time_window": meta.get("time_window"),
    }

    for fpr in fprs:
        thr, tpr, fpr_emp = _tpr_at_fpr(pos, neg, float(fpr))
        out[f"thr@{fpr:g}"] = thr
        out[f"tpr@{fpr:g}"] = tpr
        out[f"fpr@{fpr:g}"] = fpr_emp
    return out


def _run_static_mode(args: argparse.Namespace) -> List[Dict[str, Any]]:
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    scenario_specs = _scenario_specs_default()
    rows: List[Dict[str, Any]] = []

    for scenario in scenarios:
        if scenario not in scenario_specs:
            raise SystemExit(f"Unknown scenario {scenario!r}. Known: {sorted(scenario_specs.keys())}")
        specs = scenario_specs[scenario]

        ref_flow = ref_bg = None
        for spec in specs:
            flow, bg = _load_mask(spec.bundle_dir)
            if ref_flow is None:
                ref_flow, ref_bg = flow, bg
            else:
                if flow.shape != ref_flow.shape or bg.shape != ref_bg.shape:
                    raise RuntimeError(f"[{scenario}] Mask shape mismatch across bundles")
                if np.count_nonzero(flow != ref_flow) or np.count_nonzero(bg != ref_bg):
                    raise RuntimeError(
                        f"[{scenario}] mask_flow/mask_bg differ across bundles; not a 1:1 comparison"
                    )

        for spec in specs:
            score_kind = "base" if spec.role == "baseline" else "stap"
            rows.append(
                _evaluate_method_row(
                    scenario=scenario,
                    regime=None,
                    seed=None,
                    window_offset=None,
                    window_length=None,
                    method=spec.method,
                    role=spec.role,
                    bundle_dir=spec.bundle_dir,
	                    score_kind=score_kind,
	                    fprs=args.fprs,
	                    legacy_default=args.legacy_default,
	                    eval_score=args.eval_score,
	                )
	            )
    return rows


def _run_matrix_mode(args: argparse.Namespace) -> List[Dict[str, Any]]:
    regimes = _parse_csv(args.matrix_regimes)
    if not regimes:
        raise ValueError("matrix mode requires at least one regime")

    methods_requested = _parse_csv(args.methods)
    if not methods_requested:
        raise ValueError("matrix mode requires at least one method key")
    method_specs: List[MatrixMethodSpec] = []
    for key in methods_requested:
        if key not in MATRIX_METHODS:
            raise ValueError(f"Unknown method key '{key}'. Known: {sorted(MATRIX_METHODS.keys())}")
        method_specs.append(MATRIX_METHODS[key])

    window_length = int(args.window_length)
    offsets = sorted(set(_parse_int_csv(args.window_offsets)))
    if not offsets:
        raise ValueError("matrix mode requires at least one window offset")

    seeds_by_regime: Dict[str, List[int]] = {}
    for regime in regimes:
        if regime == "open":
            seeds_by_regime[regime] = _parse_int_csv(args.matrix_seeds_open)
        elif regime == "aliascontract":
            seeds_by_regime[regime] = _parse_int_csv(args.matrix_seeds_aliascontract)
        elif regime == "skullor":
            seeds_by_regime[regime] = _parse_int_csv(args.matrix_seeds_skullor)
        else:
            raise ValueError(f"Unsupported regime '{regime}' in matrix mode")

    template_by_regime = {
        "open": args.src_template_open,
        "aliascontract": args.src_template_aliascontract,
        "skullor": args.src_template_skullor,
    }
    profile_by_regime = {
        "open": "Brain-OpenSkull",
        "aliascontract": "Brain-AliasContract",
        "skullor": "Brain-SkullOR",
    }

    inject_meta_dir_by_regime: Dict[str, Path | None] = {}
    if str(args.matrix_inject_policy).strip().lower() == "frozen_per_regime":
        for regime in regimes:
            inject_meta_dir_by_regime[regime] = _resolve_injection_meta_dir(
                regime, seeds_by_regime.get(regime, []), template_by_regime[regime]
            )

    needed_run_keys = sorted({m.run_key for m in method_specs})
    run_root = args.generated_root
    run_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for regime in regimes:
        seeds = seeds_by_regime[regime]
        for seed in seeds:
            source_dir = _resolve_source_dir(regime, seed, template_by_regime[regime])
            inject_meta_dir = inject_meta_dir_by_regime.get(regime) or source_dir
            inject_args = _injection_cli_args_from_source(
                inject_meta_dir,
                clutter_mode_override=str(args.matrix_clutter_mode),
                clutter_rank_override=int(args.matrix_clutter_rank),
            )
            if args.autogen_missing and not inject_args:
                print(
                    f"[warn] [{regime} seed={seed}] no overlay injection params found under {inject_meta_dir}; "
                    "autogenerated bundles may be spectrally degenerate (STAP ~ scalar). "
                    "Prefer an r4c Brain-* pilot dir or regenerate pilots with overlays.",
                    flush=True,
                )
            run_bundles: Dict[str, Dict[int, Path]] = {}

            for run_key in needed_run_keys:
                cfg = RUN_CONFIGS[run_key]
                out_dir = run_root / f"{regime}_seed{seed}_{run_key}"
                if args.regen_runs and out_dir.exists():
                    shutil.rmtree(out_dir)
                bundles = _bundle_map_by_window(out_dir, window_length) if out_dir.exists() else {}
                missing = [off for off in offsets if off not in bundles]
                if missing and args.autogen_missing:
                    _run_replay_generation(
                        python_exe=args.python_exe,
                        source_dir=source_dir,
                        out_dir=out_dir,
                        profile=profile_by_regime.get(regime) if args.matrix_use_profile else None,
                        baseline=cfg.baseline,
                        stap_disable=bool(getattr(cfg, "stap_disable", False)),
                        inject_args=inject_args,
                        conditional=cfg.conditional,
                        window_length=window_length,
                        offsets=missing,
                        stap_device=args.stap_device,
                        stap_detector_variant=str(getattr(cfg, "detector_variant", "msd_ratio")),
                        stap_whiten_gamma=float(getattr(cfg, "whiten_gamma", 1.0) or 1.0),
                        stap_cov_trim_q=float(getattr(cfg, "stap_cov_trim_q", 0.0) or 0.0),
                        synth_amp_jitter=args.matrix_synth_amp_jitter,
                        synth_phase_jitter=args.matrix_synth_phase_jitter,
                        synth_noise_level=args.matrix_synth_noise_level,
                        synth_shift_max_px=args.matrix_synth_shift_max_px,
                        reg_enable=bool(args.matrix_reg_enable),
                        mcsvd_reg_enable=bool(args.matrix_mcsvd_reg_enable),
                        mcsvd_energy_frac=float(args.matrix_mcsvd_energy_frac),
                        mcsvd_baseline_support=str(args.matrix_mcsvd_baseline_support),
                        rpca_lambda=args.matrix_rpca_lambda,
                        rpca_max_iters=int(args.matrix_rpca_max_iters),
                        hosvd_spatial_downsample=int(args.matrix_hosvd_spatial_downsample),
                        hosvd_energy_fracs=str(args.matrix_hosvd_energy_fracs),
                    )
                    bundles = _bundle_map_by_window(out_dir, window_length)
                    missing = [off for off in offsets if off not in bundles]

                if missing and not args.allow_incomplete:
                    raise RuntimeError(
                        f"[{regime} seed={seed} run={run_key}] missing offsets {missing} in {out_dir}. "
                        "Re-run with --autogen-missing."
                    )
                run_bundles[run_key] = bundles

            for offset in offsets:
                selected: List[Tuple[MatrixMethodSpec, Path]] = []
                for ms in method_specs:
                    bundle = run_bundles.get(ms.run_key, {}).get(offset)
                    if bundle is not None:
                        selected.append((ms, bundle))
                if not selected:
                    continue
                if len(selected) != len(method_specs) and not args.allow_incomplete:
                    raise RuntimeError(
                        f"[{regime} seed={seed} offset={offset}] incomplete methods for 1:1 comparison"
                    )

                ref_flow = ref_bg = None
                for _, bundle in selected:
                    flow, bg = _load_mask(bundle)
                    if ref_flow is None:
                        ref_flow, ref_bg = flow, bg
                    else:
                        if flow.shape != ref_flow.shape or bg.shape != ref_bg.shape:
                            raise RuntimeError(
                                f"[{regime} seed={seed} offset={offset}] mask shape mismatch"
                            )
                        if np.count_nonzero(flow != ref_flow) or np.count_nonzero(bg != ref_bg):
                            raise RuntimeError(
                                f"[{regime} seed={seed} offset={offset}] masks differ across methods"
                            )

                scenario = f"{regime}_seed{seed}_off{offset}"
                for ms, bundle in selected:
                    rows.append(
                        _evaluate_method_row(
                            scenario=scenario,
                            regime=regime,
                            seed=seed,
                            window_offset=offset,
                            window_length=window_length,
                            method=ms.method,
                            role=ms.role,
                            bundle_dir=bundle,
	                            score_kind=ms.score_kind,
	                            fprs=args.fprs,
	                            legacy_default=args.legacy_default,
	                            eval_score=args.eval_score,
	                        )
	                    )
    return rows


def _write_outputs(
    rows: List[Dict[str, Any]],
    *,
    out_csv: Path,
    out_json: Path,
    fprs: Sequence[float],
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2, sort_keys=True))

    key_set = set()
    for row in rows:
        key_set.update(row.keys())
    preferred = [
        "scenario",
        "regime",
        "seed",
        "window_offset",
        "window_length",
        "method",
        "role",
        "bundle_dir",
        "eval_score",
        "score",
        "pd_mode_roc_convention",
        "score_name",
        "n_pos",
        "n_neg",
        "fpr_floor",
        "baseline_ms",
        "stap_total_ms",
        "total_ms",
        "prf_hz",
        "Lt",
        "tile_hw",
        "tile_stride",
        "time_window",
    ]
    fpr_cols: List[str] = []
    for fpr in fprs:
        fpr_cols += [f"thr@{fpr:g}", f"tpr@{fpr:g}"]
    other = sorted(k for k in key_set if k not in preferred + fpr_cols)
    header = preferred + fpr_cols + other

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fair filter comparison across acceptance bundles.")
    ap.add_argument(
        "--eval-score",
        type=str,
        default="pd",
        choices=["pd", "vnext"],
        help=(
            "Which score to evaluate for ROC points. "
            "'pd' uses score_pd_*.npy (or legacy pd_*.npy + meta convention); "
            "'vnext' uses score_base.npy / score_stap.npy (right-tail) when present."
        ),
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="static",
        choices=["static", "matrix"],
        help="Comparison mode: static (existing bundles) or matrix (seed/window + optional autogen).",
    )
    ap.add_argument(
        "--fprs",
        type=float,
        nargs="+",
        default=[1e-4, 3e-4, 1e-3],
        help="FPR targets (default: 1e-4 3e-4 1e-3).",
    )
    ap.add_argument(
        "--legacy-default",
        type=str,
        default="lower",
        choices=["lower", "upper"],
        help=(
            "When score_pd_*.npy and meta.pd_mode are missing, choose whether PD ROC uses "
            "lower-tail-on-pd (score=-pd) or upper-tail-on-pd (score=pd). Default: lower."
        ),
    )
    ap.add_argument("--out-csv", type=Path, required=True, help="Output CSV path.")
    ap.add_argument("--out-json", type=Path, required=True, help="Output JSON path.")

    ap.add_argument(
        "--scenarios",
        type=str,
        default="open_seed1,skullor_seed2",
        help="Static mode only: comma-separated scenario names.",
    )

    ap.add_argument(
        "--matrix-regimes",
        type=str,
        default="open,aliascontract,skullor",
        help="Matrix mode: comma-separated regimes from {open,aliascontract,skullor}.",
    )
    ap.add_argument(
        "--matrix-inject-policy",
        type=str,
        default="frozen_per_regime",
        choices=["frozen_per_regime", "per_seed"],
        help=(
            "Matrix mode: how to choose overlay injection params. "
            "'frozen_per_regime' (default) sources injection flags from a single "
            "bundle meta per regime and reuses them across seeds/windows; "
            "'per_seed' sources injection flags from each seed's source bundle meta."
        ),
    )
    ap.add_argument(
        "--matrix-seeds-open",
        type=str,
        default="1",
        help="Matrix mode: comma-separated open-regime seeds.",
    )
    ap.add_argument(
        "--matrix-seeds-aliascontract",
        type=str,
        default="2",
        help="Matrix mode: comma-separated aliascontract-regime seeds.",
    )
    ap.add_argument(
        "--matrix-seeds-skullor",
        type=str,
        default="2",
        help="Matrix mode: comma-separated skullor-regime seeds.",
    )
    ap.add_argument(
        "--window-length",
        type=int,
        default=64,
        help="Matrix mode: replay window length.",
    )
    ap.add_argument(
        "--window-offsets",
        type=str,
        default="0,64,128,192,256",
        help="Matrix mode: comma-separated replay window offsets.",
    )
    ap.add_argument(
        "--matrix-synth-amp-jitter",
        type=float,
        default=0.0,
        help=(
            "Matrix mode: replay --synth-amp-jitter value. "
            "Default: 0.0 (matches Brain-* fixed-profile discipline)."
        ),
    )
    ap.add_argument(
        "--matrix-synth-phase-jitter",
        type=float,
        default=0.0,
        help=(
            "Matrix mode: replay --synth-phase-jitter value. "
            "Default: 0.0 (matches Brain-* fixed-profile discipline)."
        ),
    )
    ap.add_argument(
        "--matrix-synth-noise-level",
        type=float,
        default=0.0,
        help=(
            "Matrix mode: replay --synth-noise-level value. "
            "Default: 0.0 (matches Brain-* fixed-profile discipline)."
        ),
    )
    ap.add_argument(
        "--matrix-synth-shift-max-px",
        type=int,
        default=0,
        help=(
            "Matrix mode: replay --synth-shift-max-px value. "
            "Default: 0 (matches Brain-* fixed-profile discipline)."
        ),
    )
    ap.add_argument(
        "--methods",
        type=str,
        default="mcsvd,rpca,hosvd,stap_full",
        help=(
            "Matrix mode methods: subset of mcsvd,svd_similarity,local_svd,rpca,hosvd,"
            "rpca_pair,rpca_stap,hosvd_pair,hosvd_stap,stap_full,"
            "stap_covtrim_q05,stap_det_whitened_power,stap_det_unwhitened_ratio,stap_cond."
        ),
    )
    ap.add_argument(
        "--matrix-reg-enable",
        dest="matrix_reg_enable",
        action="store_true",
        help="Matrix mode: enable registration for non-SVD baselines (RPCA/HOSVD).",
    )
    ap.add_argument(
        "--matrix-reg-disable",
        dest="matrix_reg_enable",
        action="store_false",
        help="Matrix mode: disable registration for non-SVD baselines (RPCA/HOSVD).",
    )
    ap.set_defaults(matrix_reg_enable=True)
    ap.add_argument(
        "--matrix-mcsvd-reg-enable",
        dest="matrix_mcsvd_reg_enable",
        action="store_true",
        help="Matrix mode: enable registration for MC-SVD baselines (MC-SVD+STAP).",
    )
    ap.add_argument(
        "--matrix-mcsvd-reg-disable",
        dest="matrix_mcsvd_reg_enable",
        action="store_false",
        help="Matrix mode: disable registration for MC-SVD baselines.",
    )
    ap.set_defaults(matrix_mcsvd_reg_enable=True)
    ap.add_argument(
        "--matrix-mcsvd-energy-frac",
        type=float,
        default=0.90,
        help="Matrix mode: MC-SVD energy fraction cutoff (default: 0.90).",
    )
    ap.add_argument(
        "--matrix-mcsvd-baseline-support",
        type=str,
        default="full",
        choices=["window", "full"],
        help=(
            "Matrix mode: slow-time support used to fit MC--SVD. "
            "'full' fits the clutter subspace on the full slow-time stack and applies it to each window "
            "(stronger, more conservative baseline). "
            "'window' fits only on the requested window (more streaming-like)."
        ),
    )
    ap.add_argument(
        "--matrix-use-profile",
        action="store_true",
        help=(
            "Matrix mode: pass --profile Brain-* to replay_stap_from_run.py. "
            "Note this may override explicit CLI knobs (e.g. registration defaults)."
        ),
    )
    ap.add_argument(
        "--matrix-rpca-lambda",
        type=float,
        default=None,
        help="Matrix mode: RPCA lambda override (default: RPCA internal 1/sqrt(max(m,n))).",
    )
    ap.add_argument(
        "--matrix-rpca-max-iters",
        type=int,
        default=250,
        help="Matrix mode: RPCA max iters override (default: 250; implementation may cap internally).",
    )
    ap.add_argument(
        "--matrix-hosvd-spatial-downsample",
        type=int,
        default=2,
        help="Matrix mode: HOSVD spatial downsample factor (default: 2).",
    )
    ap.add_argument(
        "--matrix-hosvd-energy-fracs",
        type=str,
        default="0.99,0.99,0.99",
        help="Matrix mode: HOSVD energy fractions as 'fT,fH,fW' (default: 0.99,0.99,0.99).",
    )
    ap.add_argument(
        "--matrix-clutter-mode",
        type=str,
        default="lowrank",
        choices=["source", "fullrank", "lowrank"],
        help=(
            "Matrix mode: override temporal clutter injection model when clutter is enabled. "
            "'source' uses the source bundle meta (if present); "
            "'fullrank' injects independent per-pixel clutter; "
            "'lowrank' injects a small number of global temporal modes (recommended)."
        ),
    )
    ap.add_argument(
        "--matrix-clutter-rank",
        type=int,
        default=3,
        help="Matrix mode: temporal clutter rank when --matrix-clutter-mode=lowrank.",
    )
    ap.add_argument(
        "--generated-root",
        type=Path,
        default=REPO / "runs" / "pilot" / "fair_filter_matrix",
        help="Matrix mode: root directory for generated replay runs.",
    )
    ap.add_argument(
        "--autogen-missing",
        action="store_true",
        help="Matrix mode: generate missing replay runs/windows.",
    )
    ap.add_argument(
        "--regen-runs",
        action="store_true",
        help="Matrix mode: delete existing generated run dirs before replay.",
    )
    ap.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Matrix mode: keep partial rows if some methods/windows are missing.",
    )
    ap.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Python executable used to invoke replay_stap_from_run.py.",
    )
    ap.add_argument(
        "--stap-device",
        type=str,
        default="cpu",
        help="Matrix mode: replay --stap-device value (cuda or cpu).",
    )
    ap.add_argument(
        "--src-template-open",
        type=str,
        default=DEFAULT_SRC_TEMPLATE_OPEN,
        help="Matrix mode: comma-separated source templates for open regime.",
    )
    ap.add_argument(
        "--src-template-aliascontract",
        type=str,
        default=DEFAULT_SRC_TEMPLATE_ALIAS,
        help="Matrix mode: comma-separated source templates for aliascontract regime.",
    )
    ap.add_argument(
        "--src-template-skullor",
        type=str,
        default=DEFAULT_SRC_TEMPLATE_SKULL,
        help="Matrix mode: comma-separated source templates for skullor regime.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "static":
        rows = _run_static_mode(args)
    else:
        rows = _run_matrix_mode(args)
    _write_outputs(rows, out_csv=args.out_csv, out_json=args.out_json, fprs=args.fprs)
    print(f"[fair_filter_comparison] wrote {len(rows)} rows to {args.out_csv} and {args.out_json}")


if __name__ == "__main__":
    main()
