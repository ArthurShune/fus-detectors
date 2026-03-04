"""Replay telemetry and KA option shaping helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def build_guard_opts(args: argparse.Namespace) -> dict[str, float | bool]:
    guard_opts: dict[str, float | bool] = {
        "guard_target_med": float(args.bg_guard_target_med),
        "guard_target_low": float(args.bg_guard_target_low),
        "guard_percentile_low": float(args.bg_guard_percentile_low),
        "guard_tile_coverage_min": float(args.bg_guard_coverage_min),
        "guard_max_scale": float(args.bg_guard_max_scale),
        "bg_guard_enabled": bool(args.bg_guard_enabled),
        "bg_guard_target_p90": float(args.bg_guard_target_p90),
        "bg_guard_min_alpha": float(args.bg_guard_min_alpha),
        "bg_guard_metric": str(args.bg_guard_metric),
    }
    if args.alias_cap_enable:
        guard_opts.update(
            {
                "alias_cap_enable": True,
                "alias_cap_alias_thresh": float(args.alias_cap_alias_thresh),
                "alias_cap_band_med_thresh": float(args.alias_cap_band_med_thresh),
                "alias_cap_smin": float(args.alias_cap_smin),
                "alias_cap_c0": float(args.alias_cap_c0),
                "alias_cap_exp": float(args.alias_cap_exp),
            }
        )
    ka_gate_enabled = bool(args.ka_gate_enable or args.feasibility_mode == "updated")
    if ka_gate_enabled:
        gate_opts: dict[str, float | bool] = {
            "ka_gate_enable": True,
            "ka_gate_alias_rmin": float(args.ka_gate_alias_rmin),
            "ka_gate_flow_cov_min": float(args.ka_gate_flow_cov_min),
            "ka_gate_depth_min_frac": float(args.ka_gate_depth_min_frac),
            "ka_gate_depth_max_frac": float(args.ka_gate_depth_max_frac),
        }
        if args.ka_gate_pd_min is not None:
            gate_opts["ka_gate_pd_min"] = float(args.ka_gate_pd_min)
        if args.ka_gate_reg_psr_max is not None:
            gate_opts["ka_gate_reg_psr_max"] = float(args.ka_gate_reg_psr_max)
        guard_opts.update(gate_opts)
    return guard_opts


def build_ka_opts_extra(
    args: argparse.Namespace,
    *,
    guard_opts: dict[str, float | bool],
) -> dict[str, Any]:
    ka_opts_extra: dict[str, Any] = dict(guard_opts)
    if args.ka_beta_fixed is not None:
        ka_opts_extra["beta"] = float(args.ka_beta_fixed)
    if args.ka_score_model_json is not None:
        ka_opts_extra["score_model_json"] = str(args.ka_score_model_json)
    if args.ka_score_alpha is not None and args.ka_score_alpha > 0.0:
        ka_opts_extra["score_alpha"] = float(args.ka_score_alpha)
    if args.ka_score_contract_v2 or args.ka_score_contract_v2_force:
        ka_opts_extra["score_contract_v2"] = 1.0
        ka_opts_extra["score_contract_v2_mode"] = str(
            getattr(args, "ka_score_contract_v2_mode", "safety")
        )
        ka_opts_extra["score_contract_v2_proxy_source"] = str(
            getattr(args, "ka_contract_v2_proxy_source", "pd")
        )
        ka_opts_extra["score_contract_v2_alias_source"] = str(
            getattr(args, "ka_contract_v2_alias_source", "peak")
        )
    if args.ka_score_contract_v2_force:
        ka_opts_extra["score_contract_v2_force"] = 1.0
    return ka_opts_extra


def build_base_meta_extra(
    args: argparse.Namespace,
    *,
    src_root: Path,
) -> dict[str, Any]:
    return {
        "source": "replay",
        "orig_run": str(src_root),
        "profile": getattr(args, "profile", None),
        "aperture_phase_std": float(args.aperture_phase_std),
        "aperture_phase_corr_len": float(args.aperture_phase_corr_len),
        "clutter_beta": float(args.clutter_beta),
        "clutter_snr_db": float(args.clutter_snr_db),
        "clutter_mode": str(args.clutter_mode),
        "clutter_rank": int(args.clutter_rank),
        "clutter_depth_min_frac": float(args.clutter_depth_min_frac),
        "clutter_depth_max_frac": float(args.clutter_depth_max_frac),
        "flow_alias_hz": (float(args.flow_alias_hz) if args.flow_alias_hz is not None else None),
        "flow_alias_fraction": float(args.flow_alias_fraction),
        "flow_alias_depth_min_frac": args.flow_alias_depth_min_frac,
        "flow_alias_depth_max_frac": args.flow_alias_depth_max_frac,
        "flow_alias_jitter_hz": float(args.flow_alias_jitter_hz),
        "bg_alias_hz": float(args.bg_alias_hz) if args.bg_alias_hz is not None else None,
        "bg_alias_fraction": float(args.bg_alias_fraction),
        "bg_alias_depth_min_frac": args.bg_alias_depth_min_frac,
        "bg_alias_depth_max_frac": args.bg_alias_depth_max_frac,
        "bg_alias_jitter_hz": float(args.bg_alias_jitter_hz),
        "flow_doppler_min_hz": (
            float(args.flow_doppler_min_hz) if args.flow_doppler_min_hz is not None else None
        ),
        "flow_doppler_max_hz": (
            float(args.flow_doppler_max_hz) if args.flow_doppler_max_hz is not None else None
        ),
        "vibration_hz": float(args.vibration_hz) if args.vibration_hz is not None else None,
        "vibration_amp": float(args.vibration_amp),
        "vibration_depth_min_frac": float(args.vibration_depth_min_frac),
        "vibration_depth_decay_frac": float(args.vibration_depth_decay_frac),
    }


def compose_window_meta_extra(
    *,
    base_meta_extra: dict[str, Any],
    offset: int | None,
    length: int | None,
    total_frames_full: int,
) -> dict[str, Any]:
    meta_extra = dict(base_meta_extra)
    if length is not None and offset is not None:
        meta_extra["time_window"] = {
            "offset": int(offset),
            "length": int(length),
            "total_length": int(total_frames_full),
        }
    return meta_extra

