"""Replay profile and CLI preset helpers."""

from __future__ import annotations

import argparse
import os
from typing import List, Sequence, Tuple

from fus_detectors.defaults import (
    ADAPTIVE_GUARD_DEFAULTS,
    CLINICAL_REPLAY_DEFAULTS,
    PUBLIC_DETECTOR_DEFAULTS,
)


def parse_coords(coord_list: Sequence[str]) -> List[Tuple[int, int]]:
    coords: List[Tuple[int, int]] = []
    for entry in coord_list:
        clean = entry.replace(":", ",")
        parts = [p.strip() for p in clean.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError(f"Invalid coord '{entry}'")
        coords.append((int(parts[0]), int(parts[1])))
    return coords


def apply_brain_profile_defaults(args: argparse.Namespace) -> None:
    """Apply high-level Brain-* profile defaults."""
    profile = getattr(args, "profile", None)
    if profile is None:
        return

    if getattr(args, "baseline", "svd") == "svd":
        args.baseline = "mc_svd"
    args.reg_enable = True
    args.reg_method = "phasecorr"
    args.reg_subpixel = 4
    args.reg_reference = "median"
    args.svd_profile = "literature"
    if args.svd_rank is None and args.svd_energy_frac is None:
        args.svd_energy_frac = 0.90

    args.stap_profile = "clinical"
    args.score_mode = "pd"

    if getattr(args, "synth_amp_jitter", 0.05) == 0.05:
        args.synth_amp_jitter = 0.0
    if getattr(args, "synth_phase_jitter", 0.25) == 0.25:
        args.synth_phase_jitter = 0.0
    if getattr(args, "synth_noise_level", 0.01) == 0.01:
        args.synth_noise_level = 0.0
    if getattr(args, "synth_shift_max_px", 1) == 1:
        args.synth_shift_max_px = 0

    if getattr(args, "clutter_mode", "fullrank") == "fullrank":
        args.clutter_mode = "lowrank"
    if getattr(args, "clutter_rank", 3) == 3:
        args.clutter_rank = 3

    args.flow_mask_mode = "default"
    if getattr(args, "flow_mask_pd_quantile", 0.995) == 0.995:
        args.flow_mask_pd_quantile = 0.995
    if getattr(args, "flow_mask_depth_min_frac", 0.25) == 0.25:
        args.flow_mask_depth_min_frac = 0.25
    if getattr(args, "flow_mask_depth_max_frac", 0.85) == 0.85:
        args.flow_mask_depth_max_frac = 0.85
    if getattr(args, "flow_mask_dilate_iters", 2) == 2:
        args.flow_mask_dilate_iters = 2
    if getattr(args, "flow_mask_union_default", True) is True:
        args.flow_mask_union_default = False

    if profile == "Brain-Pial128":
        args.flow_mask_suppress_alias_depth = True


def apply_stap_profile_defaults(args: argparse.Namespace) -> None:
    if getattr(args, "stap_profile", "lab") != "clinical":
        return

    if getattr(args, "tile_h", 12) == 12:
        args.tile_h = int(PUBLIC_DETECTOR_DEFAULTS.tile_shape[0])
    if getattr(args, "tile_w", 12) == 12:
        args.tile_w = int(PUBLIC_DETECTOR_DEFAULTS.tile_shape[1])
    if getattr(args, "tile_stride", 6) == 6:
        args.tile_stride = int(PUBLIC_DETECTOR_DEFAULTS.tile_stride)
    if getattr(args, "lt", 4) == 4:
        args.lt = int(PUBLIC_DETECTOR_DEFAULTS.temporal_support)

    if getattr(args, "diag_load", 1e-2) == 1e-2:
        args.diag_load = float(PUBLIC_DETECTOR_DEFAULTS.diag_load)
    args.cov_estimator = str(PUBLIC_DETECTOR_DEFAULTS.covariance_estimator)
    args.huber_c = float(PUBLIC_DETECTOR_DEFAULTS.huber_c)
    args.fd_span_mode = str(CLINICAL_REPLAY_DEFAULTS.fd_span_mode)
    args.fd_span_rel = (
        f"{PUBLIC_DETECTOR_DEFAULTS.fd_span_rel[0]:0.2f},"
        f"{PUBLIC_DETECTOR_DEFAULTS.fd_span_rel[1]:0.2f}"
    )
    args.fd_fixed_span_hz = float(CLINICAL_REPLAY_DEFAULTS.fd_fixed_span_hz)
    args.grid_step_rel = float(PUBLIC_DETECTOR_DEFAULTS.grid_step_rel)
    args.max_pts = int(PUBLIC_DETECTOR_DEFAULTS.max_frequency_bins)
    args.fd_min_pts = int(PUBLIC_DETECTOR_DEFAULTS.min_frequency_bins)
    args.constraint_mode = str(CLINICAL_REPLAY_DEFAULTS.constraint_mode)
    args.constraint_ridge = float(CLINICAL_REPLAY_DEFAULTS.constraint_ridge)
    args.mvdr_load_mode = str(CLINICAL_REPLAY_DEFAULTS.mvdr_load_mode)
    args.mvdr_auto_kappa = float(CLINICAL_REPLAY_DEFAULTS.mvdr_auto_kappa)

    args.msd_lambda = PUBLIC_DETECTOR_DEFAULTS.msd_lambda
    args.msd_ridge = float(PUBLIC_DETECTOR_DEFAULTS.msd_ridge)
    args.msd_agg = str(PUBLIC_DETECTOR_DEFAULTS.msd_aggregation)
    args.msd_ratio_rho = float(PUBLIC_DETECTOR_DEFAULTS.msd_ratio_rho)
    args.msd_contrast_alpha = float(CLINICAL_REPLAY_DEFAULTS.msd_contrast_alpha)

    if getattr(args, "baseline", "mc_svd") in {"mc_svd", "svd"}:
        args.band_ratio_mode = str(CLINICAL_REPLAY_DEFAULTS.band_ratio_mode)
        args.psd_br_flow_low = float(ADAPTIVE_GUARD_DEFAULTS.flow_band_hz[0])
        args.psd_br_flow_high = float(ADAPTIVE_GUARD_DEFAULTS.flow_band_hz[1])
        args.psd_br_alias_center = float(ADAPTIVE_GUARD_DEFAULTS.alias_center_hz)
        args.psd_br_alias_width = float(ADAPTIVE_GUARD_DEFAULTS.alias_width_hz)

    if args.time_window_length is None:
        args.time_window_length = int(CLINICAL_REPLAY_DEFAULTS.time_window_length)

    os.environ.setdefault("STAP_SNAPSHOT_STRIDE", str(CLINICAL_REPLAY_DEFAULTS.snapshot_stride_env))
    os.environ.setdefault("STAP_MAX_SNAPSHOTS", str(CLINICAL_REPLAY_DEFAULTS.max_snapshots_env))
    os.environ.setdefault("STAP_FAST_PATH", str(CLINICAL_REPLAY_DEFAULTS.fast_path_env))

    if getattr(args, "score_mode", "pd") == "pd":
        os.environ.setdefault("STAP_FAST_PD_ONLY", str(CLINICAL_REPLAY_DEFAULTS.fast_pd_only_env))


def apply_svd_profile_defaults(args: argparse.Namespace) -> None:
    if getattr(args, "svd_profile", "default") != "literature":
        return
    if args.svd_rank is None and args.svd_energy_frac is None:
        args.svd_energy_frac = 0.95
