"""Replay profile and CLI preset helpers."""

from __future__ import annotations

import argparse
import os
from typing import List, Sequence, Tuple


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
        args.tile_h = 8
    if getattr(args, "tile_w", 12) == 12:
        args.tile_w = 8
    if getattr(args, "tile_stride", 6) == 6:
        args.tile_stride = 3
    if getattr(args, "lt", 4) == 4:
        args.lt = 8

    if getattr(args, "diag_load", 1e-2) == 1e-2:
        args.diag_load = 0.07
    args.cov_estimator = "tyler_pca"
    args.huber_c = 5.0
    args.fd_span_mode = "fixed"
    args.fd_span_rel = "0.30,1.10"
    args.fd_fixed_span_hz = 250.0
    args.grid_step_rel = 0.20
    args.max_pts = 15
    args.fd_min_pts = 9
    args.constraint_mode = "exp+deriv"
    args.constraint_ridge = 0.18
    args.mvdr_load_mode = "auto"
    args.mvdr_auto_kappa = 120.0

    args.msd_lambda = 0.05
    args.msd_ridge = 0.10
    args.msd_agg = "median"
    args.msd_ratio_rho = 0.05
    args.msd_contrast_alpha = 0.6

    if getattr(args, "baseline", "mc_svd") in {"mc_svd", "svd"}:
        args.band_ratio_mode = "whitened"
        args.psd_br_flow_low = 30.0
        args.psd_br_flow_high = 250.0
        args.psd_br_alias_center = 575.0
        args.psd_br_alias_width = 175.0

    if args.time_window_length is None:
        args.time_window_length = 32

    os.environ.setdefault("STAP_SNAPSHOT_STRIDE", "4")
    os.environ.setdefault("STAP_MAX_SNAPSHOTS", "64")
    os.environ.setdefault("STAP_FAST_PATH", "1")

    if getattr(args, "score_mode", "pd") == "pd":
        os.environ.setdefault("STAP_FAST_PD_ONLY", "1")


def apply_svd_profile_defaults(args: argparse.Namespace) -> None:
    if getattr(args, "svd_profile", "default") != "literature":
        return
    if args.svd_rank is None and args.svd_energy_frac is None:
        args.svd_energy_frac = 0.95
