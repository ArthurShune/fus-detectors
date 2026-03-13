#!/usr/bin/env python3
"""
Steady-state CUDA latency check for real-data pipelines (Shin RatBrain and Twinkling/Gammex).

Key principle: run multiple windows/frames in a *single* Python process so that
one-time overheads (CUDA init, Triton JIT, CUDA-graph capture) are paid once.
We report:
  - cold(win1): first window/frame
  - steady(avg win2..N): mean over subsequent windows/frames

The reported numbers are taken from compute-time telemetry written to each
bundle's meta.json (exclude disk I/O and any figure rendering).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from pipeline.realdata.shin_ratbrain import load_shin_iq, load_shin_metadata
from pipeline.realdata.twinkling_artifact import (
    RawBCFPar,
    decode_rawbcf_cfm_cube,
    parse_rawbcf_par,
    read_rawbcf_frame,
)
from pipeline.realdata.twinkling_bmode_mask import build_tube_masks_from_rawbcf
from sim.kwave.icube_bundle import write_acceptance_bundle_from_icube


def _parse_slice_list(spec: str) -> List[Tuple[List[int], str]]:
    """
    Parse comma-separated frame slices like: "0:128,64:192,122:250".
    Returns list of (frame_indices, tag).
    """
    out: List[Tuple[List[int], str]] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid slice {part!r}; expected 'start:stop'.")
        pieces = [p.strip() for p in part.split(":")]
        if len(pieces) != 2:
            raise ValueError(f"Invalid slice {part!r}; expected 'start:stop'.")
        start = int(pieces[0]) if pieces[0] else 0
        stop = int(pieces[1]) if pieces[1] else None
        if stop is None:
            raise ValueError(f"Slice {part!r} missing stop.")
        frames = list(range(int(start), int(stop)))
        tag = f"f{int(start)}_{int(stop)}"
        out.append((frames, tag))
    if not out:
        raise ValueError("No slices provided.")
    return out


def _parse_indices(spec: str, n_max: int) -> List[int]:
    """
    Parse either:
      - comma list: "0,1,2"
      - slice: "0:10" or "0:10:2"
      - single int: "5"
    """
    spec = (spec or "").strip()
    if not spec:
        return list(range(n_max))
    if "," in spec:
        out: List[int] = []
        for part in spec.split(","):
            part = part.strip()
            if part:
                out.append(int(part))
        return out
    if ":" in spec:
        parts = [p.strip() for p in spec.split(":")]
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid slice spec: {spec!r}")
        start = int(parts[0]) if parts[0] else 0
        stop = int(parts[1]) if parts[1] else n_max
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        return list(range(start, min(stop, n_max), step))
    return [int(spec)]


def _slugify(text: str) -> str:
    s = re.sub(r"[\s/]+", "_", str(text).strip())
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s.strip("._-") or "seq"


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _load_telemetry(meta_path: Path) -> Dict[str, Any]:
    meta = json.loads(meta_path.read_text())
    tele = meta.get("stap_fallback_telemetry") or {}
    return tele if isinstance(tele, dict) else {}


def _mean_over(entries: List[Dict[str, Any]], key: str) -> float | None:
    vals: List[float] = []
    for d in entries:
        if key not in d:
            continue
        try:
            vals.append(float(d[key]))
        except Exception:
            pass
    if not vals:
        return None
    return float(sum(vals) / float(len(vals)))


def _detector_time_key(entries: List[Dict[str, Any]]) -> str:
    if any("stap_total_ms" in d for d in entries):
        return "stap_total_ms"
    return "stap_ms"


def _print_cold_steady(label: str, teles: List[Dict[str, Any]], *, keys: List[str]) -> None:
    if not teles:
        raise ValueError(f"No telemetry entries for {label}")
    cold = teles[0]
    steady = teles[1:] if len(teles) > 1 else []

    print(f"\n[{label} cold(win1)]")
    for k in keys:
        if k in cold:
            print(f"  {k}: {cold[k]}")

    if not steady:
        return
    print(f"\n[{label} steady(avg win2..N)]")
    for k in keys:
        m = _mean_over(steady, k)
        if m is not None:
            print(f"  {k}: {m}")


def _mean_stage_map(teles: List[Dict[str, Any]]) -> Dict[str, float]:
    acc: Dict[str, List[float]] = {}
    for tele in teles:
        stage_map = tele.get("stap_cuda_stage_ms")
        if not isinstance(stage_map, dict):
            continue
        for key, value in stage_map.items():
            try:
                acc.setdefault(str(key), []).append(float(value))
            except Exception:
                pass
    return {key: float(sum(vals) / float(len(vals))) for key, vals in acc.items() if vals}


def _print_cuda_profile(label: str, teles: List[Dict[str, Any]]) -> None:
    if not teles:
        return
    steady = teles[1:] if len(teles) > 1 else teles
    stage_mean = _mean_stage_map(steady)
    print(f"\n[{label} steady CUDA profile]")
    if stage_mean:
        top_level_keys = [
            "stap:tiling:prep",
            "stap:tiling:cube_unfold",
            "stap:tiling:active_index",
            "stap:tiling:gather",
            "stap:core",
            "stap:tiling:fold",
            "stap:tiling:to_cpu",
        ]
        core_keys = [
            "stap:hankel",
            "stap:covariance:train_trim",
            "stap:covariance",
            "stap:shrinkage",
            "stap:lambda",
            "stap:fd_grid",
            "stap:constraints",
            "stap:band_energy",
            "stap:aggregate",
            "stap:telemetry",
        ]
        top_present = [(k, stage_mean[k]) for k in top_level_keys if k in stage_mean]
        if top_present:
            print("  top_level_ms:")
            for key, value in top_present:
                print(f"    {key}: {value:.3f}")
        core_present = [(k, stage_mean[k]) for k in core_keys if k in stage_mean]
        if core_present:
            print("  core_substage_ms:")
            for key, value in core_present:
                print(f"    {key}: {value:.3f}")
    else:
        fallback_keys = [
            "stap_hankel_ms_mean",
            "stap_cov_ms_mean",
            "stap_shrink_ms_mean",
            "stap_fdgrid_ms_mean",
            "stap_msd_ms_mean",
        ]
        fallback_present = [(k, _mean_over(steady, k)) for k in fallback_keys]
        fallback_present = [(k, v) for k, v in fallback_present if v is not None]
        if fallback_present:
            print("  tile_phase_mean_ms:")
            for key, value in fallback_present:
                print(f"    {key}: {value:.3f}")
    for scalar_key in (
        "stap_cuda_max_memory_allocated_mb",
        "stap_cuda_max_memory_reserved_mb",
        "stap_fast_active_tile_fraction",
        "stap_fast_chunk_size_mean",
        "stap_fast_chunk_size_max",
        "stap_fast_chunk_count",
    ):
        m = _mean_over(steady, scalar_key)
        if m is not None:
            print(f"  {scalar_key}: {m:.3f}")


def _stride_auto(H: int, W: int, tile_hw: Tuple[int, int], *, max_stride: int, min_tiles: int) -> int:
    th, tw = int(tile_hw[0]), int(tile_hw[1])
    if H < th or W < tw:
        raise ValueError(f"Tile larger than grid: tile_hw={tile_hw} grid={(H, W)}")
    for s in range(int(max_stride), 0, -1):
        n_y = (int(H) - th) // s + 1
        n_x = (int(W) - tw) // s + 1
        if int(n_y * n_x) >= int(min_tiles):
            return int(s)
    return 1


def _maybe_set_tile_batch(tile_batch: int | None) -> None:
    if tile_batch is None:
        return
    tb = int(tile_batch)
    if tb <= 0:
        raise ValueError("--tile-batch must be > 0")
    os.environ["STAP_TILE_BATCH"] = str(tb)


def _run_shin(args: argparse.Namespace) -> Dict[str, Any]:
    data_root = Path(args.data_root)
    iq_path = data_root / str(args.iq_file)
    if not iq_path.is_file():
        raise FileNotFoundError(f"Missing Shin IQ file: {iq_path}")

    info = load_shin_metadata(data_root)
    windows = _parse_slice_list(str(args.windows))
    for frames, tag in windows:
        if frames and (max(frames) >= int(info.frames) or min(frames) < 0):
            raise ValueError(f"Window {tag} out of range for frames={info.frames}.")

    _maybe_set_tile_batch(args.tile_batch)

    out_root = Path(args.out_root)
    run_root_full = out_root / "shin" / "stap_full"
    run_root_cond = out_root / "shin" / "stap_conditional"
    if args.clean:
        _clean_dir(run_root_full)
        _clean_dir(run_root_cond)
    else:
        run_root_full.mkdir(parents=True, exist_ok=True)
        run_root_cond.mkdir(parents=True, exist_ok=True)

    def _run_variant(*, variant_root: Path, conditional: bool) -> List[Dict[str, Any]]:
        teles: List[Dict[str, Any]] = []
        for frames, tag in windows:
            Icube = load_shin_iq(iq_path, info, frames=frames)
            dataset_name = f"shin_{iq_path.stem}_{tag}"
            meta_extra = {
                "orig_data": {
                    "dataset": "ShinRatBrain_Fig3",
                    "iq_file": str(iq_path),
                    "sizeinfo": asdict(info),
                    "frames_tag": tag,
                    "frames_spec": str(args.windows),
                }
            }
            paths = write_acceptance_bundle_from_icube(
                out_root=variant_root,
                dataset_name=dataset_name,
                Icube=Icube,
                prf_hz=float(args.prf_hz),
                tile_hw=(int(args.tile_h), int(args.tile_w)),
                tile_stride=int(args.tile_stride),
                Lt=int(args.Lt),
                diag_load=float(args.diag_load),
                cov_estimator=str(args.cov_estimator),
                baseline_type=str(args.baseline_type),
                svd_energy_frac=float(args.svd_energy_frac),
                flow_mask_mode=str(args.flow_mask_mode),
                flow_mask_pd_quantile=float(args.flow_mask_pd_quantile),
                flow_mask_min_pixels=int(args.flow_mask_min_pixels),
                flow_mask_union_default=bool(args.flow_mask_union_default),
                band_ratio_flow_low_hz=float(args.flow_low_hz),
                band_ratio_flow_high_hz=float(args.flow_high_hz),
                band_ratio_alias_center_hz=float(args.alias_center_hz),
                band_ratio_alias_width_hz=float(args.alias_width_hz),
                stap_detector_variant=str(args.stap_detector_variant),
                stap_whiten_gamma=float(args.stap_whiten_gamma),
                hybrid_rescue_rule=str(args.hybrid_rescue_rule),
                stap_device=str(args.stap_device),
                run_stap=True,
                stap_conditional_enable=bool(conditional),
                score_ka_v2_enable=False,
                meta_extra=meta_extra,
            )
            tele = _load_telemetry(Path(paths["meta"]))
            teles.append(tele)
        return teles

    tele_full = _run_variant(variant_root=run_root_full, conditional=False)
    tele_cond = _run_variant(variant_root=run_root_cond, conditional=True)

    detector_key_full = _detector_time_key(tele_full)
    detector_key_cond = _detector_time_key(tele_cond)
    keys_full = ["baseline_ms", "reg_ms", "svd_ms", detector_key_full, "stap_fast_path_used", "tile_count"]
    keys_cond = ["baseline_ms", "reg_ms", "svd_ms", detector_key_cond, "stap_fast_path_used", "tile_count"]
    _print_cold_steady("shin stap_full", tele_full, keys=keys_full)
    _print_cold_steady("shin stap_conditional", tele_cond, keys=keys_cond)
    _print_cuda_profile("shin stap_full", tele_full)
    _print_cuda_profile("shin stap_conditional", tele_cond)

    base_steady_ms = _mean_over(tele_full[1:] if len(tele_full) > 1 else tele_full, "baseline_ms")
    stap_full_steady_ms = _mean_over(
        tele_full[1:] if len(tele_full) > 1 else tele_full, detector_key_full
    )
    stap_cond_steady_ms = _mean_over(
        tele_cond[1:] if len(tele_cond) > 1 else tele_cond, detector_key_cond
    )

    return {
        "regime": "Shin RatBrain Fig3",
        "windows": [tag for _frames, tag in windows],
        "baseline_s_steady": (float(base_steady_ms) / 1000.0) if base_steady_ms is not None else None,
        "stap_full_s_steady": (float(stap_full_steady_ms) / 1000.0)
        if stap_full_steady_ms is not None
        else None,
        "stap_conditional_s_steady": (float(stap_cond_steady_ms) / 1000.0)
        if stap_cond_steady_ms is not None
        else None,
    }


def _run_gammex(args: argparse.Namespace) -> Dict[str, Any]:
    prf_hz = float(args.prf_hz)
    _maybe_set_tile_batch(args.tile_batch)

    out_root = Path(args.out_root)
    if args.clean:
        _clean_dir(out_root / "gammex")
    else:
        (out_root / "gammex").mkdir(parents=True, exist_ok=True)

    def _run_view(
        *,
        view_name: str,
        seq_dir: Path,
        par_path: Path,
        dat_path: Path,
        frames_spec: str,
        tile_stride_override: int | None,
        expected_stride: int | None,
    ) -> Dict[str, Any]:
        par_dict = parse_rawbcf_par(par_path)
        par = RawBCFPar.from_dict(par_dict)
        par.validate()
        frame_indices = _parse_indices(frames_spec, int(par.num_frames))
        if len(frame_indices) < 2:
            raise ValueError(f"{view_name}: need >=2 frames for cold/steady, got {frame_indices}")

        H = int(par.cfm_beam_samples)
        W = int(par.num_cfm_beams)
        tile_hw = (int(args.tile_h), int(args.tile_w))
        if tile_stride_override is not None:
            tile_stride = int(tile_stride_override)
        else:
            tile_stride = _stride_auto(
                H,
                W,
                tile_hw,
                max_stride=int(args.tile_stride_auto_max),
                min_tiles=int(args.tile_stride_auto_min_tiles),
            )
        if expected_stride is not None and int(tile_stride) != int(expected_stride):
            raise RuntimeError(
                f"{view_name}: auto stride mismatch (got {tile_stride}, expected {expected_stride})."
            )

        Lt_req = int(args.Lt)
        T = int(par.num_cfm_shots)
        Lt = max(2, min(Lt_req, T - 1))
        if Lt != Lt_req:
            raise ValueError(f"{view_name}: requested Lt={Lt_req} invalid for NumOfCFShots={T}.")

        mask_flow = None
        mask_bg = None
        if str(args.mask_mode).strip().lower() == "bmode_tube":
            ref_frames = _parse_indices(str(args.mask_ref_frames), int(par.num_frames))
            tube = build_tube_masks_from_rawbcf(dat_path, par, ref_frame_indices=ref_frames)
            mask_flow = tube.mask_flow
            mask_bg = tube.mask_bg

        view_slug = _slugify(f"{view_name}__{par_path.stem}")
        variant_root = out_root / "gammex" / view_slug / "stap_full"
        variant_root.mkdir(parents=True, exist_ok=True)

        teles: List[Dict[str, Any]] = []
        for frame_idx in frame_indices:
            frame = read_rawbcf_frame(dat_path, par, int(frame_idx))
            Icube = decode_rawbcf_cfm_cube(frame, par, order="beam_major")
            dataset_name = f"{view_slug}/frame{int(frame_idx):03d}"
            meta_extra = {
                "twinkling_rawbcf": {
                    "seq_dir": str(seq_dir),
                    "par_path": str(par_path),
                    "dat_path": str(dat_path),
                    "frame_idx": int(frame_idx),
                    "decode_cfm_order": "beam_major",
                    "par_keys": {k: str(v) for k, v in (par.raw or {}).items()},
                    "prf_hz_note": "prf_hz is user-provided/assumed unless independently verified.",
                },
                "latency_profile": {
                    "tile_stride_policy": "auto_max_stride",
                    "tile_stride_auto_max": int(args.tile_stride_auto_max),
                    "tile_stride_auto_min_tiles": int(args.tile_stride_auto_min_tiles),
                },
            }
            paths = write_acceptance_bundle_from_icube(
                out_root=variant_root,
                dataset_name=dataset_name,
                Icube=Icube,
                prf_hz=prf_hz,
                tile_hw=tile_hw,
                tile_stride=int(tile_stride),
                Lt=int(Lt),
                diag_load=float(args.diag_load),
                cov_estimator=str(args.cov_estimator),
                score_mode=str(args.score_mode),
                baseline_type=str(args.baseline_type),
                svd_keep_min=int(args.svd_keep_min),
                svd_keep_max=int(args.svd_keep_max) if args.svd_keep_max is not None else None,
                band_ratio_flow_low_hz=float(args.flow_low_hz),
                band_ratio_flow_high_hz=float(args.flow_high_hz),
                band_ratio_alias_center_hz=float(args.alias_center_hz),
                band_ratio_alias_width_hz=float(args.alias_width_hz),
                stap_detector_variant=str(args.stap_detector_variant),
                stap_whiten_gamma=float(args.stap_whiten_gamma),
                hybrid_rescue_rule=str(args.hybrid_rescue_rule),
                mask_flow_override=mask_flow,
                mask_bg_override=mask_bg,
                stap_conditional_enable=(str(args.stap_conditional).strip().lower() == "on"),
                stap_device=str(args.stap_device),
                run_stap=True,
                defer_raw_pd_base=(
                    str(args.baseline_type).strip().lower() in {"raw", "none", "identity"}
                ),
                score_ka_v2_enable=False,
                meta_extra=meta_extra,
            )
            teles.append(_load_telemetry(Path(paths["meta"])))

        detector_key = _detector_time_key(teles)
        keys = ["baseline_ms", "svd_ms", detector_key, "stap_fast_path_used", "tile_count"]
        _print_cold_steady(f"gammex {view_name}", teles, keys=keys)
        _print_cuda_profile(f"gammex {view_name}", teles)
        base_steady_ms = _mean_over(teles[1:], "baseline_ms")
        stap_steady_ms = _mean_over(teles[1:], detector_key)
        return {
            "view": view_name,
            "frames": frame_indices,
            "tile_stride": int(tile_stride),
            "baseline_s_steady": (float(base_steady_ms) / 1000.0) if base_steady_ms is not None else None,
            "stap_full_s_steady": (float(stap_steady_ms) / 1000.0) if stap_steady_ms is not None else None,
        }

    # Defaults match the reproducibility manifest / manuscript configs.
    seq_root = Path(args.seq_root)
    along_dir = seq_root / "Flow in Gammex phantom (along - linear probe)"
    across_dir = seq_root / "Flow in Gammex phantom (across - linear probe)"
    along = _run_view(
        view_name="along_linear17",
        seq_dir=along_dir,
        par_path=along_dir / "RawBCFCine.par",
        dat_path=along_dir / "RawBCFCine.dat",
        frames_spec=str(args.frames_along),
        tile_stride_override=(
            int(args.along_tile_stride) if args.along_tile_stride is not None else None
        ),
        expected_stride=(
            None if args.along_tile_stride is not None else 6
        ),
    )
    across_par = across_dir / "RawBCFCine_08062017_145434_17.par"
    across_dat = across_dir / "RawBCFCine_08062017_145434_17.dat"
    across = _run_view(
        view_name="across_linear17",
        seq_dir=across_dir,
        par_path=across_par,
        dat_path=across_dat,
        frames_spec=str(args.frames_across),
        tile_stride_override=(
            int(args.across_tile_stride) if args.across_tile_stride is not None else None
        ),
        expected_stride=(
            None if args.across_tile_stride is not None else 4
        ),
    )
    return {"regime": "Gammex flow phantom", "along": along, "across": across}


def main() -> None:
    ap = argparse.ArgumentParser(description="Real-data steady-state latency check (Shin + Gammex).")
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("runs/latency_realdata_cuda"),
        help="Output root for bundle artifacts (default: %(default)s).",
    )
    ap.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete output folders before running (default: %(default)s).",
    )
    ap.add_argument(
        "--stap-device",
        type=str,
        default="cuda",
        help="STAP device (cpu|cuda) used inside bundle writers (default: %(default)s).",
    )
    ap.add_argument(
        "--tile-batch",
        type=int,
        default=None,
        help="Optional override for STAP tile batch size (sets STAP_TILE_BATCH).",
    )
    ap.add_argument(
        "--profile-cuda-stages",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable CUDA-event stage profiling inside the STAP fast path (default: %(default)s).",
    )
    ap.add_argument(
        "--stap-detector-variant",
        type=str,
        default="adaptive_guard",
        choices=["msd_ratio", "whitened_power", "unwhitened_ratio", "hybrid_rescue", "adaptive_guard"],
        help="Detector-family mode to profile (default: %(default)s).",
    )
    ap.add_argument(
        "--stap-whiten-gamma",
        type=float,
        default=1.0,
        help="Whitening exponent for msd_ratio / hybrid specialist branch (default: %(default)s).",
    )
    ap.add_argument(
        "--hybrid-rescue-rule",
        type=str,
        default="guard_promote_v1",
        choices=[
            "guard_frac_v1",
            "alias_rescue_v1",
            "band_ratio_v1",
            "guard_promote_v1",
            "guard_promote_tile_v1",
        ],
        help="Frozen routing rule used by hybrid/adaptive detector variants (default: %(default)s).",
    )

    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_shin = sub.add_parser("shin", help="Run Shin RatBrain steady-state latency (multiple windows).")
    ap_shin.add_argument("--data-root", type=Path, default=Path("data/shin_zenodo_10711806/ratbrain_fig3_raw"))
    ap_shin.add_argument("--iq-file", type=str, default="IQData001.dat")
    ap_shin.add_argument("--windows", type=str, default="0:128,64:192,122:250")
    ap_shin.add_argument("--prf-hz", type=float, default=1000.0)
    ap_shin.add_argument("--Lt", type=int, default=64)
    ap_shin.add_argument("--tile-h", type=int, default=8)
    ap_shin.add_argument("--tile-w", type=int, default=8)
    ap_shin.add_argument("--tile-stride", type=int, default=3)
    ap_shin.add_argument("--diag-load", type=float, default=0.07)
    ap_shin.add_argument("--cov-estimator", type=str, default="tyler_pca")
    ap_shin.add_argument("--baseline-type", type=str, default="mc_svd")
    ap_shin.add_argument("--svd-energy-frac", type=float, default=0.97)
    ap_shin.add_argument("--flow-mask-mode", type=str, default="pd_auto")
    ap_shin.add_argument("--flow-mask-pd-quantile", type=float, default=0.99)
    ap_shin.add_argument("--flow-mask-min-pixels", type=int, default=64)
    ap_shin.add_argument("--flow-mask-union-default", action=argparse.BooleanOptionalAction, default=False)
    # Shin-U band profile (appendix_supp_results.tex)
    ap_shin.add_argument("--flow-low-hz", type=float, default=60.0)
    ap_shin.add_argument("--flow-high-hz", type=float, default=250.0)
    ap_shin.add_argument("--alias-center-hz", type=float, default=400.0)
    ap_shin.add_argument("--alias-width-hz", type=float, default=100.0)

    ap_g = sub.add_parser("gammex", help="Run Gammex flow phantom steady-state latency (multiple frames).")
    ap_g.add_argument(
        "--seq-root",
        type=Path,
        default=Path("data/twinkling_artifact/Flow in Gammex phantom"),
        help="Root containing along/across subfolders (default: %(default)s).",
    )
    ap_g.add_argument("--prf-hz", type=float, default=2500.0)
    ap_g.add_argument("--frames-along", type=str, default="0:6")
    ap_g.add_argument("--frames-across", type=str, default="0:6")
    ap_g.add_argument("--Lt", type=int, default=16)
    ap_g.add_argument("--tile-h", type=int, default=8)
    ap_g.add_argument("--tile-w", type=int, default=8)
    ap_g.add_argument("--tile-stride-auto-max", type=int, default=6)
    ap_g.add_argument("--tile-stride-auto-min-tiles", type=int, default=500)
    ap_g.add_argument(
        "--along-tile-stride",
        type=int,
        default=None,
        help="Optional fixed along-view tile stride override (default: auto with expected stride 6).",
    )
    ap_g.add_argument(
        "--across-tile-stride",
        type=int,
        default=None,
        help="Optional fixed across-view tile stride override (default: auto with expected stride 4).",
    )
    ap_g.add_argument("--diag-load", type=float, default=0.07)
    ap_g.add_argument("--cov-estimator", type=str, default="tyler_pca")
    ap_g.add_argument("--baseline-type", type=str, default="svd_bandpass")
    ap_g.add_argument("--svd-keep-min", type=int, default=2)
    ap_g.add_argument("--svd-keep-max", type=int, default=17)
    ap_g.add_argument("--score-mode", type=str, default="msd", choices=["pd", "msd", "band_ratio"])
    ap_g.add_argument("--mask-mode", type=str, default="bmode_tube", choices=["none", "bmode_tube"])
    ap_g.add_argument("--mask-ref-frames", type=str, default="0:20")
    ap_g.add_argument(
        "--stap-conditional",
        type=str,
        default="off",
        choices=["off", "on"],
        help="Whether to enable conditional execution for the phantom latency run (default: off).",
    )
    # Gammex bands (stap_fus_methodology.tex)
    ap_g.add_argument("--flow-low-hz", type=float, default=150.0)
    ap_g.add_argument("--flow-high-hz", type=float, default=450.0)
    ap_g.add_argument("--alias-center-hz", type=float, default=950.0)
    ap_g.add_argument("--alias-width-hz", type=float, default=250.0)

    args = ap.parse_args()
    os.environ.setdefault("STAP_FAST_PATH", "1")
    os.environ.setdefault("STAP_TILING_UNFOLD", "1")
    os.environ.setdefault("STAP_FAST_CUDA_GRAPH", "1")
    os.environ.setdefault("STAP_FAST_PD_ONLY", "1")
    os.environ.setdefault("STAP_FAST_TELEMETRY", "0")
    os.environ.setdefault("STAP_LATENCY_MODE", "1")
    os.environ.setdefault("MC_SVD_TORCH", "1")
    os.environ.setdefault("MC_SVD_TORCH_RETURN_CUBE", "1")
    if bool(args.profile_cuda_stages):
        os.environ["STAP_CUDA_EVENT_TIMING"] = "1"

    if args.cmd == "shin":
        summary = _run_shin(args)
    else:
        summary = _run_gammex(args)

    print("\n[summary]")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
