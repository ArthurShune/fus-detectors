import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.realdata.twinkling_artifact import (
    RawBCFPar,
    decode_rawbcf_cfm_cube,
    parse_rawbcf_par,
    read_rawbcf_frame,
)
from pipeline.realdata.twinkling_bmode_mask import build_tube_masks_from_rawbcf
from sim.kwave.icube_bundle import write_acceptance_bundle_from_icube


def _slugify(text: str) -> str:
    s = re.sub(r"[\s/]+", "_", text.strip())
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s.strip("._-") or "seq"


def _parse_indices(spec: str, n_max: int) -> list[int]:
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
        out: list[int] = []
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

def _save_mask_debug(
    out_dir: Path,
    I_ref_norm: np.ndarray,
    mask_flow: np.ndarray,
    mask_bg: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    img = (np.clip(I_ref_norm, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(img).save(out_dir / "bmode_roi_ref_norm.png")
    Image.fromarray((mask_flow.astype(np.uint8) * 255)).save(out_dir / "mask_flow.png")
    Image.fromarray((mask_bg.astype(np.uint8) * 255)).save(out_dir / "mask_bg.png")
    rgb = np.stack([img, img, img], axis=2)
    rgb[mask_bg] = [255, 64, 64]
    rgb[mask_flow] = [64, 255, 64]
    Image.fromarray(rgb.astype(np.uint8)).save(out_dir / "overlay_flow_bg.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode Twinkling RawBCFCine sequences and write STAP acceptance bundles from CFM IQ."
    )
    parser.add_argument(
        "--seq-dir",
        type=str,
        required=True,
        help="Sequence directory containing RawBCFCine.dat / RawBCFCine.par.",
    )
    parser.add_argument(
        "--par-path",
        type=str,
        default=None,
        help="Optional explicit .par path (default: auto-detect in seq-dir).",
    )
    parser.add_argument(
        "--dat-path",
        type=str,
        default=None,
        help="Optional explicit .dat path (default: auto-detect in seq-dir).",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="runs/real/twinkling",
        help="Output root for acceptance bundles (default: runs/real/twinkling).",
    )
    parser.add_argument(
        "--frames",
        type=str,
        default="0:3",
        help="Frame indices to export: '0:3' | '0,1,2' | '5' (default: 0:3).",
    )
    parser.add_argument(
        "--prf-hz",
        type=float,
        required=True,
        help="CFM pulse-repetition frequency in Hz (must be provided; do not assume).",
    )
    parser.add_argument(
        "--Lt",
        type=int,
        default=8,
        help=(
            "STAP temporal aperture Lt (default: 8). Will be clamped to < NumOfCFShots "
            "(implementation requires 2 <= Lt < T, where T=NumOfCFShots)."
        ),
    )
    parser.add_argument(
        "--tile-hw",
        type=int,
        nargs=2,
        default=(8, 8),
        metavar=("H", "W"),
        help="Tile height/width in pixels (default: 8 8).",
    )
    parser.add_argument(
        "--tile-stride",
        type=int,
        default=3,
        help="Tile stride in pixels (default: 3).",
    )
    parser.add_argument(
        "--tile-stride-auto-max",
        type=int,
        default=None,
        help=(
            "If set, override --tile-stride and choose tile_stride as the largest stride "
            "<= this value such that the implied tile count is >= --tile-stride-auto-min-tiles. "
            "This is a deterministic, geometry-only rule intended to avoid contract "
            "sample-support failures on small CFM ROIs while keeping overlap modest. "
            "(default: disabled)"
        ),
    )
    parser.add_argument(
        "--tile-stride-auto-min-tiles",
        type=int,
        default=500,
        help="Minimum tile count when --tile-stride-auto-max is set (default: 500).",
    )
    parser.add_argument(
        "--diag-load",
        type=float,
        default=0.07,
        help="STAP covariance diagonal loading (default: 0.07).",
    )
    parser.add_argument(
        "--cov-estimator",
        type=str,
        default="tyler_pca",
        choices=["scm", "huber", "tyler", "tyler_pca"],
        help=(
            "Robust covariance estimator (default: tyler_pca). "
            "For very short ensembles, 'scm' is often the most stable."
        ),
    )
    parser.add_argument(
        "--stap-conditional-enable",
        action="store_true",
        help="Enable conditional STAP (compute heuristic). Default is off for structural-mask runs.",
    )
    parser.add_argument(
        "--baseline-type",
        type=str,
        default="svd_bandpass",
        choices=["svd_bandpass", "mc_svd", "svd_similarity", "local_svd", "raw", "none", "identity"],
        help=(
            "Baseline clutter filter (default: svd_bandpass). "
            "Use baseline-type=raw/none to run STAP directly on raw (optionally registered) IQ."
        ),
    )
    parser.add_argument(
        "--score-mode",
        type=str,
        default="pd",
        choices=["pd", "msd", "band_ratio"],
        help=(
            "Primary score family used for contract gating / optional score-space KA "
            "(default: pd). Use 'msd' to have KA act on score_stap_preka."
        ),
    )
    parser.add_argument(
        "--score-ka-v2-enable",
        action="store_true",
        help="Enable score-space KA contract v2 shrink-only veto (default off).",
    )
    parser.add_argument(
        "--score-ka-v2-mode",
        type=str,
        default="auto",
        choices=["safety", "uplift", "auto"],
        help="KA v2 mode when enabled (default: auto).",
    )
    parser.add_argument(
        "--stap-device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help=(
            "Device for STAP compute inside bundle writer (default: auto). "
            "Use cpu for paper-repro stability and cuda for latency profiling."
        ),
    )
    parser.add_argument(
        "--svd-keep-min",
        type=int,
        default=2,
        help="SVD bandpass keep_min (default: 2).",
    )
    parser.add_argument(
        "--svd-keep-max",
        type=int,
        default=None,
        help="SVD bandpass keep_max (default: None).",
    )
    parser.add_argument(
        "--svd-energy-frac",
        type=float,
        default=0.95,
        help=(
            "SVD energy fraction removed for baseline-type=mc_svd or local_svd "
            "(ignored by baseline-type=svd_bandpass; default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--flow-band-hz",
        type=float,
        nargs=2,
        default=(150.0, 450.0),
        metavar=("F_LO", "F_HI"),
        help=(
            "Telemetry flow band in Hz used for band-ratio telemetry + KA contract inputs "
            "(default: 150 450; tuned for PRF=2500, NumOfCFShots=17 so Pf bins are non-empty)."
        ),
    )
    parser.add_argument(
        "--alias-band-hz",
        type=float,
        nargs=2,
        default=(700.0, 1200.0),
        metavar=("A_LO", "A_HI"),
        help=(
            "Telemetry alias band in Hz used for band-ratio telemetry + KA contract inputs "
            "(default: 700 1200; tuned for PRF=2500, NumOfCFShots=17 so Pa bins are non-empty)."
        ),
    )
    parser.add_argument(
        "--mask-mode",
        type=str,
        default="bmode_tube",
        choices=["none", "bmode_tube"],
        help="Evaluation mask strategy (default: bmode_tube).",
    )
    parser.add_argument(
        "--mask-ref-frames",
        type=str,
        default="0:20",
        help="Frames used to build B-mode reference for mask generation (default: 0:20).",
    )
    args = parser.parse_args()

    seq_dir = Path(args.seq_dir)
    if args.par_path is not None:
        par_path = Path(args.par_path)
    else:
        cand = seq_dir / "RawBCFCine.par"
        if cand.exists():
            par_path = cand
        else:
            pars = sorted(seq_dir.glob("*.par"))
            if len(pars) != 1:
                raise FileNotFoundError(f"Could not auto-detect a single .par in {seq_dir}: {pars}")
            par_path = pars[0]

    if args.dat_path is not None:
        dat_path = Path(args.dat_path)
    else:
        cand = seq_dir / "RawBCFCine.dat"
        if cand.exists():
            dat_path = cand
        else:
            dats = sorted(seq_dir.glob("*.dat"))
            if len(dats) != 1:
                raise FileNotFoundError(f"Could not auto-detect a single .dat in {seq_dir}: {dats}")
            dat_path = dats[0]

    par_dict = parse_rawbcf_par(par_path)
    par = RawBCFPar.from_dict(par_dict)
    par.validate()

    frame_indices = _parse_indices(args.frames, par.num_frames)
    out_root = Path(args.out_root)

    rel = str(seq_dir)
    try:
        if seq_dir.is_relative_to(Path.cwd()):
            rel = str(seq_dir.relative_to(Path.cwd()))
    except Exception:
        pass
    seq_slug = _slugify(f"{rel}__{par_path.stem}")
    flow_lo, flow_hi = float(args.flow_band_hz[0]), float(args.flow_band_hz[1])
    alias_lo, alias_hi = float(args.alias_band_hz[0]), float(args.alias_band_hz[1])
    alias_center = 0.5 * (alias_lo + alias_hi)
    alias_width = 0.5 * (alias_hi - alias_lo)

    T = int(par.num_cfm_shots)
    if T < 3:
        raise ValueError(f"Need NumOfCFShots >= 3 for STAP (got NumOfCFShots={T}).")
    Lt_req = int(args.Lt)
    Lt = max(2, min(Lt_req, T - 1))
    if Lt != Lt_req:
        print(f"[twinkling_make_bundles] Clamped Lt from {Lt_req} to {Lt} (NumOfCFShots={T}).")
    tile_hw = (int(args.tile_hw[0]), int(args.tile_hw[1]))
    tile_stride = int(args.tile_stride)
    tile_stride_auto_max = args.tile_stride_auto_max
    tile_stride_auto_min_tiles = int(args.tile_stride_auto_min_tiles)
    tile_stride_policy: str = "fixed"
    if tile_stride_auto_max is not None:
        th, tw = tile_hw
        H, W = int(par.cfm_beam_samples), int(par.num_cfm_beams)
        if H < th or W < tw:
            raise ValueError(
                f"Tile larger than CFM grid: tile_hw={tile_hw}, cfm_hw={(H, W)}."
            )
        best: int | None = None
        for s in range(int(tile_stride_auto_max), 0, -1):
            n_y = (H - th) // s + 1
            n_x = (W - tw) // s + 1
            n_tiles = int(n_y * n_x)
            if n_tiles >= tile_stride_auto_min_tiles:
                best = s
                break
        if best is None:
            best = 1
        tile_stride = int(best)
        tile_stride_policy = "auto_max_stride"
    diag_load = float(args.diag_load)
    cov_estimator = str(args.cov_estimator)

    mask_flow_override = None
    mask_bg_override = None
    mask_meta = None
    if str(args.mask_mode).strip().lower() == "bmode_tube":
        ref_frames = _parse_indices(args.mask_ref_frames, par.num_frames)
        tube = build_tube_masks_from_rawbcf(dat_path, par, ref_frame_indices=ref_frames)
        mask_flow_override = tube.mask_flow
        mask_bg_override = tube.mask_bg
        mask_meta = {
            "mask_source": "bmode_structural_tube",
            "b0_bbeam_0based": int(tube.mapping.b0),
            "s0_sample_0based": int(tube.mapping.s0),
            "cfm_density_Q": int(tube.mapping.Q),
            "valid_beam_fraction": float(np.mean(tube.mapping.valid_beams)),
            "ref_frames": ref_frames,
            "params": tube.params,
            "qc": tube.qc,
        }
        _save_mask_debug(out_root / f"{seq_slug}__mask_debug", tube.I_ref_norm, tube.mask_flow, tube.mask_bg)
    written: list[str] = []
    for frame_idx in frame_indices:
        frame = read_rawbcf_frame(dat_path, par, int(frame_idx))
        Icube = decode_rawbcf_cfm_cube(frame, par, order="beam_major")
        dataset_name = f"{seq_slug}/frame{int(frame_idx):03d}"
        meta_extra = {
            "twinkling_rawbcf": {
                "seq_dir": str(seq_dir),
                "par_path": str(par_path),
                "dat_path": str(dat_path),
                "frame_idx": int(frame_idx),
                "decode_cfm_order": "beam_major",
                "dtype": "int32_iq_pairs_le",
                "par_keys": {k: str(v) for k, v in (par.raw or {}).items()},
                "prf_hz_note": "prf_hz is user-provided/assumed unless independently verified.",
            },
            "twinkling_eval_masks": mask_meta,
        }
        write_acceptance_bundle_from_icube(
            out_root=out_root,
            dataset_name=dataset_name,
            Icube=Icube,
            prf_hz=float(args.prf_hz),
            tile_hw=tile_hw,
            tile_stride=tile_stride,
            Lt=Lt,
            diag_load=diag_load,
            cov_estimator=cov_estimator,
            score_mode=str(args.score_mode),
            baseline_type=str(args.baseline_type),
            svd_energy_frac=float(args.svd_energy_frac),
            svd_keep_min=int(args.svd_keep_min),
            svd_keep_max=int(args.svd_keep_max) if args.svd_keep_max is not None else None,
            band_ratio_flow_low_hz=flow_lo,
            band_ratio_flow_high_hz=flow_hi,
            band_ratio_alias_center_hz=alias_center,
            band_ratio_alias_width_hz=alias_width,
            score_ka_v2_enable=bool(args.score_ka_v2_enable),
            score_ka_v2_mode=str(args.score_ka_v2_mode),
            mask_flow_override=mask_flow_override,
            mask_bg_override=mask_bg_override,
            stap_conditional_enable=bool(args.stap_conditional_enable),
            stap_device=str(args.stap_device),
            meta_extra=meta_extra,
        )
        written.append(str(out_root / dataset_name))

    out_report = {
        "seq_dir": str(seq_dir),
        "out_root": str(out_root),
        "frames": frame_indices,
        "Lt": Lt,
        "tile_hw": list(tile_hw),
        "tile_stride": tile_stride,
        "tile_stride_policy": tile_stride_policy,
        "tile_stride_auto_max": int(tile_stride_auto_max)
        if tile_stride_auto_max is not None
        else None,
        "tile_stride_auto_min_tiles": tile_stride_auto_min_tiles
        if tile_stride_auto_max is not None
        else None,
        "diag_load": float(diag_load),
        "cov_estimator": str(cov_estimator),
        "score_mode": str(args.score_mode),
        "score_ka_v2_enable": bool(args.score_ka_v2_enable),
        "score_ka_v2_mode": str(args.score_ka_v2_mode),
        "baseline_type": str(args.baseline_type),
        "prf_hz": float(args.prf_hz),
        "written_bundles": written,
    }
    report_path = out_root / f"{seq_slug}_bundles_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(out_report, indent=2))
    print(json.dumps(out_report, indent=2))


if __name__ == "__main__":
    main()
