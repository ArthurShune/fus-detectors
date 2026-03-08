#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.realdata import mace_data_root, shin_ratbrain_data_root
from pipeline.realdata.mace_wholebrain import find_mace_scans
from pipeline.realdata.shin_ratbrain import load_shin_iq, load_shin_metadata
from pipeline.realdata.twinkling_artifact import (
    RawBCFPar,
    decode_rawbcf_cfm_cube,
    parse_rawbcf_par,
    read_rawbcf_frame,
    twinkling_artifact_data_root,
)
from pipeline.realdata.twinkling_bmode_mask import build_tube_masks_from_rawbcf
from pipeline.realdata.ulm_zenodo_7883227 import load_ulm_block_iq, ulm_zenodo_7883227_root
from scripts.physical_doppler_sanity_link import BandEdges, TileSpec, summarize_icube
from sim.simus.bundle import estimate_simus_policy_features

IQ_METRICS = (
    "flow_malias_q10",
    "flow_malias_q50",
    "flow_malias_q90",
    "bg_malias_q10",
    "bg_malias_q50",
    "bg_malias_q90",
    "flow_fpeak_q10",
    "flow_fpeak_q50",
    "flow_fpeak_q90",
    "bg_fpeak_q10",
    "bg_fpeak_q50",
    "bg_fpeak_q90",
    "flow_coh1_q10",
    "flow_coh1_q50",
    "flow_coh1_q90",
    "bg_coh1_q10",
    "bg_coh1_q50",
    "bg_coh1_q90",
    "svd_flow_cum_r1",
    "svd_flow_cum_r2",
    "svd_flow_cum_r5",
    "svd_flow_cum_r10",
    "svd_bg_cum_r1",
    "svd_bg_cum_r2",
    "svd_bg_cum_r5",
    "svd_bg_cum_r10",
    "reg_shift_rms",
    "reg_shift_p90",
    "reg_psr_median",
)

FUNCTIONAL_METRICS = (
    "frac_pf_peak_pos",
    "frac_pf_peak_neg",
    "median_log_alias_pos",
    "median_log_alias_neg",
    "pd_pauc",
    "br_pauc",
    "gate_kept_frac",
    "pd_fp_at_tpr",
    "gated_pd_fp_at_tpr",
)

DEFAULT_ACCEPTANCE_METRICS = (
    "flow_malias_q50",
    "bg_malias_q50",
    "flow_fpeak_q50",
    "bg_fpeak_q50",
    "flow_coh1_q50",
    "bg_coh1_q50",
    "svd_flow_cum_r1",
    "svd_flow_cum_r2",
    "svd_bg_cum_r1",
    "svd_bg_cum_r2",
    "reg_shift_rms",
    "reg_shift_p90",
    "reg_psr_median",
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_indices(spec: str, n_max: int) -> list[int]:
    text = (spec or "").strip()
    if not text:
        return list(range(n_max))
    if "," in text:
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    if ":" in text:
        parts = [part.strip() for part in text.split(":")]
        if len(parts) not in (2, 3):
            raise ValueError(f"invalid slice spec: {spec!r}")
        start = int(parts[0]) if parts[0] else 0
        stop = int(parts[1]) if parts[1] else n_max
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        return list(range(start, min(stop, n_max), step))
    return [int(text)]


def _metric_quantiles(values: list[float]) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "min": None, "q10": None, "q50": None, "q90": None, "max": None}
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def build_envelopes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["anchor_kind"]), []).append(row)

    out: dict[str, Any] = {}
    for kind, items in sorted(grouped.items()):
        mode = str(items[0].get("anchor_mode", "iq"))
        metrics = IQ_METRICS if mode == "iq" else FUNCTIONAL_METRICS
        metric_payload: dict[str, Any] = {}
        for metric in metrics:
            metric_payload[metric] = _metric_quantiles(
                [float(item[metric]) for item in items if item.get(metric) is not None]
            )
        out[kind] = {
            "anchor_mode": mode,
            "n_rows": int(len(items)),
            "metrics": metric_payload,
        }
    return out


def flatten_summary_row(
    *,
    anchor_kind: str,
    case_key: str,
    report: dict[str, Any],
    motion_features: dict[str, Any],
) -> dict[str, Any]:
    flow = (report.get("summary") or {}).get("flow") or {}
    bg = (report.get("summary") or {}).get("bg") or {}
    svd = report.get("svd") or {}

    def q(group: dict[str, Any], metric: str, key: str) -> float | None:
        node = group.get(metric) or {}
        val = node.get(key)
        return None if val is None else float(val)

    def sv(group_name: str, key: str) -> float | None:
        node = (svd.get(group_name) or {}) if isinstance(svd, dict) else {}
        val = node.get(key)
        return None if val is None else float(val)

    return {
        "anchor_kind": str(anchor_kind),
        "anchor_mode": "iq",
        "case_key": str(case_key),
        "flow_malias_q10": q(flow, "malias", "q10"),
        "flow_malias_q50": q(flow, "malias", "q50"),
        "flow_malias_q90": q(flow, "malias", "q90"),
        "bg_malias_q10": q(bg, "malias", "q10"),
        "bg_malias_q50": q(bg, "malias", "q50"),
        "bg_malias_q90": q(bg, "malias", "q90"),
        "flow_fpeak_q10": q(flow, "fpeak_hz", "q10"),
        "flow_fpeak_q50": q(flow, "fpeak_hz", "q50"),
        "flow_fpeak_q90": q(flow, "fpeak_hz", "q90"),
        "bg_fpeak_q10": q(bg, "fpeak_hz", "q10"),
        "bg_fpeak_q50": q(bg, "fpeak_hz", "q50"),
        "bg_fpeak_q90": q(bg, "fpeak_hz", "q90"),
        "flow_coh1_q10": q(flow, "coh1", "q10"),
        "flow_coh1_q50": q(flow, "coh1", "q50"),
        "flow_coh1_q90": q(flow, "coh1", "q90"),
        "bg_coh1_q10": q(bg, "coh1", "q10"),
        "bg_coh1_q50": q(bg, "coh1", "q50"),
        "bg_coh1_q90": q(bg, "coh1", "q90"),
        "svd_flow_cum_r1": sv("flow", "cum_r1"),
        "svd_flow_cum_r2": sv("flow", "cum_r2"),
        "svd_flow_cum_r5": sv("flow", "cum_r5"),
        "svd_flow_cum_r10": sv("flow", "cum_r10"),
        "svd_bg_cum_r1": sv("bg", "cum_r1"),
        "svd_bg_cum_r2": sv("bg", "cum_r2"),
        "svd_bg_cum_r5": sv("bg", "cum_r5"),
        "svd_bg_cum_r10": sv("bg", "cum_r10"),
        "reg_shift_rms": None if motion_features.get("reg_shift_rms") is None else float(motion_features["reg_shift_rms"]),
        "reg_shift_p90": None if motion_features.get("reg_shift_p90") is None else float(motion_features["reg_shift_p90"]),
        "reg_psr_median": None if motion_features.get("reg_psr_median") is None else float(motion_features["reg_psr_median"]),
    }


def summarize_iq_anchor(
    *,
    anchor_kind: str,
    case_key: str,
    icube: np.ndarray,
    prf_hz: float,
    tile: TileSpec,
    bands: BandEdges,
    mask_flow: np.ndarray | None,
    mask_bg: np.ndarray | None,
    derive_masks: bool,
) -> dict[str, Any]:
    report = summarize_icube(
        name=case_key,
        Icube=icube,
        prf_hz=float(prf_hz),
        tile=tile,
        bands=bands,
        mask_flow=mask_flow,
        mask_bg=mask_bg,
        derive_masks=bool(derive_masks),
        derive_vessel_q=0.99,
        derive_bg_q=0.20,
    )
    motion = estimate_simus_policy_features(icube, reg_subpixel=4, reg_reference="median")
    return flatten_summary_row(
        anchor_kind=anchor_kind,
        case_key=case_key,
        report=report,
        motion_features=motion,
    )


def summarize_mace_phase2_report(report_csv: Path) -> list[dict[str, Any]]:
    if not report_csv.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with report_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for src in reader:
            row = {
                "anchor_kind": "mace_phase2",
                "anchor_mode": "functional_readout",
                "case_key": f"{src.get('scan_name', 'scan')}::plane{src.get('plane_idx', '0')}",
            }
            for metric in FUNCTIONAL_METRICS:
                val = src.get(metric)
                row[metric] = None if val in (None, "") else float(val)
            rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Freeze SIMUS v2 real-data anchor envelopes from Shin, Gammex, ULM, and Macé-derived summaries.")
    ap.add_argument("--out-dir", type=Path, default=Path("reports/simus_v2/anchors"))
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--pf", type=float, nargs=2, default=(30.0, 250.0))
    ap.add_argument("--pg", type=float, nargs=2, default=(250.0, 400.0))
    ap.add_argument("--pa-lo", type=float, default=400.0)
    ap.add_argument("--tile-hw", type=int, nargs=2, default=(8, 8))
    ap.add_argument("--tile-stride", type=int, default=3)
    ap.add_argument("--skip-shin", action="store_true")
    ap.add_argument("--skip-gammex", action="store_true")
    ap.add_argument("--skip-ulm", action="store_true")
    ap.add_argument("--skip-mace", action="store_true")
    ap.add_argument("--shin-root", type=Path, default=None)
    ap.add_argument("--shin-iq-file", type=str, default="IQData001.dat")
    ap.add_argument("--shin-frames", type=str, default="0:128")
    ap.add_argument("--shin-prf-hz", type=float, default=1000.0)
    ap.add_argument("--gammex-root", type=Path, default=None)
    ap.add_argument("--gammex-frames-along", type=str, default="0:6")
    ap.add_argument("--gammex-frames-across", type=str, default="0:6")
    ap.add_argument("--gammex-prf-hz", type=float, default=2500.0)
    ap.add_argument("--gammex-across-par", type=str, default="RawBCFCine_08062017_145434_17.par")
    ap.add_argument("--gammex-across-dat", type=str, default="RawBCFCine_08062017_145434_17.dat")
    ap.add_argument("--ulm-root", type=Path, default=None)
    ap.add_argument("--ulm-blocks", type=str, default="1,2,3")
    ap.add_argument("--ulm-frames", type=str, default="0:128")
    ap.add_argument("--ulm-prf-hz", type=float, default=1000.0)
    ap.add_argument("--ulm-cache-dir", type=Path, default=Path("tmp/ulm_zenodo_7883227"))
    ap.add_argument("--mace-report-csv", type=Path, default=Path("reports/mace_phase2_summary.csv"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv) if args.out_csv is not None else out_dir / "simus_v2_anchor_rows.csv"
    out_json = Path(args.out_json) if args.out_json is not None else out_dir / "simus_v2_anchor_envelopes.json"

    bands = BandEdges(
        pf_lo_hz=float(args.pf[0]),
        pf_hi_hz=float(args.pf[1]),
        pg_lo_hz=float(args.pg[0]),
        pg_hi_hz=float(args.pg[1]),
        pa_lo_hz=float(args.pa_lo),
    )
    tile = TileSpec(h=int(args.tile_hw[0]), w=int(args.tile_hw[1]), stride=int(args.tile_stride))

    rows: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}

    if not bool(args.skip_shin):
        shin_root = Path(args.shin_root) if args.shin_root is not None else shin_ratbrain_data_root()
        if shin_root.is_dir():
            info = load_shin_metadata(shin_root)
            frames = _parse_indices(str(args.shin_frames), int(info.frames))
            iq = load_shin_iq(shin_root / str(args.shin_iq_file), info, frames=frames)
            rows.append(
                summarize_iq_anchor(
                    anchor_kind="shin",
                    case_key=f"shin::{Path(args.shin_iq_file).stem}::{str(args.shin_frames)}",
                    icube=iq,
                    prf_hz=float(args.shin_prf_hz),
                    tile=tile,
                    bands=bands,
                    mask_flow=None,
                    mask_bg=None,
                    derive_masks=True,
                )
            )
        else:
            skipped["shin"] = f"missing root: {shin_root}"

    if not bool(args.skip_gammex):
        gammex_root = Path(args.gammex_root) if args.gammex_root is not None else twinkling_artifact_data_root() / "Flow in Gammex phantom"
        if gammex_root.is_dir():
            along_dir = gammex_root / "Flow in Gammex phantom (along - linear probe)"
            along_par = RawBCFPar.from_dict(parse_rawbcf_par(along_dir / "RawBCFCine.par"))
            along_par.validate()
            along_tube = build_tube_masks_from_rawbcf(
                along_dir / "RawBCFCine.dat",
                along_par,
                ref_frame_indices=_parse_indices(str(args.gammex_frames_along), int(along_par.num_frames)),
            )
            for idx in _parse_indices(str(args.gammex_frames_along), int(along_par.num_frames)):
                frame = read_rawbcf_frame(along_dir / "RawBCFCine.dat", along_par, int(idx))
                icube = decode_rawbcf_cfm_cube(frame, along_par, order="beam_major")
                rows.append(
                    summarize_iq_anchor(
                        anchor_kind="gammex_along",
                        case_key=f"gammex_along::frame{int(idx):03d}",
                        icube=icube,
                        prf_hz=float(args.gammex_prf_hz),
                        tile=tile,
                        bands=bands,
                        mask_flow=along_tube.mask_flow,
                        mask_bg=along_tube.mask_bg,
                        derive_masks=False,
                    )
                )

            across_dir = gammex_root / "Flow in Gammex phantom (across - linear probe)"
            across_par = RawBCFPar.from_dict(parse_rawbcf_par(across_dir / str(args.gammex_across_par)))
            across_par.validate()
            across_dat = across_dir / str(args.gammex_across_dat)
            across_tube = build_tube_masks_from_rawbcf(
                across_dat,
                across_par,
                ref_frame_indices=_parse_indices(str(args.gammex_frames_across), int(across_par.num_frames)),
            )
            for idx in _parse_indices(str(args.gammex_frames_across), int(across_par.num_frames)):
                frame = read_rawbcf_frame(across_dat, across_par, int(idx))
                icube = decode_rawbcf_cfm_cube(frame, across_par, order="beam_major")
                rows.append(
                    summarize_iq_anchor(
                        anchor_kind="gammex_across",
                        case_key=f"gammex_across::frame{int(idx):03d}",
                        icube=icube,
                        prf_hz=float(args.gammex_prf_hz),
                        tile=tile,
                        bands=bands,
                        mask_flow=across_tube.mask_flow,
                        mask_bg=across_tube.mask_bg,
                        derive_masks=False,
                    )
                )
        else:
            skipped["gammex"] = f"missing root: {gammex_root}"

    if not bool(args.skip_ulm):
        ulm_root = Path(args.ulm_root) if args.ulm_root is not None else ulm_zenodo_7883227_root()
        zip_path = ulm_root / "IQ_001_to_025.zip"
        if ulm_root.is_dir() and zip_path.is_file():
            blocks = _parse_indices(str(args.ulm_blocks), 9999)
            for block_id in blocks:
                icube = load_ulm_block_iq(
                    int(block_id),
                    frames=slice(*[int(x) if x else None for x in (str(args.ulm_frames).split(":") + ["", ""])[:3]]) if ":" in str(args.ulm_frames) else _parse_indices(str(args.ulm_frames), 9999),
                    root=ulm_root,
                    cache_dir=Path(args.ulm_cache_dir),
                )
                rows.append(
                    summarize_iq_anchor(
                        anchor_kind="ulm_7883227",
                        case_key=f"ulm_block{int(block_id):03d}::{str(args.ulm_frames)}",
                        icube=icube,
                        prf_hz=float(args.ulm_prf_hz),
                        tile=tile,
                        bands=bands,
                        mask_flow=None,
                        mask_bg=None,
                        derive_masks=True,
                    )
                )
        else:
            skipped["ulm_7883227"] = f"missing root or zip: {ulm_root}"

    if not bool(args.skip_mace):
        mace_root = mace_data_root()
        if mace_root.is_dir():
            mace_rows = summarize_mace_phase2_report(Path(args.mace_report_csv))
            if mace_rows:
                rows.extend(mace_rows)
            else:
                skipped["mace_phase2"] = f"missing report: {args.mace_report_csv}"
        else:
            skipped["mace_phase2"] = f"missing root: {mace_root}"

    if not rows:
        raise SystemExit("no anchor rows generated")

    payload = {
        "schema_version": "simus_v2_anchor_envelopes.v1",
        "bands_hz": {
            "Pf": [float(bands.pf_lo_hz), float(bands.pf_hi_hz)],
            "Pg": [float(bands.pg_lo_hz), float(bands.pg_hi_hz)],
            "Pa": [float(bands.pa_lo_hz), None],
        },
        "tile": {"h": int(tile.h), "w": int(tile.w), "stride": int(tile.stride)},
        "iq_metrics": list(IQ_METRICS),
        "functional_metrics": list(FUNCTIONAL_METRICS),
        "default_acceptance_metrics": list(DEFAULT_ACCEPTANCE_METRICS),
        "rows": rows,
        "envelopes": build_envelopes(rows),
        "skipped": skipped,
        "mace_note": "Macé rows are exported as functional_readout anchors only and are not part of the hard IQ/motion acceptance gate.",
    }
    _write_csv(out_csv, rows)
    _write_json(out_json, payload)
    print(f"[simus-v2-anchor-envelopes] wrote {out_csv}")
    print(f"[simus-v2-anchor-envelopes] wrote {out_json}")


if __name__ == "__main__":
    main()
