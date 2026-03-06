#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.realdata import shin_ratbrain_data_root
from pipeline.realdata.shin_ratbrain import load_shin_iq, load_shin_metadata
from pipeline.realdata.twinkling_artifact import (
    RawBCFPar,
    decode_rawbcf_cfm_cube,
    parse_rawbcf_par,
    read_rawbcf_frame,
    twinkling_artifact_data_root,
)
from pipeline.realdata.twinkling_bmode_mask import build_tube_masks_from_rawbcf
from scripts.physical_doppler_sanity_link import BandEdges, TileSpec, summarize_icube
from sim.simus.bundle import load_canonical_run


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


def _slugify(text: str) -> str:
    s = re.sub(r"[\s/]+", "_", str(text).strip())
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s.strip("._-") or "run"


def _parse_indices(spec: str, n_max: int) -> list[int]:
    spec = (spec or "").strip()
    if not spec:
        return list(range(n_max))
    if "," in spec:
        return [int(part.strip()) for part in spec.split(",") if part.strip()]
    if ":" in spec:
        parts = [p.strip() for p in spec.split(":")]
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid slice spec: {spec!r}")
        start = int(parts[0]) if parts[0] else 0
        stop = int(parts[1]) if parts[1] else n_max
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        return list(range(start, min(stop, n_max), step))
    return [int(spec)]


def _discover_sim_runs(root: Path) -> list[Path]:
    root = Path(root)
    if (root / "dataset" / "meta.json").is_file():
        return [root]
    return [p for p in sorted(root.iterdir()) if p.is_dir() and (p / "dataset" / "meta.json").is_file()]


def flatten_summary_row(key: str, report: dict[str, Any], *, kind: str, motion_disp_rms_px: float | None = None, phase_rms_rad: float | None = None) -> dict[str, Any]:
    meta = report.get("meta") or {}
    flow = (report.get("summary") or {}).get("flow") or {}
    bg = (report.get("summary") or {}).get("bg") or {}
    svd = report.get("svd") or {}
    return {
        "key": str(key),
        "kind": str(kind),
        "T": int(meta.get("shape", [0])[0]) if isinstance(meta.get("shape"), list) else None,
        "H": int(meta.get("shape", [0, 0])[1]) if isinstance(meta.get("shape"), list) else None,
        "W": int(meta.get("shape", [0, 0, 0])[2]) if isinstance(meta.get("shape"), list) else None,
        "prf_hz": meta.get("prf_hz"),
        "flow_tiles": flow.get("n_tiles") if isinstance(flow, dict) else None,
        "bg_tiles": bg.get("n_tiles") if isinstance(bg, dict) else None,
        "flow_malias_q50": ((flow.get("malias") or {}).get("q50") if isinstance(flow, dict) else None),
        "bg_malias_q50": ((bg.get("malias") or {}).get("q50") if isinstance(bg, dict) else None),
        "flow_fpeak_q50": ((flow.get("fpeak_hz") or {}).get("q50") if isinstance(flow, dict) else None),
        "bg_fpeak_q50": ((bg.get("fpeak_hz") or {}).get("q50") if isinstance(bg, dict) else None),
        "flow_coh1_q50": ((flow.get("coh1") or {}).get("q50") if isinstance(flow, dict) else None),
        "bg_coh1_q50": ((bg.get("coh1") or {}).get("q50") if isinstance(bg, dict) else None),
        "svd_flow_cum_r1": ((svd.get("flow") or {}).get("cum_r1") if isinstance(svd, dict) else None),
        "svd_flow_cum_r2": ((svd.get("flow") or {}).get("cum_r2") if isinstance(svd, dict) else None),
        "svd_bg_cum_r1": ((svd.get("bg") or {}).get("cum_r1") if isinstance(svd, dict) else None),
        "svd_bg_cum_r2": ((svd.get("bg") or {}).get("cum_r2") if isinstance(svd, dict) else None),
        "motion_disp_rms_px": motion_disp_rms_px,
        "phase_rms_rad": phase_rms_rad,
    }


def build_delta_rows(rows: list[dict[str, Any]], *, metrics: list[str]) -> list[dict[str, Any]]:
    sim_rows = [r for r in rows if r.get("kind") == "sim"]
    ref_rows = [r for r in rows if r.get("kind") != "sim"]
    out: list[dict[str, Any]] = []
    for sim in sim_rows:
        for ref in ref_rows:
            row: dict[str, Any] = {
                "sim_key": sim["key"],
                "ref_key": ref["key"],
                "ref_kind": ref["kind"],
            }
            deltas: list[float] = []
            for metric in metrics:
                sv = sim.get(metric)
                rv = ref.get(metric)
                if sv is None or rv is None:
                    row[f"delta_{metric}"] = None
                    continue
                try:
                    delta = float(sv) - float(rv)
                except Exception:
                    row[f"delta_{metric}"] = None
                    continue
                row[f"delta_{metric}"] = delta
                deltas.append(abs(delta))
            row["mean_abs_delta_selected"] = float(np.mean(deltas)) if deltas else None
            out.append(row)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SIMUS sanity-link telemetry against Shin and Gammex real IQ.")
    ap.add_argument("--sim-root", type=Path, action="append", default=None, help="SIMUS run root or a root of run subdirectories.")
    ap.add_argument("--run", type=Path, action="append", default=None, help="Explicit SIMUS run directory.")
    ap.add_argument("--out-dir", type=Path, default=Path("reports/simus_sanity_link"))
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--pf", type=float, nargs=2, default=(30.0, 250.0))
    ap.add_argument("--pg", type=float, nargs=2, default=(250.0, 400.0))
    ap.add_argument("--pa-lo", type=float, default=400.0)
    ap.add_argument("--tile-hw", type=int, nargs=2, default=(8, 8))
    ap.add_argument("--tile-stride", type=int, default=3)
    ap.add_argument("--shin-root", type=Path, default=None)
    ap.add_argument("--shin-iq-file", type=str, default="IQData001.dat")
    ap.add_argument("--shin-frames", type=str, default="0:128")
    ap.add_argument("--shin-prf-hz", type=float, default=1000.0)
    ap.add_argument("--gammex-root", type=Path, default=None)
    ap.add_argument("--gammex-frames-along", type=str, default="0")
    ap.add_argument("--gammex-frames-across", type=str, default="0")
    ap.add_argument("--gammex-prf-hz", type=float, default=2500.0)
    ap.add_argument("--gammex-mask-ref-frames", type=str, default="0:6")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bands = BandEdges(
        pf_lo_hz=float(args.pf[0]),
        pf_hi_hz=float(args.pf[1]),
        pg_lo_hz=float(args.pg[0]),
        pg_hi_hz=float(args.pg[1]),
        pa_lo_hz=float(args.pa_lo),
    )
    tile = TileSpec(h=int(args.tile_hw[0]), w=int(args.tile_hw[1]), stride=int(args.tile_stride))

    summaries: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    tag_parts: list[str] = []

    sim_runs: list[Path] = []
    for root in args.sim_root or []:
        sim_runs.extend(_discover_sim_runs(Path(root)))
    for run in args.run or []:
        sim_runs.append(Path(run))
    seen: set[str] = set()
    for run_dir in sim_runs:
        key = str(Path(run_dir).resolve())
        if key in seen:
            continue
        seen.add(key)
        icube, masks, meta = load_canonical_run(Path(run_dir))
        report = summarize_icube(
            name=Path(run_dir).name,
            Icube=icube,
            prf_hz=float(meta.get("acquisition", {}).get("prf_hz", meta.get("config", {}).get("prf_hz", 0.0))),
            tile=tile,
            bands=bands,
            mask_flow=masks.get("mask_flow"),
            mask_bg=masks.get("mask_bg"),
            derive_masks=False,
            derive_vessel_q=0.99,
            derive_bg_q=0.20,
        )
        name = f"sim_{Path(run_dir).name}"
        summaries[name] = report
        rows.append(
            flatten_summary_row(
                name,
                report,
                kind="sim",
                motion_disp_rms_px=meta.get("motion", {}).get("telemetry", {}).get("disp_rms_px"),
                phase_rms_rad=meta.get("phase_screen", {}).get("telemetry", {}).get("phase_rms_rad"),
            )
        )
        tag_parts.append(name)

    shin_root = Path(args.shin_root) if args.shin_root is not None else shin_ratbrain_data_root()
    if shin_root.is_dir():
        info = load_shin_metadata(shin_root)
        frames = _parse_indices(str(args.shin_frames), int(info.frames))
        iq = load_shin_iq(shin_root / str(args.shin_iq_file), info, frames=frames)
        name = f"shin_{_slugify(Path(args.shin_iq_file).stem)}_{_slugify(args.shin_frames)}"
        report = summarize_icube(
            name=name,
            Icube=iq,
            prf_hz=float(args.shin_prf_hz),
            tile=tile,
            bands=bands,
            mask_flow=None,
            mask_bg=None,
            derive_masks=True,
            derive_vessel_q=0.99,
            derive_bg_q=0.20,
        )
        summaries[name] = report
        rows.append(flatten_summary_row(name, report, kind="shin"))
        tag_parts.append(name)

    gammex_root = Path(args.gammex_root) if args.gammex_root is not None else twinkling_artifact_data_root() / "Flow in Gammex phantom"
    if gammex_root.is_dir():
        along_dir = gammex_root / "Flow in Gammex phantom (along - linear probe)"
        across_dir = gammex_root / "Flow in Gammex phantom (across - linear probe)"

        def _run_gammex(name: str, par_path: Path, dat_path: Path, frames_spec: str) -> None:
            par = RawBCFPar.from_dict(parse_rawbcf_par(par_path))
            par.validate()
            ref_frames = _parse_indices(str(args.gammex_mask_ref_frames), int(par.num_frames))
            tube = build_tube_masks_from_rawbcf(dat_path, par, ref_frame_indices=ref_frames)
            for idx in _parse_indices(frames_spec, int(par.num_frames)):
                frame = read_rawbcf_frame(dat_path, par, int(idx))
                icube = decode_rawbcf_cfm_cube(frame, par, order="beam_major")
                key = f"{name}_frame{int(idx):03d}"
                report = summarize_icube(
                    name=key,
                    Icube=icube,
                    prf_hz=float(args.gammex_prf_hz),
                    tile=tile,
                    bands=bands,
                    mask_flow=tube.mask_flow,
                    mask_bg=tube.mask_bg,
                    derive_masks=False,
                    derive_vessel_q=0.99,
                    derive_bg_q=0.20,
                )
                summaries[key] = report
                rows.append(flatten_summary_row(key, report, kind="gammex"))
                tag_parts.append(key)

        _run_gammex(
            "gammex_along",
            along_dir / "RawBCFCine.par",
            along_dir / "RawBCFCine.dat",
            str(args.gammex_frames_along),
        )
        _run_gammex(
            "gammex_across",
            across_dir / "RawBCFCine_08062017_145434_17.par",
            across_dir / "RawBCFCine_08062017_145434_17.dat",
            str(args.gammex_frames_across),
        )

    if not summaries:
        raise SystemExit("No SIMUS or real-IQ inputs found.")

    selected_metrics = [
        "flow_malias_q50",
        "bg_malias_q50",
        "flow_fpeak_q50",
        "bg_fpeak_q50",
        "flow_coh1_q50",
        "bg_coh1_q50",
        "svd_flow_cum_r1",
        "svd_bg_cum_r1",
    ]
    delta_rows = build_delta_rows(rows, metrics=selected_metrics)
    tag = str(args.tag).strip() if args.tag else _slugify("__".join(tag_parts)[:120])
    _write_json(out_dir / f"{tag}_summary.json", {"schema_version": "simus_sanity_link.v1", "summaries": summaries, "table_rows": rows, "delta_rows": delta_rows})
    _write_csv(out_dir / f"{tag}_table.csv", rows)
    _write_csv(out_dir / f"{tag}_deltas.csv", delta_rows)
    _write_json(out_dir / f"{tag}_table.json", {"rows": rows})
    _write_json(out_dir / f"{tag}_deltas.json", {"rows": delta_rows})
    print(f"[simus-sanity-link] wrote {out_dir / f'{tag}_summary.json'}")
    print(f"[simus-sanity-link] wrote {out_dir / f'{tag}_table.csv'}")
    print(f"[simus-sanity-link] wrote {out_dir / f'{tag}_deltas.csv'}")


if __name__ == "__main__":
    main()
