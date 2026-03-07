#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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
from sim.simus.bundle import estimate_simus_policy_features


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


def summarize_rows(rows: list[dict[str, Any]], *, threshold: float) -> dict[str, Any]:
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_kind.setdefault(str(row["kind"]), []).append(row)

    out: dict[str, Any] = {}
    for kind, items in sorted(by_kind.items()):
        reg_p90 = np.asarray([float(r["reg_shift_p90"]) for r in items], dtype=np.float64)
        reg_rms = np.asarray([float(r["reg_shift_rms"]) for r in items], dtype=np.float64)
        psr = np.asarray([float(r["reg_psr_median"]) for r in items], dtype=np.float64)
        out[kind] = {
            "n": int(len(items)),
            "reg_shift_p90_median": float(np.median(reg_p90)),
            "reg_shift_p90_max": float(np.max(reg_p90)),
            "reg_shift_rms_median": float(np.median(reg_rms)),
            "reg_psr_median_q50": float(np.median(psr)),
            "fraction_reg_shift_p90_gt_threshold": float(np.mean(reg_p90 > float(threshold))),
        }

    all_reg_p90 = np.asarray([float(r["reg_shift_p90"]) for r in rows], dtype=np.float64) if rows else np.zeros((0,), dtype=np.float64)
    all_reg_rms = np.asarray([float(r["reg_shift_rms"]) for r in rows], dtype=np.float64) if rows else np.zeros((0,), dtype=np.float64)
    overall = {
        "n": int(len(rows)),
        "threshold": float(threshold),
        "reg_shift_p90_max": float(np.max(all_reg_p90)) if all_reg_p90.size else None,
        "reg_shift_p90_q95": float(np.quantile(all_reg_p90, 0.95)) if all_reg_p90.size else None,
        "reg_shift_rms_max": float(np.max(all_reg_rms)) if all_reg_rms.size else None,
    }
    return {"by_kind": out, "overall": overall}


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Measure real-data motion-proxy telemetry (reg_shift_rms/p90, reg_psr_median) on Shin and Gammex IQ.")
    ap.add_argument("--shin-root", type=Path, default=None)
    ap.add_argument("--shin-iq-file", type=str, default="IQData001.dat")
    ap.add_argument("--shin-frames", type=str, default="0:128")
    ap.add_argument("--gammex-root", type=Path, default=None)
    ap.add_argument("--gammex-along-frames", type=str, default="0:6")
    ap.add_argument("--gammex-across-frames", type=str, default="0:6")
    ap.add_argument("--gammex-across-par", type=str, default="RawBCFCine_08062017_145434_17.par")
    ap.add_argument("--gammex-across-dat", type=str, default="RawBCFCine_08062017_145434_17.dat")
    ap.add_argument("--threshold", type=float, default=2.194)
    ap.add_argument("--out-csv", type=Path, default=Path("reports/simus_sanity_link/real_motion_proxy_telemetry.csv"))
    ap.add_argument("--out-json", type=Path, default=Path("reports/simus_sanity_link/real_motion_proxy_telemetry.json"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []

    shin_root = Path(args.shin_root) if args.shin_root is not None else shin_ratbrain_data_root()
    if shin_root.is_dir():
        info = load_shin_metadata(shin_root)
        frames = _parse_indices(str(args.shin_frames), int(info.frames))
        iq = load_shin_iq(shin_root / str(args.shin_iq_file), info, frames=frames)
        feat = estimate_simus_policy_features(iq, reg_subpixel=4, reg_reference="median")
        rows.append(
            {
                "kind": "shin",
                "case_key": f"shin_{Path(args.shin_iq_file).stem}_{str(args.shin_frames).replace(':', '_')}",
                "n_frames": int(iq.shape[0]),
                "reg_shift_rms": float(feat["reg_shift_rms"]),
                "reg_shift_p90": float(feat["reg_shift_p90"]),
                "reg_psr_median": float(feat["reg_psr_median"]),
                "reg_psr_p10": float(feat["reg_psr_p10"]),
                "reg_psr_p90": float(feat["reg_psr_p90"]),
                "reg_ms_prepass": float(feat["reg_ms_prepass"]),
            }
        )

    gammex_root = Path(args.gammex_root) if args.gammex_root is not None else twinkling_artifact_data_root() / "Flow in Gammex phantom"
    if gammex_root.is_dir():
        along_dir = gammex_root / "Flow in Gammex phantom (along - linear probe)"
        along_par = RawBCFPar.from_dict(parse_rawbcf_par(along_dir / "RawBCFCine.par"))
        along_par.validate()
        for idx in _parse_indices(str(args.gammex_along_frames), int(along_par.num_frames)):
            frame = read_rawbcf_frame(along_dir / "RawBCFCine.dat", along_par, int(idx))
            icube = decode_rawbcf_cfm_cube(frame, along_par, order="beam_major")
            feat = estimate_simus_policy_features(icube, reg_subpixel=4, reg_reference="median")
            rows.append(
                {
                    "kind": "gammex_along",
                    "case_key": f"gammex_along_frame{int(idx):03d}",
                    "n_frames": int(icube.shape[0]),
                    "reg_shift_rms": float(feat["reg_shift_rms"]),
                    "reg_shift_p90": float(feat["reg_shift_p90"]),
                    "reg_psr_median": float(feat["reg_psr_median"]),
                    "reg_psr_p10": float(feat["reg_psr_p10"]),
                    "reg_psr_p90": float(feat["reg_psr_p90"]),
                    "reg_ms_prepass": float(feat["reg_ms_prepass"]),
                }
            )

        across_dir = gammex_root / "Flow in Gammex phantom (across - linear probe)"
        across_par = RawBCFPar.from_dict(parse_rawbcf_par(across_dir / str(args.gammex_across_par)))
        across_par.validate()
        across_dat = across_dir / str(args.gammex_across_dat)
        for idx in _parse_indices(str(args.gammex_across_frames), int(across_par.num_frames)):
            frame = read_rawbcf_frame(across_dat, across_par, int(idx))
            icube = decode_rawbcf_cfm_cube(frame, across_par, order="beam_major")
            feat = estimate_simus_policy_features(icube, reg_subpixel=4, reg_reference="median")
            rows.append(
                {
                    "kind": "gammex_across",
                    "case_key": f"gammex_across_frame{int(idx):03d}",
                    "n_frames": int(icube.shape[0]),
                    "reg_shift_rms": float(feat["reg_shift_rms"]),
                    "reg_shift_p90": float(feat["reg_shift_p90"]),
                    "reg_psr_median": float(feat["reg_psr_median"]),
                    "reg_psr_p10": float(feat["reg_psr_p10"]),
                    "reg_psr_p90": float(feat["reg_psr_p90"]),
                    "reg_ms_prepass": float(feat["reg_ms_prepass"]),
                }
            )

    payload = {
        "schema_version": "real_motion_proxy_telemetry.v1",
        "threshold": float(args.threshold),
        "rows": rows,
    }
    payload.update(summarize_rows(rows, threshold=float(args.threshold)))
    _write_csv(Path(args.out_csv), rows)
    _write_json(Path(args.out_json), payload)
    print(f"[real-motion-proxy-telemetry] wrote {args.out_csv}")
    print(f"[real-motion-proxy-telemetry] wrote {args.out_json}")


if __name__ == "__main__":
    main()
