#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _safe_float(x: Any) -> float | None:
    if x in (None, ""):
        return None
    return float(x)


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


def summarize_rows(rows: list[dict[str, Any]], *, real_max: float, real_q95: float | None) -> dict[str, Any]:
    vals = np.asarray([float(r["reg_shift_p90"]) for r in rows], dtype=np.float64) if rows else np.zeros((0,), dtype=np.float64)
    by_scale: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scale.setdefault(str(row["motion_scale"]), []).append(row)
    summary_by_scale: dict[str, Any] = {}
    for scale, items in sorted(by_scale.items(), key=lambda kv: float(kv[0])):
        hits = [bool(r["within_real_reg_shift_envelope"]) for r in items]
        summary_by_scale[scale] = {
            "n": int(len(items)),
            "fraction_within_real_reg_shift_envelope": float(np.mean(hits)) if hits else None,
            "reg_shift_p90_min": float(np.min([float(r["reg_shift_p90"]) for r in items])),
            "reg_shift_p90_max": float(np.max([float(r["reg_shift_p90"]) for r in items])),
        }
    return {
        "real_reg_shift_p90_max": float(real_max),
        "real_reg_shift_p90_q95": float(real_q95) if real_q95 is not None else None,
        "simus_n_cases": int(len(rows)),
        "simus_reg_shift_p90_min": float(np.min(vals)) if vals.size else None,
        "simus_reg_shift_p90_max": float(np.max(vals)) if vals.size else None,
        "fraction_within_real_reg_shift_envelope": float(np.mean([bool(r["within_real_reg_shift_envelope"]) for r in rows])) if rows else None,
        "summary_by_motion_scale": summary_by_scale,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Check whether SIMUS motion-search cases fall inside the measured real-data reg_shift_p90 envelope.")
    ap.add_argument("--search-csv", type=Path, required=True)
    ap.add_argument("--real-telemetry-json", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    search_rows = list(csv.DictReader(Path(args.search_csv).open()))
    real_payload = json.loads(Path(args.real_telemetry_json).read_text(encoding="utf-8"))
    overall = real_payload.get("overall") or {}
    real_max = _safe_float(overall.get("reg_shift_p90_max"))
    real_q95 = _safe_float(overall.get("reg_shift_p90_q95"))
    if real_max is None:
        raise ValueError(f"{args.real_telemetry_json}: missing overall.reg_shift_p90_max")

    by_case: dict[str, dict[str, Any]] = {}
    for row in search_rows:
        if str(row.get("role")) != "baseline":
            continue
        case_key = str(row["case_key"])
        reg_shift_p90 = _safe_float(row.get("reg_shift_p90"))
        if reg_shift_p90 is None:
            continue
        by_case[case_key] = {
            "case_key": case_key,
            "simus_profile": row.get("simus_profile"),
            "seed": int(row["seed"]),
            "motion_scale": float(row["motion_scale"]),
            "reg_shift_p90": reg_shift_p90,
            "reg_shift_rms": _safe_float(row.get("reg_shift_rms")),
            "motion_disp_rms_px": _safe_float(row.get("motion_disp_rms_px")),
            "within_real_reg_shift_envelope": bool(reg_shift_p90 <= float(real_max)),
            "margin_to_real_reg_shift_max": float(reg_shift_p90 - float(real_max)),
        }

    rows = list(sorted(by_case.values(), key=lambda r: (str(r["simus_profile"]), int(r["seed"]), float(r["motion_scale"]))))
    payload = {
        "schema_version": "simus_real_envelope_check.v1",
        "rows": rows,
        "summary": summarize_rows(rows, real_max=float(real_max), real_q95=real_q95),
    }
    _write_csv(Path(args.out_csv), rows)
    _write_json(Path(args.out_json), payload)
    print(f"[simus-real-envelope-check] wrote {args.out_csv}")
    print(f"[simus-real-envelope-check] wrote {args.out_json}")


if __name__ == "__main__":
    main()
