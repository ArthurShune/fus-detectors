#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


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


_SIM_KEY_RE = re.compile(
    r"^sim_simus_(?P<profile>.+?)_motionx(?P<motion>[0-9p]+)_phasex(?P<phase>[0-9p]+)_seed(?P<seed>\d+)$"
)


def _decode_scale(text: str) -> float:
    return float(str(text).replace("p", "."))


def parse_sim_key(sim_key: str) -> dict[str, Any]:
    m = _SIM_KEY_RE.match(str(sim_key))
    if not m:
        raise ValueError(f"Unrecognized sim key: {sim_key!r}")
    return {
        "sim_key": str(sim_key),
        "profile_slug": str(m.group("profile")),
        "motion_scale": _decode_scale(m.group("motion")),
        "phase_scale": _decode_scale(m.group("phase")),
        "seed": int(m.group("seed")),
    }


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value: Any) -> float | None:
    if value in (None, "", "None", "nan", "NaN"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def summarize_calibration(
    *,
    table_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sim_meta: dict[str, dict[str, Any]] = {}
    for row in table_rows:
        if row.get("kind") != "sim":
            continue
        parsed = parse_sim_key(str(row["key"]))
        sim_meta[str(row["key"])] = {
            **parsed,
            "motion_disp_rms_px": _to_float(row.get("motion_disp_rms_px")),
            "phase_rms_rad": _to_float(row.get("phase_rms_rad")),
            "flow_malias_q50": _to_float(row.get("flow_malias_q50")),
            "bg_malias_q50": _to_float(row.get("bg_malias_q50")),
            "flow_fpeak_q50": _to_float(row.get("flow_fpeak_q50")),
            "bg_fpeak_q50": _to_float(row.get("bg_fpeak_q50")),
            "flow_coh1_q50": _to_float(row.get("flow_coh1_q50")),
            "bg_coh1_q50": _to_float(row.get("bg_coh1_q50")),
        }

    rows_out: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "schema_version": "simus_motion_calibration.v1",
        "sim_keys": sorted(sim_meta),
    }

    # Best reference for each SIM run.
    best_ref_by_sim: dict[str, tuple[float, dict[str, Any]]] = {}
    # Best SIM run for each reference.
    best_sim_by_ref: dict[str, tuple[float, dict[str, Any]]] = {}
    # Mean delta grouped by (profile_slug, ref_kind, motion_scale).
    grouped: dict[tuple[str, str, float], list[tuple[float, dict[str, Any]]]] = {}

    for row in delta_rows:
        sim_key = str(row["sim_key"])
        ref_key = str(row["ref_key"])
        ref_kind = str(row["ref_kind"])
        mad = _to_float(row.get("mean_abs_delta_selected"))
        if mad is None or sim_key not in sim_meta:
            continue
        payload = {
            **sim_meta[sim_key],
            "ref_key": ref_key,
            "ref_kind": ref_kind,
            "mean_abs_delta_selected": mad,
        }
        if sim_key not in best_ref_by_sim or mad < best_ref_by_sim[sim_key][0]:
            best_ref_by_sim[sim_key] = (mad, payload)
        if ref_key not in best_sim_by_ref or mad < best_sim_by_ref[ref_key][0]:
            best_sim_by_ref[ref_key] = (mad, payload)
        grouped.setdefault((payload["profile_slug"], ref_kind, payload["motion_scale"]), []).append((mad, payload))

    for sim_key, (_, payload) in sorted(best_ref_by_sim.items(), key=lambda kv: kv[0]):
        rows_out.append({"summary_type": "sim_best_ref", **payload})

    for ref_key, (_, payload) in sorted(best_sim_by_ref.items(), key=lambda kv: kv[0]):
        rows_out.append({"summary_type": "ref_best_sim", **payload})

    profile_refkind_best: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
    profile_refkind_curve: dict[str, list[dict[str, Any]]] = {}
    for (profile_slug, ref_kind, motion_scale), items in sorted(grouped.items()):
        vals = [mad for mad, _ in items]
        base = items[0][1]
        payload = {
            "summary_type": "profile_refkind_curve",
            "profile_slug": profile_slug,
            "ref_kind": ref_kind,
            "motion_scale": motion_scale,
            "phase_scale": base["phase_scale"],
            "seed": base["seed"],
            "motion_disp_rms_px": base["motion_disp_rms_px"],
            "phase_rms_rad": base["phase_rms_rad"],
            "mean_abs_delta_selected": sum(vals) / float(len(vals)),
            "n_refs": len(vals),
        }
        rows_out.append(payload)
        profile_refkind_curve.setdefault(f"{profile_slug}:{ref_kind}", []).append(payload)
        key = (profile_slug, ref_kind)
        if key not in profile_refkind_best or payload["mean_abs_delta_selected"] < profile_refkind_best[key][0]:
            profile_refkind_best[key] = (float(payload["mean_abs_delta_selected"]), payload)

    details["best_by_profile_ref_kind"] = {
        f"{profile}:{ref_kind}": payload for (profile, ref_kind), (_, payload) in sorted(profile_refkind_best.items())
    }
    details["curves_by_profile_ref_kind"] = profile_refkind_curve
    return rows_out, details


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize which SIMUS motion scales best match Shin/Gammex telemetry.")
    ap.add_argument(
        "--table-csv",
        type=Path,
        default=Path("reports/simus_sanity_link/phase4_motion_ladders_seed21_table.csv"),
    )
    ap.add_argument(
        "--deltas-csv",
        type=Path,
        default=Path("reports/simus_sanity_link/phase4_motion_ladders_seed21_deltas.csv"),
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/simus_motion/simus_phase4_calibration_summary.csv"),
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/simus_motion/simus_phase4_calibration_summary.json"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    table_rows = _read_csv(Path(args.table_csv))
    delta_rows = _read_csv(Path(args.deltas_csv))
    rows, details = summarize_calibration(table_rows=table_rows, delta_rows=delta_rows)
    _write_csv(Path(args.out_csv), rows)
    _write_json(Path(args.out_json), {"rows": rows, "details": details})
    print(f"[simus-motion-calibration] wrote {args.out_csv}")
    print(f"[simus-motion-calibration] wrote {args.out_json}")


if __name__ == "__main__":
    main()
