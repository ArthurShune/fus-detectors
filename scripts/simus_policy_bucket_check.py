#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


FEATURES = ("flow_fpeak_q50", "bg_fpeak_q50", "flow_coh1_q50", "bg_coh1_q50")


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


def _safe_float(x: Any) -> float | None:
    if x in (None, ""):
        return None
    return float(x)


def _norm_case_name(text: str) -> str:
    s = str(text).split("|")[0].lower().replace("+", "_").replace("-", "_")
    return re.sub(r"[^a-z0-9_]+", "_", s)


def _find_matching_sim_row(case_key: str, seed: int, sim_rows: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    want = _norm_case_name(case_key)
    for key, row in sim_rows.items():
        if f"seed{seed}" in key and want in key:
            return row
    return None


def _dist(sim_row: dict[str, Any], real_row: dict[str, Any]) -> float:
    total = 0.0
    used = 0
    for feat in FEATURES:
        a = _safe_float(sim_row.get(feat))
        b = _safe_float(real_row.get(feat))
        if a is None or b is None:
            continue
        scale = 1000.0 if "fpeak" in feat else 1.0
        total += ((a - b) / scale) ** 2
        used += 1
    return math.sqrt(total) if used else float("inf")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare SIMUS policy-selected cases to existing real-data telemetry buckets.")
    ap.add_argument("--policy-csv", type=Path, required=True)
    ap.add_argument("--sanity-table-csv", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    policy_rows = list(csv.DictReader(Path(args.policy_csv).open()))
    table_rows = list(csv.DictReader(Path(args.sanity_table_csv).open()))
    sim_rows = {str(r["key"]): r for r in table_rows if str(r.get("kind")) == "sim"}
    real_rows = [r for r in table_rows if str(r.get("kind")) in {"shin", "gammex"}]

    out_rows: list[dict[str, Any]] = []
    for row in policy_rows:
        if str(row["seed"]) != "21":
            continue
        sim_row = _find_matching_sim_row(str(row["case_key"]), int(row["seed"]), sim_rows)
        if sim_row is None:
            continue
        distances = sorted(
            (
                {
                    "real_key": str(real_row["key"]),
                    "real_kind": str(real_row["kind"]),
                    "distance": _dist(sim_row, real_row),
                }
                for real_row in real_rows
            ),
            key=lambda x: float(x["distance"]),
        )
        top3 = distances[:3]
        out_rows.append(
            {
                "case_key": row["case_key"],
                "policy_profile": row["policy_profile"],
                "motion_scale": row["motion_scale"],
                "motion_disp_rms_px": row["motion_disp_rms_px"],
                "nearest_real_key": top3[0]["real_key"],
                "nearest_real_kind": top3[0]["real_kind"],
                "nearest_distance": top3[0]["distance"],
                "top3": json.dumps(top3, sort_keys=True),
            }
        )

    summary = {
        "schema_version": "simus_policy_bucket_check.v1",
        "rows": out_rows,
        "nearest_kind_counts": {},
    }
    for row in out_rows:
        kind = str(row["nearest_real_kind"])
        summary["nearest_kind_counts"][kind] = summary["nearest_kind_counts"].get(kind, 0) + 1

    _write_csv(Path(args.out_csv), out_rows)
    _write_json(Path(args.out_json), summary)
    print(f"[simus-policy-bucket-check] wrote {args.out_csv}")
    print(f"[simus-policy-bucket-check] wrote {args.out_json}")


if __name__ == "__main__":
    main()
