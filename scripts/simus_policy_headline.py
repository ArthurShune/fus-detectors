#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from sim.simus.bundle import select_simus_stap_profile


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


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize a frozen SIMUS STAP policy from a compromise-search CSV.")
    ap.add_argument("--search-csv", type=Path, required=True)
    ap.add_argument("--policy", type=str, default="Brain-SIMUS-Clin-MotionDisp-v0")
    ap.add_argument("--requested-profile", type=str, default="Brain-SIMUS-Clin")
    ap.add_argument("--current-profile", type=str, default="Brain-SIMUS-Clin")
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(Path(args.search_csv).open()))
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_key"]), []).append(row)

    out_rows: list[dict[str, Any]] = []
    summary_groups: dict[str, list[dict[str, Any]]] = {}
    for case_key, items in sorted(by_case.items()):
        baseline = next(r for r in items if r["role"] == "baseline")
        candidates = {str(r["stap_profile"]): r for r in items if r["role"] == "stap"}
        motion_disp = _safe_float(baseline.get("motion_disp_rms_px"))
        feature_values = {
            "motion_disp_rms_px": motion_disp,
            "reg_shift_rms": _safe_float(baseline.get("reg_shift_rms")),
            "reg_shift_p90": _safe_float(baseline.get("reg_shift_p90")),
            "reg_psr_median": _safe_float(baseline.get("reg_psr_median")),
        }
        applied_profile, policy_info = select_simus_stap_profile(
            requested_profile=str(args.requested_profile),
            policy=str(args.policy),
            feature_values=feature_values,
            motion_disp_rms_px=motion_disp,
        )
        current = candidates[str(args.current_profile)]
        selected = candidates[applied_profile]

        row = {
            "case_key": case_key,
            "seed": int(baseline["seed"]),
            "simus_profile": baseline["simus_profile"],
            "motion_scale": float(baseline["motion_scale"]),
            "motion_disp_rms_px": motion_disp,
            "policy": str(args.policy),
            "requested_profile": str(args.requested_profile),
            "current_profile": str(args.current_profile),
            "policy_profile": applied_profile,
            "current_auc_main_vs_bg": current["auc_main_vs_bg"],
            "current_auc_main_vs_nuisance": current["auc_main_vs_nuisance"],
            "current_fpr_nuisance_match@0p5": current["fpr_nuisance_match@0p5"],
            "policy_auc_main_vs_bg": selected["auc_main_vs_bg"],
            "policy_auc_main_vs_nuisance": selected["auc_main_vs_nuisance"],
            "policy_fpr_nuisance_match@0p5": selected["fpr_nuisance_match@0p5"],
            "policy_delta_vs_current_auc_bg": float(selected["auc_main_vs_bg"]) - float(current["auc_main_vs_bg"]),
            "policy_delta_vs_current_auc_nuis": float(selected["auc_main_vs_nuisance"]) - float(current["auc_main_vs_nuisance"]),
            "policy_delta_vs_current_fpr_nuis_05": float(selected["fpr_nuisance_match@0p5"]) - float(current["fpr_nuisance_match@0p5"]),
        }
        out_rows.append(row)
        summary_groups.setdefault(str(row["motion_scale"]), []).append(row)

    summary_by_scale: dict[str, Any] = {}
    for scale, items in summary_groups.items():
        summary_by_scale[scale] = {
            "n_cases": len(items),
            "mean_policy_auc_main_vs_bg": _mean(items, "policy_auc_main_vs_bg"),
            "mean_policy_auc_main_vs_nuisance": _mean(items, "policy_auc_main_vs_nuisance"),
            "mean_policy_fpr_nuisance_match@0p5": _mean(items, "policy_fpr_nuisance_match@0p5"),
            "mean_policy_delta_vs_current_auc_bg": _mean(items, "policy_delta_vs_current_auc_bg"),
            "mean_policy_delta_vs_current_auc_nuis": _mean(items, "policy_delta_vs_current_auc_nuis"),
            "mean_policy_delta_vs_current_fpr_nuis_05": _mean(items, "policy_delta_vs_current_fpr_nuis_05"),
        }

    payload = {
        "schema_version": "simus_policy_headline.v1",
        "policy": str(args.policy),
        "requested_profile": str(args.requested_profile),
        "current_profile": str(args.current_profile),
        "rows": out_rows,
        "summary_by_motion_scale": summary_by_scale,
    }
    _write_csv(Path(args.out_csv), out_rows)
    _write_json(Path(args.out_json), payload)
    print(f"[simus-policy-headline] wrote {args.out_csv}")
    print(f"[simus-policy-headline] wrote {args.out_json}")


if __name__ == "__main__":
    main()
