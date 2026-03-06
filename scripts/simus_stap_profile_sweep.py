#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.simus_eval_structural import evaluate_structural_metrics
from sim.simus.bundle import (
    SUPPORTED_SIMUS_STAP_PROFILES,
    bundle_profile_kwargs,
    derive_bundle_from_run,
    load_canonical_run,
    slugify,
)


def _split_csv_list(spec: str) -> list[str]:
    return [s.strip() for s in (spec or "").split(",") if s.strip()]


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


def _parse_case(spec: str) -> tuple[str, Path]:
    parts = str(spec).split("::", 1)
    if len(parts) != 2:
        raise ValueError(f"--case expects name::run_dir, got {spec!r}")
    return parts[0], Path(parts[1])


def _score_metrics(
    *,
    score: np.ndarray,
    mask_h1_pf_main: np.ndarray,
    mask_h0_bg: np.ndarray,
    mask_h0_nuisance_pa: np.ndarray | None,
    mask_h1_alias_qc: np.ndarray | None,
    fprs: list[float],
    match_tprs: list[float],
) -> dict[str, Any]:
    return evaluate_structural_metrics(
        score=score,
        mask_h1_pf_main=mask_h1_pf_main,
        mask_h0_bg=mask_h0_bg,
        mask_h0_nuisance_pa=mask_h0_nuisance_pa,
        mask_h1_alias_qc=mask_h1_alias_qc,
        fprs=fprs,
        match_tprs=match_tprs,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sweep named SIMUS STAP profiles on canonical runs.")
    ap.add_argument(
        "--case",
        type=str,
        action="append",
        required=True,
        help="Case spec: name::run_dir",
    )
    ap.add_argument(
        "--stap-profiles",
        type=str,
        default="Brain-SIMUS-Clin,Brain-SIMUS-Clin-MotionWide-v0,Brain-SIMUS-Clin-MotionShort-v0,Brain-SIMUS-Clin-MotionLong-v0,Brain-SIMUS-Clin-MotionRobust-v0",
        help="Comma-separated SIMUS STAP profiles to evaluate.",
    )
    ap.add_argument("--baseline-type", type=str, default="mc_svd")
    ap.add_argument("--stap-device", type=str, default="cuda")
    ap.add_argument("--out-root", type=Path, default=Path("runs/sim_eval/simus_stap_profile_sweep"))
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/simus_motion/simus_stap_profile_sweep.csv"),
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/simus_motion/simus_stap_profile_sweep.json"),
    )
    ap.add_argument("--fprs", type=str, default="1e-4,1e-3")
    ap.add_argument("--match-tprs", type=str, default="0.25,0.5,0.75")
    ap.add_argument("--reuse-bundles", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    fprs = [float(x) for x in _split_csv_list(str(args.fprs))]
    match_tprs = [float(x) for x in _split_csv_list(str(args.match_tprs))]
    stap_profiles = _split_csv_list(str(args.stap_profiles))
    unknown = [p for p in stap_profiles if p not in SUPPORTED_SIMUS_STAP_PROFILES]
    if unknown:
        raise ValueError(f"Unknown --stap-profiles entries: {unknown}")

    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "schema_version": "simus_stap_profile_sweep.v1",
        "stap_profiles": stap_profiles,
        "baseline_type": str(args.baseline_type),
        "cases": {},
    }

    for case_spec in args.case:
        case_name, run_dir = _parse_case(case_spec)
        icube, masks, meta = load_canonical_run(run_dir)
        mask_h1_pf_main = masks.get("mask_h1_pf_main")
        mask_h0_bg = masks.get("mask_h0_bg")
        if mask_h1_pf_main is None or mask_h0_bg is None:
            raise ValueError(f"{run_dir}: missing mask_h1_pf_main.npy or mask_h0_bg.npy")
        mask_h0_nuisance_pa = masks.get("mask_h0_nuisance_pa")
        mask_h1_alias_qc = masks.get("mask_h1_alias_qc")

        case_rows: list[dict[str, Any]] = []
        case_detail: dict[str, Any] = {
            "run_dir": str(run_dir),
            "profile": meta.get("simus", {}).get("profile"),
            "tier": meta.get("simus", {}).get("tier"),
            "shape": list(icube.shape),
            "motion_disp_rms_px": meta.get("motion", {}).get("telemetry", {}).get("disp_rms_px"),
            "phase_rms_rad": meta.get("phase_screen", {}).get("telemetry", {}).get("phase_rms_rad"),
            "profiles": {},
        }

        baseline_written = False
        for stap_profile in stap_profiles:
            dataset_name = f"{slugify(case_name)}_{slugify(stap_profile)}"
            bundle_root = Path(args.out_root) / slugify(case_name)
            bundle_dir = bundle_root / slugify(dataset_name)
            if not bool(args.reuse_bundles) or not (bundle_dir / "meta.json").is_file():
                bundle_dir = derive_bundle_from_run(
                    run_dir=run_dir,
                    out_root=bundle_root,
                    dataset_name=dataset_name,
                    stap_profile=str(stap_profile),
                    baseline_type=str(args.baseline_type),
                    run_stap=True,
                    stap_device=str(args.stap_device),
                    meta_extra={
                        "simus_stap_profile_sweep": True,
                        "case_name": str(case_name),
                        "stap_profile_candidate": str(stap_profile),
                    },
                )

            score_base = np.load(bundle_dir / "score_base.npy").astype(np.float32, copy=False)
            score_stap = np.load(bundle_dir / "score_stap_preka.npy").astype(np.float32, copy=False)
            profile_cfg = bundle_profile_kwargs(
                str(stap_profile),
                T=int(icube.shape[0]),
                baseline_type=str(args.baseline_type),
            )
            case_detail["profiles"][str(stap_profile)] = {
                "bundle_dir": str(bundle_dir),
                "profile_kwargs": profile_cfg,
            }

            if not baseline_written:
                baseline_metrics = _score_metrics(
                    score=score_base,
                    mask_h1_pf_main=mask_h1_pf_main,
                    mask_h0_bg=mask_h0_bg,
                    mask_h0_nuisance_pa=mask_h0_nuisance_pa,
                    mask_h1_alias_qc=mask_h1_alias_qc,
                    fprs=fprs,
                    match_tprs=match_tprs,
                )
                base_row = {
                    "case": case_name,
                    "simus_profile": meta.get("simus", {}).get("profile"),
                    "tier": meta.get("simus", {}).get("tier"),
                    "role": "baseline",
                    "method_label": "MC-SVD",
                    "score_label": "baseline_score",
                    "stap_profile": None,
                    "bundle_dir": str(bundle_dir),
                    "motion_disp_rms_px": case_detail["motion_disp_rms_px"],
                    "phase_rms_rad": case_detail["phase_rms_rad"],
                }
                base_row.update(baseline_metrics)
                rows.append(base_row)
                case_rows.append(base_row)
                baseline_written = True

            stap_metrics = _score_metrics(
                score=score_stap,
                mask_h1_pf_main=mask_h1_pf_main,
                mask_h0_bg=mask_h0_bg,
                mask_h0_nuisance_pa=mask_h0_nuisance_pa,
                mask_h1_alias_qc=mask_h1_alias_qc,
                fprs=fprs,
                match_tprs=match_tprs,
            )
            stap_row = {
                "case": case_name,
                "simus_profile": meta.get("simus", {}).get("profile"),
                "tier": meta.get("simus", {}).get("tier"),
                "role": "stap",
                "method_label": f"MC-SVD -> STAP [{stap_profile}]",
                "score_label": "stap_detector",
                "stap_profile": str(stap_profile),
                "bundle_dir": str(bundle_dir),
                "motion_disp_rms_px": case_detail["motion_disp_rms_px"],
                "phase_rms_rad": case_detail["phase_rms_rad"],
            }
            stap_row.update(stap_metrics)
            rows.append(stap_row)
            case_rows.append(stap_row)

        stap_only = [r for r in case_rows if r["role"] == "stap"]
        case_detail["best_auc_main_vs_bg"] = None
        case_detail["best_auc_main_vs_nuisance"] = None
        case_detail["best_fpr_nuisance_match@0p5"] = None
        if stap_only:
            best_bg = max(stap_only, key=lambda r: float(r["auc_main_vs_bg"]))
            best_nuis = max(stap_only, key=lambda r: float(r["auc_main_vs_nuisance"]))
            best_fpr = min(stap_only, key=lambda r: float(r["fpr_nuisance_match@0p5"]))
            case_detail["best_auc_main_vs_bg"] = {
                "stap_profile": best_bg["stap_profile"],
                "value": float(best_bg["auc_main_vs_bg"]),
            }
            case_detail["best_auc_main_vs_nuisance"] = {
                "stap_profile": best_nuis["stap_profile"],
                "value": float(best_nuis["auc_main_vs_nuisance"]),
            }
            case_detail["best_fpr_nuisance_match@0p5"] = {
                "stap_profile": best_fpr["stap_profile"],
                "value": float(best_fpr["fpr_nuisance_match@0p5"]),
            }
        details["cases"][case_name] = case_detail

    _write_csv(Path(args.out_csv), rows)
    _write_json(Path(args.out_json), {"rows": rows, "details": details})
    print(f"[simus-stap-profile-sweep] wrote {args.out_csv}")
    print(f"[simus-stap-profile-sweep] wrote {args.out_json}")


if __name__ == "__main__":
    main()
