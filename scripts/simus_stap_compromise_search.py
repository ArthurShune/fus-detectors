#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from scripts.simus_eval_motion import _run_slug, make_motion_ladder_config
from scripts.simus_eval_structural import evaluate_structural_metrics
from sim.simus.bundle import (
    SUPPORTED_SIMUS_STAP_PROFILES,
    bundle_profile_kwargs,
    derive_bundle_from_run,
    load_canonical_run,
    slugify,
)
from sim.simus.pilot_pymust_simus import SUPPORTED_SIMUS_PROFILES, write_simus_run


def _split_csv_list(spec: str) -> list[str]:
    return [s.strip() for s in (spec or "").split(",") if s.strip()]


def _parse_case(spec: str) -> tuple[str, Path]:
    parts = str(spec).split("::", 1)
    if len(parts) != 2:
        raise ValueError(f"--case expects name::run_dir, got {spec!r}")
    return parts[0], Path(parts[1])


def _profile_motion_scale(profile_name: str | None) -> float | None:
    text = str(profile_name or "")
    m = re.search(r"motionx([0-9p]+)", text)
    if not m:
        return None
    return float(m.group(1).replace("p", "."))


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


def _telemetry_fields(meta: dict[str, Any]) -> dict[str, Any]:
    tele = meta.get("stap_fallback_telemetry", {}) or {}
    out = {}
    for key in (
        "reg_shift_rms",
        "reg_shift_p90",
        "reg_psr_median",
        "reg_psr_p10",
        "reg_psr_p90",
        "flow_cov_ge_50_fraction",
        "flow_cov_ge_80_fraction",
        "tile_flow_coverage_p90",
        "tile_flow_coverage_p50",
        "cov_train_trim_q",
        "median_motion_half_span_hz",
        "median_msd_lambda",
        "median_msd_lambda_conditioned",
        "median_msd_lambda_needed",
        "constraint_ridge",
        "reg_ms",
        "baseline_ms",
        "stap_ms",
    ):
        out[key] = tele.get(key)
    return out


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


def _summary_by_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_profile: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["role"] != "stap":
            continue
        by_profile.setdefault(str(row["stap_profile"]), []).append(row)
    out: dict[str, Any] = {}
    for profile, items in by_profile.items():
        auc_bg = np.asarray([float(r["auc_main_vs_bg"]) for r in items], dtype=np.float64)
        auc_n = np.asarray([float(r["auc_main_vs_nuisance"]) for r in items], dtype=np.float64)
        fpr = np.asarray([float(r["fpr_nuisance_match@0p5"]) for r in items], dtype=np.float64)
        d_auc_bg = np.asarray([float(r["delta_auc_main_vs_bg"]) for r in items], dtype=np.float64)
        d_auc_n = np.asarray([float(r["delta_auc_main_vs_nuisance"]) for r in items], dtype=np.float64)
        d_fpr = np.asarray([float(r["delta_fpr_nuisance_match@0p5"]) for r in items], dtype=np.float64)
        out[profile] = {
            "count": int(len(items)),
            "mean_auc_main_vs_bg": float(np.mean(auc_bg)),
            "mean_auc_main_vs_nuisance": float(np.mean(auc_n)),
            "mean_fpr_nuisance_match@0p5": float(np.mean(fpr)),
            "mean_delta_auc_main_vs_bg": float(np.mean(d_auc_bg)),
            "mean_delta_auc_main_vs_nuisance": float(np.mean(d_auc_n)),
            "mean_delta_fpr_nuisance_match@0p5": float(np.mean(d_fpr)),
            "pareto_improved_cases": int(
                np.sum((d_auc_bg > 0.0) & (d_auc_n > 0.0) & (d_fpr < 0.0))
            ),
        }
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Search for a single compromise SIMUS STAP profile across seeds and motion levels.")
    ap.add_argument(
        "--case",
        type=str,
        action="append",
        default=None,
        help="Explicit existing run: name::run_dir. When provided, these runs are reused directly.",
    )
    ap.add_argument(
        "--simus-profiles",
        type=str,
        default="ClinIntraOp-Pf-v1,ClinMobile-Pf-v1",
        help="Comma-separated SIMUS generator profiles.",
    )
    ap.add_argument("--tier", type=str, default="paper", choices=["smoke", "paper"])
    ap.add_argument("--seeds", type=str, default="21,22,23")
    ap.add_argument("--motion-scales", type=str, default="0.25,0.5,1.0")
    ap.add_argument("--phase-scale-mode", type=str, default="same", choices=["same", "zero", "fixed"])
    ap.add_argument("--phase-scale-fixed", type=float, default=1.0)
    ap.add_argument(
        "--stap-profiles",
        type=str,
        default=(
            "Brain-SIMUS-Clin,"
            "Brain-SIMUS-Clin-MotionShort-v0,"
            "Brain-SIMUS-Clin-MotionMid-v0,"
            "Brain-SIMUS-Clin-MotionLong-v0,"
            "Brain-SIMUS-Clin-MotionRobust-v0,"
            "Brain-SIMUS-Clin-MotionMidRobust-v0"
        ),
    )
    ap.add_argument("--baseline-type", type=str, default="mc_svd")
    ap.add_argument("--stap-device", type=str, default="cuda")
    ap.add_argument("--out-root", type=Path, default=Path("runs/sim_eval/simus_stap_compromise_search"))
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/simus_motion/simus_stap_compromise_search.csv"),
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/simus_motion/simus_stap_compromise_search.json"),
    )
    ap.add_argument("--fprs", type=str, default="1e-4,1e-3")
    ap.add_argument("--match-tprs", type=str, default="0.25,0.5,0.75")
    ap.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--reuse-bundles", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    simus_profiles = _split_csv_list(str(args.simus_profiles))
    unsupported_simus = [p for p in simus_profiles if p not in SUPPORTED_SIMUS_PROFILES]
    if unsupported_simus:
        raise ValueError(f"Unknown --simus-profiles entries: {unsupported_simus}")
    stap_profiles = _split_csv_list(str(args.stap_profiles))
    unsupported_stap = [p for p in stap_profiles if p not in SUPPORTED_SIMUS_STAP_PROFILES]
    if unsupported_stap:
        raise ValueError(f"Unknown --stap-profiles entries: {unsupported_stap}")
    seeds = [int(x) for x in _split_csv_list(str(args.seeds))]
    motion_scales = [float(x) for x in _split_csv_list(str(args.motion_scales))]
    fprs = [float(x) for x in _split_csv_list(str(args.fprs))]
    match_tprs = [float(x) for x in _split_csv_list(str(args.match_tprs))]

    out_root = Path(args.out_root)
    runs_root = out_root / "runs"
    eval_root = out_root / "eval"
    runs_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "schema_version": "simus_stap_compromise_search.v1",
        "simus_profiles": simus_profiles,
        "explicit_cases": args.case or [],
        "seeds": seeds,
        "motion_scales": motion_scales,
        "stap_profiles": stap_profiles,
        "cases": {},
    }

    generated_cases: list[tuple[str, Path]] = []
    for simus_profile in simus_profiles:
        for seed in seeds:
            for motion_scale in motion_scales:
                if args.phase_scale_mode == "zero":
                    phase_scale = 0.0
                elif args.phase_scale_mode == "fixed":
                    phase_scale = float(args.phase_scale_fixed)
                else:
                    phase_scale = float(motion_scale)
                cfg = make_motion_ladder_config(
                    profile=str(simus_profile),
                    tier=str(args.tier),
                    seed=int(seed),
                    motion_scale=float(motion_scale),
                    phase_scale=float(phase_scale),
                )
                run_slug = _run_slug(str(simus_profile), int(seed), float(motion_scale), float(phase_scale))
                run_dir = runs_root / run_slug
                if not bool(args.reuse_existing) or not (run_dir / "dataset" / "meta.json").is_file():
                    write_simus_run(out_root=run_dir, cfg=cfg, skip_bundle=True)
                generated_cases.append((run_slug, run_dir))

    explicit_cases = [_parse_case(spec) for spec in (args.case or [])]
    all_cases = explicit_cases + generated_cases

    for case_name, run_dir in all_cases:
        icube, masks, run_meta = load_canonical_run(run_dir)
        simus_profile = str(run_meta.get("simus", {}).get("profile") or "")
        seed = int(run_meta.get("config", {}).get("seed", 0))
        motion_scale = _profile_motion_scale(simus_profile)
        if motion_scale is None:
            motion_scale = float("nan")
        phase_scale = motion_scale
        mask_h1_pf_main = masks.get("mask_h1_pf_main")
        mask_h0_bg = masks.get("mask_h0_bg")
        if mask_h1_pf_main is None or mask_h0_bg is None:
            raise ValueError(f"{run_dir}: missing mask_h1_pf_main.npy or mask_h0_bg.npy")
        mask_h0_nuisance_pa = masks.get("mask_h0_nuisance_pa")
        mask_h1_alias_qc = masks.get("mask_h1_alias_qc")

        case_key = f"{simus_profile}|seed{seed}|motion{motion_scale:g}"
        details["cases"][case_key] = {
            "run_dir": str(run_dir),
            "simus_profile": str(simus_profile),
            "seed": int(seed),
            "motion_scale": float(motion_scale),
            "phase_scale": float(phase_scale),
            "shape": list(icube.shape),
            "motion_disp_rms_px": run_meta.get("motion", {}).get("telemetry", {}).get("disp_rms_px"),
            "phase_rms_rad": run_meta.get("phase_screen", {}).get("telemetry", {}).get("phase_rms_rad"),
            "profiles": {},
        }

        baseline_row: dict[str, Any] | None = None
        for stap_profile in stap_profiles:
            dataset_name = f"{slugify(case_name)}_{slugify(stap_profile)}"
            bundle_root = eval_root / slugify(case_name)
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
                        "simus_stap_compromise_search": True,
                        "simus_profile": str(simus_profile),
                        "motion_scale": float(motion_scale),
                        "phase_scale": float(phase_scale),
                        "seed": int(seed),
                    },
                )

            bundle_meta = json.loads((bundle_dir / "meta.json").read_text(encoding="utf-8"))
            tele = _telemetry_fields(bundle_meta)
            profile_cfg = bundle_profile_kwargs(
                str(stap_profile), T=int(icube.shape[0]), baseline_type=str(args.baseline_type)
            )
            details["cases"][case_key]["profiles"][str(stap_profile)] = {
                "bundle_dir": str(bundle_dir),
                "profile_kwargs": profile_cfg,
                "telemetry": tele,
            }

            if baseline_row is None:
                score_base = np.load(bundle_dir / "score_base.npy").astype(np.float32, copy=False)
                base_metrics = _score_metrics(
                    score=score_base,
                    mask_h1_pf_main=mask_h1_pf_main,
                    mask_h0_bg=mask_h0_bg,
                    mask_h0_nuisance_pa=mask_h0_nuisance_pa,
                    mask_h1_alias_qc=mask_h1_alias_qc,
                    fprs=fprs,
                    match_tprs=match_tprs,
                )
                baseline_row = {
                    "case_key": case_key,
                    "simus_profile": str(simus_profile),
                    "seed": int(seed),
                    "motion_scale": float(motion_scale),
                    "phase_scale": float(phase_scale),
                    "role": "baseline",
                    "stap_profile": None,
                    "bundle_dir": str(bundle_dir),
                    "motion_disp_rms_px": details["cases"][case_key]["motion_disp_rms_px"],
                    "phase_rms_rad": details["cases"][case_key]["phase_rms_rad"],
                    **tele,
                    **base_metrics,
                }
                rows.append(baseline_row)

            score_stap = np.load(bundle_dir / "score_stap_preka.npy").astype(np.float32, copy=False)
            stap_metrics = _score_metrics(
                score=score_stap,
                mask_h1_pf_main=mask_h1_pf_main,
                mask_h0_bg=mask_h0_bg,
                mask_h0_nuisance_pa=mask_h0_nuisance_pa,
                mask_h1_alias_qc=mask_h1_alias_qc,
                fprs=fprs,
                match_tprs=match_tprs,
            )
            row = {
                "case_key": case_key,
                "simus_profile": str(simus_profile),
                "seed": int(seed),
                "motion_scale": float(motion_scale),
                "phase_scale": float(phase_scale),
                "role": "stap",
                "stap_profile": str(stap_profile),
                "bundle_dir": str(bundle_dir),
                "motion_disp_rms_px": details["cases"][case_key]["motion_disp_rms_px"],
                "phase_rms_rad": details["cases"][case_key]["phase_rms_rad"],
                **tele,
                **stap_metrics,
            }
            if baseline_row is not None:
                row["delta_auc_main_vs_bg"] = float(row["auc_main_vs_bg"]) - float(
                    baseline_row["auc_main_vs_bg"]
                )
                row["delta_auc_main_vs_nuisance"] = float(row["auc_main_vs_nuisance"]) - float(
                    baseline_row["auc_main_vs_nuisance"]
                )
                row["delta_fpr_nuisance_match@0p5"] = float(
                    row["fpr_nuisance_match@0p5"]
                ) - float(baseline_row["fpr_nuisance_match@0p5"])
            rows.append(row)

    details["summary_by_profile"] = _summary_by_profile(rows)
    _write_csv(Path(args.out_csv), rows)
    _write_json(Path(args.out_json), {"rows": rows, "details": details})
    print(f"[simus-stap-compromise-search] wrote {args.out_csv}")
    print(f"[simus-stap-compromise-search] wrote {args.out_json}")


if __name__ == "__main__":
    main()
