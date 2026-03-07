#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from scripts.simus_eval_motion import make_motion_ladder_config
from scripts.simus_eval_structural import evaluate_structural_metrics
from scripts.simus_motion_proxy_warp_check import PROFILE_TO_NOMOTION_RUN, _warp_icube
from sim.simus.bundle import derive_bundle_from_run, estimate_simus_policy_features, load_canonical_run, select_simus_stap_profile
from sim.simus.motion import build_motion_artifacts


def _split_csv_list(spec: str) -> list[str]:
    return [s.strip() for s in (spec or "").split(",") if s.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _save_proxy_run(
    run_dir: Path,
    *,
    moved: np.ndarray,
    masks: dict[str, np.ndarray],
    base_meta: dict[str, Any],
    source_run_dir: Path,
    motion_scale: float,
    art_telemetry: dict[str, Any],
    proxy_features: dict[str, Any],
) -> None:
    ds = run_dir / "dataset"
    dbg = ds / "debug"
    dbg.mkdir(parents=True, exist_ok=True)
    np.save(ds / "icube.npy", np.asarray(moved, dtype=np.complex64))
    for name, arr in masks.items():
        np.save(ds / f"{name}.npy", np.asarray(arr, dtype=bool))
    meta = json.loads(json.dumps(base_meta))
    meta["simus"] = dict(meta.get("simus") or {})
    meta["simus"]["profile"] = f"{meta['simus'].get('profile', 'unknown')}+proxywarp_motionx{str(motion_scale).replace('.', 'p')}"
    meta["proxy_motion"] = {
        "mode": "beamformed_iq_warp",
        "source_nomotion_run_dir": str(source_run_dir),
        "motion_scale": float(motion_scale),
        "motion_telemetry": {k: float(v) for k, v in art_telemetry.items()},
        "policy_features": {k: (float(v) if v is not None else None) for k, v in proxy_features.items()},
    }
    _write_json(ds / "meta.json", meta)
    _write_json(ds / "config.json", meta.get("config") or {})


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("role") == "baseline":
            continue
        by_label.setdefault(str(row["method_label"]), []).append(row)
    for label, items in sorted(by_label.items()):
        out[label] = {
            "count": len(items),
            "mean_auc_main_vs_bg": float(np.mean([float(r["auc_main_vs_bg"]) for r in items])),
            "mean_auc_main_vs_nuisance": float(np.mean([float(r["auc_main_vs_nuisance"]) for r in items])),
            "mean_fpr_nuisance_match@0p5": float(np.mean([float(r["fpr_nuisance_match@0p5"]) for r in items])),
            "mean_delta_auc_main_vs_bg": float(np.mean([float(r.get("delta_auc_main_vs_bg") or 0.0) for r in items])),
            "mean_delta_auc_main_vs_nuisance": float(np.mean([float(r.get("delta_auc_main_vs_nuisance") or 0.0) for r in items])),
            "mean_delta_fpr_nuisance_match@0p5": float(np.mean([float(r.get("delta_fpr_nuisance_match@0p5") or 0.0) for r in items])),
        }
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Approximate corrected SIMUS motion benchmarking by warping existing no-motion beamformed IQ cubes, then re-running bundle baselines and STAP.")
    ap.add_argument("--profiles", type=str, default="ClinIntraOp-Pf-v1,ClinMobile-Pf-v1")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--tier", type=str, default="paper", choices=["smoke", "paper"])
    ap.add_argument("--motion-scales", type=str, default="0.25,1.0")
    ap.add_argument(
        "--stap-profiles",
        type=str,
        default="Brain-SIMUS-Clin,Brain-SIMUS-Clin-MotionShort-v0,Brain-SIMUS-Clin-MotionRobust-v0,Brain-SIMUS-Clin-MotionMidRobust-v0",
    )
    ap.add_argument(
        "--policies",
        type=str,
        default="Brain-SIMUS-Clin-MotionDisp-v0,Brain-SIMUS-Clin-RegShiftP90-v0",
    )
    ap.add_argument("--baseline-type", type=str, default="mc_svd")
    ap.add_argument("--stap-device", type=str, default="cuda")
    ap.add_argument("--fprs", type=str, default="1e-4,1e-3")
    ap.add_argument("--match-tprs", type=str, default="0.25,0.5,0.75")
    ap.add_argument("--out-root", type=Path, default=Path("runs/sim_eval/simus_motion_proxy_bundle_bench"))
    ap.add_argument("--out-csv", type=Path, default=Path("reports/simus_motion/simus_motion_proxy_bundle_bench.csv"))
    ap.add_argument("--out-json", type=Path, default=Path("reports/simus_motion/simus_motion_proxy_bundle_bench.json"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("STAP_FAST_CUDA_GRAPH", "0")

    simus_profiles = _split_csv_list(str(args.profiles))
    motion_scales = [float(x) for x in _split_csv_list(str(args.motion_scales))]
    stap_profiles = _split_csv_list(str(args.stap_profiles))
    policies = _split_csv_list(str(args.policies))
    fprs = [float(x) for x in _split_csv_list(str(args.fprs))]
    match_tprs = [float(x) for x in _split_csv_list(str(args.match_tprs))]

    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}

    for simus_profile in simus_profiles:
        base_run = PROFILE_TO_NOMOTION_RUN[str(simus_profile)]
        icube0, masks, meta = load_canonical_run(base_run)
        mask_h1_pf_main = masks["mask_h1_pf_main"]
        mask_h0_bg = masks["mask_h0_bg"]
        mask_h0_nuisance_pa = masks.get("mask_h0_nuisance_pa")
        mask_h1_alias_qc = masks.get("mask_h1_alias_qc")

        for motion_scale in motion_scales:
            cfg = make_motion_ladder_config(
                profile=str(simus_profile),
                tier=str(args.tier),
                seed=int(args.seed),
                motion_scale=float(motion_scale),
                phase_scale=float(motion_scale),
            )
            art = build_motion_artifacts(cfg=cfg, seed=int(args.seed))
            moved = _warp_icube(icube0, art.dx_px, art.dz_px)
            proxy_features = estimate_simus_policy_features(moved, reg_subpixel=4, reg_reference="median")

            case_slug = f"{simus_profile.lower().replace('-', '_')}__seed{int(args.seed)}__motionx{str(motion_scale).replace('.', 'p')}__proxywarp"
            case_run = Path(args.out_root) / "runs" / case_slug
            _save_proxy_run(
                case_run,
                moved=moved,
                masks=masks,
                base_meta=meta,
                source_run_dir=base_run,
                motion_scale=float(motion_scale),
                art_telemetry=art.telemetry,
                proxy_features=proxy_features,
            )

            baseline_bundle = derive_bundle_from_run(
                run_dir=case_run,
                out_root=Path(args.out_root) / "eval" / case_slug / "baseline",
                dataset_name=f"{case_slug}_baseline",
                stap_profile="Brain-SIMUS-Clin",
                baseline_type=str(args.baseline_type),
                run_stap=False,
                stap_device=str(args.stap_device),
                meta_extra={"proxy_motion": True, "proxy_case_slug": case_slug},
            )
            baseline_score = np.load(baseline_bundle / "score_base.npy")
            baseline_metrics = evaluate_structural_metrics(
                score=baseline_score,
                mask_h1_pf_main=mask_h1_pf_main,
                mask_h0_bg=mask_h0_bg,
                mask_h0_nuisance_pa=mask_h0_nuisance_pa,
                mask_h1_alias_qc=mask_h1_alias_qc,
                fprs=fprs,
                match_tprs=match_tprs,
            )
            base_row = {
                "case_slug": case_slug,
                "simus_profile": str(simus_profile),
                "seed": int(args.seed),
                "motion_scale": float(motion_scale),
                "approximation": "beamformed_iq_warp",
                "requested_profile": None,
                "applied_profile": None,
                "policy": None,
                "role": "baseline",
                "method_label": "MC-SVD",
                "reg_shift_p90": float(proxy_features["reg_shift_p90"]),
                "reg_shift_rms": float(proxy_features["reg_shift_rms"]),
                "reg_psr_median": float(proxy_features["reg_psr_median"]),
                "motion_disp_rms_px": float(art.telemetry.get("disp_rms_px") or 0.0),
                "motion_disp_p90_px": float(art.telemetry.get("disp_p90_px") or 0.0),
                **baseline_metrics,
            }
            rows.append(base_row)

            cache: dict[str, Path] = {}
            for stap_profile in stap_profiles:
                bundle = derive_bundle_from_run(
                    run_dir=case_run,
                    out_root=Path(args.out_root) / "eval" / case_slug / stap_profile,
                    dataset_name=f"{case_slug}_{stap_profile}",
                    stap_profile=str(stap_profile),
                    baseline_type=str(args.baseline_type),
                    run_stap=True,
                    stap_device=str(args.stap_device),
                    meta_extra={"proxy_motion": True, "proxy_case_slug": case_slug},
                )
                cache[str(stap_profile)] = bundle
                score = np.load(bundle / "score_stap_preka.npy")
                metrics = evaluate_structural_metrics(
                    score=score,
                    mask_h1_pf_main=mask_h1_pf_main,
                    mask_h0_bg=mask_h0_bg,
                    mask_h0_nuisance_pa=mask_h0_nuisance_pa,
                    mask_h1_alias_qc=mask_h1_alias_qc,
                    fprs=fprs,
                    match_tprs=match_tprs,
                )
                rows.append(
                    {
                        "case_slug": case_slug,
                        "simus_profile": str(simus_profile),
                        "seed": int(args.seed),
                        "motion_scale": float(motion_scale),
                        "approximation": "beamformed_iq_warp",
                        "requested_profile": str(stap_profile),
                        "applied_profile": str(stap_profile),
                        "policy": None,
                        "role": "stap",
                        "method_label": f"MC-SVD -> STAP ({stap_profile})",
                        "reg_shift_p90": float(proxy_features["reg_shift_p90"]),
                        "reg_shift_rms": float(proxy_features["reg_shift_rms"]),
                        "reg_psr_median": float(proxy_features["reg_psr_median"]),
                        "motion_disp_rms_px": float(art.telemetry.get("disp_rms_px") or 0.0),
                        "motion_disp_p90_px": float(art.telemetry.get("disp_p90_px") or 0.0),
                        "delta_auc_main_vs_bg": float(metrics["auc_main_vs_bg"]) - float(base_row["auc_main_vs_bg"]),
                        "delta_auc_main_vs_nuisance": float(metrics["auc_main_vs_nuisance"]) - float(base_row["auc_main_vs_nuisance"]),
                        "delta_fpr_nuisance_match@0p5": float(metrics["fpr_nuisance_match@0p5"]) - float(base_row["fpr_nuisance_match@0p5"]),
                        **metrics,
                    }
                )

            for policy in policies:
                applied, policy_meta = select_simus_stap_profile(
                    requested_profile="Brain-SIMUS-Clin",
                    policy=str(policy),
                    feature_values=proxy_features,
                    motion_disp_rms_px=float(art.telemetry.get("disp_rms_px") or 0.0),
                )
                bundle = cache[applied]
                score = np.load(bundle / "score_stap_preka.npy")
                metrics = evaluate_structural_metrics(
                    score=score,
                    mask_h1_pf_main=mask_h1_pf_main,
                    mask_h0_bg=mask_h0_bg,
                    mask_h0_nuisance_pa=mask_h0_nuisance_pa,
                    mask_h1_alias_qc=mask_h1_alias_qc,
                    fprs=fprs,
                    match_tprs=match_tprs,
                )
                rows.append(
                    {
                        "case_slug": case_slug,
                        "simus_profile": str(simus_profile),
                        "seed": int(args.seed),
                        "motion_scale": float(motion_scale),
                        "approximation": "beamformed_iq_warp",
                        "requested_profile": "Brain-SIMUS-Clin",
                        "applied_profile": str(applied),
                        "policy": str(policy),
                        "role": "stap_policy",
                        "method_label": f"MC-SVD -> STAP policy ({policy})",
                        "reg_shift_p90": float(proxy_features["reg_shift_p90"]),
                        "reg_shift_rms": float(proxy_features["reg_shift_rms"]),
                        "reg_psr_median": float(proxy_features["reg_psr_median"]),
                        "motion_disp_rms_px": float(art.telemetry.get("disp_rms_px") or 0.0),
                        "motion_disp_p90_px": float(art.telemetry.get("disp_p90_px") or 0.0),
                        "delta_auc_main_vs_bg": float(metrics["auc_main_vs_bg"]) - float(base_row["auc_main_vs_bg"]),
                        "delta_auc_main_vs_nuisance": float(metrics["auc_main_vs_nuisance"]) - float(base_row["auc_main_vs_nuisance"]),
                        "delta_fpr_nuisance_match@0p5": float(metrics["fpr_nuisance_match@0p5"]) - float(base_row["fpr_nuisance_match@0p5"]),
                        **policy_meta,
                        **metrics,
                    }
                )

            details[case_slug] = {
                "source_nomotion_run_dir": str(base_run),
                "proxy_features": proxy_features,
                "motion_telemetry": art.telemetry,
            }

    summary = summarize_rows(rows)
    _write_csv(Path(args.out_csv), rows)
    _write_json(
        Path(args.out_json),
        {
            "schema_version": "simus_motion_proxy_bundle_bench.v1",
            "approximation": "beamformed_iq_warp",
            "seed": int(args.seed),
            "tier": str(args.tier),
            "motion_scales": motion_scales,
            "simus_profiles": simus_profiles,
            "stap_profiles": stap_profiles,
            "policies": policies,
            "details": details,
            "summary": summary,
            "rows": rows,
        },
    )
    print(f"[simus-motion-proxy-bundle-bench] wrote {args.out_csv}")
    print(f"[simus-motion-proxy-bundle-bench] wrote {args.out_json}")


if __name__ == "__main__":
    main()
