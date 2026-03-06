#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from scripts.simus_eval_structural import (
    MethodSpec,
    _baseline_label,
    _headline_label,
    _pipeline_label,
    _score_label,
    _score_semantics,
    _split_csv_list,
    evaluate_structural_metrics,
)
from sim.simus.bundle import (
    SUPPORTED_SIMUS_STAP_POLICIES,
    SUPPORTED_SIMUS_STAP_PROFILES,
    derive_bundle_from_run,
    estimate_simus_policy_features,
    load_canonical_run,
    select_simus_stap_profile,
    slugify,
)
from sim.simus.config import MotionSpec, PhaseScreenSpec, SimusConfig, default_profile_config
from sim.simus.pilot_pymust_simus import SUPPORTED_SIMUS_PROFILES, write_simus_run


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


def _scale_tag(value: float) -> str:
    s = f"{float(value):.2f}"
    s = s.rstrip("0").rstrip(".")
    return s.replace(".", "p")


def scale_motion_spec(spec: MotionSpec, scale: float) -> MotionSpec:
    scale = float(scale)
    if (not spec.enabled) or scale <= 0.0:
        return MotionSpec(enabled=False)
    return MotionSpec(
        enabled=True,
        breathing_hz=float(spec.breathing_hz),
        breathing_amp_x_px=float(spec.breathing_amp_x_px) * scale,
        breathing_amp_z_px=float(spec.breathing_amp_z_px) * scale,
        cardiac_hz=float(spec.cardiac_hz),
        cardiac_amp_x_px=float(spec.cardiac_amp_x_px) * scale,
        cardiac_amp_z_px=float(spec.cardiac_amp_z_px) * scale,
        random_walk_sigma_px=float(spec.random_walk_sigma_px) * scale,
        drift_x_px=float(spec.drift_x_px) * scale,
        drift_z_px=float(spec.drift_z_px) * scale,
        elastic_amp_px=float(spec.elastic_amp_px) * scale,
        elastic_sigma_px=float(spec.elastic_sigma_px),
        elastic_depth_decay_frac=float(spec.elastic_depth_decay_frac),
        elastic_temporal_rho=float(spec.elastic_temporal_rho),
        elastic_lateral_scale=float(spec.elastic_lateral_scale),
        elastic_axial_scale=float(spec.elastic_axial_scale),
    )


def scale_phase_spec(spec: PhaseScreenSpec, scale: float) -> PhaseScreenSpec:
    scale = float(scale)
    if (not spec.enabled) or scale <= 0.0:
        return PhaseScreenSpec(enabled=False)
    return PhaseScreenSpec(
        enabled=True,
        std_rad=float(spec.std_rad) * scale,
        corr_len_elem=float(spec.corr_len_elem),
        drift_rho=float(spec.drift_rho),
        drift_sigma_rad=float(spec.drift_sigma_rad) * scale,
    )


def make_motion_ladder_config(
    *,
    profile: str,
    tier: str,
    seed: int,
    motion_scale: float,
    phase_scale: float | None = None,
) -> SimusConfig:
    cfg = default_profile_config(profile=profile, tier=tier, seed=seed)  # type: ignore[arg-type]
    phase_scale = float(motion_scale if phase_scale is None else phase_scale)
    cfg = dataclasses.replace(
        cfg,
        profile=f"{cfg.profile}+motionx{_scale_tag(motion_scale)}_phasex{_scale_tag(phase_scale)}",
        motion=scale_motion_spec(cfg.motion, motion_scale),
        phase_screen=scale_phase_spec(cfg.phase_screen, phase_scale),
    )
    return cfg


def _run_slug(profile: str, seed: int, motion_scale: float, phase_scale: float) -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", profile.lower().replace("-", "_"))
    return f"simus_{slug}_motionx{_scale_tag(motion_scale)}_phasex{_scale_tag(phase_scale)}_seed{int(seed)}"


def _score_path(bundle_dir: Path, *, eval_score: str, role: str) -> Path:
    if eval_score == "pd":
        candidates = [bundle_dir / "score_pd_base.npy"] if role == "baseline" else [bundle_dir / "score_pd_stap.npy"]
        candidates += [bundle_dir / "score_base.npy"] if role == "baseline" else [bundle_dir / "score_pd_base.npy"]
    else:
        candidates = [bundle_dir / "score_base.npy"] if role == "baseline" else [bundle_dir / "score_stap_preka.npy", bundle_dir / "score_stap.npy"]
    return next((p for p in candidates if p.is_file()), candidates[0])


def _build_methods(baselines: str, stap_baseline: str) -> list[MethodSpec]:
    out = [MethodSpec(key=f"baseline_{b}", baseline_type=b, run_stap=False, role="baseline") for b in _split_csv_list(baselines)]
    out.append(MethodSpec(key="stap", baseline_type=str(stap_baseline), run_stap=True, role="stap"))
    return out


def _summarize_scale(rows: list[dict[str, Any]], *, match_tag: str) -> dict[str, Any]:
    by_scale: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scale.setdefault(str(row["motion_scale"]), []).append(row)
    summary: dict[str, Any] = {}
    for scale, items in sorted(by_scale.items(), key=lambda kv: float(kv[0])):
        best_auc_nuis = max(items, key=lambda r: float(r.get("auc_main_vs_nuisance") or float("-inf")))
        best_auc_bg = max(items, key=lambda r: float(r.get("auc_main_vs_bg") or float("-inf")))
        lowest_match = min(items, key=lambda r: float(r.get(f"fpr_nuisance_match@{match_tag}") or float("inf")))
        summary[scale] = {
            "best_auc_main_vs_bg": {
                "method": best_auc_bg["method_label"],
                "value": best_auc_bg.get("auc_main_vs_bg"),
            },
            "best_auc_main_vs_nuisance": {
                "method": best_auc_nuis["method_label"],
                "value": best_auc_nuis.get("auc_main_vs_nuisance"),
            },
            f"lowest_fpr_nuisance_match@{match_tag}": {
                "method": lowest_match["method_label"],
                "value": lowest_match.get(f"fpr_nuisance_match@{match_tag}"),
                "tpr_main_match": lowest_match.get(f"tpr_main_match@{match_tag}"),
            },
        }
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate a SIMUS motion/phase nuisance ladder against structural metrics.")
    ap.add_argument("--profile", type=str, default="ClinIntraOp-Pf-v1", choices=list(SUPPORTED_SIMUS_PROFILES))
    ap.add_argument("--tier", type=str, default="paper", choices=["smoke", "paper"])
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--motion-scales", type=str, default="0,0.5,1.0")
    ap.add_argument("--phase-scale-mode", type=str, default="same", choices=["same", "zero", "fixed"])
    ap.add_argument("--phase-scale-fixed", type=float, default=1.0)
    ap.add_argument("--out-root", type=Path, default=Path("runs/sim_eval/simus_motion_ladder"))
    ap.add_argument("--out-csv", type=Path, default=Path("reports/simus_motion/simus_motion_ladder.csv"))
    ap.add_argument("--out-json", type=Path, default=Path("reports/simus_motion/simus_motion_ladder.json"))
    ap.add_argument(
        "--stap-profile",
        type=str,
        default="Brain-SIMUS-Clin",
        choices=list(SUPPORTED_SIMUS_STAP_PROFILES),
    )
    ap.add_argument(
        "--stap-policy",
        type=str,
        default=None,
        choices=[None, *SUPPORTED_SIMUS_STAP_POLICIES],
    )
    ap.add_argument("--eval-score", type=str, default="pd", choices=["pd", "vnext"])
    ap.add_argument("--baselines", type=str, default="mc_svd,svd_similarity,local_svd,rpca,hosvd")
    ap.add_argument("--stap-baseline", type=str, default="mc_svd")
    ap.add_argument("--stap-device", type=str, default="cpu")
    ap.add_argument("--fprs", type=str, default="1e-4,1e-3")
    ap.add_argument("--match-tprs", type=str, default="0.25,0.5,0.75")
    ap.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--reuse-bundles", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    sim_root = out_root / "runs"
    eval_root = out_root / "eval"
    sim_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    methods = _build_methods(str(args.baselines), str(args.stap_baseline))
    fprs = [float(x) for x in _split_csv_list(str(args.fprs))]
    match_tprs = [float(x) for x in _split_csv_list(str(args.match_tprs))]
    motion_scales = [float(x) for x in _split_csv_list(str(args.motion_scales))]
    details: dict[str, Any] = {
        "schema_version": "simus_motion_ladder.v1",
        "profile": str(args.profile),
        "tier": str(args.tier),
        "seed": int(args.seed),
        "motion_scales": motion_scales,
        "phase_scale_mode": str(args.phase_scale_mode),
        "phase_scale_fixed": float(args.phase_scale_fixed),
        "runs": {},
    }
    rows: list[dict[str, Any]] = []

    os.environ.setdefault("STAP_FAST_CUDA_GRAPH", "0")

    for motion_scale in motion_scales:
        if args.phase_scale_mode == "zero":
            phase_scale = 0.0
        elif args.phase_scale_mode == "fixed":
            phase_scale = float(args.phase_scale_fixed)
        else:
            phase_scale = float(motion_scale)
        cfg = make_motion_ladder_config(
            profile=str(args.profile),
            tier=str(args.tier),
            seed=int(args.seed),
            motion_scale=float(motion_scale),
            phase_scale=float(phase_scale),
        )
        run_slug = _run_slug(str(args.profile), int(args.seed), float(motion_scale), float(phase_scale))
        run_dir = sim_root / run_slug
        if not bool(args.reuse_existing) or not (run_dir / "dataset" / "meta.json").is_file():
            outputs = write_simus_run(out_root=run_dir, cfg=cfg, skip_bundle=True)
            dataset_dir = outputs["dataset_dir"]
        else:
            dataset_dir = run_dir / "dataset"

        icube, masks, meta = load_canonical_run(run_dir)
        mask_h1_pf_main = masks.get("mask_h1_pf_main")
        mask_h0_bg = masks.get("mask_h0_bg")
        if mask_h1_pf_main is None or mask_h0_bg is None:
            raise ValueError(f"{run_dir}: dataset is missing mask_h1_pf_main.npy or mask_h0_bg.npy")
        mask_h0_nuisance_pa = masks.get("mask_h0_nuisance_pa")
        mask_h1_alias_qc = masks.get("mask_h1_alias_qc")

        details["runs"][run_slug] = {
            "run_dir": str(run_dir),
            "dataset_dir": str(dataset_dir),
            "motion_scale": float(motion_scale),
            "phase_scale": float(phase_scale),
            "shape": list(icube.shape),
            "motion_disp_rms_px": meta.get("motion", {}).get("telemetry", {}).get("disp_rms_px"),
            "phase_rms_rad": meta.get("phase_screen", {}).get("telemetry", {}).get("phase_rms_rad"),
            "methods": {},
        }
        policy_features = {
            "motion_disp_rms_px": meta.get("motion", {}).get("telemetry", {}).get("disp_rms_px"),
        }
        if args.stap_policy == "Brain-SIMUS-Clin-RegShiftP90-v0":
            policy_features.update(
                estimate_simus_policy_features(
                    icube,
                    reg_subpixel=4,
                    reg_reference="median",
                )
            )
        applied_stap_profile, stap_policy_info = select_simus_stap_profile(
            requested_profile=str(args.stap_profile),
            policy=args.stap_policy,
            feature_values=policy_features,
            motion_disp_rms_px=meta.get("motion", {}).get("telemetry", {}).get("disp_rms_px"),
        )
        details["runs"][run_slug]["stap_policy_features"] = policy_features
        details["runs"][run_slug]["stap_policy"] = stap_policy_info

        for method in methods:
            dataset_name = f"{run_slug}_{method.key}"
            bundle_root = eval_root / run_slug
            bundle_dir = bundle_root / slugify(dataset_name)
            if not bool(args.reuse_bundles) or not (bundle_dir / "meta.json").is_file():
                bundle_dir = derive_bundle_from_run(
                    run_dir=run_dir,
                    out_root=bundle_root,
                    dataset_name=dataset_name,
                    stap_profile=applied_stap_profile,
                    baseline_type=str(method.baseline_type),
                    run_stap=bool(method.run_stap),
                    stap_device=str(args.stap_device),
                    meta_extra={
                        "simus_motion_ladder": True,
                        "motion_scale": float(motion_scale),
                        "phase_scale": float(phase_scale),
                        "compare_method_key": str(method.key),
                        "compare_role": str(method.role),
                        "stap_policy": stap_policy_info,
                    },
                )
            score_path = _score_path(bundle_dir, eval_score=str(args.eval_score), role=str(method.role))
            score = np.load(score_path).astype(np.float32, copy=False)
            metrics = evaluate_structural_metrics(
                score=score,
                mask_h1_pf_main=mask_h1_pf_main,
                mask_h0_bg=mask_h0_bg,
                mask_h0_nuisance_pa=mask_h0_nuisance_pa,
                mask_h1_alias_qc=mask_h1_alias_qc,
                fprs=fprs,
                match_tprs=match_tprs,
            )
            row = {
                "run": run_slug,
                "profile": str(args.profile),
                "tier": str(args.tier),
                "seed": int(args.seed),
                "motion_scale": float(motion_scale),
                "phase_scale": float(phase_scale),
                "method": method.key,
                "method_label": _pipeline_label(method),
                "pipeline_label": _pipeline_label(method),
                "headline_label": _headline_label(str(args.eval_score), method),
                "upstream_baseline_label": _baseline_label(method.baseline_type),
                "baseline_type": method.baseline_type,
                "role": method.role,
                "run_stap": int(method.run_stap),
                "eval_score": str(args.eval_score),
                "score_label": _score_label(str(args.eval_score), method),
                "score_semantics": _score_semantics(str(args.eval_score), method),
                "bundle_dir": str(bundle_dir),
                "score_file": score_path.name,
                "stap_profile_requested": str(args.stap_profile),
                "stap_profile_applied": applied_stap_profile if method.run_stap else None,
                "stap_policy": str(args.stap_policy) if args.stap_policy else None,
                "T": int(icube.shape[0]),
                "H": int(icube.shape[1]),
                "W": int(icube.shape[2]),
                "prf_hz": meta.get("acquisition", {}).get("prf_hz"),
                "motion_disp_rms_px": meta.get("motion", {}).get("telemetry", {}).get("disp_rms_px"),
                "phase_rms_rad": meta.get("phase_screen", {}).get("telemetry", {}).get("phase_rms_rad"),
            }
            row.update(metrics)
            rows.append(row)
            details["runs"][run_slug]["methods"][method.key] = {
                "pipeline_label": _pipeline_label(method),
                "headline_label": _headline_label(str(args.eval_score), method),
                "score_label": _score_label(str(args.eval_score), method),
                "score_semantics": _score_semantics(str(args.eval_score), method),
                "bundle_dir": str(bundle_dir),
                "score_file": score_path.name,
                "metrics": metrics,
            }

    match_tag = "0p5" if any(abs(t - 0.5) < 1e-6 for t in match_tprs) else re.sub(r"[^\w]", "_", f"{match_tprs[0]:.3f}")
    details["scale_summary"] = _summarize_scale(rows, match_tag=match_tag)
    _write_csv(Path(args.out_csv), rows)
    _write_json(Path(args.out_json), {"rows": rows, "details": details})
    print(f"[simus-eval-motion] wrote {args.out_csv}")
    print(f"[simus-eval-motion] wrote {args.out_json}")


if __name__ == "__main__":
    main()
