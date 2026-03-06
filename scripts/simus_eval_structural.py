#!/usr/bin/env python3
"""
Evaluate SIMUS/PyMUST structural benchmarks using the explicit H1/H0 label contract.

Primary task:
  H1 = mask_h1_pf_main
  H0 tail calibration = mask_h0_bg

Additional diagnostics:
  - nuisance-region FPR on mask_h0_nuisance_pa at the same bg-calibrated threshold
  - alias QC hit rate on mask_h1_alias_qc at the same threshold
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sim.simus.bundle import (
    SUPPORTED_SIMUS_STAP_PROFILES,
    derive_bundle_from_run,
    load_canonical_run,
    slugify,
)


@dataclass(frozen=True)
class MethodSpec:
    key: str
    baseline_type: str
    run_stap: bool
    role: str


def _baseline_label(baseline_type: str) -> str:
    raw = str(baseline_type).strip().lower()
    mapping = {
        "mc_svd": "MC-SVD",
        "svd_similarity": "Adaptive Local SVD",
        "local_svd": "Local SVD",
        "rpca": "RPCA",
        "hosvd": "HOSVD",
    }
    return mapping.get(raw, raw.replace("_", " ").upper())


def _pipeline_label(method: MethodSpec) -> str:
    base = _baseline_label(method.baseline_type)
    if method.role == "stap":
        return f"{base} -> STAP"
    return base


def _score_semantics(eval_score: str, method: MethodSpec) -> str:
    mode = str(eval_score).strip().lower()
    if method.role == "stap":
        if mode == "vnext":
            return "stap_detector_score_pre_ka"
        return "pd_after_stap"
    if mode == "vnext":
        return "baseline_score"
    return "baseline_pd"


def _score_label(eval_score: str, method: MethodSpec) -> str:
    mode = str(eval_score).strip().lower()
    if method.role == "stap":
        if mode == "vnext":
            return "STAP detector"
        return "PD-after-STAP"
    if mode == "vnext":
        return "Baseline score"
    return "Baseline PD"


def _headline_label(eval_score: str, method: MethodSpec) -> str:
    return f"{_pipeline_label(method)} ({_score_label(eval_score, method)})"


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


def _finite(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _auc_pos_vs_neg(pos: np.ndarray, neg: np.ndarray) -> float | None:
    pos = _finite(pos)
    neg = _finite(neg)
    if pos.size == 0 or neg.size == 0:
        return None
    neg_sorted = np.sort(neg)
    less = np.searchsorted(neg_sorted, pos, side="left")
    right = np.searchsorted(neg_sorted, pos, side="right")
    equal = right - less
    return float((float(np.sum(less)) + 0.5 * float(np.sum(equal))) / float(pos.size * neg.size))


def _threshold_from_neg(neg: np.ndarray, fpr: float) -> tuple[float | None, float | None]:
    neg = _finite(neg)
    if neg.size == 0:
        return None, None
    q = float(np.clip(1.0 - float(fpr), 0.0, 1.0))
    thr = float(np.quantile(neg, q))
    fpr_emp = float(np.mean(neg >= thr))
    return thr, fpr_emp


def _threshold_from_pos(pos: np.ndarray, tpr: float) -> tuple[float | None, float | None]:
    pos = _finite(pos)
    if pos.size == 0:
        return None, None
    q = float(np.clip(1.0 - float(tpr), 0.0, 1.0))
    thr = float(np.quantile(pos, q))
    tpr_emp = float(np.mean(pos >= thr))
    return thr, tpr_emp


def _rate_tag(rate: float) -> str:
    return f"{float(rate):.3f}".rstrip("0").rstrip(".").replace(".", "p")


def evaluate_structural_metrics(
    *,
    score: np.ndarray,
    mask_h1_pf_main: np.ndarray,
    mask_h0_bg: np.ndarray,
    mask_h0_nuisance_pa: np.ndarray | None,
    mask_h1_alias_qc: np.ndarray | None,
    fprs: list[float],
    match_tprs: list[float] | None = None,
) -> dict[str, Any]:
    score = np.asarray(score, dtype=np.float64)
    pos_main = score[np.asarray(mask_h1_pf_main, dtype=bool)]
    neg_bg = score[np.asarray(mask_h0_bg, dtype=bool)]
    neg_nuisance = (
        score[np.asarray(mask_h0_nuisance_pa, dtype=bool)] if mask_h0_nuisance_pa is not None else np.asarray([], dtype=np.float64)
    )
    pos_alias = (
        score[np.asarray(mask_h1_alias_qc, dtype=bool)] if mask_h1_alias_qc is not None else np.asarray([], dtype=np.float64)
    )
    out: dict[str, Any] = {
        "n_h1_pf_main": int(np.asarray(mask_h1_pf_main, dtype=bool).sum()),
        "n_h0_bg": int(np.asarray(mask_h0_bg, dtype=bool).sum()),
        "n_h0_nuisance_pa": int(np.asarray(mask_h0_nuisance_pa, dtype=bool).sum()) if mask_h0_nuisance_pa is not None else 0,
        "n_h1_alias_qc": int(np.asarray(mask_h1_alias_qc, dtype=bool).sum()) if mask_h1_alias_qc is not None else 0,
        "auc_main_vs_bg": _auc_pos_vs_neg(pos_main, neg_bg),
        "auc_main_vs_nuisance": _auc_pos_vs_neg(pos_main, neg_nuisance) if neg_nuisance.size else None,
        "fpr_floor_bg": (1.0 / float(max(1, neg_bg.size))) if neg_bg.size else None,
        "fpr_floor_nuisance": (1.0 / float(max(1, neg_nuisance.size))) if neg_nuisance.size else None,
    }
    for fpr in fprs:
        tag = f"{float(fpr):.0e}"
        thr, bg_emp = _threshold_from_neg(neg_bg, fpr)
        if thr is None:
            out[f"thr@{tag}"] = None
            out[f"tpr_main@{tag}"] = None
            out[f"fpr_bg@{tag}"] = None
            out[f"fpr_nuisance@{tag}"] = None
            out[f"tpr_alias_qc@{tag}"] = None
            continue
        out[f"thr@{tag}"] = thr
        out[f"tpr_main@{tag}"] = float(np.mean(pos_main >= thr)) if pos_main.size else None
        out[f"fpr_bg@{tag}"] = bg_emp
        out[f"fpr_nuisance@{tag}"] = float(np.mean(neg_nuisance >= thr)) if neg_nuisance.size else None
        out[f"tpr_alias_qc@{tag}"] = float(np.mean(pos_alias >= thr)) if pos_alias.size else None
    for tpr in match_tprs or []:
        tag = _rate_tag(tpr)
        thr, tpr_emp = _threshold_from_pos(pos_main, tpr)
        if thr is None:
            out[f"thr_match_tpr@{tag}"] = None
            out[f"tpr_main_match@{tag}"] = None
            out[f"fpr_bg_match@{tag}"] = None
            out[f"fpr_nuisance_match@{tag}"] = None
            out[f"tpr_alias_qc_match@{tag}"] = None
            continue
        out[f"thr_match_tpr@{tag}"] = thr
        out[f"tpr_main_match@{tag}"] = tpr_emp
        out[f"fpr_bg_match@{tag}"] = float(np.mean(neg_bg >= thr)) if neg_bg.size else None
        out[f"fpr_nuisance_match@{tag}"] = float(np.mean(neg_nuisance >= thr)) if neg_nuisance.size else None
        out[f"tpr_alias_qc_match@{tag}"] = float(np.mean(pos_alias >= thr)) if pos_alias.size else None
    return out


def _discover_runs(sim_root: Path) -> list[Path]:
    root = Path(sim_root)
    if (root / "dataset" / "meta.json").is_file():
        return [root]
    runs = [p for p in sorted(root.iterdir()) if p.is_dir() and (p / "dataset" / "meta.json").is_file()]
    if not runs:
        raise FileNotFoundError(f"{root}: no run directories with dataset/meta.json")
    return runs


def _split_csv_list(spec: str) -> list[str]:
    return [s.strip() for s in (spec or "").split(",") if s.strip()]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Structural SIMUS evaluation using H1/H0 masks.")
    ap.add_argument("--sim-root", type=Path, action="append", default=None, help="Root containing per-run subdirectories.")
    ap.add_argument("--run", type=Path, action="append", default=None, help="Explicit run directory (repeatable).")
    ap.add_argument("--out-root", type=Path, default=Path("runs/sim_eval/simus_structural_eval"))
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/simus_structural/simus_structural_eval.csv"),
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/simus_structural/simus_structural_eval.json"),
    )
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument(
        "--stap-profile",
        type=str,
        default="Brain-SIMUS-Clin",
        choices=list(SUPPORTED_SIMUS_STAP_PROFILES),
    )
    ap.add_argument("--eval-score", type=str, default="pd", choices=["pd", "vnext"])
    ap.add_argument("--baselines", type=str, default="mc_svd,svd_similarity,local_svd,rpca,hosvd")
    ap.add_argument("--stap-baseline", type=str, default="mc_svd")
    ap.add_argument("--stap-device", type=str, default="cpu")
    ap.add_argument("--fprs", type=str, default="1e-4,3e-4,1e-3")
    ap.add_argument("--match-tprs", type=str, default="0.25,0.5,0.75")
    ap.add_argument("--reuse-bundles", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    runs: list[Path] = []
    for root in args.sim_root or []:
        runs.extend(_discover_runs(Path(root)))
    for run in args.run or []:
        runs.append(Path(run))
    if not runs:
        raise ValueError("Provide --sim-root or at least one --run")

    baselines = _split_csv_list(str(args.baselines))
    methods = [MethodSpec(key=f"baseline_{b}", baseline_type=b, run_stap=False, role="baseline") for b in baselines]
    methods.append(MethodSpec(key="stap", baseline_type=str(args.stap_baseline), run_stap=True, role="stap"))
    fprs = [float(x) for x in _split_csv_list(str(args.fprs))]
    match_tprs = [float(x) for x in _split_csv_list(str(args.match_tprs))]
    eval_score = str(args.eval_score)
    tag = str(args.tag).strip() if args.tag else ""

    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {"schema_version": "simus_structural_eval.v2", "runs": {}, "match_tprs": match_tprs}

    os.environ.setdefault("STAP_FAST_CUDA_GRAPH", "0")

    seen: set[str] = set()
    for run_dir in runs:
        run_dir = Path(run_dir)
        key = str(run_dir.resolve())
        if key in seen:
            continue
        seen.add(key)

        icube, masks, meta = load_canonical_run(run_dir)
        mask_h1_pf_main = masks.get("mask_h1_pf_main")
        mask_h0_bg = masks.get("mask_h0_bg")
        if mask_h1_pf_main is None or mask_h0_bg is None:
            raise ValueError(f"{run_dir}: dataset is missing mask_h1_pf_main.npy or mask_h0_bg.npy")
        mask_h0_nuisance_pa = masks.get("mask_h0_nuisance_pa")
        mask_h1_alias_qc = masks.get("mask_h1_alias_qc")

        run_key = run_dir.name
        details["runs"][run_key] = {
            "run_dir": str(run_dir),
            "profile": meta.get("simus", {}).get("profile"),
            "tier": meta.get("simus", {}).get("tier"),
            "prf_hz": meta.get("acquisition", {}).get("prf_hz"),
            "shape": list(icube.shape),
            "motion_disp_rms_px": meta.get("motion", {}).get("telemetry", {}).get("disp_rms_px"),
            "phase_rms_rad": meta.get("phase_screen", {}).get("telemetry", {}).get("phase_rms_rad"),
        }

        for method in methods:
            parts = [run_key]
            if tag:
                parts.append(tag)
            parts.append(method.key)
            dataset_name = "_".join(parts)
            bundle_root = Path(args.out_root) / run_key
            bundle_dir = bundle_root / slugify(dataset_name)
            if not bool(args.reuse_bundles) or not (bundle_dir / "meta.json").is_file():
                bundle_dir = derive_bundle_from_run(
                    run_dir=run_dir,
                    out_root=bundle_root,
                    dataset_name=dataset_name,
                    stap_profile=str(args.stap_profile),
                    baseline_type=str(method.baseline_type),
                    run_stap=bool(method.run_stap),
                    stap_device=str(args.stap_device),
                    meta_extra={
                        "simus_structural_eval": True,
                        "compare_method_key": str(method.key),
                        "compare_role": str(method.role),
                    },
                )

            if eval_score == "pd":
                candidates = [bundle_dir / "score_pd_base.npy"] if method.role == "baseline" else [bundle_dir / "score_pd_stap.npy"]
                candidates += [bundle_dir / "score_base.npy"] if method.role == "baseline" else [bundle_dir / "score_pd_base.npy"]
            else:
                candidates = [bundle_dir / "score_base.npy"] if method.role == "baseline" else [bundle_dir / "score_stap_preka.npy", bundle_dir / "score_stap.npy"]
            score_path = next((p for p in candidates if p.is_file()), candidates[0])
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
                "run": run_key,
                "method": method.key,
                "method_label": _pipeline_label(method),
                "pipeline_label": _pipeline_label(method),
                "headline_label": _headline_label(eval_score, method),
                "upstream_baseline_label": _baseline_label(method.baseline_type),
                "baseline_type": method.baseline_type,
                "role": method.role,
                "run_stap": int(method.run_stap),
                "eval_score": eval_score,
                "score_label": _score_label(eval_score, method),
                "score_semantics": _score_semantics(eval_score, method),
                "stap_profile": str(args.stap_profile),
                "bundle_dir": str(bundle_dir),
                "score_file": score_path.name,
                "profile": meta.get("simus", {}).get("profile"),
                "tier": meta.get("simus", {}).get("tier"),
                "T": int(icube.shape[0]),
                "H": int(icube.shape[1]),
                "W": int(icube.shape[2]),
                "prf_hz": meta.get("acquisition", {}).get("prf_hz"),
                "motion_disp_rms_px": meta.get("motion", {}).get("telemetry", {}).get("disp_rms_px"),
                "phase_rms_rad": meta.get("phase_screen", {}).get("telemetry", {}).get("phase_rms_rad"),
            }
            row.update(metrics)
            rows.append(row)
            details["runs"][run_key].setdefault("methods", {})[method.key] = {
                "bundle_dir": str(bundle_dir),
                "score_file": score_path.name,
                "pipeline_label": _pipeline_label(method),
                "headline_label": _headline_label(eval_score, method),
                "score_label": _score_label(eval_score, method),
                "score_semantics": _score_semantics(eval_score, method),
                "upstream_baseline_label": _baseline_label(method.baseline_type),
                "metrics": metrics,
            }

    _write_csv(Path(args.out_csv), rows)
    _write_json(Path(args.out_json), {"rows": rows, "details": details})
    print(f"[simus-eval-structural] wrote {args.out_csv}")
    print(f"[simus-eval-structural] wrote {args.out_json}")


if __name__ == "__main__":
    main()
