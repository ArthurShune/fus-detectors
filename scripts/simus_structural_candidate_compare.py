#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.simus_eval_functional import _derive_bundle_timed
from scripts.simus_eval_structural import _baseline_label, evaluate_structural_metrics
from scripts.simus_functional_stap_head_search import (
    _load_json,
    _override_builder_lookup,
    _selected_residual_specs,
)
from sim.simus.bundle import slugify


@dataclass(frozen=True)
class Candidate:
    name: str
    detector_variant: str
    whiten_gamma: float
    diag_load: float
    cov_estimator: str
    huber_c: float
    stap_cov_train_trim_q: float
    mvdr_auto_kappa: float
    constraint_ridge: float


DEFAULT_CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        name="unwhitened_ref",
        detector_variant="unwhitened_ratio",
        whiten_gamma=0.0,
        diag_load=0.07,
        cov_estimator="tyler_pca",
        huber_c=5.0,
        stap_cov_train_trim_q=0.0,
        mvdr_auto_kappa=120.0,
        constraint_ridge=0.18,
    ),
    Candidate(
        name="huber_trim8",
        detector_variant="msd_ratio",
        whiten_gamma=1.0,
        diag_load=0.10,
        cov_estimator="huber",
        huber_c=5.0,
        stap_cov_train_trim_q=0.08,
        mvdr_auto_kappa=200.0,
        constraint_ridge=0.25,
    ),
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_candidate(spec: str) -> Candidate:
    parts = [x.strip() for x in str(spec).split(",")]
    if len(parts) != 9:
        raise ValueError(
            "--candidate expects "
            "name,detector_variant,whiten_gamma,diag_load,cov_estimator,huber_c,stap_cov_train_trim_q,mvdr_auto_kappa,constraint_ridge"
        )
    return Candidate(
        name=parts[0],
        detector_variant=parts[1],
        whiten_gamma=float(parts[2]),
        diag_load=float(parts[3]),
        cov_estimator=parts[4],
        huber_c=float(parts[5]),
        stap_cov_train_trim_q=float(parts[6]),
        mvdr_auto_kappa=float(parts[7]),
        constraint_ridge=float(parts[8]),
    )


def _candidate_bundle_overrides(base_overrides: dict[str, Any], cand: Candidate) -> dict[str, Any]:
    out = dict(base_overrides)
    out.update(
        {
            "stap_detector_variant": cand.detector_variant,
            "stap_whiten_gamma": float(cand.whiten_gamma),
            "diag_load": float(cand.diag_load),
            "cov_estimator": str(cand.cov_estimator),
            "huber_c": float(cand.huber_c),
            "stap_cov_train_trim_q": float(cand.stap_cov_train_trim_q),
            "mvdr_auto_kappa": float(cand.mvdr_auto_kappa),
            "constraint_ridge": float(cand.constraint_ridge),
        }
    )
    return out


def _setting_label(run_dir: Path) -> str:
    name = str(run_dir.name).lower()
    if "clin_mobile" in name:
        return "Mobile"
    if "clin_intraop_parenchyma" in name:
        return "Intra-operative parenchymal"
    return str(run_dir.name)


def _selection_score(row: dict[str, Any]) -> float:
    return (
        float(row.get("auc_main_vs_bg") or 0.0)
        + float(row.get("auc_main_vs_nuisance") or 0.0)
        - float(row.get("fpr_nuisance_match@0p5") or 0.0)
    )


def _summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["setting"]), str(row["method_family"]), str(row["candidate"]))
        grouped.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (setting, method_family, candidate), items in sorted(grouped.items()):
        def _mean(key: str) -> float | None:
            vals = np.asarray(
                [float(r[key]) for r in items if r.get(key) is not None and np.isfinite(r.get(key))],
                dtype=np.float64,
            )
            return float(np.mean(vals)) if vals.size else None

        rec = {
            "setting": setting,
            "method_family": method_family,
            "baseline_label": items[0]["baseline_label"],
            "config_name": items[0]["config_name"],
            "candidate": candidate,
            "count": int(len(items)),
            "mean_auc_main_vs_bg": _mean("auc_main_vs_bg"),
            "mean_auc_main_vs_nuisance": _mean("auc_main_vs_nuisance"),
            "mean_tpr_main@1e-04": _mean("tpr_main@1e-04"),
            "mean_tpr_main@1e-03": _mean("tpr_main@1e-03"),
            "mean_fpr_nuisance@1e-04": _mean("fpr_nuisance@1e-04"),
            "mean_fpr_nuisance@1e-03": _mean("fpr_nuisance@1e-03"),
            "mean_fpr_nuisance_match@0p5": _mean("fpr_nuisance_match@0p5"),
            "mean_runtime_ms": _mean("runtime_ms"),
        }
        rec["selection_score"] = (
            float(rec["mean_auc_main_vs_bg"] or 0.0)
            + float(rec["mean_auc_main_vs_nuisance"] or 0.0)
            - float(rec["mean_fpr_nuisance_match@0p5"] or 0.0)
        )
        out.append(rec)
    return out


def _compare_pairs(summary_rows: list[dict[str, Any]], candidate_names: list[str]) -> list[dict[str, Any]]:
    if len(candidate_names) != 2:
        return []
    a_name, b_name = candidate_names
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in summary_rows:
        key = (str(row["setting"]), str(row["method_family"]))
        grouped.setdefault(key, {})[str(row["candidate"])] = row
    out: list[dict[str, Any]] = []
    for (setting, method_family), items in sorted(grouped.items()):
        a = items.get(a_name)
        b = items.get(b_name)
        if a is None or b is None:
            continue
        out.append(
            {
                "setting": setting,
                "method_family": method_family,
                "baseline_label": a["baseline_label"],
                "config_name": a["config_name"],
                "candidate_a": a_name,
                "candidate_b": b_name,
                "delta_selection_b_minus_a": float(b["selection_score"]) - float(a["selection_score"]),
                "delta_auc_bg_b_minus_a": float(b["mean_auc_main_vs_bg"]) - float(a["mean_auc_main_vs_bg"]),
                "delta_auc_nuis_b_minus_a": float(b["mean_auc_main_vs_nuisance"]) - float(a["mean_auc_main_vs_nuisance"]),
                "delta_fpr_nuis_match_b_minus_a": float(b["mean_fpr_nuisance_match@0p5"]) - float(a["mean_fpr_nuisance_match@0p5"]),
                "delta_tpr_main_1e-03_b_minus_a": float(b["mean_tpr_main@1e-03"]) - float(a["mean_tpr_main@1e-03"]),
            }
        )
    return out


def _policy_summary(comparison_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in comparison_rows:
        grouped.setdefault(str(row["setting"]), []).append(row)
    out: list[dict[str, Any]] = []
    for setting, items in sorted(grouped.items()):
        deltas = np.asarray([float(r["delta_selection_b_minus_a"]) for r in items], dtype=np.float64)
        delta_fpr = np.asarray([float(r["delta_fpr_nuis_match_b_minus_a"]) for r in items], dtype=np.float64)
        delta_auc_bg = np.asarray([float(r["delta_auc_bg_b_minus_a"]) for r in items], dtype=np.float64)
        delta_auc_n = np.asarray([float(r["delta_auc_nuis_b_minus_a"]) for r in items], dtype=np.float64)
        wins_b = int(np.sum(deltas > 0.0))
        losses_b = int(np.sum(deltas < 0.0))
        ties = int(np.sum(np.isclose(deltas, 0.0)))
        recommended = "huber_trim8" if (float(np.mean(deltas)) > 0.0 and wins_b >= losses_b) else "unwhitened_ref"
        rationale = (
            "better mean selection score across residual families"
            if recommended == "huber_trim8"
            else "lower nuisance leakage / higher overall structural score across residual families"
        )
        out.append(
            {
                "setting": setting,
                "family_count": int(len(items)),
                "wins_huber_trim8": wins_b,
                "losses_huber_trim8": losses_b,
                "ties": ties,
                "mean_delta_selection_huber_minus_unwhitened": float(np.mean(deltas)),
                "mean_delta_auc_bg_huber_minus_unwhitened": float(np.mean(delta_auc_bg)),
                "mean_delta_auc_nuis_huber_minus_unwhitened": float(np.mean(delta_auc_n)),
                "mean_delta_fpr_nuis_match_huber_minus_unwhitened": float(np.mean(delta_fpr)),
                "recommended_candidate": recommended,
                "rationale": rationale,
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Head-to-head structural comparison for accepted SIMUS detector candidates.")
    ap.add_argument(
        "--search-json",
        type=Path,
        default=Path("reports/simus_v2/simus_fair_profile_search_seed125_126_to_127_128.json"),
    )
    ap.add_argument(
        "--stap-profile",
        type=str,
        default=None,
        help="Defaults to the selected accepted STAP profile from --search-json.",
    )
    ap.add_argument("--stap-device", type=str, default="cuda")
    ap.add_argument("--reuse-bundles", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--candidate",
        type=str,
        action="append",
        default=None,
        help=(
            "Optional candidate: "
            "name,detector_variant,whiten_gamma,diag_load,cov_estimator,huber_c,stap_cov_train_trim_q,mvdr_auto_kappa,constraint_ridge"
        ),
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("runs/sim_eval/simus_structural_candidate_compare"),
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/simus_v2/simus_structural_candidate_compare_rows.csv"),
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/simus_v2/simus_structural_candidate_compare_rows.json"),
    )
    ap.add_argument(
        "--out-summary-csv",
        type=Path,
        default=Path("reports/simus_v2/simus_structural_candidate_compare_summary.csv"),
    )
    ap.add_argument(
        "--out-summary-json",
        type=Path,
        default=Path("reports/simus_v2/simus_structural_candidate_compare_summary.json"),
    )
    ap.add_argument(
        "--out-policy-csv",
        type=Path,
        default=Path("reports/simus_v2/simus_structural_candidate_policy.csv"),
    )
    ap.add_argument(
        "--out-policy-json",
        type=Path,
        default=Path("reports/simus_v2/simus_structural_candidate_policy.json"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    candidates = [_parse_candidate(spec) for spec in args.candidate] if args.candidate else list(DEFAULT_CANDIDATES)
    payload = _load_json(Path(args.search_json))
    details = payload.get("details", {})
    eval_cases = [dict(x) for x in details.get("eval_cases") or []]
    if not eval_cases:
        raise ValueError(f"{args.search_json}: no details.eval_cases found")
    residual_specs = _selected_residual_specs(payload)
    override_lookup = _override_builder_lookup()
    selected_stap_profile = str(
        args.stap_profile
        or ((details.get("selected_configs") or {}).get("stap") or {}).get("config_name")
        or "Brain-SIMUS-Clin"
    )

    rows: list[dict[str, Any]] = []
    bundles_root = Path(args.out_root) / "bundles"
    bundles_root.mkdir(parents=True, exist_ok=True)

    for case in eval_cases:
        run_dir = Path(case["run_dir"])
        ds = run_dir / "dataset"
        setting = _setting_label(run_dir)
        mask_h1_pf_main = np.load(ds / "mask_h1_pf_main.npy").astype(bool, copy=False)
        mask_h0_bg = np.load(ds / "mask_h0_bg.npy").astype(bool, copy=False)
        mask_h0_nuisance_pa = np.load(ds / "mask_h0_nuisance_pa.npy").astype(bool, copy=False)
        mask_h1_alias_qc = np.load(ds / "mask_h1_alias_qc.npy").astype(bool, copy=False)
        icube_shape = tuple(int(v) for v in np.load(ds / "icube.npy", mmap_mode="r").shape)

        for residual in residual_specs:
            base_overrides = override_lookup[(residual["method_family"], residual["config_name"])](icube_shape)
            for cand in candidates:
                dataset_name = (
                    f"{run_dir.name}_{slugify(residual['method_family'])}_{slugify(residual['config_name'])}_{slugify(cand.name)}"
                )
                bundle_dir, elapsed_ms = _derive_bundle_timed(
                    run_dir=run_dir,
                    out_root=bundles_root / slugify(setting),
                    dataset_name=dataset_name,
                    baseline_type=str(residual["baseline_type"]),
                    bundle_overrides=_candidate_bundle_overrides(base_overrides, cand),
                    run_stap=True,
                    stap_profile=selected_stap_profile,
                    stap_device=str(args.stap_device),
                    meta_extra={
                        "simus_structural_candidate_compare": True,
                        "setting": setting,
                        "candidate": cand.name,
                    },
                    reuse_bundle=bool(args.reuse_bundles),
                )
                score = np.load(bundle_dir / "score_stap_preka.npy").astype(np.float32, copy=False)
                metrics = evaluate_structural_metrics(
                    score=score,
                    mask_h1_pf_main=mask_h1_pf_main,
                    mask_h0_bg=mask_h0_bg,
                    mask_h0_nuisance_pa=mask_h0_nuisance_pa,
                    mask_h1_alias_qc=mask_h1_alias_qc,
                    fprs=[1e-4, 1e-3],
                    match_tprs=[0.5],
                )
                row = {
                    "case_name": str(case["name"]),
                    "run_dir": str(run_dir),
                    "setting": setting,
                    "method_family": str(residual["method_family"]),
                    "config_name": str(residual["config_name"]),
                    "baseline_type": str(residual["baseline_type"]),
                    "baseline_label": _baseline_label(str(residual["baseline_type"])),
                    "candidate": cand.name,
                    "stap_profile": selected_stap_profile,
                    "bundle_dir": str(bundle_dir),
                    "runtime_ms": elapsed_ms,
                    "auc_main_vs_bg": metrics.get("auc_main_vs_bg"),
                    "auc_main_vs_nuisance": metrics.get("auc_main_vs_nuisance"),
                    "tpr_main@1e-04": metrics.get("tpr_main@1e-04"),
                    "tpr_main@1e-03": metrics.get("tpr_main@1e-03"),
                    "fpr_nuisance@1e-04": metrics.get("fpr_nuisance@1e-04"),
                    "fpr_nuisance@1e-03": metrics.get("fpr_nuisance@1e-03"),
                    "fpr_bg_match@0p5": metrics.get("fpr_bg_match@0p5"),
                    "fpr_nuisance_match@0p5": metrics.get("fpr_nuisance_match@0p5"),
                    "tpr_alias_qc_match@0p5": metrics.get("tpr_alias_qc_match@0p5"),
                }
                row["selection_score"] = _selection_score(row)
                rows.append(row)

    summary_rows = _summarize_rows(rows)
    candidate_names = [cand.name for cand in candidates]
    comparison_rows = _compare_pairs(summary_rows, candidate_names)
    policy_rows = _policy_summary(comparison_rows)

    full_payload = {
        "schema_version": "simus_structural_candidate_compare.v1",
        "search_json": str(args.search_json),
        "selected_stap_profile": selected_stap_profile,
        "eval_cases": eval_cases,
        "candidates": [asdict(cand) for cand in candidates],
        "rows": rows,
        "summary": summary_rows,
        "comparison": comparison_rows,
        "policy": policy_rows,
    }
    _write_csv(Path(args.out_csv), rows)
    _write_json(Path(args.out_json), full_payload)
    _write_csv(Path(args.out_summary_csv), summary_rows)
    _write_json(Path(args.out_summary_json), {"summary": summary_rows, "comparison": comparison_rows})
    _write_csv(Path(args.out_policy_csv), policy_rows)
    _write_json(Path(args.out_policy_json), {"policy": policy_rows, "comparison": comparison_rows})


if __name__ == "__main__":
    main()
