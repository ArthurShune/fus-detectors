#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.simus_eval_functional import (
    FPR_TAGS,
    _corr,
    _derive_bundle_timed,
    _glm_tmap,
    _pipeline_label,
    _score_selection,
    _split_csv_list,
    _threshold_from_null,
)
from scripts.simus_eval_structural import evaluate_structural_metrics
from scripts.simus_functional_stap_head_search import (
    _load_json,
    _load_score,
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


def _summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["split"]),
            str(row["base_profile"]),
            str(row["method_family"]),
            str(row["candidate"]),
            str(row["detector_head"]),
        )
        grouped.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        split, base_profile, method_family, candidate, detector_head = key
        rec = {
            "split": split,
            "base_profile": base_profile,
            "method_family": method_family,
            "candidate": candidate,
            "detector_head": detector_head,
            "count": len(items),
            "pipeline_label": f"{method_family} -> {candidate}::{detector_head}",
        }
        for metric in (
            "auc_activation_vs_bg",
            "auc_activation_vs_nuisance",
            "runtime_ms",
            "roi_corr_task",
            "roi_corr_null",
            "roi_tpr_task@1e-04",
            "roi_tpr_task@1e-03",
            "outside_frac_task@1e-04",
            "outside_frac_task@1e-03",
            "nuisance_frac_task@1e-04",
            "nuisance_frac_task@1e-03",
        ):
            vals = np.asarray(
                [float(x[metric]) for x in items if x.get(metric) is not None and np.isfinite(x.get(metric))],
                dtype=np.float64,
            )
            rec[f"mean_{metric}"] = float(np.mean(vals)) if vals.size else None
        rec["selection_score"] = _score_selection(rec)
        out.append(rec)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare fixed detector candidates on the accepted SIMUS functional audit.")
    ap.add_argument(
        "--search-json",
        type=Path,
        default=Path("reports/simus_v2/simus_fair_profile_search_seed125_126_to_127_128.json"),
    )
    ap.add_argument(
        "--functional-json",
        type=Path,
        default=Path("reports/simus_v2/simus_eval_functional_seed221_222_to_223_224_ec6_bgcdf_outside.json"),
    )
    ap.add_argument(
        "--cases-root",
        type=Path,
        default=Path("runs/sim_eval/simus_v2_functional_eval_seed221_222_to_223_224_ec6_bgcdf_outside/cases"),
    )
    ap.add_argument("--stap-profile", type=str, default="Brain-SIMUS-Clin")
    ap.add_argument("--stap-device", type=str, default="cuda")
    ap.add_argument("--reuse-bundles", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--splits", type=str, default="eval", help="Comma-separated subset of dev,eval.")
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
        default=Path("runs/sim_eval/simus_functional_candidate_compare"),
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/simus_v2/simus_functional_candidate_compare.csv"),
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/simus_v2/simus_functional_candidate_compare.json"),
    )
    ap.add_argument(
        "--out-summary-csv",
        type=Path,
        default=Path("reports/simus_v2/simus_functional_candidate_compare_summary.csv"),
    )
    ap.add_argument(
        "--out-summary-json",
        type=Path,
        default=Path("reports/simus_v2/simus_functional_candidate_compare_summary.json"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    candidates = [_parse_candidate(spec) for spec in args.candidate] if args.candidate else list(DEFAULT_CANDIDATES)
    search_payload = _load_json(Path(args.search_json))
    functional_payload = _load_json(Path(args.functional_json))
    residual_specs = _selected_residual_specs(search_payload)
    override_lookup = _override_builder_lookup()
    base_profiles = [str(x) for x in functional_payload["base_profiles"]]
    splits = {x.strip().lower() for x in _split_csv_list(str(args.splits))}

    dev_seeds = [int(x) for x in functional_payload.get("dev_seeds") or []]
    eval_seeds = [int(x) for x in functional_payload.get("eval_seeds") or []]
    cases_root = Path(args.cases_root)
    out_root = Path(args.out_root)
    bundles_root = out_root / "bundles"
    bundles_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    def _case_root(split: str, base_profile: str, seed: int, null_run: bool) -> Path:
        tag = "null" if bool(null_run) else "task"
        return cases_root / split / f"{slugify(base_profile)}_seed{int(seed)}_{tag}"

    def _eval_case(split: str, base_profile: str, seed: int) -> None:
        task_root = _case_root(split, base_profile, seed, False)
        null_root = _case_root(split, base_profile, seed, True)
        reg = np.load(task_root / "hemo_regressor.npy").astype(np.float32, copy=False)
        activation_roi = np.load(task_root / "mask_activation_roi.npy").astype(bool, copy=False)
        mask_bg = np.load(task_root / "mask_h0_bg.npy").astype(bool, copy=False)
        nuisance_mask = np.zeros_like(mask_bg, dtype=bool)
        for name in ("mask_h0_nuisance_pa.npy", "mask_h0_specular_struct.npy"):
            p = task_root / name
            if p.is_file():
                nuisance_mask |= np.load(p).astype(bool, copy=False)
        outside_mask = (mask_bg | nuisance_mask) & (~activation_roi)
        task_rows = _load_json(task_root / "ensemble_table.json")["rows"]
        null_rows = _load_json(null_root / "ensemble_table.json")["rows"]
        icube_shape = tuple(
            int(v) for v in np.load(Path(task_rows[0]["dataset_dir"]) / "icube.npy", mmap_mode="r").shape
        )

        for residual in residual_specs:
            base_overrides = override_lookup[(residual["method_family"], residual["config_name"])](icube_shape)
            for cand in candidates:
                task_scores: dict[str, list[np.ndarray]] = {"stap": [], "stap_pd": []}
                null_scores: dict[str, list[np.ndarray]] = {"stap": [], "stap_pd": []}
                head_runtimes_ms: dict[str, list[float]] = {"stap": [], "stap_pd": []}
                bundle_overrides = _candidate_bundle_overrides(base_overrides, cand)
                for task_row, null_row in zip(task_rows, null_rows):
                    dataset_stub = (
                        f"{slugify(base_profile)}_seed{int(seed)}_{slugify(residual['method_family'])}"
                        f"_{slugify(residual['config_name'])}_{slugify(cand.name)}"
                    )
                    task_bundle_root = bundles_root / split / f"{slugify(base_profile)}_seed{int(seed)}" / "task"
                    null_bundle_root = bundles_root / split / f"{slugify(base_profile)}_seed{int(seed)}" / "null"
                    task_bundle_root.mkdir(parents=True, exist_ok=True)
                    null_bundle_root.mkdir(parents=True, exist_ok=True)
                    task_bundle_dir, task_ms = _derive_bundle_timed(
                        run_dir=Path(task_row["ensemble_dir"]),
                        out_root=task_bundle_root,
                        dataset_name=f"{dataset_stub}_e{int(task_row['ensemble_index']):03d}",
                        baseline_type=residual["baseline_type"],
                        bundle_overrides=bundle_overrides,
                        run_stap=True,
                        stap_profile=str(args.stap_profile),
                        stap_device=str(args.stap_device),
                        meta_extra={
                            "simus_functional_candidate_compare": True,
                            "candidate": cand.name,
                            "split": split,
                            "base_profile": base_profile,
                        },
                        reuse_bundle=bool(args.reuse_bundles),
                    )
                    null_bundle_dir, null_ms = _derive_bundle_timed(
                        run_dir=Path(null_row["ensemble_dir"]),
                        out_root=null_bundle_root,
                        dataset_name=f"{dataset_stub}_e{int(null_row['ensemble_index']):03d}",
                        baseline_type=residual["baseline_type"],
                        bundle_overrides=bundle_overrides,
                        run_stap=True,
                        stap_profile=str(args.stap_profile),
                        stap_device=str(args.stap_device),
                        meta_extra={
                            "simus_functional_candidate_compare": True,
                            "candidate": cand.name,
                            "split": split,
                            "base_profile": base_profile,
                            "null": True,
                        },
                        reuse_bundle=bool(args.reuse_bundles),
                    )
                    if task_ms is not None:
                        head_runtimes_ms["stap"].append(float(task_ms))
                        head_runtimes_ms["stap_pd"].append(float(task_ms))
                    if null_ms is not None:
                        head_runtimes_ms["stap"].append(float(null_ms))
                        head_runtimes_ms["stap_pd"].append(float(null_ms))
                    task_scores["stap"].append(_load_score(task_bundle_dir, "stap"))
                    task_scores["stap_pd"].append(_load_score(task_bundle_dir, "stap_pd"))
                    null_scores["stap"].append(_load_score(null_bundle_dir, "stap"))
                    null_scores["stap_pd"].append(_load_score(null_bundle_dir, "stap_pd"))

                for detector_head in ("stap", "stap_pd"):
                    task_stack = np.stack(task_scores[detector_head], axis=0).astype(np.float32, copy=False)
                    null_stack = np.stack(null_scores[detector_head], axis=0).astype(np.float32, copy=False)
                    task_map = _glm_tmap(task_stack, reg)
                    null_map = _glm_tmap(null_stack, reg)
                    structural = evaluate_structural_metrics(
                        score=task_map,
                        mask_h1_pf_main=activation_roi,
                        mask_h0_bg=mask_bg & (~activation_roi),
                        mask_h0_nuisance_pa=nuisance_mask,
                        mask_h1_alias_qc=None,
                        fprs=[1e-4, 1e-3],
                        match_tprs=[0.5],
                    )
                    roi_series_task = (
                        np.mean(task_stack[:, activation_roi], axis=1)
                        if np.any(activation_roi)
                        else np.zeros((task_stack.shape[0],), dtype=np.float32)
                    )
                    roi_series_null = (
                        np.mean(null_stack[:, activation_roi], axis=1)
                        if np.any(activation_roi)
                        else np.zeros((null_stack.shape[0],), dtype=np.float32)
                    )
                    row: dict[str, Any] = {
                        "split": split,
                        "base_profile": base_profile,
                        "seed": int(seed),
                        "method_family": str(residual["method_family"]),
                        "config_name": str(residual["config_name"]),
                        "baseline_type": str(residual["baseline_type"]),
                        "candidate": cand.name,
                        "detector_head": detector_head,
                        "stap_profile": str(args.stap_profile),
                        "pipeline_label": _pipeline_label(str(residual["baseline_type"]), detector_head),
                        "runtime_ms": float(np.sum(head_runtimes_ms[detector_head])) if head_runtimes_ms[detector_head] else None,
                        "auc_activation_vs_bg": float(structural["auc_main_vs_bg"]),
                        "auc_activation_vs_nuisance": float(structural["auc_main_vs_nuisance"]),
                        "roi_corr_task": _corr(roi_series_task, reg),
                        "roi_corr_null": _corr(roi_series_null, reg),
                    }
                    for tag, fpr in zip(FPR_TAGS, (1e-4, 1e-3)):
                        thr, emp = _threshold_from_null(null_map, outside_mask, fpr)
                        row[f"thr_null@{tag}"] = thr
                        row[f"outside_fpr_null@{tag}"] = emp
                        if thr is None:
                            row[f"roi_tpr_task@{tag}"] = None
                            row[f"outside_frac_task@{tag}"] = None
                            row[f"nuisance_frac_task@{tag}"] = None
                        else:
                            row[f"roi_tpr_task@{tag}"] = float(np.mean(task_map[activation_roi] >= thr)) if np.any(activation_roi) else None
                            row[f"outside_frac_task@{tag}"] = float(np.mean(task_map[outside_mask] >= thr)) if np.any(outside_mask) else None
                            row[f"nuisance_frac_task@{tag}"] = float(np.mean(task_map[nuisance_mask] >= thr)) if np.any(nuisance_mask) else None
                    rows.append(row)

    if "dev" in splits:
        for base_profile in base_profiles:
            for seed in dev_seeds:
                _eval_case("dev", base_profile, int(seed))
    if "eval" in splits:
        for base_profile in base_profiles:
            for seed in eval_seeds:
                _eval_case("eval", base_profile, int(seed))

    summary_rows = _summarize_rows(rows)
    comparison: list[dict[str, Any]] = []
    pair_groups: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = {}
    for row in summary_rows:
        key = (str(row["split"]), str(row["base_profile"]), str(row["method_family"]), str(row["detector_head"]))
        pair_groups.setdefault(key, {})[str(row["candidate"])] = row
    cand_names = [cand.name for cand in candidates]
    if len(cand_names) == 2:
        a_name, b_name = cand_names
        for key, items in sorted(pair_groups.items()):
            a = items.get(a_name)
            b = items.get(b_name)
            if a is None or b is None:
                continue
            comparison.append(
                {
                    "split": key[0],
                    "base_profile": key[1],
                    "method_family": key[2],
                    "detector_head": key[3],
                    "candidate_a": a_name,
                    "candidate_b": b_name,
                    "delta_selection_b_minus_a": float(b["selection_score"]) - float(a["selection_score"]),
                    "delta_auc_bg_b_minus_a": float(b["mean_auc_activation_vs_bg"]) - float(a["mean_auc_activation_vs_bg"]),
                    "delta_auc_nuis_b_minus_a": float(b["mean_auc_activation_vs_nuisance"]) - float(a["mean_auc_activation_vs_nuisance"]),
                    "delta_nuis_frac@1e-03_b_minus_a": float(b["mean_nuisance_frac_task@1e-03"]) - float(a["mean_nuisance_frac_task@1e-03"]),
                    "delta_roi_tpr@1e-03_b_minus_a": float(b["mean_roi_tpr_task@1e-03"]) - float(a["mean_roi_tpr_task@1e-03"]),
                }
            )

    payload = {
        "schema_version": "simus_functional_candidate_compare.v1",
        "search_json": str(args.search_json),
        "functional_json": str(args.functional_json),
        "cases_root": str(args.cases_root),
        "stap_profile": str(args.stap_profile),
        "splits": sorted(splits),
        "candidates": [asdict(cand) for cand in candidates],
        "rows": rows,
        "summary": summary_rows,
        "comparison": comparison,
    }
    _write_csv(Path(args.out_csv), rows)
    _write_json(Path(args.out_json), payload)
    _write_csv(Path(args.out_summary_csv), summary_rows)
    _write_json(Path(args.out_summary_json), {"summary": summary_rows, "comparison": comparison})


if __name__ == "__main__":
    main()
