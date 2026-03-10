#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.simus_eval_functional import (
    FPR_TAGS,
    _derive_bundle_timed,
    _best_native_simple_by_family,
    _corr,
    _detector_label,
    _extract_matching_eval,
    _glm_tmap,
    _pipeline_label,
    _score_selection,
    _split_csv_list,
    _summarize_rows,
    _threshold_from_null,
    _write_csv,
    _write_json,
)
from scripts.simus_eval_structural import _baseline_label, evaluate_structural_metrics
from scripts.simus_fair_profile_search import _candidate_specs
from sim.simus.bundle import slugify


FIXED_STAP_PROFILES = (
    "Brain-SIMUS-Clin",
    "Brain-SIMUS-Clin-MotionWide-v0",
    "Brain-SIMUS-Clin-MotionShort-v0",
    "Brain-SIMUS-Clin-MotionMid-v0",
    "Brain-SIMUS-Clin-MotionLong-v0",
    "Brain-SIMUS-Clin-MotionRobust-v0",
    "Brain-SIMUS-Clin-MotionMidRobust-v0",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _override_builder_lookup() -> dict[tuple[str, str], Any]:
    out: dict[tuple[str, str], Any] = {}
    for cand in _candidate_specs():
        out[(str(cand.method_family), str(cand.config_name))] = cand.override_builder
    return out


def _selected_residual_specs(search_payload: dict[str, Any]) -> list[dict[str, str]]:
    selected = dict(search_payload.get("details", {}).get("selected_configs") or search_payload.get("selected_configs") or {})
    by_key: dict[tuple[str, str], Any] = {}
    for cand in _candidate_specs():
        by_key[(str(cand.method_family), str(cand.config_name))] = cand
    out: list[dict[str, str]] = []
    for method_family, summary in selected.items():
        if str(method_family) == "stap":
            continue
        key = (str(method_family), str(summary["config_name"]))
        cand = by_key.get(key)
        if cand is None:
            raise KeyError(f"selected config not found in candidate grid: {key}")
        out.append(
            {
                "method_family": str(cand.method_family),
                "config_name": str(cand.config_name),
                "baseline_type": str(cand.method.baseline_type),
                "override_builder_name": str(cand.override_builder.__name__),
            }
        )
    return sorted(out, key=lambda x: x["method_family"])


def _load_score(bundle_dir: Path, detector_head: str) -> np.ndarray:
    name = str(detector_head).strip().lower()
    if name == "stap":
        path = bundle_dir / "score_stap_preka.npy"
    elif name == "stap_pd":
        path = bundle_dir / "score_pd_stap.npy"
    else:
        raise ValueError(f"unsupported STAP detector head {detector_head!r}")
    return np.load(path).astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Search functional STAP profile/observable choices on accepted SIMUS v2 cases.")
    ap.add_argument("--search-json", type=Path, required=True)
    ap.add_argument("--functional-json", type=Path, required=True)
    ap.add_argument("--cases-root", type=Path, required=True)
    ap.add_argument(
        "--stap-profiles",
        type=str,
        default=",".join(FIXED_STAP_PROFILES),
        help="Comma-separated frozen STAP profiles to search.",
    )
    ap.add_argument("--stap-device", type=str, default="cuda")
    ap.add_argument("--reuse-bundles", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-headline-csv", type=Path, required=True)
    ap.add_argument("--out-headline-json", type=Path, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    search_payload = _load_json(Path(args.search_json))
    functional_payload = _load_json(Path(args.functional_json))
    residual_specs = _selected_residual_specs(search_payload)
    override_lookup = _override_builder_lookup()
    stap_profiles = [p for p in _split_csv_list(str(args.stap_profiles)) if p]

    base_profiles = [str(x) for x in functional_payload["base_profiles"]]
    dev_seeds = [int(x) for x in functional_payload["dev_seeds"]]
    eval_seeds = [int(x) for x in functional_payload["eval_seeds"]]
    native_dev_summary = [
        row
        for row in functional_payload["summary"]
        if str(row["split"]) == "dev" and str(row["detector_head"]) in {"pd", "kasai"}
    ]
    native_eval_summary = [row for row in functional_payload["summary"] if str(row["split"]) == "eval"]
    selected_native = _best_native_simple_by_family(native_dev_summary)
    eval_native = _extract_matching_eval(native_eval_summary, selected_native)

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
        p = task_root / "mask_h0_nuisance_pa.npy"
        if p.is_file():
            nuisance_mask |= np.load(p).astype(bool, copy=False)
        p = task_root / "mask_h0_specular_struct.npy"
        if p.is_file():
            nuisance_mask |= np.load(p).astype(bool, copy=False)
        outside_mask = (mask_bg | nuisance_mask) & (~activation_roi)
        task_rows = _load_json(task_root / "ensemble_table.json")["rows"]
        null_rows = _load_json(null_root / "ensemble_table.json")["rows"]
        if len(task_rows) != len(null_rows):
            raise ValueError(f"{task_root}: task/null ensemble count mismatch")
        icube_shape = tuple(int(v) for v in np.load(Path(task_rows[0]["dataset_dir"]) / "icube.npy", mmap_mode="r").shape)

        for residual in residual_specs:
            builder = override_lookup[(residual["method_family"], residual["config_name"])]
            bundle_overrides = builder(icube_shape)
            for stap_profile in stap_profiles:
                task_scores: dict[str, list[np.ndarray]] = {"stap": [], "stap_pd": []}
                null_scores: dict[str, list[np.ndarray]] = {"stap": [], "stap_pd": []}
                head_runtimes_ms: dict[str, list[float]] = {"stap": [], "stap_pd": []}
                for task_row, null_row in zip(task_rows, null_rows):
                    dataset_stub = (
                        f"{slugify(base_profile)}_seed{int(seed)}_{slugify(residual['method_family'])}"
                        f"_{slugify(residual['config_name'])}_{slugify(stap_profile)}"
                    )
                    task_bundle_root = bundles_root / split / f"{slugify(base_profile)}_seed{int(seed)}" / "task"
                    null_bundle_root = bundles_root / split / f"{slugify(base_profile)}_seed{int(seed)}" / "null"
                    task_bundle_root.mkdir(parents=True, exist_ok=True)
                    null_bundle_root.mkdir(parents=True, exist_ok=True)
                    stap_task_dir, stap_task_ms = _derive_bundle_timed(
                        run_dir=Path(task_row["ensemble_dir"]),
                        out_root=task_bundle_root,
                        dataset_name=f"{dataset_stub}_e{int(task_row['ensemble_index']):03d}_stap",
                        baseline_type=residual["baseline_type"],
                        bundle_overrides=bundle_overrides,
                        run_stap=True,
                        stap_profile=stap_profile,
                        stap_device=str(args.stap_device),
                        meta_extra={
                            "simus_functional_stap_head_search": True,
                            "split": split,
                            "base_profile": base_profile,
                            "detector_bundle": "stap",
                        },
                        reuse_bundle=bool(args.reuse_bundles),
                    )
                    stap_null_dir, stap_null_ms = _derive_bundle_timed(
                        run_dir=Path(null_row["ensemble_dir"]),
                        out_root=null_bundle_root,
                        dataset_name=f"{dataset_stub}_e{int(null_row['ensemble_index']):03d}_stap",
                        baseline_type=residual["baseline_type"],
                        bundle_overrides=bundle_overrides,
                        run_stap=True,
                        stap_profile=stap_profile,
                        stap_device=str(args.stap_device),
                        meta_extra={
                            "simus_functional_stap_head_search": True,
                            "split": split,
                            "base_profile": base_profile,
                            "detector_bundle": "stap_null",
                        },
                        reuse_bundle=bool(args.reuse_bundles),
                    )
                    if stap_task_ms is not None:
                        head_runtimes_ms["stap"].append(float(stap_task_ms))
                        head_runtimes_ms["stap_pd"].append(float(stap_task_ms))
                    if stap_null_ms is not None:
                        head_runtimes_ms["stap"].append(float(stap_null_ms))
                        head_runtimes_ms["stap_pd"].append(float(stap_null_ms))

                    task_scores["stap"].append(_load_score(stap_task_dir, "stap"))
                    task_scores["stap_pd"].append(_load_score(stap_task_dir, "stap_pd"))
                    null_scores["stap"].append(_load_score(stap_null_dir, "stap"))
                    null_scores["stap_pd"].append(_load_score(stap_null_dir, "stap_pd"))

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
                    roi_series_task = np.mean(task_stack[:, activation_roi], axis=1) if np.any(activation_roi) else np.zeros((task_stack.shape[0],), dtype=np.float32)
                    roi_series_null = np.mean(null_stack[:, activation_roi], axis=1) if np.any(activation_roi) else np.zeros((null_stack.shape[0],), dtype=np.float32)
                    row = {
                        "split": split,
                        "base_profile": base_profile,
                        "seed": int(seed),
                        "method_family": residual["method_family"],
                        "config_name": residual["config_name"],
                        "baseline_type": residual["baseline_type"],
                        "baseline_label": _baseline_label(residual["baseline_type"]),
                        "detector_head": detector_head,
                        "detector_label": _detector_label(detector_head),
                        "pipeline_label": _pipeline_label(residual["baseline_type"], detector_head),
                        "stap_profile": stap_profile,
                        "auc_activation_vs_bg": structural.get("auc_main_vs_bg"),
                        "auc_activation_vs_nuisance": structural.get("auc_main_vs_nuisance"),
                        "roi_corr_task": _corr(roi_series_task, reg),
                        "roi_corr_null": _corr(roi_series_null, reg),
                        "runtime_ms_mean": float(np.mean(head_runtimes_ms[detector_head])) if head_runtimes_ms[detector_head] else None,
                        "runtime_ms_median": float(np.median(head_runtimes_ms[detector_head])) if head_runtimes_ms[detector_head] else None,
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

    for split, seeds in (("dev", dev_seeds), ("eval", eval_seeds)):
        for base_profile in base_profiles:
            for seed in seeds:
                _eval_case(split, base_profile, int(seed))

    summary = _summarize_rows(rows)
    dev_summary = [r for r in summary if str(r["split"]) == "dev"]
    eval_summary = [r for r in summary if str(r["split"]) == "eval"]

    selected_stap: dict[tuple[str, str], dict[str, Any]] = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in dev_summary:
        grouped.setdefault((str(row["base_profile"]), str(row["method_family"])), []).append(row)
    for key, items in grouped.items():
        selected_stap[key] = max(items, key=_score_selection)

    wanted = {
        (
            str(v["base_profile"]),
            str(v["method_family"]),
            str(v["config_name"]),
            str(v["detector_head"]),
            str(v["stap_profile"]),
        )
        for v in selected_stap.values()
    }
    eval_stap = [
        row
        for row in eval_summary
        if (
            str(row["base_profile"]),
            str(row["method_family"]),
            str(row["config_name"]),
            str(row["detector_head"]),
            str(row["stap_profile"]),
        )
        in wanted
    ]

    comparison_rows: list[dict[str, Any]] = []
    for key, native_row in sorted(selected_native.items()):
        base_profile, method_family = key
        stap_row = selected_stap.get((base_profile, method_family))
        native_eval = next(
            row
            for row in eval_native
            if str(row["base_profile"]) == base_profile and str(row["method_family"]) == method_family
        )
        stap_eval = next(
            row
            for row in eval_stap
            if str(row["base_profile"]) == base_profile and str(row["method_family"]) == method_family
        )
        comparison_rows.append(
            {
                "base_profile": base_profile,
                "method_family": method_family,
                "config_name": str(native_eval["config_name"]),
                "native_detector_head_dev": str(native_row["detector_head"]),
                "native_selection_score_dev": float(native_row["selection_score"]),
                "native_selection_score_eval": float(native_eval["selection_score"]),
                "native_auc_activation_vs_bg_eval": float(native_eval["mean_auc_activation_vs_bg"]),
                "native_auc_activation_vs_nuisance_eval": float(native_eval["mean_auc_activation_vs_nuisance"]),
                "native_runtime_ms_eval": float(native_eval["mean_runtime_ms"]),
                "stap_detector_head_dev": str(stap_row["detector_head"]) if stap_row else None,
                "stap_profile_dev": str(stap_row["stap_profile"]) if stap_row else None,
                "stap_selection_score_dev": float(stap_row["selection_score"]) if stap_row else None,
                "stap_selection_score_eval": float(stap_eval["selection_score"]),
                "stap_auc_activation_vs_bg_eval": float(stap_eval["mean_auc_activation_vs_bg"]),
                "stap_auc_activation_vs_nuisance_eval": float(stap_eval["mean_auc_activation_vs_nuisance"]),
                "stap_runtime_ms_eval": float(stap_eval["mean_runtime_ms"]),
                "delta_selection_eval": float(stap_eval["selection_score"]) - float(native_eval["selection_score"]),
            }
        )

    payload = {
        "schema_version": "simus_functional_stap_head_search.v1",
        "base_profiles": base_profiles,
        "dev_seeds": dev_seeds,
        "eval_seeds": eval_seeds,
        "stap_profiles": stap_profiles,
        "rows": rows,
        "summary": summary,
        "selected_native_simple_by_family": [
            {"base_profile": key[0], "method_family": key[1], "selection": value}
            for key, value in sorted(selected_native.items())
        ],
        "selected_stap_by_family": [
            {"base_profile": key[0], "method_family": key[1], "selection": value}
            for key, value in sorted(selected_stap.items())
        ],
        "eval_native_simple_selected": eval_native,
        "eval_stap_selected": eval_stap,
        "comparison": comparison_rows,
    }
    _write_csv(Path(args.out_csv), rows)
    _write_json(Path(args.out_json), payload)
    _write_csv(Path(args.out_headline_csv), comparison_rows)
    _write_json(Path(args.out_headline_json), {"comparison": comparison_rows})
    print(f"[simus-functional-stap-head-search] wrote {args.out_csv}")
    print(f"[simus-functional-stap-head-search] wrote {args.out_json}")


if __name__ == "__main__":
    main()
