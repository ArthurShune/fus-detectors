#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.simus_eval_functional import (
    FPR_TAGS,
    READOUT_MODES,
    _corr,
    _derive_bundle_timed,
    _detector_label,
    _functional_maps,
    _load_json,
    _load_score,
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
from scripts.simus_functional_stap_head_search import FIXED_STAP_PROFILES
from sim.simus.bundle import slugify


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Search functional residual configs + detector heads on existing SIMUS v2 functional cases.")
    ap.add_argument("--cases-root", type=Path, required=True)
    ap.add_argument("--profiles", type=str, default="ClinMobile-Pf-v2,ClinIntraOpParenchyma-Pf-v3")
    ap.add_argument("--dev-seeds", type=str, default="221")
    ap.add_argument("--eval-seeds", type=str, default="222")
    ap.add_argument("--families", type=str, required=True)
    ap.add_argument("--readout-mode", type=str, default="basic", choices=READOUT_MODES)
    ap.add_argument("--stap-profiles", type=str, default=",".join(FIXED_STAP_PROFILES))
    ap.add_argument("--stap-device", type=str, default="cuda")
    ap.add_argument("--reuse-bundles", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-headline-csv", type=Path, required=True)
    ap.add_argument("--out-headline-json", type=Path, required=True)
    return ap.parse_args()


def _candidate_grid(families: set[str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for cand in _candidate_specs():
        fam = str(cand.method_family)
        if fam == "stap" or fam not in families:
            continue
        out.append(
            {
                "method_family": fam,
                "config_name": str(cand.config_name),
                "baseline_type": str(cand.method.baseline_type),
                "override_builder": cand.override_builder,
            }
        )
    return out


def main() -> None:
    args = parse_args()
    families = set(_split_csv_list(str(args.families)))
    if not families:
        raise ValueError("no families")
    base_profiles = _split_csv_list(str(args.profiles))
    dev_seeds = [int(x) for x in _split_csv_list(str(args.dev_seeds))]
    eval_seeds = [int(x) for x in _split_csv_list(str(args.eval_seeds))]
    stap_profiles = _split_csv_list(str(args.stap_profiles))
    candidates = _candidate_grid(families)

    cases_root = Path(args.cases_root)
    out_root = Path(args.out_root)
    bundles_root = out_root / "bundles"
    bundles_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    def _case_root(split: str, base_profile: str, seed: int, null_run: bool) -> Path:
        tag = "null" if bool(null_run) else "task"
        return cases_root / split / f"{slugify(base_profile)}_seed{int(seed)}_{tag}"

    for split, seeds in (("dev", dev_seeds), ("eval", eval_seeds)):
        for base_profile in base_profiles:
            for seed in seeds:
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
                bg_mask = mask_bg & (~activation_roi)
                task_rows = _load_json(task_root / "ensemble_table.json")["rows"]
                null_rows = _load_json(null_root / "ensemble_table.json")["rows"]
                icube_shape = tuple(int(v) for v in np.load(Path(task_rows[0]["dataset_dir"]) / "icube.npy", mmap_mode="r").shape)

                for cand in candidates:
                    bundle_overrides = cand["override_builder"](icube_shape)
                    native_scores = {"pd": [], "kasai": []}
                    native_null_scores = {"pd": [], "kasai": []}
                    native_runtime: list[float] = []
                    for task_row, null_row in zip(task_rows, null_rows):
                        dataset_stub = f"{slugify(base_profile)}_seed{seed}_{slugify(cand['method_family'])}_{slugify(cand['config_name'])}"
                        task_bundle_root = bundles_root / split / f"{slugify(base_profile)}_seed{seed}" / "task"
                        null_bundle_root = bundles_root / split / f"{slugify(base_profile)}_seed{seed}" / "null"
                        task_bundle_root.mkdir(parents=True, exist_ok=True)
                        null_bundle_root.mkdir(parents=True, exist_ok=True)
                        native_task_dir, native_task_ms = _derive_bundle_timed(
                            run_dir=Path(task_row["ensemble_dir"]),
                            out_root=task_bundle_root,
                            dataset_name=f"{dataset_stub}_e{int(task_row['ensemble_index']):03d}_native",
                            baseline_type=cand["baseline_type"],
                            bundle_overrides=bundle_overrides,
                            run_stap=False,
                            stap_profile=stap_profiles[0],
                            stap_device=str(args.stap_device),
                            meta_extra={"simus_functional_family_search": True, "split": split, "base_profile": base_profile, "detector_bundle": "native"},
                            reuse_bundle=bool(args.reuse_bundles),
                        )
                        native_null_dir, native_null_ms = _derive_bundle_timed(
                            run_dir=Path(null_row["ensemble_dir"]),
                            out_root=null_bundle_root,
                            dataset_name=f"{dataset_stub}_e{int(null_row['ensemble_index']):03d}_native",
                            baseline_type=cand["baseline_type"],
                            bundle_overrides=bundle_overrides,
                            run_stap=False,
                            stap_profile=stap_profiles[0],
                            stap_device=str(args.stap_device),
                            meta_extra={"simus_functional_family_search": True, "split": split, "base_profile": base_profile, "detector_bundle": "native_null"},
                            reuse_bundle=bool(args.reuse_bundles),
                        )
                        native_scores["pd"].append(_load_score(native_task_dir, "pd"))
                        native_scores["kasai"].append(_load_score(native_task_dir, "kasai"))
                        native_null_scores["pd"].append(_load_score(native_null_dir, "pd"))
                        native_null_scores["kasai"].append(_load_score(native_null_dir, "kasai"))
                        if native_task_ms is not None:
                            native_runtime.append(float(native_task_ms))
                        if native_null_ms is not None:
                            native_runtime.append(float(native_null_ms))

                    for detector_head in ("pd", "kasai"):
                        task_stack = np.stack(native_scores[detector_head], axis=0).astype(np.float32, copy=False)
                        null_stack = np.stack(native_null_scores[detector_head], axis=0).astype(np.float32, copy=False)
                        task_map, null_map = _functional_maps(
                            task_stack, null_stack, reg,
                            readout_mode=str(args.readout_mode),
                            bg_mask=bg_mask,
                            outside_mask=outside_mask,
                        )
                        structural = evaluate_structural_metrics(
                            score=task_map,
                            mask_h1_pf_main=activation_roi,
                            mask_h0_bg=bg_mask,
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
                            "method_family": cand["method_family"],
                            "config_name": cand["config_name"],
                            "baseline_type": cand["baseline_type"],
                            "baseline_label": _baseline_label(cand["baseline_type"]),
                            "detector_head": detector_head,
                            "detector_label": _detector_label(detector_head),
                            "pipeline_label": _pipeline_label(cand['baseline_type'], detector_head),
                            "stap_profile": None,
                            "readout_mode": str(args.readout_mode),
                            "auc_activation_vs_bg": structural.get("auc_main_vs_bg"),
                            "auc_activation_vs_nuisance": structural.get("auc_main_vs_nuisance"),
                            "roi_corr_task": _corr(roi_series_task, reg),
                            "roi_corr_null": _corr(roi_series_null, reg),
                            "runtime_ms_mean": float(np.mean(native_runtime)) if native_runtime else None,
                            "runtime_ms_median": float(np.median(native_runtime)) if native_runtime else None,
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

                    for stap_profile in stap_profiles:
                        stap_scores = {"stap": [], "stap_pd": []}
                        stap_null_scores = {"stap": [], "stap_pd": []}
                        stap_runtime: list[float] = []
                        for task_row, null_row in zip(task_rows, null_rows):
                            dataset_stub = f"{slugify(base_profile)}_seed{seed}_{slugify(cand['method_family'])}_{slugify(cand['config_name'])}_{slugify(stap_profile)}"
                            task_bundle_root = bundles_root / split / f"{slugify(base_profile)}_seed{seed}" / "task"
                            null_bundle_root = bundles_root / split / f"{slugify(base_profile)}_seed{seed}" / "null"
                            stap_task_dir, stap_task_ms = _derive_bundle_timed(
                                run_dir=Path(task_row["ensemble_dir"]),
                                out_root=task_bundle_root,
                                dataset_name=f"{dataset_stub}_e{int(task_row['ensemble_index']):03d}_stap",
                                baseline_type=cand["baseline_type"],
                                bundle_overrides=bundle_overrides,
                                run_stap=True,
                                stap_profile=stap_profile,
                                stap_device=str(args.stap_device),
                                meta_extra={"simus_functional_family_search": True, "split": split, "base_profile": base_profile, "detector_bundle": "stap"},
                                reuse_bundle=bool(args.reuse_bundles),
                            )
                            stap_null_dir, stap_null_ms = _derive_bundle_timed(
                                run_dir=Path(null_row["ensemble_dir"]),
                                out_root=null_bundle_root,
                                dataset_name=f"{dataset_stub}_e{int(null_row['ensemble_index']):03d}_stap",
                                baseline_type=cand["baseline_type"],
                                bundle_overrides=bundle_overrides,
                                run_stap=True,
                                stap_profile=stap_profile,
                                stap_device=str(args.stap_device),
                                meta_extra={"simus_functional_family_search": True, "split": split, "base_profile": base_profile, "detector_bundle": "stap_null"},
                                reuse_bundle=bool(args.reuse_bundles),
                            )
                            stap_scores["stap"].append(_load_score(stap_task_dir, "stap"))
                            stap_scores["stap_pd"].append(_load_score(stap_task_dir, "stap_pd"))
                            stap_null_scores["stap"].append(_load_score(stap_null_dir, "stap"))
                            stap_null_scores["stap_pd"].append(_load_score(stap_null_dir, "stap_pd"))
                            if stap_task_ms is not None:
                                stap_runtime.append(float(stap_task_ms))
                            if stap_null_ms is not None:
                                stap_runtime.append(float(stap_null_ms))

                        for detector_head in ("stap", "stap_pd"):
                            task_stack = np.stack(stap_scores[detector_head], axis=0).astype(np.float32, copy=False)
                            null_stack = np.stack(stap_null_scores[detector_head], axis=0).astype(np.float32, copy=False)
                            task_map, null_map = _functional_maps(
                                task_stack, null_stack, reg,
                                readout_mode=str(args.readout_mode),
                                bg_mask=bg_mask,
                                outside_mask=outside_mask,
                            )
                            structural = evaluate_structural_metrics(
                                score=task_map,
                                mask_h1_pf_main=activation_roi,
                                mask_h0_bg=bg_mask,
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
                                "method_family": cand["method_family"],
                                "config_name": cand["config_name"],
                                "baseline_type": cand["baseline_type"],
                                "baseline_label": _baseline_label(cand["baseline_type"]),
                                "detector_head": detector_head,
                                "detector_label": _detector_label(detector_head),
                                "pipeline_label": _pipeline_label(cand['baseline_type'], detector_head),
                                "stap_profile": stap_profile,
                                "readout_mode": str(args.readout_mode),
                                "auc_activation_vs_bg": structural.get("auc_main_vs_bg"),
                                "auc_activation_vs_nuisance": structural.get("auc_main_vs_nuisance"),
                                "roi_corr_task": _corr(roi_series_task, reg),
                                "roi_corr_null": _corr(roi_series_null, reg),
                                "runtime_ms_mean": float(np.mean(stap_runtime)) if stap_runtime else None,
                                "runtime_ms_median": float(np.median(stap_runtime)) if stap_runtime else None,
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

    summary = _summarize_rows(rows)
    dev_summary = [r for r in summary if str(r['split']) == 'dev']
    eval_summary = [r for r in summary if str(r['split']) == 'eval']

    comparison = []
    for base_profile in base_profiles:
        for family in sorted(families):
            dev_items = [r for r in dev_summary if str(r['base_profile']) == base_profile and str(r['method_family']) == family]
            native_dev = [r for r in dev_items if str(r['detector_head']) in {'pd','kasai'}]
            stap_dev = [r for r in dev_items if str(r['detector_head']) in {'stap','stap_pd'}]
            if not native_dev or not stap_dev:
                continue
            sel_native = max(native_dev, key=_score_selection)
            sel_stap = max(stap_dev, key=_score_selection)
            native_eval = next(r for r in eval_summary if str(r['base_profile'])==base_profile and str(r['method_family'])==family and str(r['config_name'])==str(sel_native['config_name']) and str(r['detector_head'])==str(sel_native['detector_head']))
            stap_eval = next(r for r in eval_summary if str(r['base_profile'])==base_profile and str(r['method_family'])==family and str(r['config_name'])==str(sel_stap['config_name']) and str(r['detector_head'])==str(sel_stap['detector_head']) and str(r.get('stap_profile') or '')==str(sel_stap.get('stap_profile') or ''))
            comparison.append({
                'base_profile': base_profile,
                'method_family': family,
                'readout_mode': str(args.readout_mode),
                'native_config_name_dev': str(sel_native['config_name']),
                'native_detector_head_dev': str(sel_native['detector_head']),
                'native_selection_score_dev': float(sel_native['selection_score']),
                'native_selection_score_eval': float(native_eval['selection_score']),
                'native_auc_activation_vs_bg_eval': float(native_eval['mean_auc_activation_vs_bg']),
                'native_auc_activation_vs_nuisance_eval': float(native_eval['mean_auc_activation_vs_nuisance']),
                'native_runtime_ms_eval': float(native_eval['mean_runtime_ms']),
                'stap_config_name_dev': str(sel_stap['config_name']),
                'stap_detector_head_dev': str(sel_stap['detector_head']),
                'stap_profile_dev': str(sel_stap['stap_profile']),
                'stap_selection_score_dev': float(sel_stap['selection_score']),
                'stap_selection_score_eval': float(stap_eval['selection_score']),
                'stap_auc_activation_vs_bg_eval': float(stap_eval['mean_auc_activation_vs_bg']),
                'stap_auc_activation_vs_nuisance_eval': float(stap_eval['mean_auc_activation_vs_nuisance']),
                'stap_runtime_ms_eval': float(stap_eval['mean_runtime_ms']),
                'delta_selection_eval': float(stap_eval['selection_score']) - float(native_eval['selection_score']),
            })

    payload = {
        'schema_version': 'simus_functional_family_search.v1',
        'profiles': base_profiles,
        'dev_seeds': dev_seeds,
        'eval_seeds': eval_seeds,
        'families': sorted(families),
        'readout_mode': str(args.readout_mode),
        'stap_profiles': stap_profiles,
        'rows': rows,
        'summary': summary,
        'comparison': comparison,
    }
    _write_csv(Path(args.out_csv), rows)
    _write_json(Path(args.out_json), payload)
    _write_csv(Path(args.out_headline_csv), comparison)
    _write_json(Path(args.out_headline_json), {'comparison': comparison})


if __name__ == '__main__':
    main()
