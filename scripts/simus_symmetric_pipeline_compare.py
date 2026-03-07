#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.simus_eval_structural import _baseline_label, evaluate_structural_metrics
from scripts.simus_fair_profile_search import _candidate_specs
from sim.simus.bundle import derive_bundle_from_run, load_canonical_run, slugify


SIMPLE_HEADS = ("pd", "kasai")


@dataclass(frozen=True)
class ResidualSpec:
    method_family: str
    config_name: str
    baseline_type: str
    override_builder_name: str


def _split_csv_list(spec: str) -> list[str]:
    return [s.strip() for s in str(spec or "").split(",") if s.strip()]


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


def _selection_score(row: dict[str, Any]) -> float:
    return (
        float(row.get("auc_main_vs_bg") or 0.0)
        + float(row.get("auc_main_vs_nuisance") or 0.0)
        - float(row.get("fpr_nuisance_match@0p5") or 0.0)
    )


def _load_search_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "details" not in payload:
        raise ValueError(f"{path}: expected top-level 'details'")
    return payload


def _selected_residual_specs(search_payload: dict[str, Any]) -> tuple[list[ResidualSpec], str]:
    details = dict(search_payload["details"])
    selected = dict(details.get("selected_configs") or {})
    if "stap" not in selected:
        raise ValueError("search payload missing selected STAP config")
    stap_profile = str(selected["stap"]["config_name"])

    by_key: dict[tuple[str, str], Any] = {}
    for cand in _candidate_specs():
        by_key[(str(cand.method_family), str(cand.config_name))] = cand

    out: list[ResidualSpec] = []
    for method_family, summary in selected.items():
        if str(method_family) == "stap":
            continue
        key = (str(method_family), str(summary["config_name"]))
        cand = by_key.get(key)
        if cand is None:
            raise KeyError(f"selected config not found in candidate grid: {key}")
        out.append(
            ResidualSpec(
                method_family=str(cand.method_family),
                config_name=str(cand.config_name),
                baseline_type=str(cand.method.baseline_type),
                override_builder_name=str(cand.override_builder.__name__),
            )
        )
    return sorted(out, key=lambda x: x.method_family), stap_profile


def _override_builder_lookup() -> dict[tuple[str, str], Any]:
    out: dict[tuple[str, str], Any] = {}
    for cand in _candidate_specs():
        out[(str(cand.method_family), str(cand.config_name))] = cand.override_builder
    return out


def _cases_from_search(search_payload: dict[str, Any], key: str) -> list[tuple[str, Path]]:
    details = dict(search_payload["details"])
    cases = details.get(key)
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"search payload missing {key}")
    out: list[tuple[str, Path]] = []
    for item in cases:
        out.append((str(item["name"]), Path(item["run_dir"])))
    return out


def _score_path(bundle_dir: Path, detector_head: str) -> Path | None:
    head = str(detector_head).strip().lower()
    if head == "pd":
        return bundle_dir / "score_base.npy"
    if head == "kasai":
        path = bundle_dir / "score_base_kasai.npy"
        return path if path.is_file() else None
    if head == "stap":
        return bundle_dir / "score_stap_preka.npy"
    raise ValueError(f"unsupported detector head {detector_head!r}")


def _detector_label(detector_head: str) -> str:
    return {
        "pd": "PD",
        "kasai": "Kasai",
        "stap": "STAP",
    }[str(detector_head).strip().lower()]


def _pipeline_label(baseline_type: str, detector_head: str) -> str:
    return f"{_baseline_label(baseline_type)} -> {_detector_label(detector_head)}"


def _summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (
                str(row["split"]),
                str(row["method_family"]),
                str(row["config_name"]),
                str(row["detector_head"]),
            ),
            [],
        ).append(row)

    out: list[dict[str, Any]] = []
    for (split, method_family, config_name, detector_head), items in sorted(grouped.items()):
        auc_bg = np.asarray([float(r["auc_main_vs_bg"]) for r in items], dtype=np.float64)
        auc_n = np.asarray([float(r["auc_main_vs_nuisance"]) for r in items], dtype=np.float64)
        fpr_n = np.asarray([float(r["fpr_nuisance_match@0p5"]) for r in items], dtype=np.float64)
        row0 = items[0]
        summary = {
            "split": split,
            "method_family": method_family,
            "config_name": config_name,
            "baseline_label": row0["baseline_label"],
            "baseline_type": row0["baseline_type"],
            "detector_head": detector_head,
            "detector_label": row0["detector_label"],
            "pipeline_label": row0["pipeline_label"],
            "stap_profile": row0.get("stap_profile"),
            "count": int(len(items)),
            "mean_auc_main_vs_bg": float(np.mean(auc_bg)),
            "mean_auc_main_vs_nuisance": float(np.mean(auc_n)),
            "mean_fpr_nuisance_match@0p5": float(np.mean(fpr_n)),
        }
        summary["selection_score"] = (
            summary["mean_auc_main_vs_bg"]
            + summary["mean_auc_main_vs_nuisance"]
            - summary["mean_fpr_nuisance_match@0p5"]
        )
        out.append(summary)
    return out


def _select_best_simple_detector(dev_summary: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in dev_summary:
        if str(row["detector_head"]) not in SIMPLE_HEADS:
            continue
        by_family.setdefault(str(row["method_family"]), []).append(row)
    selected: dict[str, dict[str, Any]] = {}
    for method_family, items in by_family.items():
        items_sorted = sorted(items, key=lambda r: float(r["selection_score"]), reverse=True)
        selected[method_family] = items_sorted[0]
    return selected


def _select_best_detector_any(dev_summary: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in dev_summary:
        by_family.setdefault(str(row["method_family"]), []).append(row)
    selected: dict[str, dict[str, Any]] = {}
    for method_family, items in by_family.items():
        items_sorted = sorted(items, key=lambda r: float(r["selection_score"]), reverse=True)
        selected[method_family] = items_sorted[0]
    return selected


def _extract_matching_eval(eval_summary: list[dict[str, Any]], selected: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {
        (
            str(v["method_family"]),
            str(v["config_name"]),
            str(v["detector_head"]),
        )
        for v in selected.values()
    }
    return [
        row
        for row in eval_summary
        if (str(row["method_family"]), str(row["config_name"]), str(row["detector_head"])) in wanted
    ]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Symmetric residualizer->detector comparison for corrected SIMUS.")
    ap.add_argument(
        "--search-json",
        type=Path,
        default=Path("reports/simus_motion/simus_fair_profile_search_seed2122to2324_expanded.json"),
        help="Expanded fair-search JSON used to freeze residualizer configs and cases.",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("runs/sim_eval/simus_symmetric_pipeline_compare"),
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/simus_motion/simus_symmetric_pipeline_compare.csv"),
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/simus_motion/simus_symmetric_pipeline_compare.json"),
    )
    ap.add_argument(
        "--out-headline-csv",
        type=Path,
        default=Path("reports/simus_motion/simus_symmetric_pipeline_compare_headline.csv"),
    )
    ap.add_argument(
        "--out-headline-json",
        type=Path,
        default=Path("reports/simus_motion/simus_symmetric_pipeline_compare_headline.json"),
    )
    ap.add_argument("--fprs", type=str, default="1e-4,1e-3")
    ap.add_argument("--match-tprs", type=str, default="0.25,0.5,0.75")
    ap.add_argument("--stap-device", type=str, default="cuda")
    ap.add_argument(
        "--stap-profile",
        type=str,
        default=None,
        help="Override fixed STAP profile. Defaults to the selected STAP profile from --search-json.",
    )
    ap.add_argument(
        "--detectors",
        type=str,
        default="pd,kasai,stap",
        help="Comma-separated detector heads to evaluate (pd, kasai, stap).",
    )
    ap.add_argument("--reuse-bundles", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    payload = _load_search_payload(Path(args.search_json))
    residual_specs, selected_stap_profile = _selected_residual_specs(payload)
    stap_profile = str(args.stap_profile or selected_stap_profile)
    dev_cases = _cases_from_search(payload, "dev_cases")
    eval_cases = _cases_from_search(payload, "eval_cases")
    override_builders = _override_builder_lookup()
    fprs = [float(x) for x in _split_csv_list(args.fprs)]
    match_tprs = [float(x) for x in _split_csv_list(args.match_tprs)]
    detectors = [d for d in _split_csv_list(args.detectors) if d in {"pd", "kasai", "stap"}]
    if not detectors:
        raise ValueError("No valid detector heads requested")

    out_root = Path(args.out_root)
    rows: list[dict[str, Any]] = []

    def eval_caseset(split: str, cases: list[tuple[str, Path]]) -> None:
        split_root = out_root / split
        split_root.mkdir(parents=True, exist_ok=True)
        for case_name, run_dir in cases:
            icube, masks, run_meta = load_canonical_run(run_dir)
            mask_h1_pf_main = masks["mask_h1_pf_main"]
            mask_h0_bg = masks["mask_h0_bg"]
            mask_h0_nuisance_pa = masks.get("mask_h0_nuisance_pa")
            mask_h1_alias_qc = masks.get("mask_h1_alias_qc")
            for spec in residual_specs:
                builder = override_builders[(spec.method_family, spec.config_name)]
                bundle_overrides = builder(tuple(int(v) for v in icube.shape))
                bundle_root = split_root / case_name
                dataset_name = f"{case_name}_{slugify(spec.method_family)}_{slugify(spec.config_name)}_sym"
                bundle_dir = bundle_root / slugify(dataset_name)
                if not bool(args.reuse_bundles) or not (bundle_dir / "meta.json").is_file():
                    bundle_dir = derive_bundle_from_run(
                        run_dir=run_dir,
                        out_root=bundle_root,
                        dataset_name=dataset_name,
                        stap_profile=stap_profile,
                        baseline_type=spec.baseline_type,
                        run_stap=True,
                        stap_device=str(args.stap_device),
                        bundle_overrides=bundle_overrides,
                        meta_extra={
                            "simus_symmetric_pipeline_compare": True,
                            "search_json": str(Path(args.search_json)),
                            "search_selected_residualizer": {
                                "method_family": spec.method_family,
                                "config_name": spec.config_name,
                            },
                            "fixed_stap_profile": stap_profile,
                            "split": split,
                            "case_name": case_name,
                        },
                    )
                for detector_head in detectors:
                    score_path = _score_path(bundle_dir, detector_head)
                    if score_path is None or not score_path.is_file():
                        continue
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
                        "split": split,
                        "case_name": case_name,
                        "run_dir": str(run_dir),
                        "bundle_dir": str(bundle_dir),
                        "method_family": spec.method_family,
                        "config_name": spec.config_name,
                        "baseline_type": spec.baseline_type,
                        "baseline_label": _baseline_label(spec.baseline_type),
                        "detector_head": detector_head,
                        "detector_label": _detector_label(detector_head),
                        "pipeline_label": _pipeline_label(spec.baseline_type, detector_head),
                        "score_file": score_path.name,
                        "stap_profile": stap_profile if detector_head == "stap" else None,
                        "T": int(icube.shape[0]),
                        "H": int(icube.shape[1]),
                        "W": int(icube.shape[2]),
                        "prf_hz": run_meta.get("acquisition", {}).get("prf_hz"),
                        "motion_disp_rms_px": run_meta.get("motion", {}).get("telemetry", {}).get("disp_rms_px"),
                        "phase_rms_rad": run_meta.get("phase_screen", {}).get("telemetry", {}).get("phase_rms_rad"),
                        "bundle_overrides_json": json.dumps(bundle_overrides, sort_keys=True),
                    }
                    row.update(metrics)
                    row["selection_score"] = _selection_score(row)
                    rows.append(row)

    eval_caseset("dev", dev_cases)
    eval_caseset("eval", eval_cases)

    dev_rows = [row for row in rows if str(row["split"]) == "dev"]
    eval_rows = [row for row in rows if str(row["split"]) == "eval"]
    dev_summary = _summarize_rows(dev_rows)
    eval_summary = _summarize_rows(eval_rows)
    selected_simple = _select_best_simple_detector(dev_summary)
    selected_any = _select_best_detector_any(dev_summary)
    eval_best_simple = _extract_matching_eval(eval_summary, selected_simple)
    eval_best_any = _extract_matching_eval(eval_summary, selected_any)

    headline_rows: list[dict[str, Any]] = []
    headline_rows.extend(eval_summary)
    for row in eval_best_simple:
        tagged = dict(row)
        tagged["summary_view"] = "best_native_simple_detector_per_family"
        headline_rows.append(tagged)
    for row in eval_best_any:
        tagged = dict(row)
        tagged["summary_view"] = "best_detector_any_per_family"
        headline_rows.append(tagged)

    _write_csv(Path(args.out_csv), rows)
    _write_json(
        Path(args.out_json),
        {
            "rows": rows,
            "details": {
                "schema_version": "simus_symmetric_pipeline_compare.v1",
                "search_json": str(Path(args.search_json)),
                "fixed_stap_profile": stap_profile,
                "residual_specs": [spec.__dict__ for spec in residual_specs],
                "dev_cases": [{"name": name, "run_dir": str(run_dir)} for name, run_dir in dev_cases],
                "eval_cases": [{"name": name, "run_dir": str(run_dir)} for name, run_dir in eval_cases],
                "dev_summary": dev_summary,
                "eval_summary": eval_summary,
                "selected_simple": selected_simple,
                "selected_any": selected_any,
                "eval_best_simple": eval_best_simple,
                "eval_best_any": eval_best_any,
            },
        },
    )
    _write_csv(Path(args.out_headline_csv), headline_rows)
    _write_json(
        Path(args.out_headline_json),
        {
            "schema_version": "simus_symmetric_pipeline_compare_headline.v1",
            "fixed_stap_profile": stap_profile,
            "eval_summary": eval_summary,
            "eval_best_simple": eval_best_simple,
            "eval_best_any": eval_best_any,
        },
    )
    print(f"[simus-symmetric-pipeline-compare] wrote {args.out_csv}")
    print(f"[simus-symmetric-pipeline-compare] wrote {args.out_json}")
    print(f"[simus-symmetric-pipeline-compare] wrote {args.out_headline_csv}")
    print(f"[simus-symmetric-pipeline-compare] wrote {args.out_headline_json}")


if __name__ == "__main__":
    main()
