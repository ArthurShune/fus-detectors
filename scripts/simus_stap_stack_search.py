#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.simus_eval_structural import evaluate_structural_metrics
from scripts.simus_fair_profile_search import _candidate_specs
from sim.simus.bundle import derive_bundle_from_run, load_canonical_run, slugify


FIXED_STAP_PROFILES = (
    "Brain-SIMUS-Clin",
    "Brain-SIMUS-Clin-MotionWide-v0",
    "Brain-SIMUS-Clin-MotionShort-v0",
    "Brain-SIMUS-Clin-MotionMid-v0",
    "Brain-SIMUS-Clin-MotionLong-v0",
    "Brain-SIMUS-Clin-MotionRobust-v0",
    "Brain-SIMUS-Clin-MotionMidRobust-v0",
)


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


def _load_search_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "details" not in payload:
        raise ValueError(f"{path}: expected top-level 'details'")
    return payload


def _override_builder_lookup() -> dict[tuple[str, str], Any]:
    out: dict[tuple[str, str], Any] = {}
    for cand in _candidate_specs():
        out[(str(cand.method_family), str(cand.config_name))] = cand.override_builder
    return out


def _selected_residual_specs(search_payload: dict[str, Any]) -> list[ResidualSpec]:
    details = dict(search_payload["details"])
    selected = dict(details.get("selected_configs") or {})
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
    return sorted(out, key=lambda x: x.method_family)


def _cases_from_search(search_payload: dict[str, Any], key: str) -> list[tuple[str, Path]]:
    details = dict(search_payload["details"])
    cases = details.get(key)
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"search payload missing {key}")
    out: list[tuple[str, Path]] = []
    for item in cases:
        out.append((str(item["name"]), Path(item["run_dir"])))
    return out


def _summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (
                str(row["split"]),
                str(row["residual_family"]),
                str(row["stap_profile"]),
            ),
            [],
        ).append(row)

    out: list[dict[str, Any]] = []
    for (split, residual_family, stap_profile), items in sorted(grouped.items()):
        auc_bg = np.asarray([float(r["auc_main_vs_bg"]) for r in items], dtype=np.float64)
        auc_n = np.asarray([float(r["auc_main_vs_nuisance"]) for r in items], dtype=np.float64)
        fpr_n = np.asarray([float(r["fpr_nuisance_match@0p5"]) for r in items], dtype=np.float64)
        row0 = items[0]
        summary = {
            "split": split,
            "residual_family": residual_family,
            "config_name": row0["config_name"],
            "baseline_label": row0["baseline_label"],
            "baseline_type": row0["baseline_type"],
            "stap_profile": stap_profile,
            "pipeline_label": row0["pipeline_label"],
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Joint upstream residualizer + STAP head search on accepted SIMUS v2.")
    ap.add_argument(
        "--search-json",
        type=Path,
        required=True,
        help="Existing fair-profile search JSON used to freeze residualizer configs and cases.",
    )
    ap.add_argument(
        "--stap-profiles",
        type=str,
        default=",".join(FIXED_STAP_PROFILES),
        help="Comma-separated fixed STAP profiles to search.",
    )
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--fprs", type=str, default="1e-4,1e-3")
    ap.add_argument("--match-tprs", type=str, default="0.25,0.5,0.75")
    ap.add_argument("--stap-device", type=str, default="cuda")
    ap.add_argument("--reuse-bundles", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    payload = _load_search_payload(Path(args.search_json))
    residual_specs = _selected_residual_specs(payload)
    dev_cases = _cases_from_search(payload, "dev_cases")
    eval_cases = _cases_from_search(payload, "eval_cases")
    fprs = [float(x) for x in _split_csv_list(str(args.fprs))]
    match_tprs = [float(x) for x in _split_csv_list(str(args.match_tprs))]
    stap_profiles = [p for p in _split_csv_list(str(args.stap_profiles)) if not p.endswith("MotionDisp-v0") and not p.endswith("RegShiftP90-v0")]
    override_lookup = _override_builder_lookup()

    rows: list[dict[str, Any]] = []
    out_root = Path(args.out_root)
    dev_root = out_root / "dev"
    eval_root = out_root / "eval"
    dev_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    def eval_caseset(split: str, cases: list[tuple[str, Path]], selected_only: dict[tuple[str, str], dict[str, Any]] | None) -> None:
        root = dev_root if split == "dev" else eval_root
        for case_name, run_dir in cases:
            icube, masks, run_meta = load_canonical_run(run_dir)
            mask_h1_pf_main = masks["mask_h1_pf_main"]
            mask_h0_bg = masks["mask_h0_bg"]
            mask_h0_nuisance_pa = masks.get("mask_h0_nuisance_pa")
            mask_h1_alias_qc = masks.get("mask_h1_alias_qc")
            for residual in residual_specs:
                for stap_profile in stap_profiles:
                    key = (residual.method_family, stap_profile)
                    if selected_only is not None and key not in selected_only:
                        continue
                    builder = override_lookup[(residual.method_family, residual.config_name)]
                    bundle_overrides = builder(tuple(int(v) for v in icube.shape))
                    dataset_name = (
                        f"{case_name}_{slugify(residual.method_family)}_{slugify(residual.config_name)}"
                        f"_stap_{slugify(stap_profile)}"
                    )
                    bundle_root = root / case_name
                    bundle_dir = bundle_root / slugify(dataset_name)
                    if not bool(args.reuse_bundles) or not (bundle_dir / "meta.json").is_file():
                        bundle_dir = derive_bundle_from_run(
                            run_dir=run_dir,
                            out_root=bundle_root,
                            dataset_name=dataset_name,
                            stap_profile=stap_profile,
                            baseline_type=str(residual.baseline_type),
                            run_stap=True,
                            stap_device=str(args.stap_device),
                            bundle_overrides=bundle_overrides,
                            meta_extra={
                                "simus_stap_stack_search": True,
                                "search_split": split,
                                "case_name": case_name,
                                "residual_family": residual.method_family,
                                "config_name": residual.config_name,
                                "stap_profile": stap_profile,
                            },
                        )
                    score = np.load(bundle_dir / "score_stap_preka.npy").astype(np.float32, copy=False)
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
                        "residual_family": residual.method_family,
                        "config_name": residual.config_name,
                        "baseline_label": residual.method_family,
                        "baseline_type": residual.baseline_type,
                        "stap_profile": stap_profile,
                        "pipeline_label": f"{residual.method_family} -> STAP",
                        "bundle_dir": str(bundle_dir),
                        "score_file": "score_stap_preka.npy",
                        "bundle_overrides_json": json.dumps(bundle_overrides, sort_keys=True),
                        "prf_hz": run_meta.get("acquisition", {}).get("prf_hz"),
                    }
                    row.update(metrics)
                    rows.append(row)

    eval_caseset("dev", dev_cases, selected_only=None)
    dev_summary = _summarize_rows([r for r in rows if r["split"] == "dev"])
    selected_key: tuple[str, str] | None = None
    selected_summary: dict[str, Any] | None = None
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    if dev_summary:
        best = max(dev_summary, key=lambda r: float(r["selection_score"]))
        selected_key = (str(best["residual_family"]), str(best["stap_profile"]))
        selected_summary = best
        selected[selected_key] = best
    eval_caseset("eval", eval_cases, selected_only=selected)
    eval_summary = _summarize_rows([r for r in rows if r["split"] == "eval"])

    payload = {
        "schema_version": "simus_stap_stack_search.v1",
        "dev_cases": [{"name": name, "run_dir": str(path)} for name, path in dev_cases],
        "eval_cases": [{"name": name, "run_dir": str(path)} for name, path in eval_cases],
        "residual_specs": [r.__dict__ for r in residual_specs],
        "stap_profiles": stap_profiles,
        "dev_summary": dev_summary,
        "selected_stack": selected_summary,
        "selected_stack_key": (
            {
                "residual_family": selected_key[0],
                "stap_profile": selected_key[1],
            }
            if selected_key is not None
            else None
        ),
        "eval_summary": eval_summary,
        "rows": rows,
    }
    _write_csv(Path(args.out_csv), rows)
    _write_json(Path(args.out_json), payload)
    print(f"[simus-stap-stack-search] wrote {args.out_csv}")
    print(f"[simus-stap-stack-search] wrote {args.out_json}")


if __name__ == "__main__":
    main()
