#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.simus_clinical_stress_benchmark import PROFILE_LABELS, _motion_speed_mm_s, _stress_axes
from scripts.simus_eval_structural import _baseline_label, evaluate_structural_metrics
from scripts.simus_fair_profile_search import _candidate_specs
from scripts.simus_structural_candidate_compare import DEFAULT_CANDIDATES
from sim.simus.bundle import derive_bundle_from_run, estimate_simus_policy_features, load_canonical_run, slugify
from sim.simus.config import default_profile_config
from sim.simus.pilot_pymust_simus import write_simus_run


@dataclass(frozen=True)
class ResidualSpec:
    method_family: str
    config_name: str
    baseline_type: str
    override_builder_name: str


@dataclass(frozen=True)
class PublicPipeline:
    key: str
    residual: ResidualSpec
    detector_head: str

    @property
    def label(self) -> str:
        return f"{_baseline_label(self.residual.baseline_type)} -> {self.detector_head.upper()}"


@dataclass(frozen=True)
class DetectorPipeline:
    key: str
    residual: ResidualSpec
    detector_name: str
    bundle_overrides: dict[str, Any]

    @property
    def label(self) -> str:
        detector_label = {
            "unwhitened_ref": "Matched-subspace default",
            "adaptive_guard_huber": "Adaptive guard",
            "huber_trim8": "Whitened specialist",
        }[self.detector_name]
        return f"{_baseline_label(self.residual.baseline_type)} -> {detector_label}"


def _split_csv_list(spec: str) -> list[str]:
    return [s.strip() for s in str(spec or "").split(",") if s.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


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


def _fmt(x: float | None, digits: int = 3) -> str:
    if x is None or not np.isfinite(float(x)):
        return "--"
    return f"{float(x):.{digits}f}"


def _load_selected_residual_specs(search_json: Path) -> tuple[dict[tuple[str, str], Any], list[ResidualSpec]]:
    payload = json.loads(Path(search_json).read_text(encoding="utf-8"))
    details = dict(payload["details"])
    selected = dict(details.get("selected_configs") or {})
    if not selected:
        raise ValueError(f"{search_json}: missing details.selected_configs")
    override_builders = {(str(c.method_family), str(c.config_name)): c.override_builder for c in _candidate_specs()}
    residual_specs: list[ResidualSpec] = []
    for method_family, summary in selected.items():
        if str(method_family) == "stap":
            continue
        key = (str(method_family), str(summary["config_name"]))
        cand = next((c for c in _candidate_specs() if (str(c.method_family), str(c.config_name)) == key), None)
        if cand is None:
            raise KeyError(f"{search_json}: selected residual config not found for {key}")
        residual_specs.append(
            ResidualSpec(
                method_family=str(cand.method_family),
                config_name=str(cand.config_name),
                baseline_type=str(cand.method.baseline_type),
                override_builder_name=str(cand.override_builder.__name__),
            )
        )
    return override_builders, residual_specs


def _candidate_override(name: str) -> dict[str, Any]:
    if name == "adaptive_guard_huber":
        huber = next(c for c in DEFAULT_CANDIDATES if c.name == "huber_trim8")
        return {
            "stap_detector_variant": "adaptive_guard",
            "stap_whiten_gamma": float(huber.whiten_gamma),
            "diag_load": float(huber.diag_load),
            "cov_estimator": str(huber.cov_estimator),
            "huber_c": float(huber.huber_c),
            "stap_cov_train_trim_q": float(huber.stap_cov_train_trim_q),
            "mvdr_auto_kappa": float(huber.mvdr_auto_kappa),
            "constraint_ridge": float(huber.constraint_ridge),
            "stap_conditional_enable": False,
        }
    cand = next(c for c in DEFAULT_CANDIDATES if c.name == name)
    return {
        "stap_detector_variant": cand.detector_variant,
        "stap_whiten_gamma": float(cand.whiten_gamma),
        "diag_load": float(cand.diag_load),
        "cov_estimator": str(cand.cov_estimator),
        "huber_c": float(cand.huber_c),
        "stap_cov_train_trim_q": float(cand.stap_cov_train_trim_q),
        "mvdr_auto_kappa": float(cand.mvdr_auto_kappa),
        "constraint_ridge": float(cand.constraint_ridge),
        "stap_conditional_enable": False,
    }


def _score_path(bundle_dir: Path, *, detector_head: str, role: str) -> Path | None:
    if role == "public":
        head = str(detector_head).strip().lower()
        if head == "pd":
            candidates = [bundle_dir / "score_pd_base.npy", bundle_dir / "score_base.npy"]
        elif head == "kasai":
            candidates = [bundle_dir / "score_base_kasai.npy"]
        else:
            raise ValueError(f"unsupported public detector head {detector_head!r}")
    else:
        candidates = [bundle_dir / "score_stap_preka.npy", bundle_dir / "score_stap.npy"]
    return next((p for p in candidates if p.is_file()), None)


def _selection_score(row: dict[str, Any]) -> float:
    return (
        float(row.get("auc_main_vs_bg") or 0.0)
        + float(row.get("auc_main_vs_nuisance") or 0.0)
        - float(row.get("fpr_nuisance_match@0p5") or 0.0)
    )


def _midpoints(values: list[float]) -> list[float]:
    vals = sorted(v for v in values if np.isfinite(v))
    if not vals:
        return [0.0]
    out = [vals[0] - 1e-6]
    out.extend((a + b) * 0.5 for a, b in zip(vals[:-1], vals[1:], strict=False))
    out.append(vals[-1] + 1e-6)
    return out


def _aggregate_by_pipeline(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["pipeline_key"]), []).append(row)
    out: list[dict[str, Any]] = []
    for pipeline_key, items in sorted(grouped.items()):
        row0 = items[0]

        def _mean(key: str) -> float | None:
            vals = np.asarray(
                [float(r[key]) for r in items if r.get(key) is not None and np.isfinite(r.get(key))],
                dtype=np.float64,
            )
            return float(np.mean(vals)) if vals.size else None

        rec = {
            "pipeline_key": pipeline_key,
            "pipeline_label": row0["pipeline_label"],
            "role": row0["role"],
            "setting": row0["setting"],
            "count": int(len(items)),
            "mean_auc_main_vs_bg": _mean("auc_main_vs_bg"),
            "mean_auc_main_vs_nuisance": _mean("auc_main_vs_nuisance"),
            "mean_fpr_nuisance_match@0p5": _mean("fpr_nuisance_match@0p5"),
            "mean_tpr_main@1e-03": _mean("tpr_main@1e-03"),
        }
        rec["selection_score"] = (
            float(rec["mean_auc_main_vs_bg"] or 0.0)
            + float(rec["mean_auc_main_vs_nuisance"] or 0.0)
            - float(rec["mean_fpr_nuisance_match@0p5"] or 0.0)
        )
        out.append(rec)
    return out


def _jsonable_case_meta(case: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in case.items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def _best_pipeline(summary_rows: list[dict[str, Any]], *, role: str) -> dict[str, Any]:
    candidates = [row for row in summary_rows if str(row["role"]) == role]
    if not candidates:
        raise ValueError(f"no {role} candidates")
    return max(candidates, key=lambda row: float(row["selection_score"]))


def _search_detector_policy(dev_rows: list[dict[str, Any]], detector_keys: list[str]) -> dict[str, Any] | None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in dev_rows:
        if str(row["role"]) != "detector":
            continue
        grouped.setdefault(str(row["case_key"]), []).append(row)
    cases = []
    for case_key, items in grouped.items():
        by_pipeline = {str(item["pipeline_key"]): item for item in items}
        if not all(key in by_pipeline for key in detector_keys):
            continue
        case0 = items[0]
        cases.append(
            {
                "case_key": case_key,
                "features": {
                    "reg_shift_p90": float(case0.get("reg_shift_p90") or np.nan),
                    "motion_speed_p90_mm_s": float(case0.get("motion_speed_p90_mm_s") or np.nan),
                },
                "pipelines": by_pipeline,
            }
        )
    if not cases:
        return None

    best: dict[str, Any] | None = None
    for feature in ("reg_shift_p90", "motion_speed_p90_mm_s"):
        values = [float(case["features"][feature]) for case in cases if np.isfinite(case["features"][feature])]
        if not values:
            continue
        for low_key in detector_keys:
            for high_key in detector_keys:
                if low_key == high_key:
                    continue
                for thr in _midpoints(values):
                    selected = [
                        case["pipelines"][low_key] if float(case["features"][feature]) <= float(thr) else case["pipelines"][high_key]
                        for case in cases
                    ]
                    utility = float(np.mean([_selection_score(row) for row in selected]))
                    cand = {
                        "feature": feature,
                        "threshold": float(thr),
                        "low_key": low_key,
                        "high_key": high_key,
                        "mean_utility": utility,
                        "mean_auc_main_vs_nuisance": float(np.mean([float(r["auc_main_vs_nuisance"]) for r in selected])),
                        "mean_fpr_nuisance_match@0p5": float(np.mean([float(r["fpr_nuisance_match@0p5"]) for r in selected])),
                        "mean_tpr_main@1e-03": float(np.mean([float(r["tpr_main@1e-03"]) for r in selected])),
                    }
                    if best is None or float(cand["mean_utility"]) > float(best["mean_utility"]):
                        best = cand
    return best


def _apply_policy_to_rows(rows: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if str(row["role"]) != "detector":
            continue
        grouped.setdefault(str(row["case_key"]), []).append(row)
    selected: list[dict[str, Any]] = []
    for items in grouped.values():
        by_key = {str(item["pipeline_key"]): item for item in items}
        low_key = str(policy["low_key"])
        high_key = str(policy["high_key"])
        if low_key not in by_key or high_key not in by_key:
            continue
        feature_value = float(items[0].get(str(policy["feature"])) or np.nan)
        chosen = low_key if feature_value <= float(policy["threshold"]) else high_key
        row = dict(by_key[chosen])
        row["policy_name"] = "motion_aware_detector_policy"
        row["policy_feature"] = str(policy["feature"])
        row["policy_threshold"] = float(policy["threshold"])
        row["policy_chosen"] = chosen
        selected.append(row)
    return selected


def _build_cases(
    *,
    split: str,
    profiles: list[str],
    axes_requested: set[str],
    levels_requested: set[str],
    seeds: list[int],
    tier: str,
    out_root: Path,
) -> list[dict[str, Any]]:
    axes = [axis for axis in _stress_axes() if axis.key in axes_requested]
    if not axes:
        raise ValueError("no stress axes selected")
    cases: list[dict[str, Any]] = []
    for profile in profiles:
        if profile not in PROFILE_LABELS:
            raise ValueError(f"unsupported profile {profile!r}")
        for axis in axes:
            for level in axis.levels:
                if level.level not in levels_requested:
                    continue
                for seed in seeds:
                    cfg = default_profile_config(profile=profile, tier=tier, seed=int(seed))  # type: ignore[arg-type]
                    cfg = level.apply(cfg)
                    run_dir = out_root / split / f"{profile}_seed{seed}_{axis.key}_{level.level}"
                    dataset_meta_path = run_dir / "dataset" / "meta.json"
                    if not dataset_meta_path.is_file():
                        write_simus_run(out_root=run_dir, cfg=cfg, skip_bundle=True)
                    cases.append(
                        {
                            "split": split,
                            "profile": profile,
                            "setting": PROFILE_LABELS[profile],
                            "axis_key": axis.key,
                            "axis_label": axis.label,
                            "level": level.level,
                            "level_label": level.label,
                            "seed": int(seed),
                            "tier": tier,
                            "run_dir": run_dir,
                            "case_key": f"{split}:{profile}:{axis.key}:{level.level}:seed{seed}",
                            "value_text": level.value_text(cfg),
                        }
                    )
    return cases


def _existing_cases(
    *,
    split: str,
    root: Path,
    profiles: list[str],
    axes_requested: set[str],
    levels_requested: set[str],
    seeds: list[int],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    allowed_profiles = set(profiles)
    allowed_seeds = set(int(x) for x in seeds)
    for run_dir in sorted(Path(root).iterdir()):
        if not run_dir.is_dir() or not (run_dir / "dataset" / "meta.json").is_file():
            continue
        parts = run_dir.name.split("_")
        if len(parts) < 4:
            continue
        profile = parts[0]
        seed_part = parts[1]
        level = parts[-1]
        axis = "_".join(parts[2:-1])
        if profile not in allowed_profiles:
            continue
        seed = int(seed_part.replace("seed", ""))
        if seed not in allowed_seeds:
            continue
        if axis not in axes_requested or level not in levels_requested:
            continue
        cfg = json.loads((run_dir / "dataset" / "config.json").read_text(encoding="utf-8"))
        value_text = ""
        for axis_spec in _stress_axes():
            if axis_spec.key != axis:
                continue
            for level_spec in axis_spec.levels:
                if level_spec.level == level:
                    value_text = level_spec.value_text(default_profile_config(profile=profile, tier=str(cfg.get("tier", "smoke")), seed=seed))
                    break
        cases.append(
            {
                "split": split,
                "profile": profile,
                "setting": PROFILE_LABELS.get(profile, profile),
                "axis_key": axis,
                "axis_label": next((a.label for a in _stress_axes() if a.key == axis), axis),
                "level": level,
                "level_label": level.capitalize(),
                "seed": seed,
                "tier": str(cfg.get("tier", "existing")),
                "run_dir": run_dir,
                "case_key": f"{split}:{profile}:{axis}:{level}:seed{seed}",
                "value_text": value_text or level,
            }
        )
    return cases


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Search for a held-out SIMUS stress-frontier headline comparison.")
    ap.add_argument("--search-json", type=Path, default=Path("reports/simus_motion/simus_fair_profile_search_seed2122to2324_expanded.json"))
    ap.add_argument("--profiles", type=str, default="ClinMobile-Pf-v2")
    ap.add_argument("--axes", type=str, default="bulk_motion,nuisance_reflectivity,cardiac_pulsation,short_ensemble")
    ap.add_argument("--levels", type=str, default="reference,moderate,hard")
    ap.add_argument("--dev-seeds", type=str, default="121,122")
    ap.add_argument("--eval-seeds", type=str, default="127,128")
    ap.add_argument("--dev-tier", type=str, default="smoke", choices=["smoke", "paper"])
    ap.add_argument("--eval-tier", type=str, default="paper", choices=["smoke", "paper"])
    ap.add_argument("--existing-dev-root", type=Path, default=None)
    ap.add_argument("--existing-eval-root", type=Path, default=None)
    ap.add_argument("--public-families", type=str, default="svd_similarity,rpca,hosvd")
    ap.add_argument("--detector-families", type=str, default="svd_similarity,rpca")
    ap.add_argument("--stap-profile", type=str, default="Brain-SIMUS-Clin-MotionRobust-v0")
    ap.add_argument("--stap-device", type=str, default="cuda")
    ap.add_argument("--out-root", type=Path, default=Path("runs/sim_eval/simus_clinical_stress_frontier"))
    ap.add_argument("--out-json", type=Path, default=Path("reports/simus_clinical_stress_frontier.json"))
    ap.add_argument("--out-csv", type=Path, default=Path("reports/simus_clinical_stress_frontier.csv"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    profiles = _split_csv_list(args.profiles)
    axes_requested = set(_split_csv_list(args.axes))
    levels_requested = set(_split_csv_list(args.levels))
    dev_seeds = [int(x) for x in _split_csv_list(args.dev_seeds)]
    eval_seeds = [int(x) for x in _split_csv_list(args.eval_seeds)]
    public_families = set(_split_csv_list(args.public_families))
    detector_families = set(_split_csv_list(args.detector_families))

    override_builders, selected_residuals = _load_selected_residual_specs(Path(args.search_json))
    public_residuals = [r for r in selected_residuals if r.method_family in public_families]
    detector_residuals = [r for r in selected_residuals if r.method_family in detector_families]
    if not public_residuals or not detector_residuals:
        raise ValueError("selected residualizer families produced an empty benchmark set")

    public_pipelines: list[PublicPipeline] = []
    for residual in public_residuals:
        for head in ("pd", "kasai"):
            public_pipelines.append(
                PublicPipeline(
                    key=f"public:{residual.method_family}:{residual.config_name}:{head}",
                    residual=residual,
                    detector_head=head,
                )
            )

    detector_pipelines: list[DetectorPipeline] = []
    for residual in detector_residuals:
        for detector_name in ("unwhitened_ref", "adaptive_guard_huber", "huber_trim8"):
            detector_pipelines.append(
                DetectorPipeline(
                    key=f"detector:{residual.method_family}:{residual.config_name}:{detector_name}",
                    residual=residual,
                    detector_name=detector_name,
                    bundle_overrides=_candidate_override(detector_name),
                )
            )

    cases = []
    if args.existing_dev_root is not None:
        cases.extend(
            _existing_cases(
                split="dev",
                root=Path(args.existing_dev_root),
                profiles=profiles,
                axes_requested=axes_requested,
                levels_requested=levels_requested,
                seeds=dev_seeds,
            )
        )
    else:
        cases.extend(
            _build_cases(
                split="dev",
                profiles=profiles,
                axes_requested=axes_requested,
                levels_requested=levels_requested,
                seeds=dev_seeds,
                tier=str(args.dev_tier),
                out_root=Path(args.out_root),
            )
        )
    if args.existing_eval_root is not None:
        cases.extend(
            _existing_cases(
                split="eval",
                root=Path(args.existing_eval_root),
                profiles=profiles,
                axes_requested=axes_requested,
                levels_requested=levels_requested,
                seeds=eval_seeds,
            )
        )
    else:
        cases.extend(
            _build_cases(
                split="eval",
                profiles=profiles,
                axes_requested=axes_requested,
                levels_requested=levels_requested,
                seeds=eval_seeds,
                tier=str(args.eval_tier),
                out_root=Path(args.out_root),
            )
        )

    rows: list[dict[str, Any]] = []
    for case in cases:
        run_dir = Path(case["run_dir"])
        icube, masks, run_meta = load_canonical_run(run_dir)
        observable = estimate_simus_policy_features(icube, reg_subpixel=4, reg_reference="median")
        motion_speed = _motion_speed_mm_s(run_dir)
        mask_h1_pf_main = masks["mask_h1_pf_main"]
        mask_h0_bg = masks["mask_h0_bg"]
        mask_h0_nuisance_pa = masks.get("mask_h0_nuisance_pa")
        mask_h1_alias_qc = masks.get("mask_h1_alias_qc")

        for pipeline in public_pipelines:
            builder = override_builders[(pipeline.residual.method_family, pipeline.residual.config_name)]
            bundle_overrides = builder(tuple(int(v) for v in icube.shape))
            bundle_dir = derive_bundle_from_run(
                run_dir=run_dir,
                out_root=Path(args.out_root) / "bundles" / str(case["split"]),
                dataset_name=f"{slugify(run_dir.name)}_{slugify(pipeline.key)}",
                stap_profile=str(args.stap_profile),
                baseline_type=str(pipeline.residual.baseline_type),
                run_stap=False,
                stap_device=str(args.stap_device),
                bundle_overrides=bundle_overrides,
                meta_extra={"simus_clinical_stress_frontier": True, "pipeline_key": pipeline.key, **_jsonable_case_meta(case)},
            )
            score_path = _score_path(bundle_dir, detector_head=pipeline.detector_head, role="public")
            if score_path is None or not score_path.is_file():
                continue
            score = np.load(score_path).astype(np.float32, copy=False)
            metrics = evaluate_structural_metrics(
                score=score,
                mask_h1_pf_main=mask_h1_pf_main,
                mask_h0_bg=mask_h0_bg,
                mask_h0_nuisance_pa=mask_h0_nuisance_pa,
                mask_h1_alias_qc=mask_h1_alias_qc,
                fprs=[1e-3],
                match_tprs=[0.5],
            )
            row = {
                **case,
                "pipeline_key": pipeline.key,
                "pipeline_label": pipeline.label,
                "role": "public",
                "method_family": pipeline.residual.method_family,
                "config_name": pipeline.residual.config_name,
                "baseline_type": pipeline.residual.baseline_type,
                "detector_head": pipeline.detector_head,
                "reg_shift_p90": observable.get("reg_shift_p90"),
                "motion_speed_p90_mm_s": motion_speed,
                "bundle_dir": str(bundle_dir),
            }
            row.update(metrics)
            row["selection_score"] = _selection_score(row)
            rows.append(row)

        for pipeline in detector_pipelines:
            builder = override_builders[(pipeline.residual.method_family, pipeline.residual.config_name)]
            bundle_overrides = builder(tuple(int(v) for v in icube.shape))
            bundle_overrides.update(dict(pipeline.bundle_overrides))
            bundle_dir = derive_bundle_from_run(
                run_dir=run_dir,
                out_root=Path(args.out_root) / "bundles" / str(case["split"]),
                dataset_name=f"{slugify(run_dir.name)}_{slugify(pipeline.key)}",
                stap_profile=str(args.stap_profile),
                baseline_type=str(pipeline.residual.baseline_type),
                run_stap=True,
                stap_device=str(args.stap_device),
                bundle_overrides=bundle_overrides,
                meta_extra={"simus_clinical_stress_frontier": True, "pipeline_key": pipeline.key, **_jsonable_case_meta(case)},
            )
            score_path = _score_path(bundle_dir, detector_head="stap", role="detector")
            if score_path is None or not score_path.is_file():
                continue
            score = np.load(score_path).astype(np.float32, copy=False)
            metrics = evaluate_structural_metrics(
                score=score,
                mask_h1_pf_main=mask_h1_pf_main,
                mask_h0_bg=mask_h0_bg,
                mask_h0_nuisance_pa=mask_h0_nuisance_pa,
                mask_h1_alias_qc=mask_h1_alias_qc,
                fprs=[1e-3],
                match_tprs=[0.5],
            )
            row = {
                **case,
                "pipeline_key": pipeline.key,
                "pipeline_label": pipeline.label,
                "role": "detector",
                "method_family": pipeline.residual.method_family,
                "config_name": pipeline.residual.config_name,
                "baseline_type": pipeline.residual.baseline_type,
                "detector_name": pipeline.detector_name,
                "reg_shift_p90": observable.get("reg_shift_p90"),
                "motion_speed_p90_mm_s": motion_speed,
                "bundle_dir": str(bundle_dir),
            }
            row.update(metrics)
            row["selection_score"] = _selection_score(row)
            rows.append(row)

    dev_rows = [row for row in rows if str(row["split"]) == "dev"]
    eval_rows = [row for row in rows if str(row["split"]) == "eval"]
    dev_summary = _aggregate_by_pipeline(dev_rows)
    eval_summary = _aggregate_by_pipeline(eval_rows)
    best_public = _best_pipeline(dev_summary, role="public")
    best_detector = _best_pipeline(dev_summary, role="detector")
    detector_keys = [row["pipeline_key"] for row in dev_summary if str(row["role"]) == "detector"]
    detector_policy = _search_detector_policy(dev_rows, detector_keys=detector_keys)

    eval_best_public_rows = [row for row in eval_rows if str(row["pipeline_key"]) == str(best_public["pipeline_key"])]
    eval_best_detector_rows = [row for row in eval_rows if str(row["pipeline_key"]) == str(best_detector["pipeline_key"])]
    eval_policy_rows = _apply_policy_to_rows(eval_rows, detector_policy) if detector_policy is not None else []

    summary = {
        "best_public_dev": best_public,
        "best_detector_dev": best_detector,
        "detector_policy_dev": detector_policy,
        "eval_best_public": _aggregate_by_pipeline(eval_best_public_rows),
        "eval_best_detector": _aggregate_by_pipeline(eval_best_detector_rows),
        "eval_detector_policy": _aggregate_by_pipeline(eval_policy_rows),
    }
    summary["headline_delta_eval"] = None
    if summary["eval_best_public"] and summary["eval_best_detector"]:
        public_eval = summary["eval_best_public"][0]
        detector_eval = summary["eval_best_detector"][0]
        summary["headline_delta_eval"] = {
            "detector_minus_public_selection": float(detector_eval["selection_score"]) - float(public_eval["selection_score"]),
            "detector_minus_public_auc_main_vs_nuisance": float(detector_eval["mean_auc_main_vs_nuisance"] or 0.0) - float(public_eval["mean_auc_main_vs_nuisance"] or 0.0),
            "detector_minus_public_fpr_nuisance_match@0p5": float(detector_eval["mean_fpr_nuisance_match@0p5"] or 0.0) - float(public_eval["mean_fpr_nuisance_match@0p5"] or 0.0),
        }

    payload = {
        "schema_version": "simus_clinical_stress_frontier.v1",
        "search_json": str(Path(args.search_json)),
        "profiles": profiles,
        "axes": sorted(axes_requested),
        "levels": sorted(levels_requested),
        "dev_seeds": dev_seeds,
        "eval_seeds": eval_seeds,
        "dev_tier": str(args.dev_tier),
        "eval_tier": str(args.eval_tier),
        "rows": rows,
        "dev_summary": dev_summary,
        "eval_summary": eval_summary,
        "summary": summary,
    }
    _write_csv(Path(args.out_csv), rows)
    _write_json(Path(args.out_json), payload)
    print(f"[simus-clinical-stress-frontier] wrote {args.out_csv}")
    print(f"[simus-clinical-stress-frontier] wrote {args.out_json}")


if __name__ == "__main__":
    main()
