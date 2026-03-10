#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.simus_eval_structural import _baseline_label, evaluate_structural_metrics
from scripts.simus_fair_profile_search import _candidate_specs
from sim.simus.bundle import derive_bundle_from_run, slugify
from sim.simus.functional import default_functional_design, write_functional_case


SIMPLE_HEADS = ("pd", "kasai")
FPR_TAGS = ("1e-04", "1e-03")


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


def _override_builder_lookup() -> dict[tuple[str, str], Any]:
    out: dict[tuple[str, str], Any] = {}
    for cand in _candidate_specs():
        out[(str(cand.method_family), str(cand.config_name))] = cand.override_builder
    return out


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _selected_residual_specs(search_payload: dict[str, Any]) -> list[ResidualSpec]:
    selected = dict(search_payload.get("details", {}).get("selected_configs") or search_payload.get("selected_configs") or {})
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


def _selected_stap_profile(stack_payload: dict[str, Any]) -> str:
    selected = dict(stack_payload.get("selected_stack") or {})
    profile = selected.get("stap_profile")
    if not profile:
        raise ValueError("stack payload missing selected STAP profile")
    return str(profile)


def _glm_tmap(stack_ehw: np.ndarray, regressor: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    Y = np.asarray(stack_ehw, dtype=np.float64)
    if Y.ndim != 3:
        raise ValueError(f"expected (E,H,W), got {Y.shape}")
    E, H, W = Y.shape
    if E < 6:
        raise ValueError(f"functional GLM requires at least 6 ensembles, got {E}")
    y = Y.reshape(E, H * W)
    t_idx = np.arange(E, dtype=np.float64)
    intercept = np.ones_like(t_idx)
    trend = (t_idx - t_idx.mean()) / (t_idx.std() + eps)
    reg = np.asarray(regressor, dtype=np.float64).reshape(E)
    reg = (reg - reg.mean()) / (reg.std() + eps)
    X = np.stack([intercept, trend, reg], axis=1)
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    dof = max(E - X.shape[1], 1)
    sigma2 = np.sum(resid * resid, axis=0) / float(dof)
    var_beta = np.diag(XtX_inv)[2] * sigma2
    t_stat = beta[2] / np.sqrt(var_beta + eps)
    return t_stat.reshape(H, W).astype(np.float32, copy=False)


def _corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size == 0:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    den = float(np.sqrt(np.sum(x * x) * np.sum(y * y)) + eps)
    return float(np.sum(x * y) / den) if den > 0.0 else 0.0


def _threshold_from_null(null_map: np.ndarray, mask: np.ndarray, rate: float) -> tuple[float | None, float | None]:
    vals = np.asarray(null_map, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, None
    thr = float(np.quantile(vals, 1.0 - float(rate)))
    emp = float(np.mean(vals >= thr))
    return thr, emp


def _score_selection(row: dict[str, Any]) -> float:
    return (
        float(row.get("mean_auc_activation_vs_bg") or 0.0)
        + float(row.get("mean_auc_activation_vs_nuisance") or 0.0)
        - float(row.get("mean_nuisance_frac_task@1e-03") or 0.0)
        - float(row.get("mean_outside_frac_task@1e-03") or 0.0)
    )


def _load_mask(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    return np.load(p).astype(bool, copy=False)


def _load_score(bundle_dir: Path, detector_head: str) -> np.ndarray:
    head = str(detector_head).strip().lower()
    if head == "pd":
        path = bundle_dir / "score_base.npy"
    elif head == "kasai":
        path = bundle_dir / "score_base_kasai.npy"
    elif head == "stap":
        path = bundle_dir / "score_stap_preka.npy"
    elif head == "stap_pd":
        path = bundle_dir / "score_pd_stap.npy"
    else:
        raise ValueError(f"unsupported detector head {detector_head!r}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path).astype(np.float32, copy=False)


def _detector_label(detector_head: str) -> str:
    return {
        "pd": "PD",
        "kasai": "Kasai",
        "stap": "STAP",
        "stap_pd": "PD-after-STAP",
    }[str(detector_head).strip().lower()]


def _pipeline_label(baseline_type: str, detector_head: str) -> str:
    return f"{_baseline_label(baseline_type)} -> {_detector_label(detector_head)}"


def _derive_bundle_timed(
    *,
    run_dir: Path,
    out_root: Path,
    dataset_name: str,
    baseline_type: str,
    bundle_overrides: dict[str, Any],
    run_stap: bool,
    stap_profile: str,
    stap_device: str,
    meta_extra: dict[str, Any],
    reuse_bundle: bool,
) -> tuple[Path, float | None]:
    bundle_dir = out_root / slugify(dataset_name)
    runtime_meta = bundle_dir / "functional_runtime.json"
    if bool(reuse_bundle) and (bundle_dir / "meta.json").is_file():
        if runtime_meta.is_file():
            payload = _load_json(runtime_meta)
            return bundle_dir, float(payload.get("elapsed_ms")) if payload.get("elapsed_ms") is not None else None
        return bundle_dir, None
    t0 = time.perf_counter()
    bundle_dir = derive_bundle_from_run(
        run_dir=run_dir,
        out_root=out_root,
        dataset_name=dataset_name,
        stap_profile=stap_profile,
        baseline_type=baseline_type,
        run_stap=bool(run_stap),
        stap_device=stap_device,
        bundle_overrides=bundle_overrides,
        meta_extra=meta_extra,
    )
    elapsed_ms = 1000.0 * (time.perf_counter() - t0)
    _write_json(
        runtime_meta,
        {
            "elapsed_ms": float(elapsed_ms),
            "run_stap": bool(run_stap),
            "baseline_type": str(baseline_type),
            "stap_profile": str(stap_profile),
        },
    )
    return bundle_dir, float(elapsed_ms)


def _summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (
                str(row["split"]),
                str(row["base_profile"]),
                str(row["method_family"]),
                str(row["config_name"]),
                str(row["detector_head"]),
            ),
            [],
        ).append(row)
    out: list[dict[str, Any]] = []
    for (split, base_profile, method_family, config_name, detector_head), items in sorted(grouped.items()):
        def _mean(key: str) -> float | None:
            vals = [float(r[key]) for r in items if r.get(key) is not None]
            return float(np.mean(np.asarray(vals, dtype=np.float64))) if vals else None
        row0 = items[0]
        summary = {
            "split": split,
            "base_profile": base_profile,
            "method_family": method_family,
            "config_name": config_name,
            "baseline_type": row0["baseline_type"],
            "baseline_label": row0["baseline_label"],
            "detector_head": detector_head,
            "detector_label": row0["detector_label"],
            "pipeline_label": row0["pipeline_label"],
            "stap_profile": row0["stap_profile"],
            "count": int(len(items)),
            "mean_auc_activation_vs_bg": _mean("auc_activation_vs_bg"),
            "mean_auc_activation_vs_nuisance": _mean("auc_activation_vs_nuisance"),
            "mean_roi_corr_task": _mean("roi_corr_task"),
            "mean_roi_corr_null": _mean("roi_corr_null"),
            "mean_runtime_ms": _mean("runtime_ms_mean"),
        }
        for tag in FPR_TAGS:
            summary[f"mean_outside_frac_task@{tag}"] = _mean(f"outside_frac_task@{tag}")
            summary[f"mean_nuisance_frac_task@{tag}"] = _mean(f"nuisance_frac_task@{tag}")
            summary[f"mean_roi_tpr_task@{tag}"] = _mean(f"roi_tpr_task@{tag}")
        summary["selection_score"] = _score_selection(summary)
        out.append(summary)
    return out


def _best_native_simple_by_family(dev_summary: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in dev_summary:
        if str(row["detector_head"]) not in SIMPLE_HEADS:
            continue
        grouped.setdefault((str(row["base_profile"]), str(row["method_family"])), []).append(row)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, items in grouped.items():
        out[key] = max(items, key=_score_selection)
    return out


def _extract_matching_eval(eval_summary: list[dict[str, Any]], selected: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {
        (
            str(v["base_profile"]),
            str(v["method_family"]),
            str(v["config_name"]),
            str(v["detector_head"]),
        )
        for v in selected.values()
    }
    return [
        row
        for row in eval_summary
        if (
            str(row["base_profile"]),
            str(row["method_family"]),
            str(row["config_name"]),
            str(row["detector_head"]),
        )
        in wanted
    ]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Functional SIMUS evaluation on accepted v2 profiles.")
    ap.add_argument("--search-json", type=Path, required=True, help="Accepted v2 frozen-profile search JSON.")
    ap.add_argument("--stack-json", type=Path, required=True, help="Accepted v2 STAP stack search JSON.")
    ap.add_argument(
        "--profiles",
        type=str,
        default="ClinMobile-Pf-v2,ClinIntraOpParenchyma-Pf-v3",
    )
    ap.add_argument("--tier", type=str, default="functional", choices=["smoke", "paper", "functional"])
    ap.add_argument("--dev-seeds", type=str, default="221")
    ap.add_argument("--eval-seeds", type=str, default="222")
    ap.add_argument("--ensemble-count", type=int, default=8)
    ap.add_argument("--max-workers", type=int, default=2)
    ap.add_argument("--threads-per-worker", type=int, default=12)
    ap.add_argument("--stap-device", type=str, default="cuda")
    ap.add_argument("--reuse-cases", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--reuse-bundles", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-headline-csv", type=Path, required=True)
    ap.add_argument("--out-headline-json", type=Path, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.ensemble_count) < 6:
        raise ValueError("--ensemble-count must be at least 6 for a valid functional GLM")
    search_payload = _load_json(Path(args.search_json))
    stack_payload = _load_json(Path(args.stack_json))
    residual_specs = _selected_residual_specs(search_payload)
    stap_profile = _selected_stap_profile(stack_payload)
    override_lookup = _override_builder_lookup()
    base_profiles = _split_csv_list(str(args.profiles))
    dev_seeds = [int(x) for x in _split_csv_list(str(args.dev_seeds))]
    eval_seeds = [int(x) for x in _split_csv_list(str(args.eval_seeds))]

    out_root = Path(args.out_root)
    cases_root = out_root / "cases"
    bundles_root = out_root / "bundles"
    cases_root.mkdir(parents=True, exist_ok=True)
    bundles_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    def _case_root(split: str, base_profile: str, seed: int, null_run: bool) -> Path:
        tag = "null" if bool(null_run) else "task"
        return cases_root / split / f"{slugify(base_profile)}_seed{int(seed)}_{tag}"

    def _build_case(split: str, base_profile: str, seed: int, null_run: bool) -> Path:
        case_root = _case_root(split, base_profile, seed, null_run)
        write_functional_case(
            out_root=case_root,
            base_profile=base_profile,
            tier=str(args.tier),
            seed=int(seed),
            null_run=bool(null_run),
            design_spec=default_functional_design(base_profile, ensemble_count=int(args.ensemble_count)),
            max_workers=int(args.max_workers),
            threads_per_worker=int(args.threads_per_worker),
            reuse_existing=bool(args.reuse_cases),
        )
        return case_root

    def _eval_case(split: str, base_profile: str, seed: int) -> None:
        task_root = _build_case(split, base_profile, seed, False)
        null_root = _build_case(split, base_profile, seed, True)
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
            builder = override_lookup[(residual.method_family, residual.config_name)]
            bundle_overrides = builder(icube_shape)
            task_scores: dict[str, list[np.ndarray]] = {"pd": [], "kasai": [], "stap": [], "stap_pd": []}
            null_scores: dict[str, list[np.ndarray]] = {"pd": [], "kasai": [], "stap": [], "stap_pd": []}
            head_runtimes_ms: dict[str, list[float]] = {"pd": [], "kasai": [], "stap": [], "stap_pd": []}
            for task_row, null_row in zip(task_rows, null_rows):
                dataset_stub = f"{slugify(base_profile)}_seed{int(seed)}_{slugify(residual.method_family)}_{slugify(residual.config_name)}"
                task_bundle_root = bundles_root / split / f"{slugify(base_profile)}_seed{int(seed)}" / "task"
                null_bundle_root = bundles_root / split / f"{slugify(base_profile)}_seed{int(seed)}" / "null"
                task_bundle_root.mkdir(parents=True, exist_ok=True)
                null_bundle_root.mkdir(parents=True, exist_ok=True)

                native_task_dir, native_task_ms = _derive_bundle_timed(
                    run_dir=Path(task_row["ensemble_dir"]),
                    out_root=task_bundle_root,
                    dataset_name=f"{dataset_stub}_e{int(task_row['ensemble_index']):03d}_native",
                    baseline_type=residual.baseline_type,
                    bundle_overrides=bundle_overrides,
                    run_stap=False,
                    stap_profile=stap_profile,
                    stap_device=str(args.stap_device),
                    meta_extra={"simus_eval_functional": True, "split": split, "base_profile": base_profile, "detector_bundle": "native"},
                    reuse_bundle=bool(args.reuse_bundles),
                )
                native_null_dir, native_null_ms = _derive_bundle_timed(
                    run_dir=Path(null_row["ensemble_dir"]),
                    out_root=null_bundle_root,
                    dataset_name=f"{dataset_stub}_e{int(null_row['ensemble_index']):03d}_native",
                    baseline_type=residual.baseline_type,
                    bundle_overrides=bundle_overrides,
                    run_stap=False,
                    stap_profile=stap_profile,
                    stap_device=str(args.stap_device),
                    meta_extra={"simus_eval_functional": True, "split": split, "base_profile": base_profile, "detector_bundle": "native_null"},
                    reuse_bundle=bool(args.reuse_bundles),
                )
                stap_task_dir, stap_task_ms = _derive_bundle_timed(
                    run_dir=Path(task_row["ensemble_dir"]),
                    out_root=task_bundle_root,
                    dataset_name=f"{dataset_stub}_e{int(task_row['ensemble_index']):03d}_stap",
                    baseline_type=residual.baseline_type,
                    bundle_overrides=bundle_overrides,
                    run_stap=True,
                    stap_profile=stap_profile,
                    stap_device=str(args.stap_device),
                    meta_extra={"simus_eval_functional": True, "split": split, "base_profile": base_profile, "detector_bundle": "stap"},
                    reuse_bundle=bool(args.reuse_bundles),
                )
                stap_null_dir, stap_null_ms = _derive_bundle_timed(
                    run_dir=Path(null_row["ensemble_dir"]),
                    out_root=null_bundle_root,
                    dataset_name=f"{dataset_stub}_e{int(null_row['ensemble_index']):03d}_stap",
                    baseline_type=residual.baseline_type,
                    bundle_overrides=bundle_overrides,
                    run_stap=True,
                    stap_profile=stap_profile,
                    stap_device=str(args.stap_device),
                    meta_extra={"simus_eval_functional": True, "split": split, "base_profile": base_profile, "detector_bundle": "stap_null"},
                    reuse_bundle=bool(args.reuse_bundles),
                )

                task_scores["pd"].append(_load_score(native_task_dir, "pd"))
                task_scores["kasai"].append(_load_score(native_task_dir, "kasai"))
                task_scores["stap"].append(_load_score(stap_task_dir, "stap"))
                task_scores["stap_pd"].append(_load_score(stap_task_dir, "stap_pd"))
                null_scores["pd"].append(_load_score(native_null_dir, "pd"))
                null_scores["kasai"].append(_load_score(native_null_dir, "kasai"))
                null_scores["stap"].append(_load_score(stap_null_dir, "stap"))
                null_scores["stap_pd"].append(_load_score(stap_null_dir, "stap_pd"))

                if native_task_ms is not None:
                    head_runtimes_ms["pd"].append(float(native_task_ms))
                    head_runtimes_ms["kasai"].append(float(native_task_ms))
                if native_null_ms is not None:
                    head_runtimes_ms["pd"].append(float(native_null_ms))
                    head_runtimes_ms["kasai"].append(float(native_null_ms))
                if stap_task_ms is not None:
                    head_runtimes_ms["stap"].append(float(stap_task_ms))
                    head_runtimes_ms["stap_pd"].append(float(stap_task_ms))
                if stap_null_ms is not None:
                    head_runtimes_ms["stap"].append(float(stap_null_ms))
                    head_runtimes_ms["stap_pd"].append(float(stap_null_ms))

            for detector_head in ("pd", "kasai", "stap", "stap_pd"):
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
                    "method_family": residual.method_family,
                    "config_name": residual.config_name,
                    "baseline_type": residual.baseline_type,
                    "baseline_label": _baseline_label(residual.baseline_type),
                    "detector_head": detector_head,
                    "detector_label": _detector_label(detector_head),
                    "pipeline_label": _pipeline_label(residual.baseline_type, detector_head),
                    "stap_profile": stap_profile if detector_head in {"stap", "stap_pd"} else None,
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
    selected_native = _best_native_simple_by_family(dev_summary)
    eval_native = _extract_matching_eval(eval_summary, selected_native)
    selected_native_json = [
        {"base_profile": key[0], "method_family": key[1], "selection": value}
        for key, value in sorted(selected_native.items())
    ]
    payload = {
        "schema_version": "simus_eval_functional.v1",
        "base_profiles": base_profiles,
        "dev_seeds": dev_seeds,
        "eval_seeds": eval_seeds,
        "ensemble_count": int(args.ensemble_count),
        "selected_residual_specs": [r.__dict__ for r in residual_specs],
        "stap_profile": stap_profile,
        "rows": rows,
        "summary": summary,
        "selected_native_simple_by_family": selected_native_json,
        "eval_native_simple_selected": eval_native,
    }
    _write_csv(Path(args.out_csv), rows)
    _write_json(Path(args.out_json), payload)
    _write_csv(Path(args.out_headline_csv), summary)
    _write_json(Path(args.out_headline_json), {"summary": summary, "selected_native_simple_by_family": selected_native_json, "eval_native_simple_selected": eval_native, "stap_profile": stap_profile})
    print(f"[simus-eval-functional] wrote {args.out_csv}")
    print(f"[simus-eval-functional] wrote {args.out_json}")


if __name__ == "__main__":
    main()
