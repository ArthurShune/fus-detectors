#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from scripts.simus_detector_family_ablation_table import _dataset_name, _setting_label
from scripts.simus_eval_structural import evaluate_structural_metrics
from sim.simus.bundle import derive_bundle_from_run

REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SweepConfig:
    name: str
    detector_variant: str
    whiten_gamma: float
    cov_estimator: str
    diag_load: float
    cov_train_trim_q: float
    huber_c: float
    mvdr_auto_kappa: float
    constraint_ridge: float


DEFAULT_CONFIGS: tuple[SweepConfig, ...] = (
    SweepConfig(
        name="unwhitened_ref",
        detector_variant="unwhitened_ratio",
        whiten_gamma=0.0,
        cov_estimator="tyler_pca",
        diag_load=0.07,
        cov_train_trim_q=0.0,
        huber_c=5.0,
        mvdr_auto_kappa=120.0,
        constraint_ridge=0.18,
    ),
    SweepConfig(
        name="whitened_default",
        detector_variant="msd_ratio",
        whiten_gamma=1.0,
        cov_estimator="tyler_pca",
        diag_load=0.07,
        cov_train_trim_q=0.0,
        huber_c=5.0,
        mvdr_auto_kappa=120.0,
        constraint_ridge=0.18,
    ),
    SweepConfig(
        name="whitened_acceptance_like",
        detector_variant="msd_ratio",
        whiten_gamma=1.0,
        cov_estimator="tyler_pca",
        diag_load=0.10,
        cov_train_trim_q=0.05,
        huber_c=5.0,
        mvdr_auto_kappa=200.0,
        constraint_ridge=0.25,
    ),
    SweepConfig(
        name="whitened_trim10_loaded",
        detector_variant="msd_ratio",
        whiten_gamma=1.0,
        cov_estimator="tyler_pca",
        diag_load=0.12,
        cov_train_trim_q=0.10,
        huber_c=5.0,
        mvdr_auto_kappa=220.0,
        constraint_ridge=0.30,
    ),
    SweepConfig(
        name="whitened_huber_acceptance_like",
        detector_variant="msd_ratio",
        whiten_gamma=1.0,
        cov_estimator="huber",
        diag_load=0.10,
        cov_train_trim_q=0.05,
        huber_c=5.0,
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


def _finite_mean(vals: Iterable[float | None]) -> float | None:
    arr = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def _simus_runs_default() -> list[Path]:
    return [
        Path("runs/sim/simus_v2_acceptance_clin_mobile_pf_v2_phase3_seed127"),
        Path("runs/sim/simus_v2_acceptance_clin_mobile_pf_v2_phase3_seed128"),
        Path("runs/sim/simus_v2_acceptance_clin_intraop_parenchyma_pf_v3_phase3_seed127"),
        Path("runs/sim/simus_v2_acceptance_clin_intraop_parenchyma_pf_v3_phase3_seed128"),
    ]


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
    return env


def _resolve_twinkling_par_dat(seq_dir: Path) -> tuple[Path, Path]:
    par_candidates = sorted(seq_dir.glob("RawBCFCine*.par"))
    if not par_candidates:
        raise FileNotFoundError(f"No RawBCFCine*.par found under {seq_dir}")
    par_path = seq_dir / "RawBCFCine.par"
    dat_path = seq_dir / "RawBCFCine.dat"
    if par_path.exists() and dat_path.exists():
        return par_path, dat_path
    preferred_par = [p for p in par_candidates if p.stem.endswith("_17")]
    par_path = preferred_par[0] if preferred_par else par_candidates[-1]
    dat_path = par_path.with_suffix(".dat")
    if not dat_path.exists():
        raise FileNotFoundError(f"Missing matching dat file for {par_path}")
    return par_path, dat_path


def _num_frames_for_seq(seq_dir: Path) -> int:
    par_path, _ = _resolve_twinkling_par_dat(seq_dir)
    text = par_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = [x.strip() for x in line.split("=", 1)]
        if key == "NumOfFrames":
            return int(value)
    raise ValueError(f"Could not parse NumOfFrames from {par_path}")


def _existing_twinkling_root_for_config(cfg: SweepConfig) -> Path | None:
    if cfg.name == "unwhitened_ref":
        return REPO / "runs/real/twinkling_gammex_across17_prf2500_str4_unwhitened_ratio"
    if cfg.name == "whitened_default":
        return REPO / "runs/real/twinkling_gammex_across17_prf2500_str4_msd_ratio_fast"
    return None


def _existing_twinkling_summary_for_config(cfg: SweepConfig) -> Path | None:
    if cfg.name == "unwhitened_ref":
        return REPO / "reports/twinkling_gammex_across17_prf2500_str4_unwhitened_ratio_structural_summary.json"
    if cfg.name == "whitened_default":
        return REPO / "reports/twinkling_gammex_across17_prf2500_str4_msd_ratio_fast_structural_summary.json"
    return None


def _run_twinkling_bundle(
    *,
    args: argparse.Namespace,
    seq_dir: Path,
    out_root: Path,
    cfg: SweepConfig,
) -> Path:
    n_frames = _num_frames_for_seq(seq_dir)
    par_path, dat_path = _resolve_twinkling_par_dat(seq_dir)
    cmd = [
        args.python_exe,
        str(REPO / "scripts/twinkling_make_bundles.py"),
        "--seq-dir",
        str(seq_dir),
        "--par-path",
        str(par_path),
        "--dat-path",
        str(dat_path),
        "--out-root",
        str(out_root),
        "--frames",
        f"0:{n_frames}",
        "--prf-hz",
        str(float(args.twinkling_prf_hz)),
        "--tile-stride",
        str(int(args.twinkling_tile_stride)),
        "--stap-device",
        str(args.stap_device),
        "--stap-detector-variant",
        str(cfg.detector_variant),
        "--stap-whiten-gamma",
        str(float(cfg.whiten_gamma)),
        "--diag-load",
        str(float(cfg.diag_load)),
        "--cov-estimator",
        str(cfg.cov_estimator),
        "--huber-c",
        str(float(cfg.huber_c)),
        "--stap-cov-trim-q",
        str(float(cfg.cov_train_trim_q)),
        "--mvdr-auto-kappa",
        str(float(cfg.mvdr_auto_kappa)),
        "--constraint-ridge",
        str(float(cfg.constraint_ridge)),
    ]
    subprocess.run(cmd, cwd=REPO, env=_subprocess_env(), check=True)
    return out_root


def _run_twinkling_eval(*, args: argparse.Namespace, root: Path, out_csv: Path, summary_json: Path) -> None:
    cmd = [
        args.python_exe,
        str(REPO / "scripts/twinkling_eval_structural.py"),
        "--root",
        str(root),
        "--out-csv",
        str(out_csv),
        "--out-summary-json",
        str(summary_json),
    ]
    subprocess.run(cmd, cwd=REPO, env=_subprocess_env(), check=True)


def _default_configs() -> list[SweepConfig]:
    return list(DEFAULT_CONFIGS)


def _parse_config(spec: str) -> SweepConfig:
    parts = [x.strip() for x in str(spec).split(",")]
    if len(parts) != 9:
        raise ValueError(
            "--config expects "
            "name,detector_variant,whiten_gamma,cov_estimator,diag_load,cov_train_trim_q,huber_c,mvdr_auto_kappa,constraint_ridge"
        )
    return SweepConfig(
        name=parts[0],
        detector_variant=parts[1],
        whiten_gamma=float(parts[2]),
        cov_estimator=parts[3],
        diag_load=float(parts[4]),
        cov_train_trim_q=float(parts[5]),
        huber_c=float(parts[6]),
        mvdr_auto_kappa=float(parts[7]),
        constraint_ridge=float(parts[8]),
    )


def _run_simus_case(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    cfg: SweepConfig,
) -> dict[str, Any]:
    ds = run_dir / "dataset"
    mask_h1_pf_main = np.load(ds / "mask_h1_pf_main.npy")
    mask_h0_bg = np.load(ds / "mask_h0_bg.npy")
    mask_h0_nuisance_pa = np.load(ds / "mask_h0_nuisance_pa.npy")
    mask_h1_alias_qc = np.load(ds / "mask_h1_alias_qc.npy")
    setting = _setting_label(run_dir)
    dataset_name = _dataset_name(run_dir, cfg.name)
    bundle_dir = derive_bundle_from_run(
        run_dir=run_dir,
        out_root=Path(args.simus_out_root) / cfg.name,
        dataset_name=dataset_name,
        stap_profile=str(args.simus_profile),
        baseline_type="mc_svd",
        run_stap=True,
        stap_device=str(args.stap_device),
        bundle_overrides={
            "stap_detector_variant": str(cfg.detector_variant),
            "stap_whiten_gamma": float(cfg.whiten_gamma),
            "diag_load": float(cfg.diag_load),
            "stap_cov_train_trim_q": float(cfg.cov_train_trim_q),
            "cov_estimator": str(cfg.cov_estimator),
            "huber_c": float(cfg.huber_c),
            "mvdr_auto_kappa": float(cfg.mvdr_auto_kappa),
            "constraint_ridge": float(cfg.constraint_ridge),
        },
        meta_extra={
            "stap_covariance_regime_sweep": True,
            "sweep_config": asdict(cfg),
        },
    )
    score = np.load(Path(bundle_dir) / "score_stap_preka.npy").astype(np.float32, copy=False)
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
        "family": "simus",
        "config": cfg.name,
        "setting": setting,
        "run": run_dir.name,
        "bundle_dir": str(bundle_dir),
        "auc_main_vs_bg": metrics.get("auc_main_vs_bg"),
        "auc_main_vs_nuisance": metrics.get("auc_main_vs_nuisance"),
        "fpr_nuisance_match@0p5": metrics.get("fpr_nuisance_match@0p5"),
        "tpr_main@1e-03": metrics.get("tpr_main@1e-03"),
    }
    row.update(
        {
            "detector_variant": cfg.detector_variant,
            "whiten_gamma": float(cfg.whiten_gamma),
            "cov_estimator": cfg.cov_estimator,
            "diag_load": float(cfg.diag_load),
            "cov_train_trim_q": float(cfg.cov_train_trim_q),
            "huber_c": float(cfg.huber_c),
            "mvdr_auto_kappa": float(cfg.mvdr_auto_kappa),
            "constraint_ridge": float(cfg.constraint_ridge),
        }
    )
    return row


def _run_twinkling_case(
    *,
    args: argparse.Namespace,
    cfg: SweepConfig,
) -> dict[str, Any]:
    seq_dir = Path(args.twinkling_seq_dir)
    root = _existing_twinkling_root_for_config(cfg)
    summary_json = _existing_twinkling_summary_for_config(cfg)
    if root is None or summary_json is None or not root.exists() or not summary_json.exists():
        root = Path(args.twinkling_out_root) / f"twinkling_gammex_across_covsweep_{cfg.name}"
        out_csv = Path(args.out_root) / "twinkling" / f"{cfg.name}_across.csv"
        summary_json = Path(args.out_root) / "twinkling" / f"{cfg.name}_across_summary.json"
        if not root.exists() or not summary_json.exists():
            root = _run_twinkling_bundle(args=args, seq_dir=seq_dir, out_root=root, cfg=cfg)
            _run_twinkling_eval(args=args, root=root, out_csv=out_csv, summary_json=summary_json)

    obj = json.loads(summary_json.read_text(encoding="utf-8"))
    methods = obj.get("pooled_roc", {}).get("methods", {})
    stap = methods.get("stap_preka", {})
    roc = stap.get("roc", [])
    point_by_fpr: dict[float, dict[str, Any]] = {}
    for pt in roc:
        if not isinstance(pt, dict):
            continue
        try:
            point_by_fpr[float(pt.get("fpr_target"))] = pt
        except Exception:
            continue
    row = {
        "family": "twinkling",
        "config": cfg.name,
        "setting": "Gammex across-view",
        "view": "across",
        "root": str(root),
        "summary_json": str(summary_json),
        "tpr@1e-4": (point_by_fpr.get(1e-4) or {}).get("tpr"),
        "tpr@3e-4": (point_by_fpr.get(3e-4) or {}).get("tpr"),
        "tpr@1e-3": (point_by_fpr.get(1e-3) or {}).get("tpr"),
        "detector_variant": cfg.detector_variant,
        "whiten_gamma": float(cfg.whiten_gamma),
        "cov_estimator": cfg.cov_estimator,
        "diag_load": float(cfg.diag_load),
        "cov_train_trim_q": float(cfg.cov_train_trim_q),
        "huber_c": float(cfg.huber_c),
        "mvdr_auto_kappa": float(cfg.mvdr_auto_kappa),
        "constraint_ridge": float(cfg.constraint_ridge),
    }
    return row


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family_setting: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family_setting[(str(row["family"]), str(row["setting"]))].append(row)

    out: dict[str, Any] = {}
    for (family, setting), items in sorted(by_family_setting.items()):
        setting_out: list[dict[str, Any]] = []
        ref = next((x for x in items if str(x["config"]) == "unwhitened_ref"), None)
        for cfg_name in sorted({str(x["config"]) for x in items}):
            cfg_items = [x for x in items if str(x["config"]) == cfg_name]
            rec: dict[str, Any] = {
                "config": cfg_name,
                "count": len(cfg_items),
            }
            if family == "simus":
                rec.update(
                    {
                        "auc_main_vs_bg_mean": _finite_mean(x.get("auc_main_vs_bg") for x in cfg_items),
                        "auc_main_vs_nuisance_mean": _finite_mean(
                            x.get("auc_main_vs_nuisance") for x in cfg_items
                        ),
                        "fpr_nuisance_match@0p5_mean": _finite_mean(
                            x.get("fpr_nuisance_match@0p5") for x in cfg_items
                        ),
                        "tpr_main@1e-03_mean": _finite_mean(x.get("tpr_main@1e-03") for x in cfg_items),
                    }
                )
                if ref is not None and cfg_name != "unwhitened_ref":
                    ref_items = [x for x in items if str(x["config"]) == "unwhitened_ref"]
                    rec["delta_auc_main_vs_bg_mean_vs_unwhitened"] = (
                        rec["auc_main_vs_bg_mean"] - _finite_mean(x.get("auc_main_vs_bg") for x in ref_items)
                        if rec["auc_main_vs_bg_mean"] is not None
                        else None
                    )
                    rec["delta_auc_main_vs_nuisance_mean_vs_unwhitened"] = (
                        rec["auc_main_vs_nuisance_mean"]
                        - _finite_mean(x.get("auc_main_vs_nuisance") for x in ref_items)
                        if rec["auc_main_vs_nuisance_mean"] is not None
                        else None
                    )
                    rec["delta_fpr_nuisance_match@0p5_mean_vs_unwhitened"] = (
                        rec["fpr_nuisance_match@0p5_mean"]
                        - _finite_mean(x.get("fpr_nuisance_match@0p5") for x in ref_items)
                        if rec["fpr_nuisance_match@0p5_mean"] is not None
                        else None
                    )
            else:
                rec.update(
                    {
                        "tpr@1e-4_mean": _finite_mean(x.get("tpr@1e-4") for x in cfg_items),
                        "tpr@3e-4_mean": _finite_mean(x.get("tpr@3e-4") for x in cfg_items),
                        "tpr@1e-3_mean": _finite_mean(x.get("tpr@1e-3") for x in cfg_items),
                    }
                )
                if ref is not None and cfg_name != "unwhitened_ref":
                    ref_items = [x for x in items if str(x["config"]) == "unwhitened_ref"]
                    rec["delta_tpr@1e-3_mean_vs_unwhitened"] = (
                        rec["tpr@1e-3_mean"] - _finite_mean(x.get("tpr@1e-3") for x in ref_items)
                        if rec["tpr@1e-3_mean"] is not None
                        else None
                    )
            setting_out.append(rec)
        out.setdefault(family, {})[setting] = setting_out
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Focused covariance-training sweep for the whitening path on the clinically anchored "
            "SIMUS benchmark and the Gammex across-view phantom."
        )
    )
    ap.add_argument("--python-exe", type=str, default=sys.executable)
    ap.add_argument("--stap-device", type=str, default="cuda")
    ap.add_argument("--out-root", type=Path, default=Path("runs/analysis/stap_covariance_regime_sweep"))
    ap.add_argument("--out-csv", type=Path, default=Path("reports/stap_covariance_regime_sweep.csv"))
    ap.add_argument("--out-json", type=Path, default=Path("reports/stap_covariance_regime_sweep.json"))
    ap.add_argument("--simus-out-root", type=Path, default=Path("runs/sim_eval/stap_covariance_regime_sweep"))
    ap.add_argument("--simus-profile", type=str, default="Brain-SIMUS-Clin-MotionRobust-v0")
    ap.add_argument("--simus-run", type=Path, action="append", default=None)
    ap.add_argument(
        "--twinkling-seq-dir",
        type=Path,
        default=Path(
            "data/twinkling_artifact/Flow in Gammex phantom/Flow in Gammex phantom (across - linear probe)"
        ),
    )
    ap.add_argument("--twinkling-out-root", type=Path, default=Path("runs/real/stap_covariance_regime_sweep"))
    ap.add_argument("--twinkling-prf-hz", type=float, default=2500.0)
    ap.add_argument("--twinkling-tile-stride", type=int, default=4)
    ap.add_argument(
        "--families",
        type=str,
        default="simus,twinkling",
        help="Comma-separated subset of simus,twinkling.",
    )
    ap.add_argument(
        "--config",
        type=str,
        action="append",
        default=None,
        help=(
            "Optional custom config: "
            "name,detector_variant,whiten_gamma,cov_estimator,diag_load,cov_train_trim_q,huber_c,mvdr_auto_kappa,constraint_ridge"
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    configs = [_parse_config(spec) for spec in args.config] if args.config else _default_configs()
    families = {x.strip().lower() for x in str(args.families).split(",") if x.strip()}

    rows: list[dict[str, Any]] = []
    if "simus" in families:
        for cfg in configs:
            for run_dir in [Path(p) for p in (args.simus_run or _simus_runs_default())]:
                rows.append(_run_simus_case(args=args, run_dir=run_dir, cfg=cfg))
    if "twinkling" in families:
        for cfg in configs:
            rows.append(_run_twinkling_case(args=args, cfg=cfg))

    summary = {
        "configs": [asdict(cfg) for cfg in configs],
        "summary": _summarize(rows),
    }
    _write_csv(Path(args.out_csv), rows)
    _write_json(Path(args.out_json), summary)


if __name__ == "__main__":
    main()
