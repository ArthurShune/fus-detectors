#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from scripts.fair_filter_comparison import (
    DEFAULT_SRC_TEMPLATE_OPEN,
    DEFAULT_SRC_TEMPLATE_SKULL,
    _bundle_map_by_window,
    _injection_cli_args_from_source,
    _resolve_source_dir,
    _run_replay_generation,
)


@dataclass(frozen=True)
class Config:
    name: str
    detector_variant: str
    whiten_gamma: float
    diag_load: float
    cov_estimator: str
    huber_c: float
    stap_cov_trim_q: float
    mvdr_auto_kappa: float
    constraint_ridge: float


DEFAULT_CONFIGS: tuple[Config, ...] = (
    Config(
        name="unwhitened_ref",
        detector_variant="unwhitened_ratio",
        whiten_gamma=0.0,
        diag_load=0.07,
        cov_estimator="tyler_pca",
        huber_c=5.0,
        stap_cov_trim_q=0.0,
        mvdr_auto_kappa=120.0,
        constraint_ridge=0.18,
    ),
    Config(
        name="whitened_default",
        detector_variant="msd_ratio",
        whiten_gamma=1.0,
        diag_load=0.07,
        cov_estimator="tyler_pca",
        huber_c=5.0,
        stap_cov_trim_q=0.0,
        mvdr_auto_kappa=120.0,
        constraint_ridge=0.18,
    ),
    Config(
        name="huber_trim8",
        detector_variant="msd_ratio",
        whiten_gamma=1.0,
        diag_load=0.10,
        cov_estimator="huber",
        huber_c=5.0,
        stap_cov_trim_q=0.08,
        mvdr_auto_kappa=200.0,
        constraint_ridge=0.25,
    ),
)


def _existing_root_for_config(regime: str, cfg_name: str) -> Path | None:
    regime = str(regime).strip().lower()
    mapping = {
        ("open", "unwhitened_ref"): Path("runs/pilot/stap_whitening_regime_sweep/open_seed1_gamma0p00"),
        ("open", "whitened_default"): Path("runs/pilot/stap_whitening_regime_sweep/open_seed1_gamma1p00"),
        ("skullor", "unwhitened_ref"): Path("runs/pilot/stap_whitening_regime_sweep/skullor_seed2_gamma0p00"),
        ("skullor", "whitened_default"): Path("runs/pilot/stap_whitening_regime_sweep/skullor_seed2_gamma1p00"),
    }
    root = mapping.get((regime, cfg_name))
    if root is None or not root.exists():
        return None
    return root


def _finite(vals: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(vals), dtype=np.float64)
    return arr[np.isfinite(arr)]


def _summarize(vals: Iterable[float]) -> dict[str, float]:
    arr = _finite(vals)
    if arr.size == 0:
        return {"median": float("nan"), "q25": float("nan"), "q75": float("nan")}
    return {
        "median": float(np.quantile(arr, 0.5)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
    }


def _tau_for_fpr(neg: np.ndarray, alpha: float) -> float:
    n = int(neg.size)
    if n <= 0:
        return float("inf")
    neg_sorted = np.sort(neg)
    q = 1.0 - float(alpha)
    k = int(np.ceil(q * n)) - 1
    k = max(0, min(n - 1, k))
    return float(neg_sorted[k])


def _eval_at_tau(pos: np.ndarray, neg: np.ndarray, tau: float) -> tuple[float, float]:
    if not np.isfinite(tau):
        return 0.0, 0.0
    tpr = float(np.mean(pos >= tau)) if pos.size else 0.0
    fpr = float(np.mean(neg >= tau)) if neg.size else 0.0
    return tpr, fpr


def _load_scores(bundle_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    score = np.load(bundle_dir / "score_stap_preka.npy").astype(np.float64, copy=False)
    mf = np.load(bundle_dir / "mask_flow.npy").astype(bool, copy=False)
    mb = np.load(bundle_dir / "mask_bg.npy").astype(bool, copy=False)
    return score[mf].ravel(), score[mb].ravel()


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


def _parse_config(spec: str) -> Config:
    parts = [x.strip() for x in str(spec).split(",")]
    if len(parts) != 9:
        raise ValueError(
            "--config expects "
            "name,detector_variant,whiten_gamma,diag_load,cov_estimator,huber_c,stap_cov_trim_q,mvdr_auto_kappa,constraint_ridge"
        )
    return Config(
        name=parts[0],
        detector_variant=parts[1],
        whiten_gamma=float(parts[2]),
        diag_load=float(parts[3]),
        cov_estimator=parts[4],
        huber_c=float(parts[5]),
        stap_cov_trim_q=float(parts[6]),
        mvdr_auto_kappa=float(parts[7]),
        constraint_ridge=float(parts[8]),
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Validate candidate whitening policies on labeled-brain stress tests and "
            "cross-window calibration transfer."
        )
    )
    ap.add_argument("--python-exe", type=str, default=sys.executable)
    ap.add_argument("--stap-device", type=str, default="cuda")
    ap.add_argument("--generated-root", type=Path, default=Path("runs/pilot/brain_whitening_policy_validation"))
    ap.add_argument("--out-csv", type=Path, default=Path("reports/brain_whitening_policy_validation.csv"))
    ap.add_argument("--out-json", type=Path, default=Path("reports/brain_whitening_policy_validation.json"))
    ap.add_argument("--window-length", type=int, default=64)
    ap.add_argument("--window-offsets", type=str, default="0,64,128,192,256")
    ap.add_argument("--alphas", type=str, default="1e-4,3e-4,1e-3")
    ap.add_argument(
        "--regimes",
        type=str,
        default="open,skullor",
        help="Comma-separated subset of open,skullor.",
    )
    ap.add_argument("--autogen-missing", action="store_true", default=True)
    ap.add_argument("--regen-runs", action="store_true", default=False)
    ap.add_argument(
        "--config",
        type=str,
        action="append",
        default=None,
        help=(
            "Optional custom config: "
            "name,detector_variant,whiten_gamma,diag_load,cov_estimator,huber_c,stap_cov_trim_q,mvdr_auto_kappa,constraint_ridge"
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    configs = [_parse_config(spec) for spec in args.config] if args.config else list(DEFAULT_CONFIGS)
    offsets = [int(x) for x in str(args.window_offsets).split(",") if x.strip()]
    alphas = [float(x) for x in str(args.alphas).split(",") if x.strip()]

    regime_specs = {
        "open": {
            "seed": 1,
            "profile": "Brain-OpenSkull",
            "label": "open-skull",
            "src_template": DEFAULT_SRC_TEMPLATE_OPEN,
        },
        "skullor": {
            "seed": 2,
            "profile": "Brain-SkullOR",
            "label": "structured-clutter",
            "src_template": DEFAULT_SRC_TEMPLATE_SKULL,
        },
    }

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "configs": [asdict(cfg) for cfg in configs],
        "regimes": {},
        "alphas": alphas,
        "window_offsets": offsets,
    }

    for regime in [x.strip().lower() for x in str(args.regimes).split(",") if x.strip()]:
        spec = regime_specs[regime]
        seed = int(spec["seed"])
        profile = str(spec["profile"])
        source_dir = _resolve_source_dir(regime, seed, str(spec["src_template"]))
        inject_args = _injection_cli_args_from_source(
            source_dir,
            clutter_mode_override="frozen",
            clutter_rank_override=3,
        )
        summary["regimes"][regime] = {"label": spec["label"], "configs": {}}

        for cfg in configs:
            out_dir = _existing_root_for_config(regime, cfg.name) or (
                Path(args.generated_root) / f"{regime}_seed{seed}_{cfg.name}"
            )
            if args.regen_runs and out_dir.exists():
                import shutil

                shutil.rmtree(out_dir)
            bundles = _bundle_map_by_window(out_dir, int(args.window_length)) if out_dir.exists() else {}
            missing = [off for off in offsets if off not in bundles]
            if missing and args.autogen_missing:
                _run_replay_generation(
                    python_exe=str(args.python_exe),
                    source_dir=source_dir,
                    out_dir=out_dir,
                    profile=profile,
                    baseline="mc_svd",
                    stap_disable=False,
                    inject_args=inject_args,
                    conditional=False,
                    window_length=int(args.window_length),
                    offsets=missing,
                    stap_device=str(args.stap_device),
                    stap_detector_variant=cfg.detector_variant,
                    stap_whiten_gamma=float(cfg.whiten_gamma),
                    stap_cov_trim_q=float(cfg.stap_cov_trim_q),
                    diag_load=float(cfg.diag_load),
                    cov_estimator=str(cfg.cov_estimator),
                    huber_c=float(cfg.huber_c),
                    mvdr_auto_kappa=float(cfg.mvdr_auto_kappa),
                    constraint_ridge=float(cfg.constraint_ridge),
                    synth_amp_jitter=0.0,
                    synth_phase_jitter=0.0,
                    synth_noise_level=0.0,
                    synth_shift_max_px=0,
                    reg_enable=True,
                    mcsvd_reg_enable=True,
                    mcsvd_energy_frac=0.90,
                    mcsvd_baseline_support="full",
                    rpca_lambda=None,
                    rpca_max_iters=250,
                    hosvd_spatial_downsample=2,
                    hosvd_energy_fracs="0.99,0.99,0.99",
                )
                bundles = _bundle_map_by_window(out_dir, int(args.window_length))

            ordered_bundles = [bundles[off] for off in offsets]
            posneg_by_win = [_load_scores(d) for d in ordered_bundles]

            cfg_summary: dict[str, Any] = {
                "config": asdict(cfg),
                "within": {},
                "cross": {},
                "fixed_cal": {},
            }

            for alpha in alphas:
                within_tpr: list[float] = []
                within_fpr: list[float] = []
                for pos, neg in posneg_by_win:
                    tau = _tau_for_fpr(neg, alpha)
                    tpr, fpr = _eval_at_tau(pos, neg, tau)
                    within_tpr.append(tpr)
                    within_fpr.append(fpr)

                cross_tpr: list[float] = []
                cross_fpr: list[float] = []
                for i, (pos_i, neg_i) in enumerate(posneg_by_win):
                    tau = _tau_for_fpr(neg_i, alpha)
                    for j, (pos_j, neg_j) in enumerate(posneg_by_win):
                        if i == j:
                            continue
                        tpr, fpr = _eval_at_tau(pos_j, neg_j, tau)
                        cross_tpr.append(tpr)
                        cross_fpr.append(fpr)

                fixed_tpr: list[float] = []
                fixed_fpr: list[float] = []
                for i, (pos_i, neg_i) in enumerate(posneg_by_win):
                    neg_pool = np.concatenate([posneg_by_win[j][1] for j in range(len(posneg_by_win)) if j != i])
                    tau = _tau_for_fpr(neg_pool, alpha)
                    tpr, fpr = _eval_at_tau(pos_i, neg_i, tau)
                    fixed_tpr.append(tpr)
                    fixed_fpr.append(fpr)

                alpha_key = str(alpha)
                cfg_summary["within"][alpha_key] = {"tpr": _summarize(within_tpr), "fpr": _summarize(within_fpr)}
                cfg_summary["cross"][alpha_key] = {"tpr": _summarize(cross_tpr), "fpr": _summarize(cross_fpr)}
                cfg_summary["fixed_cal"][alpha_key] = {"tpr": _summarize(fixed_tpr), "fpr": _summarize(fixed_fpr)}

                rows.append(
                    {
                        "regime": regime,
                        "regime_label": spec["label"],
                        "config": cfg.name,
                        "split": "within",
                        "alpha": alpha,
                        "tpr_median": cfg_summary["within"][alpha_key]["tpr"]["median"],
                        "tpr_q25": cfg_summary["within"][alpha_key]["tpr"]["q25"],
                        "tpr_q75": cfg_summary["within"][alpha_key]["tpr"]["q75"],
                        "fpr_median": cfg_summary["within"][alpha_key]["fpr"]["median"],
                        "fpr_q25": cfg_summary["within"][alpha_key]["fpr"]["q25"],
                        "fpr_q75": cfg_summary["within"][alpha_key]["fpr"]["q75"],
                    }
                )
                rows.append(
                    {
                        "regime": regime,
                        "regime_label": spec["label"],
                        "config": cfg.name,
                        "split": "cross",
                        "alpha": alpha,
                        "tpr_median": cfg_summary["cross"][alpha_key]["tpr"]["median"],
                        "tpr_q25": cfg_summary["cross"][alpha_key]["tpr"]["q25"],
                        "tpr_q75": cfg_summary["cross"][alpha_key]["tpr"]["q75"],
                        "fpr_median": cfg_summary["cross"][alpha_key]["fpr"]["median"],
                        "fpr_q25": cfg_summary["cross"][alpha_key]["fpr"]["q25"],
                        "fpr_q75": cfg_summary["cross"][alpha_key]["fpr"]["q75"],
                    }
                )
                rows.append(
                    {
                        "regime": regime,
                        "regime_label": spec["label"],
                        "config": cfg.name,
                        "split": "fixed_cal",
                        "alpha": alpha,
                        "tpr_median": cfg_summary["fixed_cal"][alpha_key]["tpr"]["median"],
                        "tpr_q25": cfg_summary["fixed_cal"][alpha_key]["tpr"]["q25"],
                        "tpr_q75": cfg_summary["fixed_cal"][alpha_key]["tpr"]["q75"],
                        "fpr_median": cfg_summary["fixed_cal"][alpha_key]["fpr"]["median"],
                        "fpr_q25": cfg_summary["fixed_cal"][alpha_key]["fpr"]["q25"],
                        "fpr_q75": cfg_summary["fixed_cal"][alpha_key]["fpr"]["q75"],
                    }
                )

            summary["regimes"][regime]["configs"][cfg.name] = cfg_summary

    _write_csv(Path(args.out_csv), rows)
    _write_json(Path(args.out_json), summary)


if __name__ == "__main__":
    main()
