#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from scripts.fair_filter_comparison import (
    DEFAULT_SRC_TEMPLATE_OPEN,
    DEFAULT_SRC_TEMPLATE_SKULL,
    _bundle_map_by_window,
    _evaluate_method_row,
    _injection_cli_args_from_source,
    _resolve_injection_meta_dir,
    _resolve_source_dir,
    _run_replay_generation,
)
from scripts.simus_detector_family_ablation_table import _dataset_name, _setting_label
from scripts.simus_eval_structural import evaluate_structural_metrics
from sim.simus.bundle import derive_bundle_from_run

REPO = Path(__file__).resolve().parents[1]


def _gamma_slug(gamma: float) -> str:
    return f"{float(gamma):.2f}".replace("-", "m").replace(".", "p")


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


def _finite_median(vals: Iterable[float | None]) -> float | None:
    arr = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.median(arr))


def _simus_runs_default() -> list[Path]:
    return [
        Path("runs/sim/simus_v2_acceptance_clin_mobile_pf_v2_phase3_seed127"),
        Path("runs/sim/simus_v2_acceptance_clin_mobile_pf_v2_phase3_seed128"),
        Path("runs/sim/simus_v2_acceptance_clin_intraop_parenchyma_pf_v3_phase3_seed127"),
        Path("runs/sim/simus_v2_acceptance_clin_intraop_parenchyma_pf_v3_phase3_seed128"),
    ]


def _run_simus_sweep(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    runs = [Path(p) for p in (args.simus_run or _simus_runs_default())]
    for gamma in args.gammas:
        for run_dir in runs:
            ds = run_dir / "dataset"
            mask_h1_pf_main = np.load(ds / "mask_h1_pf_main.npy")
            mask_h0_bg = np.load(ds / "mask_h0_bg.npy")
            mask_h0_nuisance_pa = np.load(ds / "mask_h0_nuisance_pa.npy")
            mask_h1_alias_qc = np.load(ds / "mask_h1_alias_qc.npy")
            setting = _setting_label(run_dir)
            dataset_name = _dataset_name(run_dir, f"gamma{_gamma_slug(gamma)}")
            bundle_dir = derive_bundle_from_run(
                run_dir=run_dir,
                out_root=Path(args.simus_out_root),
                dataset_name=dataset_name,
                stap_profile=str(args.simus_profile),
                baseline_type="mc_svd",
                run_stap=True,
                stap_device=str(args.stap_device),
                bundle_overrides={
                    "stap_detector_variant": "msd_ratio",
                    "stap_whiten_gamma": float(gamma),
                },
                meta_extra={
                    "stap_whitening_regime_sweep": True,
                    "regime_family": "simus",
                    "whiten_gamma": float(gamma),
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
                "gamma": float(gamma),
                "run": run_dir.name,
                "setting": setting,
                "bundle_dir": str(bundle_dir),
                "auc_main_vs_bg": metrics.get("auc_main_vs_bg"),
                "auc_main_vs_nuisance": metrics.get("auc_main_vs_nuisance"),
                "fpr_nuisance_match@0p5": metrics.get("fpr_nuisance_match@0p5"),
                "tpr_main@1e-03": metrics.get("tpr_main@1e-03"),
            }
            rows.append(row)
            grouped[(float(gamma), setting)].append(row)

    summary: list[dict[str, Any]] = []
    for (gamma, setting), items in sorted(grouped.items()):
        summary.append(
            {
                "family": "simus",
                "gamma": float(gamma),
                "setting": setting,
                "count": len(items),
                "auc_main_vs_bg_mean": _finite_mean(x.get("auc_main_vs_bg") for x in items),
                "auc_main_vs_nuisance_mean": _finite_mean(
                    x.get("auc_main_vs_nuisance") for x in items
                ),
                "fpr_nuisance_match@0p5_mean": _finite_mean(
                    x.get("fpr_nuisance_match@0p5") for x in items
                ),
                "tpr_main@1e-03_mean": _finite_mean(x.get("tpr_main@1e-03") for x in items),
            }
        )
    return rows, summary


def _run_brain_sweep(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    regime_specs = [
        ("open", 1, DEFAULT_SRC_TEMPLATE_OPEN, "Brain-OpenSkull", "open-skull"),
        ("skullor", 2, DEFAULT_SRC_TEMPLATE_SKULL, "Brain-SkullOR", "structured-clutter"),
    ]
    offsets = [0, 64, 128, 192, 256]
    window_length = 64

    for gamma in args.gammas:
        for regime, seed, template, profile, label in regime_specs:
            source_dir = _resolve_source_dir(regime, seed, template)
            inject_meta_dir = _resolve_injection_meta_dir(regime, [seed], template) or source_dir
            inject_args = _injection_cli_args_from_source(
                inject_meta_dir,
                clutter_mode_override="frozen",
                clutter_rank_override=3,
            )
            out_dir = Path(args.brain_out_root) / f"{regime}_seed{seed}_gamma{_gamma_slug(gamma)}"
            bundles = _bundle_map_by_window(out_dir, window_length) if out_dir.exists() else {}
            missing = [off for off in offsets if off not in bundles]
            if missing:
                _run_replay_generation(
                    python_exe=args.python_exe,
                    source_dir=source_dir,
                    out_dir=out_dir,
                    profile=profile,
                    baseline="mc_svd",
                    stap_disable=False,
                    inject_args=inject_args,
                    conditional=False,
                    window_length=window_length,
                    offsets=missing,
                    stap_device=args.stap_device,
                    stap_detector_variant="msd_ratio",
                    stap_whiten_gamma=float(gamma),
                    stap_cov_trim_q=0.0,
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
                bundles = _bundle_map_by_window(out_dir, window_length)
            for offset in offsets:
                bundle = bundles.get(offset)
                if bundle is None:
                    raise FileNotFoundError(
                        f"Missing bundle for regime={regime} seed={seed} gamma={gamma} offset={offset}"
                    )
                row = _evaluate_method_row(
                    scenario=f"{regime}_seed{seed}",
                    regime=regime,
                    seed=seed,
                    window_offset=offset,
                    window_length=window_length,
                    method=f"gamma={gamma:.2f}",
                    role="stap",
                    bundle_dir=bundle,
                    score_kind="stap",
                    fprs=[1e-4, 3e-4, 1e-3],
                    legacy_default="lower",
                    eval_score="vnext",
                )
                row["family"] = "brain"
                row["gamma"] = float(gamma)
                row["regime_label"] = label
                rows.append(row)
                grouped[(float(gamma), label)].append(row)

    summary: list[dict[str, Any]] = []
    for (gamma, label), items in sorted(grouped.items()):
        summary.append(
            {
                "family": "brain",
                "gamma": float(gamma),
                "regime_label": label,
                "count": len(items),
                "tpr@1e-4_median": _finite_median(x.get("tpr@0.0001") for x in items),
                "tpr@3e-4_median": _finite_median(x.get("tpr@0.0003") for x in items),
                "tpr@1e-3_median": _finite_median(x.get("tpr@0.001") for x in items),
            }
        )
    return rows, summary


def _twinkling_existing_root(view: str, gamma: float) -> Path | None:
    if abs(float(gamma)) <= 1e-8:
        if view == "along":
            return REPO / "runs/real/twinkling_gammex_alonglinear17_prf2500_str6_unwhitened_ratio"
        return REPO / "runs/real/twinkling_gammex_across17_prf2500_str4_unwhitened_ratio"
    if abs(float(gamma) - 1.0) <= 1e-8:
        if view == "along":
            return REPO / "runs/real/twinkling_gammex_alonglinear17_prf2500_str6_msd_ratio_fast"
        return REPO / "runs/real/twinkling_gammex_across17_prf2500_str4_msd_ratio_fast"
    return None


def _twinkling_existing_summary(view: str, gamma: float) -> Path | None:
    if abs(float(gamma)) <= 1e-8:
        if view == "along":
            return REPO / "reports/twinkling_gammex_alonglinear17_prf2500_str6_unwhitened_ratio_structural_summary.json"
        return REPO / "reports/twinkling_gammex_across17_prf2500_str4_unwhitened_ratio_structural_summary.json"
    if abs(float(gamma) - 1.0) <= 1e-8:
        if view == "along":
            return REPO / "reports/twinkling_gammex_alonglinear17_prf2500_str6_msd_ratio_fast_structural_summary.json"
        return REPO / "reports/twinkling_gammex_across17_prf2500_str4_msd_ratio_fast_structural_summary.json"
    return None


def _twinkling_specs() -> list[dict[str, Any]]:
    return [
        {
            "view": "along",
            "seq_dir": REPO
            / "data/twinkling_artifact/Flow in Gammex phantom/Flow in Gammex phantom (along - linear probe)",
            "prf_hz": 2500.0,
            "tile_stride": 6,
            "root_stem": "twinkling_gammex_alonglinear17_prf2500_str6",
        },
        {
            "view": "across",
            "seq_dir": REPO
            / "data/twinkling_artifact/Flow in Gammex phantom/Flow in Gammex phantom (across - linear probe)",
            "prf_hz": 2500.0,
            "tile_stride": 4,
            "root_stem": "twinkling_gammex_across17_prf2500_str4",
        },
    ]


def _resolve_twinkling_par_dat(seq_dir: Path) -> tuple[Path, Path]:
    par_candidates = sorted(seq_dir.glob("RawBCFCine*.par"))
    dat_candidates = sorted(seq_dir.glob("RawBCFCine*.dat"))
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


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
    return env


def _run_twinkling_bundle(
    *,
    args: argparse.Namespace,
    seq_dir: Path,
    out_root: Path,
    prf_hz: float,
    tile_stride: int,
    gamma: float,
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
        str(float(prf_hz)),
        "--tile-stride",
        str(int(tile_stride)),
        "--stap-device",
        str(args.stap_device),
        "--stap-detector-variant",
        "msd_ratio",
        "--stap-whiten-gamma",
        str(float(gamma)),
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


def _run_twinkling_sweep(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for gamma in args.gammas:
        for spec in _twinkling_specs():
            view = str(spec["view"])
            out_csv = Path(args.out_root) / "twinkling" / f"{spec['root_stem']}_gamma{_gamma_slug(gamma)}.csv"
            generated_summary_json = (
                Path(args.out_root)
                / "twinkling"
                / f"{spec['root_stem']}_gamma{_gamma_slug(gamma)}_summary.json"
            )
            summary_json = _twinkling_existing_summary(view, gamma)
            root = _twinkling_existing_root(view, gamma)
            if summary_json is None or root is None or not summary_json.exists() or not root.exists():
                root = Path(args.twinkling_out_root) / f"{spec['root_stem']}_gamma{_gamma_slug(gamma)}"
                summary_json = generated_summary_json
                if not summary_json.exists() or not root.exists():
                    root = _run_twinkling_bundle(
                        args=args,
                        seq_dir=Path(spec["seq_dir"]),
                        out_root=root,
                        prf_hz=float(spec["prf_hz"]),
                        tile_stride=int(spec["tile_stride"]),
                        gamma=float(gamma),
                    )
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
                "gamma": float(gamma),
                "view": view,
                "root": str(root),
                "summary_json": str(summary_json),
                "tpr@1e-4": (point_by_fpr.get(1e-4) or {}).get("tpr"),
                "tpr@3e-4": (point_by_fpr.get(3e-4) or {}).get("tpr"),
                "tpr@1e-3": (point_by_fpr.get(1e-3) or {}).get("tpr"),
            }
            rows.append(row)
            summary.append(dict(row))
    return rows, summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Sweep whitening strength across the clinically anchored SIMUS, brain stress-test, and Gammex regimes."
    )
    ap.add_argument("--python-exe", type=str, default=sys.executable)
    ap.add_argument("--stap-device", type=str, default="cuda")
    ap.add_argument("--out-root", type=Path, default=Path("runs/analysis/stap_whitening_regime_sweep"))
    ap.add_argument("--out-csv", type=Path, default=Path("reports/stap_whitening_regime_sweep.csv"))
    ap.add_argument("--out-json", type=Path, default=Path("reports/stap_whitening_regime_sweep.json"))
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--simus-out-root", type=Path, default=Path("runs/sim_eval/stap_whitening_regime_sweep"))
    ap.add_argument("--simus-profile", type=str, default="Brain-SIMUS-Clin-MotionRobust-v0")
    ap.add_argument("--simus-run", type=Path, action="append", default=None)
    ap.add_argument("--brain-out-root", type=Path, default=Path("runs/pilot/stap_whitening_regime_sweep"))
    ap.add_argument("--twinkling-out-root", type=Path, default=Path("runs/real/stap_whitening_regime_sweep"))
    ap.add_argument(
        "--families",
        type=str,
        default="simus,brain,twinkling",
        help="Comma-separated subset of simus,brain,twinkling.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    families = {x.strip().lower() for x in str(args.families).split(",") if x.strip()}
    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"gammas": [float(x) for x in args.gammas]}

    if "simus" in families:
        simus_rows, simus_summary = _run_simus_sweep(args)
        all_rows.extend(simus_rows)
        summary["simus"] = simus_summary
    if "brain" in families:
        brain_rows, brain_summary = _run_brain_sweep(args)
        all_rows.extend(brain_rows)
        summary["brain"] = brain_summary
    if "twinkling" in families:
        tw_rows, tw_summary = _run_twinkling_sweep(args)
        all_rows.extend(tw_rows)
        summary["twinkling"] = tw_summary

    _write_csv(Path(args.out_csv), all_rows)
    _write_json(Path(args.out_json), summary)


if __name__ == "__main__":
    main()
