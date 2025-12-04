"""Stress test harness over motion/prf/angle grids."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

from eval.acceptance_cli import _simulate_confirm2_scores, _simulate_scores_and_pd
from eval.metrics import partial_auc, pd_snr_db, roc_curve, tpr_at_fpr_target
from pipeline.confirm2.policy import rho_inflate_policy
from pipeline.confirm2.validator import calibrate_confirm2, evaluate_confirm2
from pipeline.sim import SimConfig
from pipeline.utils.telemetry import sample_gpu_stats, system_telemetry

try:
    import torch
except Exception:
    torch = None


def _summary_stats(arr: np.ndarray) -> Dict[str, float]:
    flat = np.asarray(arr, dtype=np.float64).ravel()
    if flat.size == 0:
        return {"count": 0}
    q5, q50, q95 = np.percentile(flat, [5, 50, 95])
    return {
        "count": int(flat.size),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "p05": float(q5),
        "p50": float(q50),
        "p95": float(q95),
    }


def _mask_stats(mask: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(mask).astype(bool)
    return {
        "count": int(arr.size),
        "true_count": int(arr.sum()),
        "false_count": int(arr.size - arr.sum()),
        "true_fraction": float(arr.mean()),
    }


def _parse_grid(entries: Sequence[str]) -> Dict[str, List[Any]]:
    grid: Dict[str, List[Any]] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid grid specification '{entry}'")
        key, val = entry.split("=", 1)
        values: List[Any] = []
        for token in val.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                num = float(token)
                values.append(num)
            except ValueError:
                values.append(token)
        if not values:
            raise ValueError(f"No values provided for grid '{key}'")
        grid[key] = values
    return grid


def _parse_tiles(spec: str) -> List[str]:
    return [token.strip() for token in spec.split(",") if token.strip()]


def _parse_float_list(spec: Optional[str]) -> List[float]:
    if spec is None or spec == "":
        return [0.0]
    tokens = []
    for part in spec.replace(";", ",").split(","):
        part = part.strip()
        if part:
            tokens.append(float(part))
    return tokens if tokens else [0.0]


def _parse_tile_shape(tile_spec: str) -> Tuple[Tuple[int, int], int]:
    if ":" not in tile_spec:
        raise ValueError(f"Invalid tile specification '{tile_spec}'. Expected 'HxW:stride'.")
    dims, stride = tile_spec.split(":", 1)
    if "x" not in dims:
        raise ValueError(f"Invalid tile specification '{tile_spec}'. Expected 'HxW:stride'.")
    h_str, w_str = dims.split("x", 1)
    return (int(h_str), int(w_str)), int(stride)


def _device_from_arg(arg: Optional[str]) -> Optional[str]:
    if arg is None:
        return "cuda" if torch is not None and torch.cuda.is_available() else None
    if arg.lower() == "none":
        return None
    if arg.startswith("cuda") and (torch is None or not torch.cuda.is_available()):
        raise RuntimeError("CUDA requested but torch with CUDA support not available")
    return arg


def run_stress(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "stress_results.csv"

    device = _device_from_arg(args.device)
    use_cuda = bool(device and torch is not None and device.startswith("cuda"))

    grid = _parse_grid(args.grid)
    tiles = _parse_tiles(args.tiles)
    parsed_tiles = []
    for tile_str in tiles:
        dims, stride = _parse_tile_shape(tile_str)
        parsed_tiles.append((tile_str, dims, stride))

    configs: List[str] = []
    for cfg_path in args.configs:
        if yaml is None:
            raise RuntimeError("PyYAML is required for --configs")
        data = yaml.safe_load(Path(cfg_path).read_text())
        configs.append(Path(cfg_path).name)
        (outdir / f"config_{Path(cfg_path).name}").write_text(json.dumps(data, indent=2))

    seeds = [args.seed_offset + i for i in range(args.seeds)]

    disable_progress = args.no_progress or os.environ.get("TQDM_DISABLE", "0") == "1"

    columns = [
        "seed",
        "motion_um",
        "prf",
        "K",
        "tile",
        "pd_snr_baseline_db",
        "pd_snr_stap_db",
        "pd_snr_delta_db",
        "tpr_at_fpr_baseline",
        "tpr_at_fpr_stap",
        "tpr_at_fpr_delta",
        "pauc_baseline",
        "pauc_stap",
        "pauc_delta",
        "timing_sec",
        "device",
        "motion_freq_hz",
        "heterogeneity",
        "prf_hz",
        "prf_jitter_pct",
        "sensor_dropout",
        "scatter_density",
        "skull_db",
        "tile_h",
        "tile_w",
        "tile_stride",
        "motion_comp",
        "steer_mode",
        "steer_grid",
        "steer_fuse",
        "angle_grouping",
        "rho_inflate",
        "rho_inflate_policy",
        "lt_auto",
        "fd_auto",
        "tbd",
        "robust_cov",
        "huber_c",
    ]

    confirm2_cols: List[str] = []
    if args.confirm2:
        tag = ("%0.3e" % args.alpha2).replace("-", "m").replace("+", "p").replace(".", "d")
        confirm2_cols.extend(
            [
                f"confirm2_emp_{tag}",
                f"confirm2_pred_{tag}",
                f"confirm2_alpha1_{tag}",
                f"confirm2_rho_eff_{tag}",
            ]
        )
    columns.extend(confirm2_cols)

    confirm2_cache: Dict[Tuple[int, float], Tuple[Dict[str, float], Dict[str, Any]]] = {}
    rows: List[Dict[str, float]] = []
    tele = system_telemetry(include_nvidia_smi=False)

    if grid:
        grid_keys = list(grid.keys())
        combos_raw = list(product(*[grid[k] for k in grid_keys]))
        combos = [dict(zip(grid_keys, combo_vals, strict=False)) for combo_vals in combos_raw]
    else:
        combos = [dict()]

    total_iters = len(combos) * max(len(tiles), 1) * max(len(seeds), 1)
    progress = tqdm(total=total_iters, desc="Stress runs", unit="case", disable=disable_progress)

    def _confirm2_for_seed(
        seed: int, rho_inflate_case: float
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        key = (seed, float(rho_inflate_case))
        cached = confirm2_cache.get(key)
        if cached is not None:
            return cached
        confirm2_start = time.perf_counter()
        pairs = args.confirm2_pairs
        s1_null, s2_null = _simulate_confirm2_scores(pairs, rho=args.confirm2_rho, seed=seed)
        half = pairs // 2
        calib = calibrate_confirm2(
            s1_null[:half],
            s2_null[:half],
            alpha2_target=args.alpha2,
            seed=seed,
            rho_inflate=rho_inflate_case,
            device=device,
        )
        ev = evaluate_confirm2(
            calib,
            s1_null[half:],
            s2_null[half:],
            alpha=args.confirm2_ci_alpha,
            device=device,
        )
        tag = ("%0.3e" % args.alpha2).replace("-", "m").replace("+", "p").replace(".", "d")
        metrics = {
            f"confirm2_emp_{tag}": ev.empirical_pair_pfa,
            f"confirm2_pred_{tag}": ev.predicted_pair_pfa,
            f"confirm2_alpha1_{tag}": calib.alpha1,
            f"confirm2_rho_eff_{tag}": calib.rho_eff,
        }
        details = {
            "alpha2_target": args.alpha2,
            "rho_hat": calib.rho_hat,
            "rho_ci": [calib.rho_lo, calib.rho_hi],
            "rho_eff": calib.rho_eff,
            "rho_inflate": float(rho_inflate_case),
            "copula_mode": calib.copula_mode,
            "lambda_u_emp": calib.lambda_u_emp,
            "lambda_u_gauss": calib.lambda_u_gauss,
            "df_t": calib.df_t,
            "pot": {
                "look1": calib.pm1.as_dict(),
                "look2": calib.pm2.as_dict(),
            },
            "thresholds": {
                "tau": [calib.tau1, calib.tau2],
                "gamma": [calib.gamma1, calib.gamma2],
                "k": [calib.k1, calib.k2],
                "n_cal": [calib.n_cal1, calib.n_cal2],
            },
            "pairs_calibration": half,
            "pairs_test": ev.n_null_pairs,
            "k_joint": ev.k_joint,
            "empirical_pair_pfa": ev.empirical_pair_pfa,
            "predicted_pair_pfa": ev.predicted_pair_pfa,
            "pair_ci": [ev.pair_ci_lo, ev.pair_ci_hi],
            "alpha1_per_look": ev.alpha1_per_look,
        }
        details["timing_sec"] = time.perf_counter() - confirm2_start
        confirm2_cache[key] = (metrics, details)
        return confirm2_cache[key]

    case_index = 0

    def _parse_fraction(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            val = float(value)
        else:
            text = str(value).strip()
            if text.endswith("%"):
                val = float(text.rstrip("%")) / 100.0
                return val
            val = float(text)
        return val / 100.0 if val > 1.0 else val

    def _parse_float(value: Any, default: float) -> float:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if text.lower().endswith("hz"):
            text = text[:-2]
        if text.lower().endswith("khz"):
            return float(text[:-3]) * 1000.0
        return float(text)

    steer_grid = _parse_float_list(getattr(args, "steer_grid", None))
    steer_fuse = args.steer_fuse
    motion_comp_flag = bool(args.motion_comp)
    grouping_strategy = args.grouping

    for combo in combos:
        motion = _parse_float(combo.get("motion_um"), 30.0)
        prf_raw = combo.get("prf", combo.get("prf_hz"))
        prf_hz = _parse_float(prf_raw, 3000.0)
        if prf_hz < 100.0:
            prf_hz *= 1000.0
        prf = prf_hz
        k_val = combo.get("K")
        motion_freq = _parse_float(combo.get("motion_freq_hz", combo.get("motion_freq")), 0.5)
        hetero = str(combo.get("heterogeneity", "medium"))
        sensor_dropout = _parse_fraction(combo.get("sensor_dropout"), 0.0)
        scatter_density = _parse_float(combo.get("scatter_density"), 1.0)
        skull_db = _parse_float(
            combo.get("skull_db", combo.get("skull_atten_db", combo.get("skull"))), 0.0
        )
        prf_jitter_pct = _parse_float(combo.get("prf_jitter_pct", combo.get("prf_jitter")), 0.0)
        for tile_str, tile_dims, tile_stride in parsed_tiles:
            for seed in seeds:
                case_start = time.perf_counter()
                phase_times: Dict[str, float] = {}
                if use_cuda:
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                case_seed = int(seed * 1_000_003 + case_index * 97)
                steer_grid_case = list(steer_grid)
                if getattr(args, "fd_auto", False):
                    if motion >= 150.0:
                        steer_grid_case = [
                            -900.0,
                            -750.0,
                            -600.0,
                            -450.0,
                            -300.0,
                            -150.0,
                            0.0,
                            150.0,
                            300.0,
                            450.0,
                            600.0,
                            750.0,
                            900.0,
                        ]
                    elif motion >= 120.0:
                        steer_grid_case = [
                            -750.0,
                            -600.0,
                            -450.0,
                            -300.0,
                            -150.0,
                            0.0,
                            150.0,
                            300.0,
                            450.0,
                            600.0,
                            750.0,
                        ]
                    elif motion >= 80.0:
                        steer_grid_case = [-600.0, -400.0, -200.0, 0.0, 200.0, 400.0, 600.0]

                sim_cfg = SimConfig(
                    motion_amp_um=motion,
                    motion_freq_hz=motion_freq,
                    prf_hz=prf_hz,
                    prf_jitter_pct=prf_jitter_pct,
                    heterogeneity=hetero,
                    skull_attenuation_db=skull_db,
                    scatter_density=scatter_density,
                    sensor_dropout=sensor_dropout,
                    angle_count=int(k_val) if k_val is not None else 9,
                    tile=tile_dims,
                    tile_stride=tile_stride,
                    seed=case_seed,
                    enable_motion_comp=motion_comp_flag,
                    doppler_grid=tuple(steer_grid_case),
                    steer_fuse=steer_fuse,
                    angle_grouping=grouping_strategy,
                    steer_mode=args.steer_mode,
                    tbd_enable=bool(args.tbd),
                )
                phase = time.perf_counter()
                (
                    base_pos,
                    base_neg,
                    stap_pos,
                    stap_neg,
                    pd_base,
                    pd_stap,
                    mask_flow,
                    mask_bg,
                ) = _simulate_scores_and_pd(
                    n_pos=args.npos,
                    n_neg=args.nneg,
                    H=args.height,
                    W=args.width,
                    seed=case_seed,
                    sim_config=sim_cfg,
                )
                phase_times["simulate_sec"] = time.perf_counter() - phase

                phase = time.perf_counter()
                fpr_b, tpr_b, _ = roc_curve(base_pos, base_neg, num_thresh=args.roc_thresholds)
                fpr_s, tpr_s, _ = roc_curve(stap_pos, stap_neg, num_thresh=args.roc_thresholds)
                phase_times["roc_sec"] = time.perf_counter() - phase

                pd_snr_baseline = pd_snr_db(pd_base, mask_flow, mask_bg)
                pd_snr_stap = pd_snr_db(pd_stap, mask_flow, mask_bg)

                confirm2_details: Optional[Dict[str, Any]] = None
                rho_inflate_case = float(args.rho_inflate)
                policy_decision = None

                row = {
                    "seed": seed,
                    "motion_um": motion,
                    "motion_freq_hz": motion_freq,
                    "heterogeneity": hetero,
                    "prf": prf_hz,
                    "prf_hz": prf_hz,
                    "prf_jitter_pct": prf_jitter_pct,
                    "sensor_dropout": sensor_dropout,
                    "scatter_density": scatter_density,
                    "skull_db": skull_db,
                    "K": k_val,
                    "tile": tile_str,
                    "tile_h": tile_dims[0],
                    "tile_w": tile_dims[1],
                    "tile_stride": tile_stride,
                    "motion_comp": int(motion_comp_flag),
                    "steer_mode": args.steer_mode,
                    "steer_grid": ";".join(f"{fd:.6g}" for fd in steer_grid_case),
                    "steer_fuse": steer_fuse,
                    "angle_grouping": grouping_strategy,
                    "robust_cov": getattr(args, "robust_cov", "scm"),
                    "huber_c": float(getattr(args, "huber_c", float("nan"))),
                    "pd_snr_baseline_db": pd_snr_baseline,
                    "pd_snr_stap_db": pd_snr_stap,
                    "pd_snr_delta_db": pd_snr_stap - pd_snr_baseline,
                    "tpr_at_fpr_baseline": tpr_at_fpr_target(fpr_b, tpr_b, args.fpr_target),
                    "tpr_at_fpr_stap": tpr_at_fpr_target(fpr_s, tpr_s, args.fpr_target),
                    "pauc_baseline": partial_auc(fpr_b, tpr_b, fpr_max=args.fpr_target),
                    "pauc_stap": partial_auc(fpr_s, tpr_s, fpr_max=args.fpr_target),
                    "device": device or "cpu",
                }
                row["tpr_at_fpr_delta"] = row["tpr_at_fpr_stap"] - row["tpr_at_fpr_baseline"]
                row["pauc_delta"] = row["pauc_stap"] - row["pauc_baseline"]

                if args.confirm2:
                    metrics, details = _confirm2_for_seed(seed, rho_inflate_case)
                    if args.rho_inflate_policy == "auto":
                        policy_decision = rho_inflate_policy(
                            rho_hat=details["rho_hat"],
                            rho_groups=None,
                            motion_um=motion,
                            dropout=sensor_dropout,
                        )
                        rho_inflate_case = policy_decision.delta
                        if abs(rho_inflate_case - details["rho_inflate"]) > 1e-9:
                            metrics, details = _confirm2_for_seed(seed, rho_inflate_case)
                    row.update(metrics)
                    confirm2_details = details
                    if policy_decision is not None:
                        confirm2_details["rho_policy"] = asdict(policy_decision)
                    phase_times["confirm2_sec"] = details.get("timing_sec", 0.0)

                row["rho_inflate"] = rho_inflate_case
                row["rho_inflate_policy"] = args.rho_inflate_policy
                row["lt_auto"] = int(getattr(args, "lt_auto", False))
                row["fd_auto"] = int(getattr(args, "fd_auto", False))
                row["tbd"] = int(getattr(args, "tbd", False))

                if use_cuda:
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - case_start
                phase_times["total_sec"] = elapsed
                row["timing_sec"] = elapsed
                rows.append(row)

                gpu_mem = torch.cuda.max_memory_allocated() / 1e6 if use_cuda else None
                payload = {
                    "seed": seed,
                    "case_seed": case_seed,
                    "device": device or "cpu",
                    "stress": {
                        "motion_um": motion,
                        "motion_freq_hz": motion_freq,
                        "heterogeneity": hetero,
                        "prf": prf,
                        "prf_hz": prf_hz,
                        "K": k_val,
                        "tile": tile_str,
                        "tile_h": tile_dims[0],
                        "tile_w": tile_dims[1],
                        "tile_stride": tile_stride,
                        "configs": configs,
                        "sensor_dropout": sensor_dropout,
                        "scatter_density": scatter_density,
                        "skull_db": skull_db,
                        "prf_jitter_pct": prf_jitter_pct,
                        "sim_config": asdict(sim_cfg),
                        "motion_comp": motion_comp_flag,
                        "steer_mode": args.steer_mode,
                        "steer_grid": steer_grid_case,
                        "steer_fuse": steer_fuse,
                        "angle_grouping": grouping_strategy,
                        "rho_inflate": float(rho_inflate_case),
                        "rho_inflate_policy": args.rho_inflate_policy,
                        "lt_auto": bool(getattr(args, "lt_auto", False)),
                        "fd_auto": bool(getattr(args, "fd_auto", False)),
                        "tbd": bool(getattr(args, "tbd", False)),
                        "robust_cov": getattr(args, "robust_cov", "scm"),
                        "huber_c": float(getattr(args, "huber_c", float("nan"))),
                    },
                    "compute": {
                        "hz": (1.0 / elapsed) if elapsed else None,
                        "ms_per_case": (elapsed * 1000.0),
                        "gpu_mem_MB": gpu_mem,
                    },
                    "timings": phase_times,
                    "gpu": sample_gpu_stats(include_nvidia_smi=False),
                    "telemetry": tele,
                    "performance": row,
                    "data_summary": {
                        "counts": {
                            "npos": int(len(base_pos)),
                            "nneg": int(len(base_neg)),
                            "pd_shape": (
                                list(pd_base.shape) if isinstance(pd_base, np.ndarray) else None
                            ),
                        },
                        "baseline": {
                            "scores_pos": _summary_stats(base_pos),
                            "scores_null": _summary_stats(base_neg),
                            "pd_map": _summary_stats(pd_base),
                        },
                        "stap": {
                            "scores_pos": _summary_stats(stap_pos),
                            "scores_null": _summary_stats(stap_neg),
                            "pd_map": _summary_stats(pd_stap),
                        },
                        "masks": {
                            "flow": _mask_stats(mask_flow),
                            "background": _mask_stats(mask_bg),
                        },
                    },
                }
                if confirm2_details is not None:
                    payload["confirm2_details"] = confirm2_details
                json_name = (
                    f"stress_seed{seed}_m{motion}_prf{prf}_K{k_val}_tile"
                    f"{tile_str.replace(':', '-')}.json"
                )
                (outdir / json_name).write_text(json.dumps(payload, indent=2))

                progress.update(1)
                case_index += 1

    progress.close()

    with csv_path.open("w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Stress-test sweep harness")
    ap.add_argument("--grid", nargs="+", default=[], help="Grid specification key=v1,v2")
    ap.add_argument("--tiles", type=str, default="8x8:4")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--npos", type=int, default=100_000)
    ap.add_argument("--nneg", type=int, default=300_000)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--fpr-target", type=float, default=1e-3)
    ap.add_argument("--roc-thresholds", type=int, default=4096)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument(
        "--motion-comp", action="store_true", help="Enable simulated tile motion compensation"
    )
    ap.add_argument(
        "--steer-mode",
        type=str,
        default="bank",
        choices=["bank", "lcmv"],
        help="Steering frontend to emulate (bank or lcmv)",
    )
    ap.add_argument(
        "--steer-grid",
        type=str,
        default=None,
        help="Comma-separated Doppler offsets (Hz) for steering bank",
    )
    ap.add_argument(
        "--steer-fuse",
        type=str,
        default="max",
        help="Fusion rule for steering responses (max|sos|lse)",
    )
    ap.add_argument(
        "--grouping",
        type=str,
        default="none",
        help="Angle grouping strategy for Confirm-2 (spectral|greedy|none)",
    )
    ap.add_argument("--confirm2", action="store_true")
    ap.add_argument("--alpha2", type=float, default=1e-4)
    ap.add_argument("--confirm2-rho", type=float, default=0.5)
    ap.add_argument("--confirm2-ci-alpha", type=float, default=0.05)
    ap.add_argument("--confirm2-pairs", type=int, default=200_000)
    ap.add_argument(
        "--rho-inflate", type=float, default=0.0, help="Additive rho inflation for Confirm-2"
    )
    ap.add_argument(
        "--rho-inflate-policy",
        type=str,
        default="none",
        choices=["none", "auto"],
        help="Heuristic rho inflation policy (none|auto)",
    )
    ap.add_argument(
        "--lt-auto", action="store_true", help="Enable Lt auto-selection flag in telemetry"
    )
    ap.add_argument("--fd-auto", action="store_true", help="Enable Doppler grid auto-sizing flag")
    ap.add_argument("--tbd", action="store_true", help="Enable temporal HMM smoothing (TBD)")
    ap.add_argument(
        "--robust-cov",
        type=str,
        default="scm",
        choices=["scm", "huber", "tyler", "tyler_pca"],
        help="Robust covariance estimator tag (telemetry only).",
    )
    ap.add_argument(
        "--huber-c", type=float, default=5.0, help="Huber c parameter when using robust covariance"
    )
    ap.add_argument(
        "--configs", nargs="*", default=["configs/sims_r1.yaml", "configs/sims_r2.yaml"]
    )
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--no-progress", action="store_true")
    return ap


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    run_stress(args)


if __name__ == "__main__":
    main()
