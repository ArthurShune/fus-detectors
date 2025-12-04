"""Monte-Carlo sweep harness for acceptance metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from tqdm import tqdm

from eval.acceptance_cli import _simulate_confirm2_scores, _simulate_scores_and_pd
from eval.metrics import partial_auc, pd_snr_db, roc_curve, tpr_at_fpr_target
from pipeline.confirm2.policy import rho_inflate_policy
from pipeline.confirm2.validator import calibrate_confirm2, evaluate_confirm2
from pipeline.utils.telemetry import sample_gpu_stats, system_telemetry

try:
    import torch
except Exception:  # torch optional
    torch = None


def _summary_stats(arr: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
    if arr is None:
        return None
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


def _sanitize_alpha(value: float) -> str:
    return ("%0.3e" % value).replace("-", "m").replace("+", "p").replace(".", "d")


def _device_from_arg(arg: Optional[str]) -> Optional[str]:
    if arg is None:
        return "cuda" if torch is not None and torch.cuda.is_available() else None
    if arg.lower() in {"none", "cpu"}:
        return None if arg.lower() == "none" else "cpu"
    if arg.startswith("cuda") and (torch is None or not torch.cuda.is_available()):
        raise RuntimeError("CUDA requested but torch with CUDA support not available")
    return arg


def run_sweep(args: argparse.Namespace) -> None:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_dir = Path(args.json_dir) if args.json_dir else None
    if json_dir:
        json_dir.mkdir(parents=True, exist_ok=True)

    device = _device_from_arg(args.device)
    use_cuda = bool(device and torch is not None and device.startswith("cuda"))
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    disable_progress = args.no_progress or os.environ.get("TQDM_DISABLE", "0") == "1"

    fields = [
        "seed",
        "method",
        "fpr_target",
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
        "rho_inflate",
        "rho_inflate_policy",
        "lt_auto",
        "fd_auto",
    ]

    confirm2_cols: List[str] = []
    alpha2_list = args.alpha2 if args.confirm2 else []
    for alpha2 in alpha2_list:
        tag = _sanitize_alpha(alpha2)
        confirm2_cols.extend(
            [
                f"confirm2_emp_{tag}",
                f"confirm2_pred_{tag}",
                f"confirm2_alpha1_{tag}",
                f"confirm2_rho_eff_{tag}",
            ]
        )
    fields.extend(confirm2_cols)

    seeds = [args.seed_offset + i for i in range(args.seeds)]

    rows: List[Dict[str, float]] = []
    tele = system_telemetry(include_nvidia_smi=False)

    for seed in tqdm(seeds, desc="Monte-Carlo", unit="run", disable=disable_progress):
        seed_start = time.perf_counter()
        phase_times: Dict[str, float] = {}
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        phase = time.perf_counter()
        sim = _simulate_scores_and_pd(
            n_pos=args.npos,
            n_neg=args.nneg,
            H=args.height,
            W=args.width,
            seed=seed,
        )
        (
            base_pos,
            base_neg,
            stap_pos,
            stap_neg,
            pd_base,
            pd_stap,
            mask_flow,
            mask_bg,
        ) = sim
        phase_times["simulate_sec"] = time.perf_counter() - phase

        # Shared ROC curves
        phase = time.perf_counter()
        fpr_b, tpr_b, _ = roc_curve(base_pos, base_neg, num_thresh=args.roc_thresholds)
        fpr_s, tpr_s, _ = roc_curve(stap_pos, stap_neg, num_thresh=args.roc_thresholds)
        phase_times["roc_sec"] = time.perf_counter() - phase

        pd_snr_baseline = pd_snr_db(pd_base, mask_flow, mask_bg)
        pd_snr_stap = pd_snr_db(pd_stap, mask_flow, mask_bg)

        # Confirm-2 metrics cached once per seed per alpha2
        confirm2_map: Dict[str, Dict[str, float]] = {}
        confirm2_elapsed = 0.0
        if args.confirm2:
            pairs = args.confirm2_pairs
            s1_null, s2_null = _simulate_confirm2_scores(pairs, rho=args.confirm2_rho, seed=seed)
            half = pairs // 2
            for alpha2 in alpha2_list:
                alpha_start = time.perf_counter()
                policy_decision = None
                rho_inflate_used = args.rho_inflate
                if args.rho_inflate_policy == "auto":
                    prelim = calibrate_confirm2(
                        s1_null[:half],
                        s2_null[:half],
                        alpha2_target=alpha2,
                        seed=seed,
                        rho_inflate=0.0,
                        device=device,
                    )
                    policy_decision = rho_inflate_policy(
                        rho_hat=prelim.rho_hat,
                        rho_groups=None,
                        motion_um=None,
                        dropout=None,
                    )
                    rho_inflate_used = policy_decision.delta
                calib = calibrate_confirm2(
                    s1_null[:half],
                    s2_null[:half],
                    alpha2_target=alpha2,
                    seed=seed,
                    rho_inflate=rho_inflate_used,
                    device=device,
                )
                ev = evaluate_confirm2(
                    calib,
                    s1_null[half:],
                    s2_null[half:],
                    alpha=args.confirm2_ci_alpha,
                    device=device,
                )
                tag = _sanitize_alpha(alpha2)
                record = {
                    "emp": ev.empirical_pair_pfa,
                    "pred": ev.predicted_pair_pfa,
                    "alpha1": calib.alpha1,
                    "rho_eff": calib.rho_eff,
                    "rho_inflate": rho_inflate_used,
                    "rho_hat": calib.rho_hat,
                    "rho_ci": [calib.rho_lo, calib.rho_hi],
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
                    "pairs_cal": half,
                    "pairs_test": ev.n_null_pairs,
                    "k_joint": ev.k_joint,
                }
                if policy_decision is not None:
                    record["rho_policy"] = asdict(policy_decision)
                confirm2_map[tag] = record
                confirm2_elapsed += time.perf_counter() - alpha_start
        if confirm2_map:
            phase_times["confirm2_sec"] = confirm2_elapsed

        if use_cuda:
            torch.cuda.synchronize()
        fpr_phase_start = time.perf_counter()

        rho_row = None
        for fpr in args.fprs:
            auc_b = partial_auc(fpr_b, tpr_b, fpr_max=fpr)
            auc_s = partial_auc(fpr_s, tpr_s, fpr_max=fpr)
            if confirm2_map:
                first_tag = next(iter(confirm2_map))
                rho_row = confirm2_map[first_tag].get("rho_inflate")
            row = {
                "seed": seed,
                "method": args.method,
                "fpr_target": fpr,
                "pd_snr_baseline_db": pd_snr_baseline,
                "pd_snr_stap_db": pd_snr_stap,
                "pd_snr_delta_db": pd_snr_stap - pd_snr_baseline,
                "tpr_at_fpr_baseline": tpr_at_fpr_target(fpr_b, tpr_b, fpr),
                "tpr_at_fpr_stap": tpr_at_fpr_target(fpr_s, tpr_s, fpr),
                "pauc_baseline": auc_b,
                "pauc_stap": auc_s,
                "pauc_delta": auc_s - auc_b,
                "device": device or "cpu",
                "rho_inflate": float(rho_row) if rho_row is not None else args.rho_inflate,
                "rho_inflate_policy": args.rho_inflate_policy,
                "lt_auto": int(getattr(args, "lt_auto", False)),
                "fd_auto": int(getattr(args, "fd_auto", False)),
            }
            row["tpr_at_fpr_delta"] = row["tpr_at_fpr_stap"] - row["tpr_at_fpr_baseline"]

            for tag, metrics in confirm2_map.items():
                row[f"confirm2_emp_{tag}"] = metrics["emp"]
                row[f"confirm2_pred_{tag}"] = metrics["pred"]
                row[f"confirm2_alpha1_{tag}"] = metrics["alpha1"]
                row[f"confirm2_rho_eff_{tag}"] = metrics["rho_eff"]

            rows.append(row)

        if use_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - fpr_phase_start
        phase_times["aggregate_rows_sec"] = elapsed
        total_elapsed = time.perf_counter() - seed_start
        phase_times["total_sec"] = total_elapsed
        for row in rows[-len(args.fprs) :]:
            row["timing_sec"] = total_elapsed
        gpu_mem = torch.cuda.max_memory_allocated() / 1e6 if use_cuda else None

        # JSON payload per seed
        if json_dir:
            payload = {
                "seed": seed,
                "device": device or "cpu",
                "sweep": {
                    "seed": seed,
                    "fpr_grid": list(args.fprs),
                    "alpha2_grid": list(alpha2_list),
                    "n_neg": args.nneg,
                    "n_pos": args.npos,
                    "rho_inflate": float(rho_row) if rho_row is not None else args.rho_inflate,
                    "rho_inflate_policy": args.rho_inflate_policy,
                    "lt_auto": bool(getattr(args, "lt_auto", False)),
                    "fd_auto": bool(getattr(args, "fd_auto", False)),
                },
                "performance": rows[-len(args.fprs) :],
                "confirm2": confirm2_map,
                "compute": {
                    "hz": (1.0 / total_elapsed) if total_elapsed else None,
                    "ms_per_fpr": (total_elapsed * 1000.0) / max(len(args.fprs), 1),
                    "gpu_mem_MB": gpu_mem,
                },
                "timings": phase_times,
                "telemetry": tele,
                "gpu": sample_gpu_stats(include_nvidia_smi=False),
                "data_summary": {
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
                "seeds": {
                    "scores": seed,
                    "confirm2": seed if args.confirm2 else None,
                },
            }
            (json_dir / f"mc_seed_{seed}.json").write_text(json.dumps(payload, indent=2))

    with out_path.open("w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Monte-Carlo acceptance sweep")
    ap.add_argument("--method", type=str, default="stap")
    ap.add_argument("--fprs", type=float, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--npos", type=int, default=100_000)
    ap.add_argument("--nneg", type=int, default=300_000)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--roc-thresholds", type=int, default=4096)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--confirm2", action="store_true")
    ap.add_argument("--alpha2", type=float, nargs="+", default=[])
    ap.add_argument("--confirm2-pairs", type=int, default=200_000)
    ap.add_argument("--confirm2-ci-alpha", type=float, default=0.05)
    ap.add_argument("--rho-inflate", type=float, default=0.0, help="Rho inflation for Confirm-2")
    ap.add_argument(
        "--rho-inflate-policy",
        type=str,
        default="none",
        choices=["none", "auto"],
        help="Heuristic rho inflation policy (none|auto)",
    )
    ap.add_argument(
        "--lt-auto", action="store_true", help="Flag Lt auto-selection in telemetry rows"
    )
    ap.add_argument(
        "--fd-auto", action="store_true", help="Flag Doppler auto-sizing in telemetry rows"
    )
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--json-dir", type=str, default=None)
    ap.add_argument("--no-progress", action="store_true")
    return ap


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    run_sweep(args)


if __name__ == "__main__":
    main()
