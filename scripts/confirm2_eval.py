#!/usr/bin/env python3
"""
Batch Confirm-2 calibration/evaluation for the motion-stress bundles.

Usage
-----
PYTHONPATH=. python scripts/confirm2_eval.py \
    --bundle alias=runs/motion/alias/pw_7.5MHz_5ang_2ens_128T_seed1 \
    --bundle pftrace=runs/motion/alias_pftrace/pw_7.5MHz_5ang_2ens_128T_seed1 \
    --alpha2 1e-5 \
    --cal-pairs 1000 \
    --output reports/motion_confirm2_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from pipeline.calib.evt_pot import MeanExcessDiagnostics, PotModel
from pipeline.confirm2.validator import calibrate_confirm2, evaluate_confirm2


def _parse_bundle_arg(arg: str) -> Tuple[str, Path]:
    if "=" not in arg:
        raise ValueError(f"--bundle entries must be label=path (got '{arg}')")
    label, _, path_str = arg.partition("=")
    label = label.strip()
    if not label:
        raise ValueError(f"Bundle label missing in '{arg}'")
    path = Path(path_str).expanduser().resolve()
    return label, path


def _load_score_pool(bundle_dir: Path, score_mode: str) -> np.ndarray:
    suffix = "" if score_mode == "msd" else f"_{score_mode}"
    path = bundle_dir / f"stap_neg{suffix}.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing (score_mode={score_mode})")
    arr = np.load(path).astype(np.float32, copy=False)
    if arr.size == 0:
        raise ValueError(f"{path} is empty")
    return arr


def _reduce_blocks(scores: np.ndarray, block_size: int) -> tuple[np.ndarray, int]:
    arr = np.asarray(scores, dtype=np.float64).ravel()
    if block_size <= 1 or arr.size == 0:
        return arr.astype(np.float32, copy=False), int(arr.size)
    blocks = arr.size // block_size
    if blocks == 0:
        return arr.astype(np.float32, copy=False), int(arr.size)
    trimmed = arr[: blocks * block_size]
    reduced = trimmed.reshape(blocks, block_size).max(axis=1)
    return reduced.astype(np.float32, copy=False), int(blocks)


def _sample_pairs(
    scores: np.ndarray,
    pair_count: int,
    *,
    seed: int,
    block_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    scores, _ = _reduce_blocks(scores, max(1, block_size))
    rng = np.random.default_rng(seed)
    needed = max(2, pair_count * 2)
    if scores.size >= needed:
        idx = rng.permutation(scores.size)[:needed]
        sample = scores[idx]
    else:
        sample = rng.choice(scores, size=needed, replace=True)
    return (
        sample[0::2].astype(np.float32, copy=False),
        sample[1::2].astype(np.float32, copy=False),
    )


def _pot_summary(pm: PotModel) -> Dict[str, float | int | None]:
    endpoint = None
    if pm.xi < 0.0:
        endpoint = pm.u - pm.beta / pm.xi
    return {
        "u": float(pm.u),
        "xi": float(pm.xi),
        "beta": float(pm.beta),
        "p_u": float(pm.p_u),
        "n_exc": int(pm.n_exc),
        "n_total": int(pm.n_total),
        "bounded_endpoint": float(endpoint) if endpoint is not None else None,
    }


def _mean_excess_summary(diag: MeanExcessDiagnostics) -> Dict[str, float | int | None]:
    return {
        "status": diag.status,
        "selected_u": float(diag.selected_u),
        "selected_index": int(diag.selected_index),
        "r2": None if diag.r2 is None else float(diag.r2),
        "r2_threshold": float(diag.r2_threshold),
        "n_exc": int(diag.n_exc),
        "min_exceedances": int(diag.min_exceedances),
    }


def evaluate_label(
    label: str,
    bundle_dirs: List[Path],
    *,
    alpha2: float,
    cal_pairs: int,
    test_pairs: int,
    ci_alpha: float,
    min_exceedances: int,
    q0: float,
    score_mode: str,
    block_size: int,
    evd_mode: str,
) -> Dict[str, float]:
    if not bundle_dirs:
        raise ValueError(f"No bundle directories provided for label '{label}'")
    pools = []
    seeds = []
    meta_paths = []
    for bundle_dir in bundle_dirs:
        meta = json.loads((bundle_dir / "meta.json").read_text())
        seeds.append(int(meta.get("seed", 0)))
        pools.append(_load_score_pool(bundle_dir, score_mode))
        meta_paths.append(str(bundle_dir))
    pool = np.concatenate(pools, axis=0)
    original_pool = int(pool.size)
    pool, pool_blocks = _reduce_blocks(pool, max(1, block_size))
    base_seed = seeds[0] if seeds else 0
    s1_cal, s2_cal = _sample_pairs(pool, cal_pairs, seed=base_seed + 101, block_size=1)
    s1_test, s2_test = _sample_pairs(
        pool,
        max(test_pairs, 50),
        seed=base_seed + 303,
        block_size=1,
    )
    calib = calibrate_confirm2(
        s1_cal,
        s2_cal,
        alpha2_target=alpha2,
        seed=base_seed + 202,
        min_exceedances=min_exceedances,
        min_exceedances_weibull=min_exceedances,
        q0=q0,
        evd_mode=evd_mode,
    )
    eval_res = evaluate_confirm2(calib, s1_test, s2_test, alpha=ci_alpha)
    return {
        "label": label,
        "bundle": bundle_dirs[0].name,
        "bundle_paths": meta_paths,
        "aggregate_count": len(bundle_dirs),
        "pool_size": original_pool,
        "pool_blocks": pool_blocks,
        "block_size": int(block_size),
        "alpha2_target": alpha2,
        "alpha1_per_look": calib.alpha1,
        "rho_hat": calib.rho_hat,
        "rho_eff": calib.rho_eff,
        "n_cal_pairs": int(s1_cal.size),
        "n_test_pairs": int(s1_test.size),
        "empirical_pair_pfa": eval_res.empirical_pair_pfa,
        "pair_ci_lo": eval_res.pair_ci_lo,
        "pair_ci_hi": eval_res.pair_ci_hi,
        "predicted_pair_pfa": eval_res.predicted_pair_pfa,
        "k_joint": eval_res.k_joint,
        "evd_mode": evd_mode,
        "pot_look1": _pot_summary(calib.pm1),
        "pot_look2": _pot_summary(calib.pm2),
        "mean_excess_look1": _mean_excess_summary(calib.mean_excess_diag1),
        "mean_excess_look2": _mean_excess_summary(calib.mean_excess_diag2),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Confirm-2 evaluator for motion-stress bundles.")
    ap.add_argument(
        "--bundle",
        action="append",
        required=True,
        help="Bundle spec label=path (e.g., alias=runs/motion/alias/pw_*).",
    )
    ap.add_argument("--alpha2", type=float, default=1e-5, help="Target pair-Pfa.")
    ap.add_argument(
        "--cal-pairs",
        type=int,
        default=1000,
        help="Number of pairs reserved for Confirm-2 calibration.",
    )
    ap.add_argument(
        "--test-pairs",
        type=int,
        default=100,
        help="Number of hold-out null pairs for evaluation.",
    )
    ap.add_argument(
        "--ci-alpha",
        type=float,
        default=0.05,
        help="Two-sided CI alpha for empirical pair-Pfa.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("reports/motion_confirm2_summary.json"),
        help="JSON file to write summary results.",
    )
    ap.add_argument(
        "--min-exceedances",
        type=int,
        default=50,
        help="Minimum exceedances for POT/Weibull fits (lowered for small bundles).",
    )
    ap.add_argument(
        "--q0",
        type=float,
        default=0.9,
        help="Quantile for mean-excess threshold selection (motion datasets typically use 0.9).",
    )
    ap.add_argument(
        "--score-mode",
        type=str,
        default="pd",
        choices=["msd", "pd", "band_ratio"],
        help="Score pool to sample Confirm-2 pairs from.",
    )
    ap.add_argument(
        "--block-size",
        type=int,
        default=1,
        help="Optional block size for declustering (use max per block before sampling).",
    )
    ap.add_argument(
        "--evd-mode",
        type=str,
        default="gpd",
        choices=["gpd", "weibull"],
        help="Extreme-value mode used in conformal thresholds.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    bundles: Dict[str, List[Path]] = {}
    for spec in args.bundle:
        label, path = _parse_bundle_arg(spec)
        bundles.setdefault(label, []).append(path)
    for label, paths in bundles.items():
        summary = evaluate_label(
            label,
            paths,
            alpha2=args.alpha2,
            cal_pairs=args.cal_pairs,
            test_pairs=args.test_pairs,
            ci_alpha=args.ci_alpha,
            min_exceedances=args.min_exceedances,
            q0=args.q0,
            score_mode=args.score_mode,
            block_size=max(1, int(args.block_size)),
            evd_mode=args.evd_mode,
        )
        results.append(summary)
        agg_note = f"(paths={len(paths)}, pool={summary['pool_size']})" if len(paths) > 1 else ""
        print(
            f"{label}: Pfa_emp={summary['empirical_pair_pfa']:.3e} "
            f"[{summary['pair_ci_lo']:.3e}, {summary['pair_ci_hi']:.3e}] "
            f"pred={summary['predicted_pair_pfa']:.3e} "
            f"(n_test={summary['n_test_pairs']}, k={summary['k_joint']}) {agg_note}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
