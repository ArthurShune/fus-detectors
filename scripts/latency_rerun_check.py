#!/usr/bin/env python3
"""
Run a small latency + no-regression check on an existing pilot run directory.

This script replays the PD/STAP stage twice:
  1) a "legacy" variant (no unfold tiling; baseline torch path off)
  2) an "optimized" variant (unfold tiling on; MC--SVD torch path on)

It then:
  - prints a latency breakdown from meta.json telemetry, and
  - asserts that key output maps are numerically identical (within tolerance).

Typical usage:

  PYTHONPATH=. conda run -n stap-fus python scripts/latency_rerun_check.py \
    --src runs/latency_pilot_open \
    --out-root runs/latency_s4_check \
    --profile Brain-OpenSkull \
    --window-length 64 --window-offset 0 \
    --stap-device cuda --stap-debug-samples 0
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from eval.metrics import partial_auc, roc_curve, tpr_at_fpr_target


def _find_single_bundle(out_dir: Path) -> Path:
    bundles = sorted(out_dir.glob("pw_*"))
    if not bundles:
        raise FileNotFoundError(f"No pw_* bundle produced under {out_dir}")
    if len(bundles) > 1:
        names = ", ".join(b.name for b in bundles[:5])
        raise RuntimeError(f"Expected one pw_* bundle under {out_dir}, found {len(bundles)}: {names}")
    return bundles[0]


def _load_telemetry(bundle_dir: Path) -> Dict:
    meta_path = bundle_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    return meta.get("stap_fallback_telemetry", {}) or {}


def _print_latency(label: str, tele: Dict, *, t_data_ms: float | None = None) -> None:
    keys = [
        "baseline_ms",
        "reg_ms",
        "svd_ms",
        "stap_total_ms",
        "stap_extract_ms",
        "stap_batch_proc_ms",
        "stap_post_ms",
        "stap_fast_path_used",
        "stap_tile_statistic_used",
        "stap_unfold_tiling_used",
        "stap_unfold_tiling_error",
    ]
    print(f"\n[{label}]")
    for k in keys:
        if k in tele:
            print(f"  {k}: {tele[k]}")
    try:
        baseline_ms = float(tele.get("baseline_ms", 0.0) or 0.0)
        stap_ms = float(tele.get("stap_total_ms", 0.0) or 0.0)
        if baseline_ms > 0.0 or stap_ms > 0.0:
            e2e_ms = baseline_ms + stap_ms
            print(f"  e2e_ms: {e2e_ms}")
            if t_data_ms is not None and t_data_ms > 0.0:
                print(f"  rtf: {e2e_ms / t_data_ms}")
    except Exception:
        pass


def _compare_maps(
    a_bundle: Path,
    b_bundle: Path,
    names: List[str],
    *,
    rtol: float,
    atol: float,
) -> None:
    for name in names:
        a_path = a_bundle / name
        b_path = b_bundle / name
        if not a_path.exists() or not b_path.exists():
            raise FileNotFoundError(f"Missing comparison file '{name}' in one of the bundles.")
        a = np.load(a_path)
        b = np.load(b_path)
        np.testing.assert_allclose(b, a, rtol=rtol, atol=atol, err_msg=f"Mismatch in {name}")


def _load_pos_neg(bundle_dir: Path, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    prefix_norm = str(prefix or "").strip().lower()
    if prefix_norm in {"stap", "stap_preka", "score_stap", "score_stap_preka"}:
        score_path = bundle_dir / "score_stap_preka.npy"
        if not score_path.exists():
            score_path = bundle_dir / "score_stap.npy"
        if not score_path.exists():
            raise FileNotFoundError(f"Missing score_stap_preka.npy/score_stap.npy under {bundle_dir}")
        score = np.load(score_path).astype(np.float64, copy=False)
        mf = np.load(bundle_dir / "mask_flow.npy").astype(bool, copy=False)
        mb = np.load(bundle_dir / "mask_bg.npy").astype(bool, copy=False)
        if score.shape != mf.shape or score.shape != mb.shape:
            raise ValueError(
                f"Shape mismatch for ROC pooling: score={score.shape} mf={mf.shape} mb={mb.shape}"
            )
        pos = score[mf].ravel()
        neg = score[mb].ravel()
    else:
        pos_path = bundle_dir / f"{prefix}_pos.npy"
        neg_path = bundle_dir / f"{prefix}_neg.npy"
        if not (pos_path.exists() and neg_path.exists()):
            raise FileNotFoundError(f"Missing {prefix}_pos.npy/{prefix}_neg.npy under {bundle_dir}")
        pos = np.load(pos_path).astype(np.float64, copy=False).ravel()
        neg = np.load(neg_path).astype(np.float64, copy=False).ravel()
    if pos.size == 0 or neg.size == 0:
        raise ValueError(f"Empty ROC pools for prefix '{prefix}': pos={pos.size}, neg={neg.size}")
    return pos, neg


def _roc_summary(
    bundle_dir: Path,
    *,
    prefix: str,
    fprs: List[float],
    num_thresh: int,
) -> Dict[str, float]:
    pos, neg = _load_pos_neg(bundle_dir, prefix)
    fpr, tpr, _ = roc_curve(pos, neg, num_thresh=int(num_thresh))
    out: Dict[str, float] = {
        "n_pos": float(pos.size),
        "n_neg": float(neg.size),
    }
    for target in fprs:
        key = f"tpr@{target:g}"
        out[key] = float(tpr_at_fpr_target(fpr, tpr, target_fpr=float(target)))
    # One partial AUC summary (up to the largest requested FPR).
    if fprs:
        fpr_max = float(max(fprs))
        out[f"pauc@{fpr_max:g}"] = float(partial_auc(fpr, tpr, fpr_max=fpr_max))
    return out


def _print_roc_compare(
    *,
    label_a: str,
    bundle_a: Path,
    label_b: str,
    bundle_b: Path,
    prefix: str,
    fprs: List[float],
    num_thresh: int,
) -> None:
    a = _roc_summary(bundle_a, prefix=prefix, fprs=fprs, num_thresh=num_thresh)
    b = _roc_summary(bundle_b, prefix=prefix, fprs=fprs, num_thresh=num_thresh)
    n_pos = int(a.get("n_pos", 0.0))
    n_neg = int(a.get("n_neg", 0.0))
    print(f"\n[roc] prefix={prefix} n_pos={n_pos} n_neg={n_neg} num_thresh={int(num_thresh)}")
    for target in fprs:
        key = f"tpr@{target:g}"
        ta = float(a.get(key, float("nan")))
        tb = float(b.get(key, float("nan")))
        print(f"  {key}: {label_a}={ta:.6f} {label_b}={tb:.6f} delta={tb-ta:+.6f}")
    if fprs:
        fpr_max = float(max(fprs))
        key = f"pauc@{fpr_max:g}"
        ta = float(a.get(key, float("nan")))
        tb = float(b.get(key, float("nan")))
        print(f"  {key}: {label_a}={ta:.6e} {label_b}={tb:.6e} delta={tb-ta:+.6e}")


def _run_replay(
    *,
    src: Path,
    out_dir: Path,
    profile: str,
    stap_profile: str,
    stap_device: str,
    stap_debug_samples: int,
    window_length: int,
    window_offset: int,
    baseline: str,
    baseline_support: str,
    extra_replay_args: List[str],
    env_overrides: Dict[str, str],
) -> Tuple[Path, Dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(env_overrides)
    cmd = [
        sys.executable,
        "scripts/replay_stap_from_run.py",
        "--src",
        str(src),
        "--out",
        str(out_dir),
        "--stap-profile",
        str(stap_profile),
        "--profile",
        str(profile),
        "--stap-device",
        str(stap_device),
        "--stap-debug-samples",
        str(int(stap_debug_samples)),
        "--baseline",
        str(baseline),
        "--baseline-support",
        str(baseline_support),
        "--time-window-length",
        str(int(window_length)),
        "--time-window-offset",
        str(int(window_offset)),
    ]
    if extra_replay_args:
        cmd.extend(list(extra_replay_args))
    print(f"[run] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)
    bundle = _find_single_bundle(out_dir)
    tele = _load_telemetry(bundle)
    return bundle, tele


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Latency replay check (legacy vs optimized).")
    ap.add_argument("--src", type=Path, required=True, help="Pilot run directory (contains meta.json).")
    ap.add_argument("--out-root", type=Path, required=True, help="Output root for legacy/optimized runs.")
    ap.add_argument(
        "--profile",
        type=str,
        required=True,
        choices=["Brain-OpenSkull", "Brain-AliasContract", "Brain-SkullOR", "Brain-Pial128"],
        help="Brain-* operating profile (matches methodology).",
    )
    ap.add_argument("--stap-profile", type=str, default="clinical", choices=["lab", "clinical"])
    ap.add_argument("--stap-device", type=str, default="cuda")
    ap.add_argument("--stap-debug-samples", type=int, default=0)
    ap.add_argument("--baseline", type=str, default="mc_svd", choices=["svd", "mc_svd", "rpca", "hosvd"])
    ap.add_argument("--baseline-support", type=str, default="window", choices=["window", "full"])
    ap.add_argument("--window-length", type=int, default=64)
    ap.add_argument("--window-offset", type=int, default=0)
    ap.add_argument(
        "--cuda-warmup-heavy",
        action="store_true",
        help="Enable heavy CUDA warmup inside replay (steady-state timings; no output changes).",
    )
    ap.add_argument(
        "--profile-tile-statistic",
        action="store_true",
        help=(
            "EXPERIMENTAL / NOT FOR PAPER RESULTS: also run a tile-statistic (cov-only) STAP replay "
            "for latency only (no parity check). Known to catastrophically regress strict-FPR ROC on "
            "Twinkling/Gammex because it replaces per-snapshot nonlinear MSD aggregation with a "
            "ratio-of-means approximation (mean(f) != f(mean))."
        ),
    )
    ap.add_argument(
        "--profile-baseline-reg-torch",
        action="store_true",
        help="Also run an MC-SVD replay with torch/CUDA registration for latency only (no parity check).",
    )
    ap.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="Relative tolerance for output parity check.",
    )
    ap.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for output parity check.",
    )
    ap.add_argument(
        "--roc-fprs",
        type=str,
        default="1e-3,3e-4,1e-4",
        help=(
            "Comma-separated list of FPR targets for optional ROC comparison "
            "(optimized snapshot scoring vs experimental tile-statistic)."
        ),
    )
    ap.add_argument(
        "--roc-thresholds",
        type=int,
        default=8192,
        help="Number of thresholds in ROC approximation (higher = more accurate, slower).",
    )
    ap.add_argument(
        "--replay-extra",
        type=str,
        default="",
        help=(
            "Extra CLI args passed through to scripts/replay_stap_from_run.py. "
            "Provide as a single quoted string, e.g. "
            "\"--flow-doppler-min-hz 60 --flow-doppler-max-hz 180 --flow-doppler-noise-amp 0.2\"."
        ),
    )
    ap.add_argument(
        "--tile-batch",
        type=int,
        default=0,
        help=(
            "Tile batch size for STAP tiling (sets STAP_TILE_BATCH for both runs). "
            "0 leaves the pipeline default unchanged. For CUDA + Brain-* profiles, "
            "values like 384/512 often improve throughput without changing outputs."
        ),
    )
    ap.add_argument(
        "--stap-conditional",
        type=str,
        default="off",
        choices=["off", "on"],
        help=(
            "Conditional STAP execution during replay. "
            "'off' matches the paper baselines (full STAP on all tiles); "
            "'on' skips tiles outside a proxy flow mask."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    roc_fprs = [float(x) for x in str(args.roc_fprs).split(",") if x.strip()]
    roc_thresh = int(args.roc_thresholds)
    extra_replay_args = []
    if str(args.stap_conditional).strip().lower() in {"on", "1", "true", "yes"}:
        extra_replay_args.append("--stap-conditional-enable")
    else:
        extra_replay_args.append("--stap-conditional-disable")
    extra_replay_args.extend(shlex.split(str(args.replay_extra or "").strip()))

    # Force debug off in both modes; debug capture disables fast paths by design.
    if args.stap_debug_samples != 0:
        raise ValueError("--stap-debug-samples must be 0 for latency checks.")

    legacy_env = {
        # Keep baseline "as-is": SVD may run on CUDA (T×T) but registration remains numpy.
        "STAP_TILING_UNFOLD": "0",
        "MC_SVD_TORCH": "0",
        "MC_SVD_REG_TORCH": "0",
        "MC_SVD_TORCH_RETURN_CUBE": "0",
        # Ensure fast path is enabled for both variants (replay may already set defaults).
        "STAP_FAST_PATH": "1",
        "STAP_FAST_PD_ONLY": "1",
    }
    opt_env = {
        "STAP_TILING_UNFOLD": "1",
        "MC_SVD_TORCH": "1",
        "MC_SVD_REG_TORCH": "0",
        "MC_SVD_TORCH_RETURN_CUBE": "1",
        "STAP_FAST_PATH": "1",
        "STAP_FAST_PD_ONLY": "1",
    }
    if int(getattr(args, "tile_batch", 0) or 0) > 0:
        tb = str(int(args.tile_batch))
        legacy_env["STAP_TILE_BATCH"] = tb
        opt_env["STAP_TILE_BATCH"] = tb
    if args.cuda_warmup_heavy:
        # Opt-in steady-state CUDA warmup for latency profiling.
        legacy_env["CUDA_WARMUP_HEAVY"] = "1"
        opt_env["CUDA_WARMUP_HEAVY"] = "1"

    legacy_out = args.out_root / "legacy"
    opt_out = args.out_root / "optimized"

    prf_hz = None
    try:
        src_meta = json.loads((args.src / "meta.json").read_text())
        prf_hz = float(src_meta.get("prf_hz"))
    except Exception:
        prf_hz = None
    t_data_ms = None
    if prf_hz is not None and prf_hz > 0.0:
        t_data_ms = 1000.0 * float(args.window_length) / float(prf_hz)
        print(f"[data] prf_hz={prf_hz} window={int(args.window_length)} -> t_data_ms={t_data_ms}")

    legacy_bundle, tele_legacy = _run_replay(
        src=args.src,
        out_dir=legacy_out,
        profile=args.profile,
        stap_profile=args.stap_profile,
        stap_device=args.stap_device,
        stap_debug_samples=args.stap_debug_samples,
        window_length=args.window_length,
        window_offset=args.window_offset,
        baseline=args.baseline,
        baseline_support=args.baseline_support,
        extra_replay_args=extra_replay_args,
        env_overrides=legacy_env,
    )
    opt_bundle, tele_opt = _run_replay(
        src=args.src,
        out_dir=opt_out,
        profile=args.profile,
        stap_profile=args.stap_profile,
        stap_device=args.stap_device,
        stap_debug_samples=args.stap_debug_samples,
        window_length=args.window_length,
        window_offset=args.window_offset,
        baseline=args.baseline,
        baseline_support=args.baseline_support,
        extra_replay_args=extra_replay_args,
        env_overrides=opt_env,
    )

    _print_latency("legacy", tele_legacy, t_data_ms=t_data_ms)
    _print_latency("optimized", tele_opt, t_data_ms=t_data_ms)

    compare_files = ["pd_base.npy", "pd_stap.npy", "score_stap.npy", "stap_score_map.npy"]
    _compare_maps(legacy_bundle, opt_bundle, compare_files, rtol=float(args.rtol), atol=float(args.atol))
    print("\n[parity] OK (maps match within tolerance).")

    if args.profile_tile_statistic:
        tile_stat_env = dict(opt_env)
        tile_stat_env["STAP_FAST_TILE_STATISTIC"] = "1"
        tile_stat_out = args.out_root / "tile_statistic"
        tile_bundle, tele_tile = _run_replay(
            src=args.src,
            out_dir=tile_stat_out,
            profile=args.profile,
            stap_profile=args.stap_profile,
            stap_device=args.stap_device,
            stap_debug_samples=args.stap_debug_samples,
            window_length=args.window_length,
            window_offset=args.window_offset,
            baseline=args.baseline,
            baseline_support=args.baseline_support,
            extra_replay_args=extra_replay_args,
            env_overrides=tile_stat_env,
        )
        _print_latency("tile_statistic", tele_tile, t_data_ms=t_data_ms)
        # Note: tile-statistic mode intentionally produces different score maps and is
        # NOT ROC-equivalent to the manuscript detector (ratio-of-means vs nonlinear
        # per-snapshot aggregation). It is kept only as a latency experiment and is
        # known to catastrophically regress strict-FPR ROC on Twinkling/Gammex, so we
        # do not enforce parity here.
        try:
            _print_roc_compare(
                label_a="optimized",
                bundle_a=opt_bundle,
                label_b="tile_statistic",
                bundle_b=tile_bundle,
                prefix="stap",
                fprs=roc_fprs,
                num_thresh=roc_thresh,
            )
        except Exception as exc:
            print(f"[roc] WARNING: failed to compute ROC comparison: {exc}", flush=True)

    if args.profile_baseline_reg_torch:
        reg_torch_env = dict(opt_env)
        reg_torch_env["MC_SVD_REG_TORCH"] = "1"
        reg_torch_out = args.out_root / "baseline_reg_torch"
        reg_bundle, tele_reg = _run_replay(
            src=args.src,
            out_dir=reg_torch_out,
            profile=args.profile,
            stap_profile=args.stap_profile,
            stap_device=args.stap_device,
            stap_debug_samples=args.stap_debug_samples,
            window_length=args.window_length,
            window_offset=args.window_offset,
            baseline=args.baseline,
            baseline_support=args.baseline_support,
            extra_replay_args=extra_replay_args,
            env_overrides=reg_torch_env,
        )
        _print_latency("baseline_reg_torch", tele_reg, t_data_ms=t_data_ms)
        # Note: torch registration can produce slightly different registered
        # cubes vs the NumPy path, so we do not enforce parity here.


if __name__ == "__main__":
    main()
