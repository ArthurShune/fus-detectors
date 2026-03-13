#!/usr/bin/env python3
"""
Run a small latency + no-regression check on an existing pilot run directory.

This script replays the PD/STAP stage twice:
  1) a "legacy" variant (no unfold tiling; baseline torch path off)
  2) an "optimized" variant (unfold tiling on; MC--SVD torch path on)

It then:
  - prints a latency breakdown from meta.json telemetry, and
  - asserts that key output maps are numerically identical (within tolerance).

For publishable CUDA timings, this script forces CUDA-graph capture off for
the legacy replay and on for the optimized replay via STAP_FAST_CUDA_GRAPH={0,1}.
Steady-state latency is reported as the mean over windows 2..N; window1 is cold
and may include one-time overheads (CUDA init, Triton JIT, CUDA-graph capture).

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
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from eval.metrics import partial_auc, roc_curve, tpr_at_fpr_target


def _generate_synthetic_pilot(
    out_root: Path,
    *,
    profile: str,
    window_length: int,
    window_offset: int,
    seed: int = 1,
) -> None:
    """
    Create a minimal synthetic "pilot run" directory compatible with replay_stap_from_run.py.

    This is intended only for latency/parity checks (not manuscript results).
    The layout matches sim/kwave/pilot_motion.py --synthetic.
    """
    out_root.mkdir(parents=True, exist_ok=True)

    profile_norm = str(profile or "").strip()
    # Frozen Brain-* acquisition profile (stap_fus_methodology.tex, App. sim-details).
    Nx = 240
    Ny = 240
    dx = 90e-6
    dy = 90e-6
    c0 = 1540.0
    rho0 = 1000.0
    pml = 16
    cfl = 0.3
    f0_hz = 7.5e6
    ncycles = 3
    base_angles_deg = [-12.0, -6.0, 0.0, 6.0, 12.0]
    prf_hz = 1500.0

    if profile_norm == "Brain-Pial128":
        ensembles = 4
        pulses_per_ensemble = 32
    else:
        ensembles = 5
        pulses_per_ensemble = 64

    # Ensure the requested window lies within the synthesized slow-time stack.
    required_T = int(window_offset) + int(window_length)
    total_T = int(ensembles) * int(pulses_per_ensemble)
    if required_T > total_T:
        # Expand by adding ensembles (keep per-ensemble pulse length fixed).
        ensembles = int(np.ceil(required_T / float(pulses_per_ensemble)))
        total_T = int(ensembles) * int(pulses_per_ensemble)

    rng = np.random.default_rng(int(seed))
    Nt = max(128, int(Ny) * 2)
    dt = float(cfl) * float(min(dx, dy)) / float(max(c0, 1.0))
    angles_used_meta: list[list[float]] = []

    for ens in range(int(ensembles)):
        used_angles: list[float] = []
        for idx, base_angle in enumerate(base_angles_deg):
            angle_dir = out_root / f"ens{ens}_angle_{int(round(base_angle))}"
            angle_dir.mkdir(parents=True, exist_ok=True)
            rf = rng.standard_normal((Nt, Nx)).astype(np.float32)
            rf += 0.25 * rng.standard_normal((Nt, Nx)).astype(np.float32)
            t = (np.arange(Nt, dtype=np.float32) * np.float32(dt)).astype(np.float32, copy=False)
            mod = 0.04 * float(ens + 1) + 0.01 * float(idx + 1)
            rf += (0.08 * np.sin(2.0 * np.pi * mod * t)).astype(np.float32, copy=False)[:, None]
            np.save(angle_dir / "rf.npy", rf.astype(np.float32, copy=False), allow_pickle=False)
            np.save(
                angle_dir / "dt.npy",
                np.array(dt, dtype=np.float32),
                allow_pickle=False,
            )
            used_angles.append(float(base_angle))
        angles_used_meta.append(used_angles)

    meta = {
        "geometry": {
            "Nx": int(Nx),
            "Ny": int(Ny),
            "dx": float(dx),
            "dy": float(dy),
            "c0": float(c0),
            "rho0": float(rho0),
            "pml": int(pml),
            "cfl": float(cfl),
        },
        "base_angles_deg": [float(a) for a in base_angles_deg],
        "angles_used_deg": angles_used_meta,
        "f0_hz": float(f0_hz),
        "ncycles": int(ncycles),
        "ensembles": int(ensembles),
        "jitter_um": 0.0,
        "seed": int(seed),
        "pulses_per_ensemble": int(pulses_per_ensemble),
        "prf_hz": float(prf_hz),
        "synthetic": True,
        "generated_by": "scripts/latency_rerun_check.py",
        "generated_for_profile": profile_norm,
        "generated_total_frames": int(total_T),
    }
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def _ensure_src_pilot_exists(args: argparse.Namespace) -> None:
    src = Path(getattr(args, "src"))
    meta_path = src / "meta.json"
    if meta_path.exists():
        return
    if src.exists():
        # Avoid clobbering a partially-populated directory that might be a real pilot.
        try:
            has_any = any(src.iterdir())
        except Exception:
            has_any = False
        if has_any:
            raise FileNotFoundError(
                f"Missing {meta_path} but {src} is not empty. "
                "Refusing to auto-generate a synthetic pilot. "
                "Either point --src to a valid pilot run directory or delete/rename this folder."
            )
    print(
        f"[src] {meta_path} missing; generating a synthetic Brain-* pilot under {src} for replay...",
        flush=True,
    )
    _generate_synthetic_pilot(
        src,
        profile=str(getattr(args, "profile", "")),
        window_length=int(getattr(args, "window_length", 64) or 64),
        window_offset=int(getattr(args, "window_offset", 0) or 0),
        seed=1,
    )


def _find_single_bundle(out_dir: Path) -> Path:
    bundles = sorted(out_dir.glob("pw_*"))
    if not bundles:
        raise FileNotFoundError(f"No pw_* bundle produced under {out_dir}")
    if len(bundles) > 1:
        names = ", ".join(b.name for b in bundles[:5])
        raise RuntimeError(f"Expected one pw_* bundle under {out_dir}, found {len(bundles)}: {names}")
    return bundles[0]


def _find_bundles(out_dir: Path) -> List[Path]:
    bundles = sorted(out_dir.glob("pw_*"))
    if not bundles:
        raise FileNotFoundError(f"No pw_* bundles produced under {out_dir}")
    return bundles


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
        "stap_fast_cuda_graph",
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


def _print_latency_windows(label: str, teles: List[Dict], *, t_data_ms: float | None = None) -> None:
    if not teles:
        raise ValueError(f"No telemetry entries provided for '{label}'.")
    if len(teles) == 1:
        _print_latency(label, teles[0], t_data_ms=t_data_ms)
        return

    cold = teles[0]
    steady = teles[1:]
    _print_latency(f"{label} cold(win1)", cold, t_data_ms=t_data_ms)

    keys_ms = [
        "baseline_ms",
        "reg_ms",
        "svd_ms",
        "stap_total_ms",
        "stap_extract_ms",
        "stap_batch_proc_ms",
        "stap_post_ms",
    ]
    keys_bool = [
        "stap_fast_path_used",
        "stap_tile_statistic_used",
        "stap_unfold_tiling_used",
    ]
    mean_tele: Dict[str, float | bool] = {}
    for k in keys_ms:
        vals: List[float] = []
        for tele in steady:
            if k not in tele:
                continue
            try:
                vals.append(float(tele[k]))
            except Exception:
                pass
        if vals:
            mean_tele[k] = float(sum(vals) / float(len(vals)))
    for k in keys_bool:
        mean_tele[k] = bool(all(bool(tele.get(k, False)) for tele in steady))

    _print_latency(f"{label} steady(avg win2..{len(teles)})", mean_tele, t_data_ms=t_data_ms)

    stap_ms = [float(tele.get("stap_total_ms", float("nan"))) for tele in teles]
    print(f"  per_window_stap_total_ms: {stap_ms}")


def _mean_stage_map(teles: List[Dict]) -> Dict[str, float]:
    acc: Dict[str, List[float]] = {}
    for tele in teles:
        stage_map = tele.get("stap_cuda_stage_ms")
        if not isinstance(stage_map, dict):
            continue
        for key, value in stage_map.items():
            try:
                acc.setdefault(str(key), []).append(float(value))
            except Exception:
                pass
    return {key: float(sum(vals) / float(len(vals))) for key, vals in acc.items() if vals}


def _mean_scalar(teles: List[Dict], key: str) -> float | None:
    vals: List[float] = []
    for tele in teles:
        try:
            if key in tele:
                vals.append(float(tele[key]))
        except Exception:
            pass
    if not vals:
        return None
    return float(sum(vals) / float(len(vals)))


def _print_cuda_profile(label: str, teles: List[Dict]) -> None:
    if not teles:
        return
    steady = teles[1:] if len(teles) > 1 else teles
    stage_mean = _mean_stage_map(steady)
    if not stage_mean:
        return
    print(f"\n[{label} steady CUDA profile]")
    top_level_keys = [
        "stap:tiling:prep",
        "stap:tiling:cube_unfold",
        "stap:tiling:active_index",
        "stap:tiling:gather",
        "stap:core",
        "stap:tiling:fold",
        "stap:tiling:to_cpu",
    ]
    core_keys = [
        "stap:hankel",
        "stap:covariance:train_trim",
        "stap:covariance",
        "stap:shrinkage",
        "stap:lambda",
        "stap:fd_grid",
        "stap:constraints",
        "stap:band_energy",
        "stap:aggregate",
        "stap:telemetry",
    ]
    top_present = [(k, stage_mean[k]) for k in top_level_keys if k in stage_mean]
    if top_present:
        print("  top_level_ms:")
        for key, value in top_present:
            print(f"    {key}: {value:.3f}")
    core_present = [(k, stage_mean[k]) for k in core_keys if k in stage_mean]
    if core_present:
        print("  core_substage_ms:")
        for key, value in core_present:
            print(f"    {key}: {value:.3f}")
    other = [
        (key, value)
        for key, value in stage_mean.items()
        if key not in set(top_level_keys) and key not in set(core_keys)
    ]
    other.sort(key=lambda kv: kv[1], reverse=True)
    if other:
        print("  other_top_ms:")
        for key, value in other[:8]:
            print(f"    {key}: {value:.3f}")
    for scalar_key in (
        "stap_cuda_max_memory_allocated_mb",
        "stap_cuda_max_memory_reserved_mb",
        "stap_fast_active_tile_fraction",
        "stap_fast_chunk_size_mean",
        "stap_fast_chunk_size_max",
        "stap_fast_chunk_count",
    ):
        val = _mean_scalar(steady, scalar_key)
        if val is None:
            continue
        print(f"  {scalar_key}: {val:.3f}")


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
    window_offset: int | List[int],
    baseline: str,
    baseline_support: str,
    extra_replay_args: List[str],
    env_overrides: Dict[str, str],
) -> Tuple[List[Path], List[Dict]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Avoid accumulating pw_* directories across runs (this script expects only the
    # bundles produced by the current invocation).
    for old in sorted(out_dir.glob("pw_*")):
        try:
            if old.is_dir():
                shutil.rmtree(old)
            else:
                old.unlink()
        except Exception:
            pass
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
    ]
    offsets: List[int]
    if isinstance(window_offset, list):
        offsets = [int(o) for o in window_offset]
    else:
        offsets = [int(window_offset)]
    if not offsets:
        raise ValueError("At least one window offset is required.")
    for off in offsets:
        cmd.extend(["--time-window-offset", str(int(off))])
    if extra_replay_args:
        cmd.extend(list(extra_replay_args))
    print(f"[run] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)
    bundles = _find_bundles(out_dir)
    teles = [_load_telemetry(b) for b in bundles]
    return bundles, teles


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
    ap.add_argument(
        "--stap-detector-variant",
        type=str,
        default="adaptive_guard",
        choices=["msd_ratio", "whitened_power", "unwhitened_ratio", "hybrid_rescue", "adaptive_guard"],
        help="Detector-family mode to replay/profile (default: %(default)s).",
    )
    ap.add_argument(
        "--stap-whiten-gamma",
        type=float,
        default=1.0,
        help="Whitening exponent for msd_ratio / hybrid specialist branch (default: %(default)s).",
    )
    ap.add_argument(
        "--hybrid-rescue-rule",
        type=str,
        default="guard_promote_v1",
        choices=["guard_frac_v1", "alias_rescue_v1", "band_ratio_v1", "guard_promote_v1"],
        help="Frozen routing rule used by hybrid/adaptive detector variants (default: %(default)s).",
    )
    ap.add_argument(
        "--baseline",
        type=str,
        default="mc_svd",
        choices=["svd", "mc_svd", "svd_similarity", "local_svd", "rpca", "hosvd"],
    )
    ap.add_argument("--baseline-support", type=str, default="window", choices=["window", "full"])
    ap.add_argument("--window-length", type=int, default=64)
    ap.add_argument("--window-offset", type=int, default=0)
    ap.add_argument(
        "--window-offsets",
        type=str,
        default="",
        help=(
            "Optional comma-separated list of window offsets to replay in a single process "
            "(e.g., '0,64,128'). When provided, overrides --window-offset/--steady-windows. "
            "Latency reporting treats window1 as cold and averages windows 2..N as steady-state."
        ),
    )
    ap.add_argument(
        "--steady-windows",
        type=int,
        default=1,
        help=(
            "If >1, replay the same window (at --window-offset) this many times in a single process "
            "and report steady-state latency as the mean over windows 2..N. "
            "This avoids counting CUDA-graph capture and other one-time effects in publishable numbers."
        ),
    )
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
    ap.add_argument(
        "--profile-cuda-stages",
        action="store_true",
        help=(
            "Enable CUDA-event stage profiling inside the STAP fast path and print "
            "steady-state stage, memory, and batch summaries."
        ),
    )
    ap.add_argument(
        "--profile-optimized-no-cuda-graph",
        action="store_true",
        help=(
            "Run an additional optimized-style replay with CUDA graphs disabled. "
            "Useful for exposing the internal fast-path core split that graph replay "
            "otherwise collapses into a single 'stap:core' bucket."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_src_pilot_exists(args)
    roc_fprs = [float(x) for x in str(args.roc_fprs).split(",") if x.strip()]
    roc_thresh = int(args.roc_thresholds)
    extra_replay_args = []
    if str(args.stap_conditional).strip().lower() in {"on", "1", "true", "yes"}:
        extra_replay_args.append("--stap-conditional-enable")
    else:
        extra_replay_args.append("--stap-conditional-disable")
    extra_replay_args.extend(
        [
            "--stap-detector-variant",
            str(args.stap_detector_variant),
            "--stap-whiten-gamma",
            str(float(args.stap_whiten_gamma)),
            "--hybrid-rescue-rule",
            str(args.hybrid_rescue_rule),
        ]
    )
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
        "STAP_FAST_TELEMETRY": "0",
        "STAP_LATENCY_MODE": "1",
        # Force CUDA-graph capture OFF for legacy timings, regardless of caller env.
        # (Optimized timings should measure replay/steady-state and exclude capture
        # cold effects via windows 2..N.)
        "STAP_FAST_CUDA_GRAPH": "0",
    }
    opt_env = {
        "STAP_TILING_UNFOLD": "1",
        "MC_SVD_TORCH": "1",
        "MC_SVD_REG_TORCH": "0",
        "MC_SVD_TORCH_RETURN_CUBE": "1",
        "STAP_FAST_PATH": "1",
        "STAP_FAST_PD_ONLY": "1",
        "STAP_FAST_TELEMETRY": "0",
        "STAP_LATENCY_MODE": "1",
        "STAP_FAST_CUDA_GRAPH_PAD": "1",
        # Force CUDA-graph capture ON for optimized timings, regardless of caller env.
        "STAP_FAST_CUDA_GRAPH": "1",
    }
    if int(getattr(args, "tile_batch", 0) or 0) > 0:
        tb = str(int(args.tile_batch))
        legacy_env["STAP_TILE_BATCH"] = tb
        opt_env["STAP_TILE_BATCH"] = tb
    if args.cuda_warmup_heavy:
        # Opt-in steady-state CUDA warmup for latency profiling.
        legacy_env["CUDA_WARMUP_HEAVY"] = "1"
        opt_env["CUDA_WARMUP_HEAVY"] = "1"
    if args.profile_cuda_stages:
        legacy_env["STAP_CUDA_EVENT_TIMING"] = "1"
        opt_env["STAP_CUDA_EVENT_TIMING"] = "1"

    legacy_out = args.out_root / "legacy"
    opt_out = args.out_root / "optimized"

    # Build the list of offsets to replay. When multiple offsets are used,
    # replay_stap_from_run.py emits multiple bundles in a single subprocess, so
    # CUDA-graph capture and other one-time effects are paid only once per variant.
    window_offsets: List[int]
    offsets_spec = str(getattr(args, "window_offsets", "") or "").strip()
    if offsets_spec:
        window_offsets = [int(float(s)) for s in offsets_spec.replace(";", ",").split(",") if s.strip()]
        if not window_offsets:
            raise ValueError("--window-offsets was provided but parsed as empty.")
    else:
        steady_n = int(getattr(args, "steady_windows", 1) or 1)
        steady_n = max(1, steady_n)
        window_offsets = [int(args.window_offset)] * steady_n

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

    legacy_bundles, tele_legacy_list = _run_replay(
        src=args.src,
        out_dir=legacy_out,
        profile=args.profile,
        stap_profile=args.stap_profile,
        stap_device=args.stap_device,
        stap_debug_samples=args.stap_debug_samples,
        window_length=args.window_length,
        window_offset=window_offsets,
        baseline=args.baseline,
        baseline_support=args.baseline_support,
        extra_replay_args=extra_replay_args,
        env_overrides=legacy_env,
    )
    opt_bundles, tele_opt_list = _run_replay(
        src=args.src,
        out_dir=opt_out,
        profile=args.profile,
        stap_profile=args.stap_profile,
        stap_device=args.stap_device,
        stap_debug_samples=args.stap_debug_samples,
        window_length=args.window_length,
        window_offset=window_offsets,
        baseline=args.baseline,
        baseline_support=args.baseline_support,
        extra_replay_args=extra_replay_args,
        env_overrides=opt_env,
    )

    _print_latency_windows("legacy", tele_legacy_list, t_data_ms=t_data_ms)
    _print_latency_windows("optimized", tele_opt_list, t_data_ms=t_data_ms)
    _print_cuda_profile("legacy", tele_legacy_list)
    _print_cuda_profile("optimized", tele_opt_list)

    compare_files = ["pd_base.npy", "pd_stap.npy", "score_stap.npy", "stap_score_map.npy"]
    if len(legacy_bundles) != len(opt_bundles):
        raise RuntimeError(
            f"Legacy/optimized bundle count mismatch: legacy={len(legacy_bundles)} opt={len(opt_bundles)}"
        )
    for lb, ob in zip(legacy_bundles, opt_bundles):
        if lb.name != ob.name:
            raise RuntimeError(
                f"Legacy/optimized bundle name mismatch: {lb.name} vs {ob.name}. "
                "Ensure both replays used the same window offsets."
            )
        _compare_maps(lb, ob, compare_files, rtol=float(args.rtol), atol=float(args.atol))
    print("\n[parity] OK (maps match within tolerance).")

    if args.profile_tile_statistic:
        tile_stat_env = dict(opt_env)
        tile_stat_env["STAP_FAST_TILE_STATISTIC"] = "1"
        tile_stat_out = args.out_root / "tile_statistic"
        tile_bundles, tele_tile_list = _run_replay(
            src=args.src,
            out_dir=tile_stat_out,
            profile=args.profile,
            stap_profile=args.stap_profile,
            stap_device=args.stap_device,
            stap_debug_samples=args.stap_debug_samples,
            window_length=args.window_length,
            window_offset=window_offsets,
            baseline=args.baseline,
            baseline_support=args.baseline_support,
            extra_replay_args=extra_replay_args,
            env_overrides=tile_stat_env,
        )
        _print_latency_windows("tile_statistic", tele_tile_list, t_data_ms=t_data_ms)
        _print_cuda_profile("tile_statistic", tele_tile_list)
        # Note: tile-statistic mode intentionally produces different score maps and is
        # NOT ROC-equivalent to the manuscript detector (ratio-of-means vs nonlinear
        # per-snapshot aggregation). It is kept only as a latency experiment and is
        # known to catastrophically regress strict-FPR ROC on Twinkling/Gammex, so we
        # do not enforce parity here.
        try:
            opt_bundle = opt_bundles[-1]
            tile_bundle = tile_bundles[-1]
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
        reg_bundles, tele_reg_list = _run_replay(
            src=args.src,
            out_dir=reg_torch_out,
            profile=args.profile,
            stap_profile=args.stap_profile,
            stap_device=args.stap_device,
            stap_debug_samples=args.stap_debug_samples,
            window_length=args.window_length,
            window_offset=window_offsets,
            baseline=args.baseline,
            baseline_support=args.baseline_support,
            extra_replay_args=extra_replay_args,
            env_overrides=reg_torch_env,
        )
        _print_latency_windows("baseline_reg_torch", tele_reg_list, t_data_ms=t_data_ms)
        _print_cuda_profile("baseline_reg_torch", tele_reg_list)
        # Note: torch registration can produce slightly different registered
        # cubes vs the NumPy path, so we do not enforce parity here.

    if args.profile_optimized_no_cuda_graph:
        opt_nograph_env = dict(opt_env)
        opt_nograph_env["STAP_FAST_CUDA_GRAPH"] = "0"
        opt_nograph_out = args.out_root / "optimized_nograph_profile"
        nograph_bundles, tele_nograph_list = _run_replay(
            src=args.src,
            out_dir=opt_nograph_out,
            profile=args.profile,
            stap_profile=args.stap_profile,
            stap_device=args.stap_device,
            stap_debug_samples=args.stap_debug_samples,
            window_length=args.window_length,
            window_offset=window_offsets,
            baseline=args.baseline,
            baseline_support=args.baseline_support,
            extra_replay_args=extra_replay_args,
            env_overrides=opt_nograph_env,
        )
        _print_latency_windows("optimized_nograph_profile", tele_nograph_list, t_data_ms=t_data_ms)
        _print_cuda_profile("optimized_nograph_profile", tele_nograph_list)
        if len(opt_bundles) != len(nograph_bundles):
            raise RuntimeError(
                "Optimized vs optimized_nograph_profile bundle count mismatch: "
                f"{len(opt_bundles)} vs {len(nograph_bundles)}"
            )
        for ob, nb in zip(opt_bundles, nograph_bundles):
            if ob.name != nb.name:
                raise RuntimeError(
                    f"Optimized/optimized_nograph_profile bundle name mismatch: {ob.name} vs {nb.name}"
                )
            _compare_maps(ob, nb, compare_files, rtol=float(args.rtol), atol=float(args.atol))
        print("\n[parity] OK (optimized vs optimized_nograph_profile maps match within tolerance).")


if __name__ == "__main__":
    main()
