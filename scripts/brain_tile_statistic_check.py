#!/usr/bin/env python3
"""
Multi-window ROC + latency check for cov-only (tile-statistic) STAP scoring.

WARNING: cov-only / tile-statistic scoring is NOT ROC-equivalent to the
manuscript detector and is known to catastrophically regress strict-FPR ROC on
Twinkling/Gammex. It replaces per-snapshot nonlinear aggregation with a
ratio-of-means approximation (mean(f) != f(mean)). Do not use for paper results
or production configs.

This script is intended only as an exploratory diagnostic before deeper latency
work (e.g., Tyler covariance changes):
  - generate (or reuse) a Brain-* pilot with multiple ensembles (T_total = 5*64)
  - replay STAP over 5 disjoint 64-frame windows with:
      (a) snapshot scoring (default)
      (b) tile-statistic scoring (STAP_FAST_TILE_STATISTIC=1)
  - report per-window TPR@{1e-4,3e-4,1e-3} for the vNext STAP detector
    (score_stap_preka.npy), plus CUDA/telemetry latency fields when present.

Example:
  PYTHONPATH=. conda run -n fus-detectors python scripts/brain_tile_statistic_check.py \
    --profile Brain-AliasContract --device cuda --synthetic \
    --out-root runs/_tile_stat_check_alias
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class WindowResult:
    offset: int
    n_neg: int
    n_pos: int
    tpr_at: Dict[float, float]
    realized_fpr_at: Dict[float, float]
    tau_at: Dict[float, float]
    telemetry: Dict[str, object]


def _float_list(spec: str) -> List[float]:
    parts = [p.strip() for p in str(spec).replace(";", ",").split(",")]
    return [float(p) for p in parts if p]


def _int_list(spec: str) -> List[int]:
    parts = [p.strip() for p in str(spec).replace(";", ",").split(",")]
    return [int(float(p)) for p in parts if p]


def _tpr_at_fpr(scores_pos: np.ndarray, scores_neg: np.ndarray, fpr: float) -> Tuple[float, float, float]:
    """
    Right-tail thresholding: choose tau on neg so that P(neg >= tau) ~= fpr.

    Returns (tau, tpr, realized_fpr).
    """
    if not (0.0 < float(fpr) < 1.0):
        raise ValueError(f"fpr must be in (0,1), got {fpr}")
    neg = np.asarray(scores_neg, dtype=np.float64).ravel()
    pos = np.asarray(scores_pos, dtype=np.float64).ravel()
    if neg.size == 0 or pos.size == 0:
        return float("nan"), float("nan"), float("nan")

    neg_sorted = np.sort(neg)
    n_neg = int(neg_sorted.size)
    q = 1.0 - float(fpr)
    idx = int(np.floor(q * n_neg))
    idx = max(0, min(idx, n_neg - 1))
    tau = float(neg_sorted[idx])
    tpr = float((pos >= tau).mean())
    realized = float((neg >= tau).mean())
    return tau, tpr, realized


def _load_score_pools(bundle_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    score_path = bundle_dir / "score_stap_preka.npy"
    if not score_path.exists():
        score_path = bundle_dir / "score_stap.npy"
    if not score_path.exists():
        raise FileNotFoundError(f"Missing score_stap_preka.npy/score_stap.npy under {bundle_dir}")
    score = np.load(score_path).astype(np.float64, copy=False)
    flow = np.load(bundle_dir / "mask_flow.npy").astype(bool, copy=False)
    bg = np.load(bundle_dir / "mask_bg.npy").astype(bool, copy=False)
    if score.shape != flow.shape or score.shape != bg.shape:
        raise ValueError(
            f"Shape mismatch: score={score.shape} flow={flow.shape} bg={bg.shape} in {bundle_dir}"
        )
    return score[flow].ravel(), score[bg].ravel()


def _median_iqr(values: List[float]) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.median(arr)), float(np.quantile(arr, 0.25)), float(np.quantile(arr, 0.75))


def _find_single_bundle(out_dir: Path) -> Path:
    bundles = sorted(out_dir.glob("pw_*"))
    if not bundles:
        raise FileNotFoundError(f"No pw_* bundle produced under {out_dir}")
    if len(bundles) != 1:
        raise RuntimeError(f"Expected one pw_* bundle under {out_dir}, found {len(bundles)}")
    return bundles[0]


def _load_telemetry(bundle_dir: Path) -> Dict[str, object]:
    meta = json.loads((bundle_dir / "meta.json").read_text())
    tele = (meta.get("stap_fallback_telemetry") or {}) if isinstance(meta, dict) else {}
    if not isinstance(tele, dict):
        tele = {}
    return tele


def _run(cmd: List[str], *, env: Dict[str, str] | None = None) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def maybe_generate_pilot(
    *,
    out_dir: Path,
    profile: str,
    synthetic: bool,
    seed: int,
    nx: int,
    ny: int,
    prf_hz: float,
    pulses: int,
    ensembles: int,
    angles: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "meta.json").exists():
        return out_dir

    # pilot_motion.py declares --angles with nargs="+", and argparse treats
    # comma-joined negative lists (e.g. "-12,-6,0,6,12") as a new option token.
    # Pass each angle as a separate argv token.
    angle_tokens = [a.strip() for a in str(angles).replace(";", ",").split(",") if a.strip()]
    if not angle_tokens:
        raise ValueError("No angles provided after parsing --angles.")

    cmd = [
        sys.executable,
        str(REPO / "sim" / "kwave" / "pilot_motion.py"),
        "--out",
        str(out_dir),
        "--profile",
        str(profile),
        "--seed",
        str(int(seed)),
        "--Nx",
        str(int(nx)),
        "--Ny",
        str(int(ny)),
        "--prf",
        str(float(prf_hz)),
        "--pulses",
        str(int(pulses)),
        "--ensembles",
        str(int(ensembles)),
        "--angles",
        *angle_tokens,
        "--baseline-type",
        "mc_svd",
        "--score-mode",
        "pd",
        "--ka-mode",
        "none",
        # Fixed-profile STAP knobs for pilot acceptance outputs (replay uses its own config).
        "--tile-h",
        "8",
        "--tile-w",
        "8",
        "--tile-stride",
        "3",
        "--lt",
        "8",
        "--cov-estimator",
        "tyler_pca",
        "--diag-load",
        "0.07",
    ]
    if synthetic:
        cmd.append("--synthetic")
    _run(cmd)
    return out_dir


def replay_window(
    *,
    src: Path,
    out_dir: Path,
    profile: str,
    device: str,
    window_length: int,
    window_offset: int,
    tile_statistic: bool,
    replay_extra: List[str],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("STAP_FAST_PATH", "1")
    env.setdefault("STAP_FAST_PD_ONLY", "1")
    env.setdefault("STAP_TILING_UNFOLD", "1")
    if tile_statistic:
        env["STAP_FAST_TILE_STATISTIC"] = "1"
    else:
        env.pop("STAP_FAST_TILE_STATISTIC", None)

    cmd = [
        sys.executable,
        str(REPO / "scripts" / "replay_stap_from_run.py"),
        "--src",
        str(src),
        "--out",
        str(out_dir),
        "--stap-profile",
        "clinical",
        "--profile",
        str(profile),
        "--stap-device",
        str(device),
        "--stap-debug-samples",
        "0",
        "--baseline",
        "mc_svd",
        "--baseline-support",
        "window",
        "--time-window-length",
        str(int(window_length)),
        "--time-window-offset",
        str(int(window_offset)),
        "--stap-conditional-disable",
        # Fixed-profile STAP knobs.
        "--tile-h",
        "8",
        "--tile-w",
        "8",
        "--tile-stride",
        "3",
        "--lt",
        "8",
        "--cov-estimator",
        "tyler_pca",
        "--diag-load",
        "0.07",
        "--score-mode",
        "pd",
    ]
    cmd.extend(list(replay_extra or []))
    _run(cmd, env=env)
    return _find_single_bundle(out_dir)


def eval_bundle(bundle_dir: Path, fprs: Iterable[float], *, offset: int) -> WindowResult:
    pos, neg = _load_score_pools(bundle_dir)
    tpr_at: Dict[float, float] = {}
    realized_at: Dict[float, float] = {}
    tau_at: Dict[float, float] = {}
    for a in fprs:
        tau, tpr, realized = _tpr_at_fpr(pos, neg, float(a))
        tpr_at[float(a)] = float(tpr)
        realized_at[float(a)] = float(realized)
        tau_at[float(a)] = float(tau)
    return WindowResult(
        offset=int(offset),
        n_neg=int(neg.size),
        n_pos=int(pos.size),
        tpr_at=tpr_at,
        realized_fpr_at=realized_at,
        tau_at=tau_at,
        telemetry=_load_telemetry(bundle_dir),
    )


def print_summary(label: str, results: List[WindowResult], fprs: List[float]) -> None:
    print(f"\n[{label}] windows={len(results)}")
    if not results:
        return
    print(f"  n_neg={results[0].n_neg} (fpr_min≈{1.0/max(1,results[0].n_neg):.3g}) n_pos={results[0].n_pos}")
    for a in fprs:
        vals = [r.tpr_at[a] for r in results]
        med, q1, q3 = _median_iqr(vals)
        print(f"  TPR@{a:g}: median {med:.4f} IQR [{q1:.4f},{q3:.4f}] perwin {[round(v,6) for v in vals]}")

    # Latency (if present).
    key = "stap_total_ms"
    if all(key in (r.telemetry or {}) for r in results):
        vals = [float(r.telemetry[key]) for r in results]  # type: ignore[index]
        med, q1, q3 = _median_iqr(vals)
        print(f"  {key}: median {med:.3f} ms IQR [{q1:.3f},{q3:.3f}] perwin {[round(v,3) for v in vals]}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "EXPERIMENTAL: Tile-statistic STAP ROC/latency check (Brain-*). "
            "Known to regress strict-FPR ROC on Twinkling/Gammex; do not use for manuscript baselines."
        )
    )
    ap.add_argument(
        "--profile",
        type=str,
        required=True,
        choices=["Brain-OpenSkull", "Brain-AliasContract", "Brain-SkullOR", "Brain-Pial128"],
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--nx", type=int, default=240)
    ap.add_argument("--ny", type=int, default=240)
    ap.add_argument("--prf-hz", type=float, default=1500.0)
    ap.add_argument("--pulses", type=int, default=64)
    ap.add_argument("--ensembles", type=int, default=5)
    ap.add_argument("--angles", type=str, default="-12,-6,0,6,12")
    ap.add_argument("--window-length", type=int, default=64)
    ap.add_argument("--window-offsets", type=str, default="0,64,128,192,256")
    ap.add_argument("--fprs", type=str, default="1e-4,3e-4,1e-3")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument(
        "--src",
        type=Path,
        default=None,
        help="Optional existing pilot run root (ens*_angle_* dirs + meta.json). If omitted, generates one.",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output root for replay bundles (snapshot vs tile_statistic).",
    )
    ap.add_argument(
        "--replay-extra",
        type=str,
        default="",
        help="Extra args passed through to replay_stap_from_run.py (single quoted string).",
    )
    args = ap.parse_args()

    fprs = _float_list(args.fprs)
    offsets = _int_list(args.window_offsets)
    replay_extra = shlex.split(str(args.replay_extra or "").strip())

    if args.src is None:
        pilot_dir = args.out_root / "_pilot"
        src = maybe_generate_pilot(
            out_dir=pilot_dir,
            profile=str(args.profile),
            synthetic=bool(args.synthetic),
            seed=int(args.seed),
            nx=int(args.nx),
            ny=int(args.ny),
            prf_hz=float(args.prf_hz),
            pulses=int(args.pulses),
            ensembles=int(args.ensembles),
            angles=str(args.angles),
        )
    else:
        src = Path(args.src)

    snapshot_results: List[WindowResult] = []
    tile_results: List[WindowResult] = []
    for off in offsets:
        snap_bundle = replay_window(
            src=src,
            out_dir=args.out_root / "snapshot" / f"off{off}",
            profile=str(args.profile),
            device=str(args.device),
            window_length=int(args.window_length),
            window_offset=int(off),
            tile_statistic=False,
            replay_extra=replay_extra,
        )
        snapshot_results.append(eval_bundle(snap_bundle, fprs, offset=int(off)))

        tile_bundle = replay_window(
            src=src,
            out_dir=args.out_root / "tile_statistic" / f"off{off}",
            profile=str(args.profile),
            device=str(args.device),
            window_length=int(args.window_length),
            window_offset=int(off),
            tile_statistic=True,
            replay_extra=replay_extra,
        )
        tile_results.append(eval_bundle(tile_bundle, fprs, offset=int(off)))

    print_summary("snapshot", snapshot_results, fprs)
    print_summary("tile_statistic", tile_results, fprs)

    print("\n[delta] tile_statistic - snapshot")
    for a in fprs:
        ds = [tile_results[i].tpr_at[a] - snapshot_results[i].tpr_at[a] for i in range(len(offsets))]
        med, q1, q3 = _median_iqr(ds)
        print(f"  ΔTPR@{a:g}: median {med:+.4f} IQR [{q1:+.4f},{q3:+.4f}] perwin {[round(v,6) for v in ds]}")


if __name__ == "__main__":
    main()
