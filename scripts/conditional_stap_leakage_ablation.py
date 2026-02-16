#!/usr/bin/env python3
"""Conditional STAP leakage ablation (Phase 2).

This script implements the reviewer-requested conditional-execution ablation:

  (i)  Full STAP (conditional execution disabled)
  (ii) Conditional STAP with a flow mask derived from the *same* window
  (iii) Conditional STAP with a flow mask derived from a *disjoint* window
        (implemented by running a cheap, "mask-only" replay on the disjoint
        window and reusing its exported mask_flow.npy)
  (iv) Conditional STAP with a *random* mask of matched tile coverage

All four conditions use the same evaluation masks (mask_flow.npy / mask_bg.npy)
for the target window; only the conditional-execution mask varies.

Outputs:
  - Per-pilot per-condition TPR@FPR metrics (CSV + JSON).
  - Bootstrap CIs (paired, over pilots) on ΔTPR relative to full STAP.

Example:
  PYTHONPATH=. python scripts/conditional_stap_leakage_ablation.py \
    --pilots runs/pilot/r4_kwave_seed1 runs/pilot/r4_kwave_seed2 \
    --profile Brain-OpenSkull \
    --out-root runs/ablation/condstap_leakage \
    --window-length 64 --window-offset 0 --disjoint-offset 64 \
    --summary-csv reports/condstap_leakage.csv \
    --summary-json reports/condstap_leakage.json
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


REPO = Path(__file__).resolve().parents[1]
REPLAY = REPO / "scripts" / "replay_stap_from_run.py"


@dataclass(frozen=True)
class Condition:
    key: str
    label: str


CONDITIONS: Tuple[Condition, ...] = (
    Condition("full", "full_stap"),
    Condition("same", "cond_same_window"),
    Condition("disjoint", "cond_disjoint_window"),
    Condition("random", "cond_random_mask"),
)


def _float_list(values: Sequence[str]) -> List[float]:
    out: List[float] = []
    for s in values:
        s = str(s).strip()
        if not s:
            continue
        out.append(float(s))
    return out


def _bundle_dir_from_out(out_dir: Path) -> Path:
    bundles = sorted(out_dir.glob("pw_*"))
    if not bundles:
        raise FileNotFoundError(f"No pw_* bundle directory found under {out_dir}")
    if len(bundles) > 1:
        raise RuntimeError(f"Expected a single pw_* bundle under {out_dir}, found {len(bundles)}")
    return bundles[0]


def _load_scores(bundle_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask_flow = np.load(bundle_dir / "mask_flow.npy").astype(bool)
    mask_bg = np.load(bundle_dir / "mask_bg.npy").astype(bool)
    score_path = bundle_dir / "score_pd_stap.npy"
    if score_path.exists():
        score = np.load(score_path).astype(np.float64, copy=False)
    else:
        pd = np.load(bundle_dir / "pd_stap.npy").astype(np.float64, copy=False)
        score = -pd
    return score, mask_flow, mask_bg


def _tpr_at_fpr(scores_pos: np.ndarray, scores_neg: np.ndarray, fpr: float) -> Tuple[float, float]:
    """Return (threshold, TPR) for right-tail scores at the requested FPR."""
    if not (0.0 < fpr < 1.0):
        raise ValueError(f"fpr must be in (0,1), got {fpr}")
    neg = np.asarray(scores_neg, dtype=np.float64).ravel()
    pos = np.asarray(scores_pos, dtype=np.float64).ravel()
    neg = neg[np.isfinite(neg)]
    pos = pos[np.isfinite(pos)]
    if neg.size == 0 or pos.size == 0:
        return float("nan"), float("nan")
    thr = float(np.quantile(neg, 1.0 - fpr, method="linear"))
    tpr = float(np.mean(pos >= thr))
    return thr, tpr


def _tile_any_fraction(mask: np.ndarray, tile_hw: Tuple[int, int], stride: int) -> float:
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    th, tw = tile_hw
    if H < th or W < tw:
        return 0.0
    total = 0
    hits = 0
    for y0 in range(0, H - th + 1, stride):
        for x0 in range(0, W - tw + 1, stride):
            total += 1
            if mask[y0 : y0 + th, x0 : x0 + tw].any():
                hits += 1
    return float(hits / total) if total else 0.0


def _random_mask_match_tile_any(
    shape: Tuple[int, int],
    tile_hw: Tuple[int, int],
    stride: int,
    target: float,
    *,
    seed: int,
    max_iters: int = 22,
) -> np.ndarray:
    """Generate a random mask whose tile-any fraction matches target (best-effort)."""
    H, W = shape
    rng = np.random.default_rng(seed)
    u = rng.random((H, W), dtype=np.float32)

    # Binary search over p so that (u < p) yields the desired tile-any fraction.
    lo = 0.0
    hi = 1.0
    best = None
    best_err = float("inf")
    for _ in range(max_iters):
        mid = 0.5 * (lo + hi)
        m = u < mid
        frac = _tile_any_fraction(m, tile_hw, stride)
        err = abs(frac - target)
        if err < best_err:
            best_err = err
            best = m
        if frac < target:
            lo = mid
        else:
            hi = mid
    if best is None:
        best = u < 0.0
    return best.astype(bool, copy=False)


def _run_replay(cmd: List[str], *, dry_run: bool) -> None:
    print("[condstap] $ " + " ".join(str(c) for c in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(REPO))


def _pilot_geom_shape(pilot_dir: Path) -> Tuple[int, int]:
    meta_path = pilot_dir / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    geom = meta.get("geometry") or {}
    ny = int(geom.get("Ny"))
    nx = int(geom.get("Nx"))
    return (ny, nx)


def _pilot_seed(pilot_dir: Path) -> int:
    meta_path = pilot_dir / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    return int(meta.get("seed") or 0)


def _bootstrap_ci_mean(values: np.ndarray, *, seed: int, n_boot: int = 2000) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = np.mean(values[idx], axis=1)
    lo = float(np.quantile(means, 0.025))
    hi = float(np.quantile(means, 0.975))
    return lo, hi


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Conditional STAP leakage ablation runner.")
    ap.add_argument(
        "--pilots",
        nargs="+",
        type=Path,
        required=True,
        help="Pilot run directories (each containing meta.json + ens*_angle_* dirs).",
    )
    ap.add_argument(
        "--profile",
        type=str,
        default="Brain-OpenSkull",
        choices=["Brain-OpenSkull", "Brain-AliasContract", "Brain-SkullOR", "Brain-Pial128"],
        help="Brain-* operating profile to use for the replay runs.",
    )
    ap.add_argument("--out-root", type=Path, required=True, help="Root directory for ablation outputs.")
    ap.add_argument("--window-length", type=int, default=64, help="Replay window length.")
    ap.add_argument("--window-offset", type=int, default=0, help="Replay window offset.")
    ap.add_argument(
        "--disjoint-offset",
        type=int,
        default=64,
        help="Disjoint window offset used to derive the conditional mask.",
    )
    ap.add_argument(
        "--fprs",
        nargs="+",
        type=str,
        default=["1e-4", "3e-4", "1e-3"],
        help="FPR targets (space-separated).",
    )
    ap.add_argument("--stap-device", type=str, default="cuda", help="STAP device passed to replay.")
    ap.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("reports/condstap_leakage.csv"),
        help="Write per-pilot metrics to CSV.",
    )
    ap.add_argument(
        "--summary-json",
        type=Path,
        default=Path("reports/condstap_leakage.json"),
        help="Write summary JSON.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    ap.add_argument("--bootstrap-seed", type=int, default=1337, help="Seed for bootstrap CIs.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not REPLAY.exists():
        raise FileNotFoundError(f"Missing {REPLAY}")

    fprs = _float_list(args.fprs)
    for f in fprs:
        if not (0.0 < f < 1.0):
            raise ValueError(f"Invalid FPR {f}; must be in (0,1).")

    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    per_pilot_rows: List[Dict[str, Any]] = []
    per_condition_tpr: Dict[str, Dict[float, List[float]]] = {
        cond.key: {fpr: [] for fpr in fprs} for cond in CONDITIONS
    }

    for pilot_dir in args.pilots:
        pilot_dir = pilot_dir.resolve()
        pilot_name = pilot_dir.name
        pilot_seed = _pilot_seed(pilot_dir)
        shape = _pilot_geom_shape(pilot_dir)

        # Condition output roots.
        out_full = out_root / f"{pilot_name}_full"
        out_same = out_root / f"{pilot_name}_same"
        out_disjoint_mask = out_root / f"{pilot_name}_disjoint_mask"
        out_disjoint = out_root / f"{pilot_name}_disjoint"
        out_random = out_root / f"{pilot_name}_random"

        # 1) Full STAP (conditional disabled)
        cmd_full = [
            sys.executable,
            str(REPLAY),
            "--src",
            str(pilot_dir),
            "--out",
            str(out_full),
            "--profile",
            str(args.profile),
            "--stap-device",
            str(args.stap_device),
            "--time-window-length",
            str(args.window_length),
            "--time-window-offset",
            str(args.window_offset),
            # Evaluation masks are simulator truth; conditional masks use the
            # PD-derived proxy mask exported as mask_flow_pd.npy.
            "--flow-mask-mode",
            "default",
            "--stap-conditional-disable",
            "--stap-conditional-mask-tag",
            "full",
        ]
        if args.dry_run:
            _run_replay(cmd_full, dry_run=True)
        else:
            _run_replay(cmd_full, dry_run=False)
        if args.dry_run:
            # Still print the remaining commands, but do not attempt to load bundles.
            bundle_full = None
        else:
            bundle_full = _bundle_dir_from_out(out_full)

        if bundle_full is not None:
            # Load evaluation mask + compute target tile coverage for random mask.
            mask_eval = np.load(bundle_full / "mask_flow.npy").astype(bool)
            # Match coverage to the PD-derived proxy mask actually used for
            # conditional execution (not the sim-truth evaluation mask).
            mask_proxy = np.load(bundle_full / "mask_flow_pd.npy").astype(bool)
            with open(bundle_full / "meta.json") as f:
                meta_full = json.load(f)
            tile_hw_raw = meta_full.get("tile_hw") or [8, 8]
            tile_hw = (int(tile_hw_raw[0]), int(tile_hw_raw[1]))
            tile_stride = int(meta_full.get("tile_stride") or 3)
            target_tile_any = _tile_any_fraction(mask_proxy, tile_hw, tile_stride)
        else:
            tile_hw = (8, 8)
            tile_stride = 3
            target_tile_any = 0.0

        # 2) Conditional (same window; default conditional mask == evaluation mask)
        cmd_same = [
            sys.executable,
            str(REPLAY),
            "--src",
            str(pilot_dir),
            "--out",
            str(out_same),
            "--profile",
            str(args.profile),
            "--stap-device",
            str(args.stap_device),
            "--time-window-length",
            str(args.window_length),
            "--time-window-offset",
            str(args.window_offset),
            "--flow-mask-mode",
            "default",
            "--stap-conditional-mask-tag",
            "same_window",
        ]
        _run_replay(cmd_same, dry_run=args.dry_run)
        bundle_same = None if args.dry_run else _bundle_dir_from_out(out_same)

        # 3) Disjoint-window mask bundle (cheap "mask-only" run with all-false conditional mask)
        out_disjoint_mask.mkdir(parents=True, exist_ok=True)
        false_mask_path = out_disjoint_mask / "mask_all_false.npy"
        if not false_mask_path.exists():
            np.save(false_mask_path, np.zeros(shape, dtype=np.bool_))
        cmd_disjoint_mask = [
            sys.executable,
            str(REPLAY),
            "--src",
            str(pilot_dir),
            "--out",
            str(out_disjoint_mask),
            "--profile",
            str(args.profile),
            "--stap-device",
            str(args.stap_device),
            "--time-window-length",
            str(args.window_length),
            "--time-window-offset",
            str(args.disjoint_offset),
            "--flow-mask-mode",
            "default",
            "--stap-conditional-mask",
            str(false_mask_path),
            "--stap-conditional-mask-tag",
            "maskonly_allfalse",
        ]
        _run_replay(cmd_disjoint_mask, dry_run=args.dry_run)
        if args.dry_run:
            disjoint_mask_path = false_mask_path
        else:
            bundle_disjoint_mask = _bundle_dir_from_out(out_disjoint_mask)
            disjoint_mask_path = bundle_disjoint_mask / "mask_flow_pd.npy"
            if not disjoint_mask_path.exists():
                raise FileNotFoundError(disjoint_mask_path)

        # Conditional (disjoint window mask)
        cmd_disjoint = [
            sys.executable,
            str(REPLAY),
            "--src",
            str(pilot_dir),
            "--out",
            str(out_disjoint),
            "--profile",
            str(args.profile),
            "--stap-device",
            str(args.stap_device),
            "--time-window-length",
            str(args.window_length),
            "--time-window-offset",
            str(args.window_offset),
            "--flow-mask-mode",
            "default",
            "--stap-conditional-mask",
            str(disjoint_mask_path),
            "--stap-conditional-mask-tag",
            f"disjoint_off{int(args.disjoint_offset)}",
        ]
        _run_replay(cmd_disjoint, dry_run=args.dry_run)
        bundle_disjoint = None if args.dry_run else _bundle_dir_from_out(out_disjoint)

        # 4) Conditional (random mask with matched tile coverage)
        out_random.mkdir(parents=True, exist_ok=True)
        random_mask_path = out_random / "mask_random_match_tilecov.npy"
        if not random_mask_path.exists():
            random_mask = _random_mask_match_tile_any(
                shape,
                tile_hw,
                tile_stride,
                target_tile_any,
                seed=pilot_seed + 911,
            )
            np.save(random_mask_path, random_mask.astype(np.bool_, copy=False))
        cmd_random = [
            sys.executable,
            str(REPLAY),
            "--src",
            str(pilot_dir),
            "--out",
            str(out_random),
            "--profile",
            str(args.profile),
            "--stap-device",
            str(args.stap_device),
            "--time-window-length",
            str(args.window_length),
            "--time-window-offset",
            str(args.window_offset),
            "--flow-mask-mode",
            "default",
            "--stap-conditional-mask",
            str(random_mask_path),
            "--stap-conditional-mask-tag",
            "random_match_tilecov",
        ]
        _run_replay(cmd_random, dry_run=args.dry_run)
        bundle_random = None if args.dry_run else _bundle_dir_from_out(out_random)

        if args.dry_run:
            continue

        bundles = {
            "full": bundle_full,
            "same": bundle_same,
            "disjoint": bundle_disjoint,
            "random": bundle_random,
        }

        # Verify evaluation masks are fixed across conditions.
        mask_flow_ref = np.load(bundle_full / "mask_flow.npy").astype(bool)
        mask_bg_ref = np.load(bundle_full / "mask_bg.npy").astype(bool)
        mask_mismatch: Dict[str, int] = {}
        for key, bdir in bundles.items():
            mf = np.load(bdir / "mask_flow.npy").astype(bool)
            mb = np.load(bdir / "mask_bg.npy").astype(bool)
            mismatch = int(np.count_nonzero(mf != mask_flow_ref) + np.count_nonzero(mb != mask_bg_ref))
            mask_mismatch[key] = mismatch

        # Compute TPR@FPR per condition.
        for cond in CONDITIONS:
            bdir = bundles[cond.key]
            score, mf, mb = _load_scores(bdir)
            pos = score[mf]
            neg = score[mb]

            entry: Dict[str, Any] = {
                "pilot": pilot_name,
                "pilot_seed": int(pilot_seed),
                "profile": str(args.profile),
                "window_offset": int(args.window_offset),
                "window_length": int(args.window_length),
                "disjoint_offset": int(args.disjoint_offset),
                "condition": cond.key,
                "condition_label": cond.label,
                "eval_mask_mismatch_pixels": int(mask_mismatch.get(cond.key, 0)),
            }
            for fpr in fprs:
                thr, tpr = _tpr_at_fpr(pos, neg, fpr)
                entry[f"tpr@{fpr:g}"] = tpr
                entry[f"thr@{fpr:g}"] = thr
                per_condition_tpr[cond.key][fpr].append(tpr)
            per_pilot_rows.append(entry)

    if args.dry_run:
        print("[condstap] Dry run complete (no outputs written).", flush=True)
        return

    # Write per-pilot metrics CSV.
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = [
        "pilot",
        "pilot_seed",
        "profile",
        "window_offset",
        "window_length",
        "disjoint_offset",
        "condition",
        "condition_label",
        "eval_mask_mismatch_pixels",
    ]
    for fpr in fprs:
        fieldnames.extend([f"tpr@{fpr:g}", f"thr@{fpr:g}"])
    with open(args.summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in per_pilot_rows:
            w.writerow(row)

    # Aggregate summary: paired deltas vs full STAP.
    summary: Dict[str, Any] = {
        "profile": str(args.profile),
        "window_offset": int(args.window_offset),
        "window_length": int(args.window_length),
        "disjoint_offset": int(args.disjoint_offset),
        "fprs": fprs,
        "n_pilots": len(set(r["pilot"] for r in per_pilot_rows)),
        "conditions": [c.key for c in CONDITIONS],
        "delta_tpr_vs_full": {},
    }
    for fpr in fprs:
        # Reconstruct paired order by iterating pilots as they appeared.
        full_vals = np.array(per_condition_tpr["full"][fpr], dtype=np.float64)
        for cond in ("same", "disjoint", "random"):
            vals = np.array(per_condition_tpr[cond][fpr], dtype=np.float64)
            if vals.size != full_vals.size:
                continue
            delta = vals - full_vals
            lo, hi = _bootstrap_ci_mean(delta, seed=args.bootstrap_seed + int(1e6 * fpr))
            summary["delta_tpr_vs_full"].setdefault(cond, {})[f"{fpr:g}"] = {
                "mean": float(np.mean(delta)),
                "median": float(np.median(delta)),
                "ci95_mean": [lo, hi],
            }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[condstap] Wrote {args.summary_csv} and {args.summary_json}", flush=True)


if __name__ == "__main__":
    main()
