#!/usr/bin/env python3
"""Brain-* MC--SVD(e) baseline sweep (Phase 3).

Purpose
-------
Reviewer feedback flagged that MC--SVD/RPCA/HOSVD baselines look "too weak" at
ultra-low FPR, suggesting the baseline might be under-tuned. This script performs
a *tune-once-then-freeze* sweep of the MC--SVD energy-fraction rule on a single,
pre-committed Brain-* calibration configuration.

We run baseline-only replays by forcing conditional STAP to skip all tiles using
an all-false conditional mask. This keeps runtime low and isolates the baseline.

Outputs
-------
- A per-candidate CSV with TPR@FPR metrics using the exported PD-mode score map
  (`score_pd_base.npy` when present, else `-pd_base.npy`).
- A JSON summary that records the calibration configuration and the chosen
  "best-effort" energy fraction under a single objective.

Example
-------
PYTHONPATH=. python scripts/brain_mcsvd_energy_sweep.py \
  --pilot runs/pilot/r4_kwave_seed1 \
  --profile Brain-OpenSkull \
  --out-root runs/sweep/mcsvd_energy_brain_seed1 \
  --window-length 64 --window-offset 0 \
  --energy-fracs 0.90,0.95,0.97,0.975,0.98,0.99 \
  --fprs 1e-4,3e-4,1e-3 \
  --summary-csv reports/brain_mcsvd_energy_sweep.csv \
  --summary-json reports/brain_mcsvd_energy_sweep.json
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


REPO = Path(__file__).resolve().parents[1]
REPLAY = REPO / "scripts" / "replay_stap_from_run.py"


@dataclass(frozen=True)
class SweepRow:
    energy_frac: float
    out_dir: Path
    bundle_dir: Path
    svd_rank_removed: int | None
    svd_energy_removed_frac: float | None
    baseline_ms: float | None
    tpr_by_fpr: Dict[float, float]
    thr_by_fpr: Dict[float, float]
    objective: float


def _float_list_csv(s: str) -> List[float]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    return [float(p) for p in parts]


def _bundle_dir_from_out(out_dir: Path) -> Path:
    bundles = sorted(out_dir.glob("pw_*"))
    if not bundles:
        raise FileNotFoundError(f"No pw_* bundle directory found under {out_dir}")
    if len(bundles) > 1:
        raise RuntimeError(f"Expected a single pw_* bundle under {out_dir}, found {len(bundles)}")
    return bundles[0]


def _load_score_base(bundle_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load right-tail PD-mode score + flow/bg masks from a bundle."""
    mask_flow = np.load(bundle_dir / "mask_flow.npy").astype(bool)
    mask_bg = np.load(bundle_dir / "mask_bg.npy").astype(bool)
    score_path = bundle_dir / "score_pd_base.npy"
    if score_path.exists():
        score = np.load(score_path).astype(np.float64, copy=False)
    else:
        pd = np.load(bundle_dir / "pd_base.npy").astype(np.float64, copy=False)
        score = -pd
    return score, mask_flow, mask_bg


def _tpr_at_fpr(scores_pos: np.ndarray, scores_neg: np.ndarray, fpr: float) -> Tuple[float, float]:
    """Return (threshold, TPR) for right-tail scores at the requested FPR."""
    neg = np.asarray(scores_neg, dtype=np.float64).ravel()
    pos = np.asarray(scores_pos, dtype=np.float64).ravel()
    neg = neg[np.isfinite(neg)]
    pos = pos[np.isfinite(pos)]
    if neg.size == 0 or pos.size == 0:
        return float("nan"), float("nan")
    thr = float(np.quantile(neg, 1.0 - float(fpr), method="linear"))
    tpr = float(np.mean(pos >= thr))
    return thr, tpr


def _objective_from_tprs(tpr_by_fpr: Dict[float, float], fprs: Sequence[float]) -> float:
    vals = [float(tpr_by_fpr.get(float(f), float("nan"))) for f in fprs]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(np.sum(vals))


def _pilot_seed(pilot_dir: Path) -> int:
    meta_path = pilot_dir / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    return int(meta.get("seed") or 0)


def _pilot_geom_shape(pilot_dir: Path) -> Tuple[int, int]:
    meta_path = pilot_dir / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    geom = meta.get("geometry") or meta.get("sim_geom") or {}
    ny = int(geom.get("Ny") or geom.get("ny") or 0)
    nx = int(geom.get("Nx") or geom.get("nx") or 0)
    if ny <= 0 or nx <= 0:
        raise RuntimeError("Could not infer (Ny,Nx) from pilot meta.json geometry/sim_geom")
    return (ny, nx)


def _run(cmd: List[str], *, dry_run: bool) -> None:
    print("[mcsvd_sweep] $ " + " ".join(str(c) for c in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(REPO))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Brain-* MC--SVD(e) baseline sweep.")
    ap.add_argument("--pilot", type=Path, required=True, help="Pilot run directory under runs/pilot/...")
    ap.add_argument(
        "--profile",
        type=str,
        default="Brain-OpenSkull",
        choices=["Brain-OpenSkull", "Brain-AliasContract", "Brain-SkullOR", "Brain-Pial128"],
        help="Brain-* profile to use for the replay run.",
    )
    ap.add_argument("--out-root", type=Path, required=True, help="Root directory for sweep outputs.")
    ap.add_argument("--window-length", type=int, default=64, help="Replay window length.")
    ap.add_argument("--window-offset", type=int, default=0, help="Replay window offset.")
    ap.add_argument(
        "--energy-fracs",
        type=str,
        default="0.90,0.95,0.97,0.975,0.98,0.99",
        help="Comma-separated MC--SVD energy fractions to sweep.",
    )
    ap.add_argument(
        "--fprs",
        type=str,
        default="1e-4,3e-4,1e-3",
        help="Comma-separated FPR targets (right-tail score).",
    )
    ap.add_argument("--stap-device", type=str, default="cpu", help="STAP device passed to replay.")
    ap.add_argument("--summary-csv", type=Path, required=True, help="Write per-candidate metrics to CSV.")
    ap.add_argument("--summary-json", type=Path, required=True, help="Write summary JSON.")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not REPLAY.exists():
        raise FileNotFoundError(f"Missing {REPLAY}")

    pilot_dir = args.pilot.resolve()
    pilot_name = pilot_dir.name
    pilot_seed = _pilot_seed(pilot_dir)
    shape = _pilot_geom_shape(pilot_dir)

    energy_fracs = _float_list_csv(args.energy_fracs)
    fprs = _float_list_csv(args.fprs)
    for f in fprs:
        if not (0.0 < f < 1.0):
            raise ValueError(f"Invalid FPR {f}; must be in (0,1).")

    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    # All-false conditional mask to force STAP skip (baseline-only run).
    mask_path = out_root / "mask_all_false.npy"
    if not mask_path.exists():
        mask = np.zeros(shape, dtype=bool)
        np.save(mask_path, mask, allow_pickle=False)

    rows: List[SweepRow] = []
    for ef in energy_fracs:
        ef = float(ef)
        out_dir = out_root / f"{pilot_name}_mcsvd_e{ef:.4f}".replace(".", "p")
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(REPLAY),
            "--src",
            str(pilot_dir),
            "--out",
            str(out_dir),
            "--profile",
            str(args.profile),
            "--stap-device",
            str(args.stap_device),
            "--time-window-length",
            str(int(args.window_length)),
            "--time-window-offset",
            str(int(args.window_offset)),
            "--svd-energy-frac",
            str(ef),
            "--stap-conditional-mask",
            str(mask_path),
            "--stap-conditional-mask-tag",
            "baseline_only_allfalse",
        ]
        _run(cmd, dry_run=bool(args.dry_run))

        bundle_dir = _bundle_dir_from_out(out_dir)
        score, mask_flow, mask_bg = _load_score_base(bundle_dir)
        pos = score[mask_flow]
        neg = score[mask_bg]

        tpr_by_fpr: Dict[float, float] = {}
        thr_by_fpr: Dict[float, float] = {}
        for fpr in fprs:
            thr, tpr = _tpr_at_fpr(pos, neg, fpr)
            tpr_by_fpr[float(fpr)] = tpr
            thr_by_fpr[float(fpr)] = thr

        meta = json.loads((bundle_dir / "meta.json").read_text())
        tele = meta.get("stap_fallback_telemetry") or {}
        svd_rank_removed = tele.get("svd_rank_removed")
        if isinstance(svd_rank_removed, (int, float)):
            svd_rank_removed = int(svd_rank_removed)
        else:
            svd_rank_removed = None
        svd_energy_removed = tele.get("svd_energy_removed_frac")
        svd_energy_removed = float(svd_energy_removed) if svd_energy_removed is not None else None
        baseline_ms = tele.get("baseline_ms")
        baseline_ms = float(baseline_ms) if baseline_ms is not None else None

        obj = _objective_from_tprs(tpr_by_fpr, fprs)
        rows.append(
            SweepRow(
                energy_frac=ef,
                out_dir=out_dir,
                bundle_dir=bundle_dir,
                svd_rank_removed=svd_rank_removed,
                svd_energy_removed_frac=svd_energy_removed,
                baseline_ms=baseline_ms,
                tpr_by_fpr=tpr_by_fpr,
                thr_by_fpr=thr_by_fpr,
                objective=obj,
            )
        )

    # Choose best (maximize objective = sum TPRs at requested FPR points).
    best = None
    for row in rows:
        if not np.isfinite(row.objective):
            continue
        if best is None or row.objective > best.objective:
            best = row

    # Write CSV (one row per energy fraction).
    header: List[str] = [
        "pilot",
        "pilot_seed",
        "profile",
        "window_offset",
        "window_length",
        "svd_energy_frac_target",
        "svd_rank_removed",
        "svd_energy_removed_frac",
        "baseline_ms",
        "objective_sum_tpr",
    ]
    for fpr in fprs:
        header.append(f"tpr@{fpr:g}")
        header.append(f"thr@{fpr:g}")

    with open(args.summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in rows:
            rec: Dict[str, Any] = {
                "pilot": pilot_name,
                "pilot_seed": pilot_seed,
                "profile": args.profile,
                "window_offset": int(args.window_offset),
                "window_length": int(args.window_length),
                "svd_energy_frac_target": row.energy_frac,
                "svd_rank_removed": row.svd_rank_removed,
                "svd_energy_removed_frac": row.svd_energy_removed_frac,
                "baseline_ms": row.baseline_ms,
                "objective_sum_tpr": row.objective,
            }
            for fpr in fprs:
                rec[f"tpr@{fpr:g}"] = row.tpr_by_fpr.get(float(fpr))
                rec[f"thr@{fpr:g}"] = row.thr_by_fpr.get(float(fpr))
            w.writerow(rec)

    summary = {
        "calibration": {
            "pilot": str(pilot_dir),
            "pilot_name": pilot_name,
            "pilot_seed": pilot_seed,
            "profile": args.profile,
            "window_offset": int(args.window_offset),
            "window_length": int(args.window_length),
            "fprs": [float(f) for f in fprs],
            "objective": "sum_tpr_over_fprs",
            "energy_fracs": [float(e) for e in energy_fracs],
        },
        "best": None
        if best is None
        else {
            "svd_energy_frac_target": best.energy_frac,
            "objective_sum_tpr": best.objective,
            "svd_rank_removed": best.svd_rank_removed,
            "svd_energy_removed_frac": best.svd_energy_removed_frac,
            "baseline_ms": best.baseline_ms,
            "bundle_dir": str(best.bundle_dir),
        },
    }
    with open(args.summary_json, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    if best is None:
        print("[mcsvd_sweep] No finite objective values; see outputs for diagnostics.")
    else:
        print(
            f"[mcsvd_sweep] Best energy_frac={best.energy_frac:.4f} "
            f"(objective_sum_tpr={best.objective:.6f})"
        )


if __name__ == "__main__":
    main()

