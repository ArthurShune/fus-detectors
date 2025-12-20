#!/usr/bin/env python3
"""
Replay clinical STAP PD on MaceBridge-OpenSkull k-Wave runs and summarize ROC.

This script assumes that:
  - scripts/macebridge_open_sim.py has already been run, producing one
    directory per slice under a root such as runs/macebridge_open, with
    layout:
        scan{scan_idx}_plane{plane_idx}/
          angle_{deg}/rf.npy, dt.npy
          meta.json
  - Each slice meta.json contains a 'sim_geom' entry compatible with
    scripts/replay_stap_from_run.py.

For each slice directory it:
  1) Invokes replay_stap_from_run.py with a clinical-style PD configuration
     (MC-SVD baseline, STAP clinical preset, PD-only fast path) on the
     Macé-aligned grid and tile geometry (8x8 tiles, stride 3, Lt=8).
  2) Locates the resulting acceptance bundle (pw_* directory) and runs
     hab_contract_check.py --score-mode pd to extract baseline/STAP TPR
     at user-specified low FPRs.
  3) Writes a JSON summary with one row per slice.

Usage (example):

    PYTHONPATH=. python scripts/macebridge_open_replay.py \\
        --sim-root runs/macebridge_open \\
        --out-root runs/macebridge_open_replay \\
        --summary-json reports/macebridge_open_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


REPO = Path(__file__).resolve().parents[1]


@dataclass
class SlicePoint:
    scan_idx: int
    plane_idx: int
    path: Path

    def tag(self) -> str:
        return f"scan{self.scan_idx}_plane{self.plane_idx}"


def _discover_slices(sim_root: Path) -> List[SlicePoint]:
    slices: List[SlicePoint] = []
    for d in sorted(sim_root.glob("scan*_plane*")):
        if not d.is_dir():
            continue
        name = d.name
        # Expect names like scan0_plane3.
        try:
            scan_part, plane_part = name.split("_", 1)
            scan_idx = int(scan_part.replace("scan", ""))
            plane_idx = int(plane_part.replace("plane", ""))
        except Exception:
            continue
        slices.append(SlicePoint(scan_idx=scan_idx, plane_idx=plane_idx, path=d))
    return slices


def _bundle_dir(root: Path) -> Path | None:
    cands = sorted(root.glob("pw_*"))
    return cands[0] if cands else None


def _parse_fprs(spec: str) -> List[float]:
    return [float(s) for s in spec.replace(";", ",").split(",") if s.strip()]


def _parse_hab_tprs(output: str) -> Dict[str, float]:
    """
    Parse hab_contract_check.py stdout for lines of the form
      fpr=...: thr_base=..., tpr_base=..., thr_stap=..., tpr_stap=...
    and return a dict {f"tpr_base@{fpr}": ..., f"tpr_stap@{fpr}": ...}.
    """

    out: Dict[str, float] = {}
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("fpr="):
            continue
        try:
            prefix, rest = line.split(":", 1)
            fpr_str = prefix.split("=")[1].strip()
            fpr_val = float(fpr_str)
            tpr_base_str = rest.split("tpr_base=")[1].split(",")[0].strip()
            tpr_stap_str = rest.split("tpr_stap=")[1].split()[0].strip()
            out[f"tpr_base@{fpr_val:g}"] = float(tpr_base_str)
            out[f"tpr_stap@{fpr_val:g}"] = float(tpr_stap_str)
        except Exception:
            continue
    return out


def run_replay_for_slice(
    sl: SlicePoint,
    out_root: Path,
    fprs: Iterable[float],
) -> Dict[str, object]:
    # Out directory for this slice's replay.
    replay_dir = out_root / sl.tag()
    replay_dir.mkdir(parents=True, exist_ok=True)

    # Environment: enable PD-only fast path and snapshot caps as in the
    # clinical STAP preset.
    env = os.environ.copy()
    env.setdefault("STAP_SNAPSHOT_STRIDE", "4")
    env.setdefault("STAP_MAX_SNAPSHOTS", "64")
    env.setdefault("STAP_FAST_PD_ONLY", "1")

    # Replay STAP PD on this slice.
    cmd: List[str] = [
        sys.executable,
        str(REPO / "scripts" / "replay_stap_from_run.py"),
        "--src",
        str(sl.path),
        "--out",
        str(replay_dir),
        "--baseline",
        "mc_svd",
        "--svd-profile",
        "literature",
        "--reg-disable",
        "--stap-device",
        "cuda",
        "--score-mode",
        "pd",
        "--time-window-length",
        "64",
        # Tile/STAP geometry aligned with the clinical brain profile.
        "--tile-h",
        "8",
        "--tile-w",
        "8",
        "--tile-stride",
        "3",
        "--lt",
        "8",
        # Clinical-style covariance / MSD configuration.
        "--diag-load",
        "0.07",
        "--cov-estimator",
        "tyler_pca",
        "--huber-c",
        "5.0",
        "--fd-span-mode",
        "psd",
        "--fd-span-rel",
        "0.30,1.10",
        "--grid-step-rel",
        "0.20",
        "--max-pts",
        "3",
        "--fd-min-pts",
        "3",
        "--constraint-mode",
        "exp+deriv",
        "--constraint-ridge",
        "0.18",
        "--mvdr-load-mode",
        "auto",
        "--mvdr-auto-kappa",
        "120.0",
        "--msd-lambda",
        "0.05",
        "--msd-ridge",
        "0.10",
        "--msd-agg",
        "median",
        "--msd-ratio-rho",
        "0.05",
        "--msd-contrast-alpha",
        "0.6",
        # Whitened band-ratio telemetry parameters (for later contract checks).
        "--band-ratio-mode",
        "whitened",
        "--psd-br-flow-low",
        "30.0",
        "--psd-br-flow-high",
        "220.0",
        "--psd-br-alias-center",
        "650.0",
        "--psd-br-alias-width",
        "140.0",
        # PD-based flow masks consistent with brain profiles.
        "--flow-mask-mode",
        "pd_auto",
        "--flow-mask-pd-quantile",
        "0.995",
        "--flow-mask-depth-min-frac",
        "0.25",
        "--flow-mask-depth-max-frac",
        "0.85",
        "--flow-mask-dilate-iters",
        "2",
        # Temporal clutter with 1/f^beta spectrum over most of the depth,
        # providing headroom between MC-SVD and STAP PD.
        "--clutter-beta",
        "1.0",
        "--clutter-snr-db",
        "18.0",
        "--clutter-depth-min-frac",
        "0.20",
        "--clutter-depth-max-frac",
        "0.95",
        # Enable PSD telemetry so Doppler Pf/Pa occupancy and alias ratios
        # can be inspected via the Doppler telemetry scripts.
        "--psd-telemetry",
        "--feasibility-mode",
        "legacy",
    ]

    print("[macebridge_replay]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

    # The replay creates a pw_* bundle under replay_dir.
    bundle = _bundle_dir(replay_dir)
    if bundle is None:
        raise RuntimeError(f"No replay bundle produced under {replay_dir}")

    # Read latency/coverage telemetry from the replay bundle meta.
    meta_path = bundle / "meta.json"
    meta = json.loads(meta_path.read_text())
    tele = meta.get("stap_fallback_telemetry", {}) or {}

    baseline_ms = float(tele.get("baseline_ms", 0.0))
    stap_ms = float(tele.get("stap_total_ms", 0.0))
    coverage = float(tele.get("coverage", 0.0))

    # Run HAB contract check on PD scores for this bundle.
    fpr_args = [f"{f:g}" for f in fprs]
    cmd_hab: List[str] = [
        sys.executable,
        str(REPO / "scripts" / "hab_contract_check.py"),
        str(bundle),
        "--score-mode",
        "pd",
        "--fprs",
        *fpr_args,
    ]
    result = subprocess.run(cmd_hab, capture_output=True, text=True, check=True)
    tpr_by_fpr = _parse_hab_tprs(result.stdout)

    row: Dict[str, object] = {
        "tag": sl.tag(),
        "scan_idx": sl.scan_idx,
        "plane_idx": sl.plane_idx,
        "bundle": str(bundle),
        "baseline_ms": baseline_ms,
        "stap_ms": stap_ms,
        "coverage": coverage,
    }
    for fpr in fprs:
        key_base = f"tpr_base@{fpr:g}"
        key_stap = f"tpr_stap@{fpr:g}"
        base_val = float(tpr_by_fpr.get(key_base, float("nan")))
        stap_val = float(tpr_by_fpr.get(key_stap, float("nan")))
        row[key_base] = base_val
        row[key_stap] = stap_val
        row[f"delta_tpr@{fpr:g}"] = stap_val - base_val
    return row


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Replay clinical STAP PD on MaceBridge-OpenSkull runs and summarize ROC."
    )
    ap.add_argument(
        "--sim-root",
        type=Path,
        default=REPO / "runs" / "macebridge_open",
        help="Root directory containing MaceBridge-OpenSkull slice runs.",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=REPO / "runs" / "macebridge_open_replay",
        help="Root directory for replay bundles.",
    )
    ap.add_argument(
        "--fprs",
        type=str,
        default="1e-4,3e-4,1e-3",
        help="Comma-separated FPR values at which to record TPR.",
    )
    ap.add_argument(
        "--summary-json",
        type=Path,
        default=REPO / "reports" / "macebridge_open_summary.json",
        help="Path for aggregated replay summary JSON.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    sim_root = args.sim_root
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    fprs = _parse_fprs(args.fprs)
    slices = _discover_slices(sim_root)
    if not slices:
        raise SystemExit(f"No slice directories found under {sim_root}")

    rows: List[Dict[str, object]] = []
    for sl in slices:
        row = run_replay_for_slice(sl, out_root, fprs)
        rows.append(row)

    # Write JSON summary.
    summary = {
        "sim_root": str(sim_root),
        "out_root": str(out_root),
        "fprs": fprs,
        "rows": rows,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[macebridge_replay] wrote summary to {args.summary_json}")


if __name__ == "__main__":
    main()
