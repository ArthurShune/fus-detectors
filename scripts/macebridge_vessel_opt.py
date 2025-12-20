#!/usr/bin/env python3
"""
MaceBridge vessel-parameter search on a single slice.

This script performs a simple stochastic search over microvascular and alias
vessel placement probabilities for a single Macé-aligned MaceBridge slice.
For each candidate parameter vector it:

  1) Regenerates micro_vessels.npy / alias_vessels.npy for the chosen slice
     using mace_bridge.vessels.generate_microvascular_vessels /
     generate_alias_vessels.
  2) Runs replay_stap_from_run.py with the clinical STAP PD configuration
     on that slice only, writing a candidate-specific replay bundle.
  3) Runs macebridge_open_doppler_telemetry.py on the candidate replay root
     to extract Doppler Pf/Pa occupancy and alias-score telemetry.
  4) Optionally parses hab_contract_check.py output (via the replay summary)
     to incorporate PD ROC behavior into the objective.

The objective combines:

  - a target Pf peak fraction in H1 (default 0.5),
  - a small Pf peak fraction in H0 (target 0.1),
  - a target delta_log_alias > 0 (default 0.5),
  - and a reward for STAP PD TPR at FPR=1e-3.

This is not a sophisticated GA/SQP implementation; it is a simple random
search over a bounded box in the vessel probability space. The intent is
to provide a reproducible, scriptable way to explore vessel parameters; it
can be run overnight on a single slice and the best candidates inspected
manually.

Usage (example)
---------------

    PYTHONPATH=. python scripts/macebridge_vessel_opt.py \\
        --sim-root runs/macebridge_open \\
        --scan-idx 0 --plane-idx 3 \\
        --out-root runs/macebridge_vessel_opt \\
        --n-iters 20
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from mace_bridge import (
    alias_vessels_to_array,
    generate_alias_vessels,
    generate_microvascular_vessels,
    micro_vessels_to_array,
)


REPO = Path(__file__).resolve().parents[1]


@dataclass
class VesselParams:
    # Microvascular placement probabilities
    p_H1: float
    p_H0: float
    p_BG: float
    # Alias placement probabilities
    p_alias_H1: float
    p_alias_H0: float
    p_alias_BG: float


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Stochastic search over MaceBridge vessel parameters for a single slice."
    )
    ap.add_argument(
        "--sim-root",
        type=Path,
        required=True,
        help="Root directory containing MaceBridge sims (scan*_plane*).",
    )
    ap.add_argument(
        "--scan-idx",
        type=int,
        default=0,
        help="Scan index (0-based) for the slice to tune.",
    )
    ap.add_argument(
        "--plane-idx",
        type=int,
        default=3,
        help="Plane index (0-based) for the slice to tune.",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Root directory for candidate replay runs.",
    )
    ap.add_argument(
        "--n-iters",
        type=int,
        default=20,
        help="Number of random candidates to evaluate.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for parameter sampling.",
    )
    ap.add_argument(
        "--target-frac-pf-pos",
        type=float,
        default=0.5,
        help="Target Pf peak fraction in H1 tiles.",
    )
    ap.add_argument(
        "--target-frac-pf-neg",
        type=float,
        default=0.1,
        help="Target Pf peak fraction in H0 tiles.",
    )
    ap.add_argument(
        "--target-delta-log-alias",
        type=float,
        default=0.5,
        help="Target delta_log_alias = median_neg - median_pos.",
    )
    ap.add_argument(
        "--fpr-tpr",
        type=float,
        default=1e-3,
        help="FPR at which to read STAP PD TPR (for objective).",
    )
    ap.add_argument(
        "--results-json",
        type=Path,
        default=REPO / "reports" / "macebridge_vessel_opt_results.json",
        help="Path to write a JSON list of candidate results.",
    )
    return ap.parse_args()


def _sim_slice_dir(sim_root: Path, scan_idx: int, plane_idx: int) -> Path:
    return sim_root / f"scan{scan_idx}_plane{plane_idx}"


def _regen_vessels_for_candidate(
    sim_slice_dir: Path,
    params: VesselParams,
    rng_seed: int,
) -> None:
    """Regenerate micro_vessels.npy / alias_vessels.npy for a candidate."""

    roi_H1 = np.load(sim_slice_dir / "roi_H1.npy").astype(bool)
    roi_H0 = np.load(sim_slice_dir / "roi_H0.npy").astype(bool)
    rng = np.random.default_rng(rng_seed)

    micro = generate_microvascular_vessels(
        roi_H1,
        roi_H0,
        rng=rng,
        p_H1=params.p_H1,
        p_H0=params.p_H0,
        p_BG=params.p_BG,
    )
    alias = generate_alias_vessels(
        roi_H1,
        roi_H0,
        rng=rng,
        p_alias_H1=params.p_alias_H1,
        p_alias_H0=params.p_alias_H0,
        p_alias_BG=params.p_alias_BG,
    )

    if micro:
        mv_arr = micro_vessels_to_array(micro)
        np.save(sim_slice_dir / "micro_vessels.npy", mv_arr, allow_pickle=False)
    else:
        (sim_slice_dir / "micro_vessels.npy").unlink(missing_ok=True)
    if alias:
        av_arr = alias_vessels_to_array(alias)
        np.save(sim_slice_dir / "alias_vessels.npy", av_arr, allow_pickle=False)
    else:
        (sim_slice_dir / "alias_vessels.npy").unlink(missing_ok=True)


def _run_replay_for_candidate(
    sim_slice_dir: Path,
    out_root: Path,
    scan_idx: int,
    plane_idx: int,
) -> Path:
    """Run clinical STAP PD replay for a single slice and return bundle path."""

    tag = f"scan{scan_idx}_plane{plane_idx}"
    replay_slice_root = out_root / tag
    replay_slice_root.mkdir(parents=True, exist_ok=True)

    # Inherit environment and enable the PD-only fast path and modest
    # snapshot caps so that replay runs quickly and avoids relying on
    # the optional robust_covariance extension.
    env = dict(**os.environ) if "os" in globals() else None
    import os as _os  # local import to avoid polluting top-level too early

    env = _os.environ.copy()
    env.setdefault("STAP_SNAPSHOT_STRIDE", "4")
    env.setdefault("STAP_MAX_SNAPSHOTS", "64")
    env.setdefault("STAP_FAST_PD_ONLY", "1")

    cmd: List[str] = [
        sys.executable,
        str(REPO / "scripts" / "replay_stap_from_run.py"),
        "--src",
        str(sim_slice_dir),
        "--out",
        str(replay_slice_root),
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
        "--tile-h",
        "8",
        "--tile-w",
        "8",
        "--tile-stride",
        "3",
        "--lt",
        "8",
        "--diag-load",
        "0.07",
        "--cov-estimator",
        "scm",
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
        "--clutter-beta",
        "1.0",
        "--clutter-snr-db",
        "18.0",
        "--clutter-depth-min-frac",
        "0.20",
        "--clutter-depth-max-frac",
        "0.95",
        "--psd-telemetry",
        "--feasibility-mode",
        "legacy",
    ]
    print("[vessel-opt replay]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

    # Find the pw_* bundle under replay_slice_root.
    cands = sorted(replay_slice_root.glob("pw_*"))
    if not cands:
        raise RuntimeError(f"No replay bundle produced under {replay_slice_root}")
    return cands[0]


def _run_telemetry_for_candidate(replay_root: Path) -> Dict[str, float]:
    """Run Doppler telemetry on a candidate replay root and return metrics."""

    csv_path = replay_root.parent / f"{replay_root.name}_telemetry.csv"
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "macebridge_open_doppler_telemetry.py"),
        "--replay-root",
        str(replay_root.parent),
        "--out-csv",
        str(csv_path),
    ]
    print("[vessel-opt telemetry]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=None)

    # There should be exactly one row for this replay root.
    import csv

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No telemetry rows found in {csv_path}")
    row = rows[0]

    def ffloat(name: str) -> float:
        try:
            return float(row[name])
        except Exception:
            return float("nan")

    return {
        "frac_pf_peak_pos": ffloat("frac_pf_peak_pos"),
        "frac_pf_peak_neg": ffloat("frac_pa_peak_neg"),  # note: BG alias column name
        "median_log_alias_pos": ffloat("median_log_alias_pos"),
        "median_log_alias_neg": ffloat("median_log_alias_neg"),
        "delta_log_alias": ffloat("delta_log_alias"),
    }


def _read_tpr_from_bundle(bundle: Path, fpr: float) -> Tuple[float, float]:
    """Return (tpr_base, tpr_stap) at a given FPR from hab_contract_check."""

    cmd = [
        sys.executable,
        str(REPO / "scripts" / "hab_contract_check.py"),
        str(bundle),
        "--score-mode",
        "pd",
        "--fprs",
        f"{fpr:g}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    tpr_base = float("nan")
    tpr_stap = float("nan")
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("fpr="):
            continue
        try:
            prefix, rest = line.split(":", 1)
            fpr_val = float(prefix.split("=")[1].strip())
            if abs(fpr_val - fpr) > 1e-10:
                continue
            tpr_base_str = rest.split("tpr_base=")[1].split(",")[0].strip()
            tpr_stap_str = rest.split("tpr_stap=")[1].split()[0].strip()
            tpr_base = float(tpr_base_str)
            tpr_stap = float(tpr_stap_str)
        except Exception:
            continue
    return tpr_base, tpr_stap


def _sample_params(rng: np.random.Generator) -> VesselParams:
    """Sample a candidate parameter vector from reasonable bounds."""

    p_H1 = float(rng.uniform(0.15, 0.5))
    p_H0 = float(rng.uniform(0.0, 0.08))
    p_BG = float(rng.uniform(0.02, 0.12))
    p_alias_H1 = float(rng.uniform(0.0, 0.02))
    p_alias_H0 = float(rng.uniform(0.03, 0.12))
    p_alias_BG = float(rng.uniform(0.0, 0.06))
    return VesselParams(
        p_H1=p_H1,
        p_H0=p_H0,
        p_BG=p_BG,
        p_alias_H1=p_alias_H1,
        p_alias_H0=p_alias_H0,
        p_alias_BG=p_alias_BG,
    )


def _objective(
    metrics: Dict[str, float],
    tpr_base: float,
    tpr_stap: float,
    *,
    target_frac_pf_pos: float,
    target_frac_pf_neg: float,
    target_delta_log_alias: float,
) -> float:
    """Compute a scalar objective (lower is better)."""

    frac_pf_pos = metrics.get("frac_pf_peak_pos", float("nan"))
    frac_pf_neg = metrics.get("frac_pf_peak_neg", float("nan"))
    delta_log_alias = metrics.get("delta_log_alias", float("nan"))

    # Penalties for Pf occupancy and alias separation.
    w_pf_pos = 1.0
    w_pf_neg = 1.0
    w_alias = 1.0
    w_tpr = 0.5

    J = 0.0
    if np.isfinite(frac_pf_pos):
        J += w_pf_pos * (frac_pf_pos - target_frac_pf_pos) ** 2
    else:
        J += 10.0
    if np.isfinite(frac_pf_neg):
        J += w_pf_neg * (frac_pf_neg - target_frac_pf_neg) ** 2
    else:
        J += 10.0
    if np.isfinite(delta_log_alias):
        gap = max(0.0, target_delta_log_alias - delta_log_alias)
        J += w_alias * gap**2
    else:
        J += 10.0

    # Encourage higher STAP TPR at fixed FPR, but do not require baseline TPR.
    if np.isfinite(tpr_stap):
        J -= w_tpr * tpr_stap
    return float(J)


def main() -> None:
    args = parse_args()
    sim_root: Path = args.sim_root
    scan_idx = int(args.scan_idx)
    plane_idx = int(args.plane_idx)
    sim_slice_dir = _sim_slice_dir(sim_root, scan_idx, plane_idx)
    if not sim_slice_dir.exists():
        raise SystemExit(f"Sim slice dir {sim_slice_dir} not found.")

    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    fpr = float(args.fpr_tpr)

    results: List[Dict[str, object]] = []
    best_obj = float("inf")
    best_idx = -1

    for i in range(int(args.n_iters)):
        params = _sample_params(rng)
        rng_seed = args.seed + i * 17
        print(f"[vessel-opt] Candidate {i+1}/{args.n_iters}: {asdict(params)}")

        # 1) Regenerate vessels for this candidate.
        _regen_vessels_for_candidate(sim_slice_dir, params, rng_seed=rng_seed)

        # 2) Replay STAP PD for this candidate.
        replay_bundle = _run_replay_for_candidate(
            sim_slice_dir,
            out_root=out_root,
            scan_idx=scan_idx,
            plane_idx=plane_idx,
        )

        # 3) Telemetry.
        metrics = _run_telemetry_for_candidate(replay_bundle.parent)

        # 4) PD ROC at fixed FPR.
        tpr_base, tpr_stap = _read_tpr_from_bundle(replay_bundle, fpr=fpr)

        obj = _objective(
            metrics,
            tpr_base,
            tpr_stap,
            target_frac_pf_pos=float(args.target_frac_pf_pos),
            target_frac_pf_neg=float(args.target_frac_pf_neg),
            target_delta_log_alias=float(args.target_delta_log_alias),
        )

        row: Dict[str, object] = {
            "candidate_index": i,
            "params": asdict(params),
            "metrics": metrics,
            "tpr_base": float(tpr_base),
            "tpr_stap": float(tpr_stap),
            "objective": float(obj),
            "bundle": str(replay_bundle),
        }
        results.append(row)
        if obj < best_obj:
            best_obj = obj
            best_idx = i
        print(
            f"[vessel-opt] cand {i} obj={obj:.4f} "
            f"tpr_stap@{fpr:g}={tpr_stap:.3f} "
            f"frac_pf_pos={metrics.get('frac_pf_peak_pos')} "
            f"delta_log_alias={metrics.get('delta_log_alias')}"
        )

    args.results_json.parent.mkdir(parents=True, exist_ok=True)
    args.results_json.write_text(json.dumps(results, indent=2))
    print(
        f"[vessel-opt] wrote {len(results)} candidates to {args.results_json}; "
        f"best index={best_idx}, best_obj={best_obj:.4f}"
    )


if __name__ == "__main__":
    main()
