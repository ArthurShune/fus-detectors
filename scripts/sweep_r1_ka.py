#!/usr/bin/env python3
"""
Run a small R1 k-Wave sweep over KA/MSD/grid settings and report acceptance metrics.

Grid:
- ka_kappa: {30, 40}
- msd_lambda: {0.02, 0.04}
- grid_step_rel: {0.08, 0.12}

Outputs under runs/pilot/r1_sweep_ka/<combo>/ ... with an acceptance summary.
"""
from __future__ import annotations

import itertools
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]


@dataclass
class Combo:
    kappa: int
    lam: float
    step: float

    def tag(self) -> str:
        return f"k{self.kappa}_lam{self.lam:g}_step{self.step:g}"


def _bundle_dir(root: Path) -> Path | None:
    # Expect a single pw_*/ subdir with meta.json
    cands = sorted((p for p in root.glob("pw_*")), key=lambda p: p.name)
    return cands[0] if cands else None


def _ensure_run(out_dir: Path, combo: Combo) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = _bundle_dir(out_dir)
    if bundle and (bundle / "meta.json").exists():
        return bundle
    cmd = [
        sys.executable,
        str(REPO / "sim/kwave/pilot_r1.py"),
        "--out",
        str(out_dir),
        "--ka-mode",
        "analytic",
        "--ka-beta-bounds",
        "0.05,0.50",
        "--ka-kappa",
        str(combo.kappa),
        "--msd-lambda",
        str(combo.lam),
        "--msd-ridge",
        "0.12",
        "--msd-ratio-rho",
        "0.05",
        "--fd-span-mode",
        "psd",
        "--grid-step-rel",
        str(combo.step),
        # Enable background guard with conservative defaults
        "--bg-guard-enabled",
        "--bg-guard-target-p90",
        "1.20",
        "--bg-guard-min-alpha",
        "0.6",
        "--bg-guard-metric",
        "tile_p90",
        "--max-pts",
        "5",
        "--cov-estimator",
        "tyler_pca",
        "--stap-device",
        "cuda",
        "--stap-debug-samples",
        "0",
    ]
    print("[run]", " ".join(cmd))
    subprocess.run(["conda", "run", "-n", "stap-fus", *cmd], check=True)
    bundle = _bundle_dir(out_dir)
    if not bundle:
        raise RuntimeError(f"No bundle emitted under {out_dir}")
    return bundle


def _acceptance_for(bundle: Path) -> Dict[str, float]:
    from eval.acceptance import DetectorDataset, Masks, acceptance_report

    pd_stats_path = bundle / "pd_stats.json"
    pd_stats = json.loads(pd_stats_path.read_text()) if pd_stats_path.exists() else {}
    base_stats = pd_stats.get("baseline")
    stap_stats = pd_stats.get("stap")
    base = DetectorDataset(
        scores_pos=np.load(bundle / "base_pos.npy"),
        scores_null=np.load(bundle / "base_neg.npy"),
        pd_map=np.load(bundle / "pd_base.npy"),
        pd_stats=base_stats,
    )
    stap = DetectorDataset(
        scores_pos=np.load(bundle / "stap_pos.npy"),
        scores_null=np.load(bundle / "stap_neg.npy"),
        pd_map=np.load(bundle / "pd_stap.npy"),
        pd_stats=stap_stats,
    )
    masks = Masks(
        mask_flow=np.load(bundle / "mask_flow.npy"),
        mask_bg=np.load(bundle / "mask_bg.npy"),
    )
    rep = acceptance_report(base, stap, masks, seed=1337)
    perf = rep.get("performance", {})
    gates = rep.get("gates", {})
    return {
        "pd_delta_db": float(perf.get("pd_snr_delta_db", float("nan"))),
        "pd_stap_db": float(perf.get("pd_snr_stap_db", float("nan"))),
        "tpr_at_fpr": float(perf.get("tpr_at_fpr_stap", float("nan"))),
        "gate_pd": bool(gates.get("gate_delta_pd_snr", False)),
        "gate_tpr": bool(gates.get("gate_delta_tpr_at_fpr", False)),
        "overall": bool(rep.get("overall_pass", False)),
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="R1 sweep over KA/MSD/grid settings")
    ap.add_argument("--out", type=str, default=str(REPO / "runs/pilot/r1_sweep_ka"))
    ap.add_argument("--kappa", type=str, default="30,40")
    ap.add_argument("--lambda", dest="lam", type=str, default="0.02,0.04")
    ap.add_argument("--step", type=str, default="0.08,0.12")
    ap.add_argument("--shard", type=int, default=0, help="Shard index (0..nshards-1)")
    ap.add_argument("--nshards", type=int, default=1, help="Total shards")
    args = ap.parse_args()

    sweep_root = Path(args.out)
    kappa_vals = [int(x) for x in args.kappa.split(",") if x.strip()]
    lam_vals = [float(x) for x in args.lam.split(",") if x.strip()]
    step_vals = [float(x) for x in args.step.split(",") if x.strip()]
    combos = [
        Combo(kappa, lam, step)
        for kappa, lam, step in itertools.product(kappa_vals, lam_vals, step_vals)
    ]
    # Optional sharding: take every nshards-th combo starting at index shard
    if args.nshards > 1:
        combos = [
            c for i, c in enumerate(combos) if i % max(1, args.nshards) == max(0, args.shard)
        ]
    results: list[tuple[Combo, Dict[str, float], Path]] = []
    for c in combos:
        out_dir = sweep_root / c.tag()
        bundle = _ensure_run(out_dir, c)
        acc = _acceptance_for(bundle)
        results.append((c, acc, bundle))
        msg = (
            f"[result] {c.tag()} -> ΔPD={acc['pd_delta_db']:.2f} dB, "
            f"TPR={acc['tpr_at_fpr']:.3e}, overall={acc['overall']}"
        )
        print(msg)

    # Choose best by PD delta, then TPR
    def key(item: tuple[Combo, Dict[str, float], Path]) -> Tuple[float, float]:
        acc = item[1]
        return (acc.get("pd_delta_db", float("-inf")), acc.get("tpr_at_fpr", 0.0))

    best = max(results, key=key)
    c_best, acc_best, bundle_best = best
    summary = {
        "best_combo": {
            "kappa": c_best.kappa,
            "msd_lambda": c_best.lam,
            "grid_step_rel": c_best.step,
        },
        "metrics": acc_best,
        "bundle": str(bundle_best),
        "all": [
            {
                "combo": r[0].__dict__,
                "metrics": r[1],
                "bundle": str(r[2]),
            }
            for r in results
        ],
    }
    out_json = sweep_root / (
        "summary.json"
        if args.nshards == 1
        else f"summary_shard{args.shard}_of_{args.nshards}.json"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print("[best]", c_best.tag(), "->", summary["metrics"])
    print("[summary]", out_json)


if __name__ == "__main__":
    main()
