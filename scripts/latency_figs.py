#!/usr/bin/env python
"""
Generate simple latency and ROC summary figures for three regimes:
open-skull brain, alias-augmented HAB, and skull/OR HAB.

This script expects each bundle to contain meta.json with stap_fallback_telemetry.
It also invokes hab_contract_check.py to extract TPR at FPR in {1e-4, 3e-4, 1e-3}.

Usage (shortened for readability):
    python scripts/latency_figs.py \
        --brain <open-skull bundle>/pw_7.5MHz_5ang_5ens_320T_seed1 \
        --hab   <alias-contract bundle>/pw_7.5MHz_5ang_5ens_320T_seed2 \
        --skull <skull/OR bundle>/pw_7.5MHz_5ang_5ens_320T_seed2 \
        --out-dir figs_latency

Outputs:
  - figs_latency/latency_bar.{png,pdf}
  - figs_latency/roc_points.{png,pdf}
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


@dataclass
class RegimeSummary:
    name: str
    bundle: Path
    baseline_ms: float
    stap_ms: float
    fprs: List[float]
    tpr_base: List[float]
    tpr_stap: List[float]


def load_meta(bundle: Path) -> Dict:
    meta_path = bundle / "meta.json"
    return json.loads(meta_path.read_text())


def parse_hab_check_output(text: str) -> Tuple[List[float], List[float], List[float]]:
    """Parse hab_contract_check stdout to extract FPR and TPR for base/STAP."""
    fprs: List[float] = []
    tpr_base: List[float] = []
    tpr_stap: List[float] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("fpr="):
            # Example: fpr=0.0001, tpr_base=0.000, tpr_stap=0.209
            try:
                parts = line.split(":")[0].split("=")[1]
                fpr_val = float(parts)
                tpr_base_str = line.split("tpr_base=")[1].split(",")[0]
                tpr_stap_str = line.split("tpr_stap=")[1].split()[0].strip()
                fprs.append(fpr_val)
                tpr_base.append(float(tpr_base_str))
                tpr_stap.append(float(tpr_stap_str))
            except Exception:
                continue
    return fprs, tpr_base, tpr_stap


def hab_check(bundle: Path) -> Tuple[List[float], List[float], List[float]]:
    cmd = [
        "python",
        "scripts/hab_contract_check.py",
        str(bundle),
        "--score-mode",
        "pd",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return parse_hab_check_output(result.stdout)


def summarize_regime(name: str, bundle_path: Path) -> RegimeSummary:
    meta = load_meta(bundle_path)
    tele = meta.get("stap_fallback_telemetry", {})
    baseline_ms = float(tele.get("baseline_ms", 0.0))
    stap_ms = float(tele.get("stap_total_ms", 0.0))
    fprs, tpr_base, tpr_stap = hab_check(bundle_path)
    return RegimeSummary(
        name=name,
        bundle=bundle_path,
        baseline_ms=baseline_ms,
        stap_ms=stap_ms,
        fprs=fprs,
        tpr_base=tpr_base,
        tpr_stap=tpr_stap,
    )


def plot_latency(summaries: List[RegimeSummary], out_dir: Path) -> None:
    names = [s.name for s in summaries]
    baseline = [s.baseline_ms / 1000.0 for s in summaries]  # s to seconds
    stap = [s.stap_ms / 1000.0 for s in summaries]
    x = range(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - width / 2 for i in x], baseline, width, label="MC–SVD baseline")
    ax.bar([i + width / 2 for i in x], stap, width, label="Clinical STAP PD (gated)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Latency (seconds)")
    ax.set_title("Latency per 320-frame volume")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / "latency_bar.png", dpi=200)
    fig.savefig(out_dir / "latency_bar.pdf")


def plot_roc_points(summaries: List[RegimeSummary], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for s in summaries:
        ax.plot(s.fprs, s.tpr_stap, marker="o", label=f"{s.name} (STAP)")
        ax.plot(
            s.fprs,
            s.tpr_base,
            marker="x",
            linestyle="--",
            alpha=0.6,
            label=f"{s.name} (base)",
        )
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 2e-3)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (PD, S = −PD)")
    ax.set_title("PD ROC points at low FPR")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_points.png", dpi=200)
    fig.savefig(out_dir / "roc_points.pdf")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--brain", type=Path, required=True, help="Bundle path for open-skull brain regime"
    )
    ap.add_argument(
        "--hab", type=Path, required=True, help="Bundle path for alias-augmented HAB regime"
    )
    ap.add_argument(
        "--skull", type=Path, required=True, help="Bundle path for skull/OR HAB regime"
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figs_latency"),
        help="Output directory for figures",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summaries = [
        summarize_regime("Brain (open-skull)", args.brain),
        summarize_regime("HAB contract", args.hab),
        summarize_regime("HAB skull/OR", args.skull),
    ]
    plot_latency(summaries, args.out_dir)
    plot_roc_points(summaries, args.out_dir)


if __name__ == "__main__":
    main()
