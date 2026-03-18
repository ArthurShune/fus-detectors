#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


METHODS = [
    ("Baseline (power Doppler)", "PD"),
    ("Baseline (Kasai lag-1 magnitude)", "Kasai"),
    ("Fixed matched-subspace detector", "Fixed"),
    ("Adaptive detector", "Adaptive"),
    ("Fully whitened detector", "Whitened"),
]

SETTINGS = [
    ("Mobile", "SIMUS-Struct-Mobile"),
    ("Intra-operative parenchymal", "SIMUS-Struct-Intraop"),
]

COLORS = {
    "PD": "#666666",
    "Kasai": "#8c8c8c",
    "Fixed": "#1f77b4",
    "Adaptive": "#2ca02c",
    "Whitened": "#d62728",
}


def _load_summary(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], dict[str, float]] = {}
    for row in payload["summary"]:
        out[(row["setting"], row["method_label"])] = row
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate SIMUS same-residual detector-family headline figure.")
    parser.add_argument(
        "--in-json",
        type=Path,
        default=Path("reports/simus_v2/simus_detector_family_ablation.json"),
        help="Summary JSON from simus_detector_family_ablation_table.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figs/paper/simus_detector_family_headline.pdf"),
        help="Output figure path",
    )
    args = parser.parse_args()

    summary = _load_summary(args.in_json)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.3), constrained_layout=True, sharey=True)
    x = np.arange(len(METHODS))

    legend_handles = [
        Patch(facecolor=COLORS[short], edgecolor="black", linewidth=0.6, label=method_label)
        for method_label, short in METHODS
    ]

    for ax, (setting_key, panel_title) in zip(axes, SETTINGS, strict=True):
        means = []
        lo_err = []
        hi_err = []
        labels = []
        colors = []
        for method_label, short in METHODS:
            row = summary[(setting_key, method_label)]
            mean = float(row["fpr_nuisance_match_0p5_mean"])
            lo = float(row["fpr_nuisance_match_0p5_min"])
            hi = float(row["fpr_nuisance_match_0p5_max"])
            means.append(mean)
            lo_err.append(max(0.0, mean - lo))
            hi_err.append(max(0.0, hi - mean))
            labels.append(short)
            colors.append(COLORS[short])

        ax.bar(x, means, color=colors, edgecolor="black", linewidth=0.6, width=0.74, zorder=2)
        ax.errorbar(
            x,
            means,
            yerr=np.vstack([lo_err, hi_err]),
            fmt="none",
            ecolor="black",
            elinewidth=0.9,
            capsize=2.5,
            zorder=3,
        )
        ax.set_xticks(x, labels)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(panel_title)
        ax.set_ylabel(r"FPR$_{\mathrm{nuis}}$ @ TPR$_{\mathrm{main}}=0.5$")
        ax.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for xi, mean in zip(x, means, strict=True):
            ax.text(xi, mean + 0.03, f"{mean:.3f}", ha="center", va="bottom", fontsize=7)

    axes[1].set_ylabel("")
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
        columnspacing=0.9,
        handlelength=1.3,
    )
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
