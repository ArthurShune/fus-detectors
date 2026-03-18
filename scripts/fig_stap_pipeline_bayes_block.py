#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def add_box(ax, center, size, text, *, fc="#f7f7f7", fontsize=8.2, weight="normal"):
    cx, cy = center
    w, h = size
    x = cx - w / 2
    y = cy - h / 2
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor=fc,
        edgecolor="#2b2b2b",
        linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        linespacing=1.15,
    )


def add_arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.3,
            color="#2b2b2b",
            shrinkA=4,
            shrinkB=6,
            connectionstyle="arc3",
        )
    )


def build(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.6, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(ax, (0.15, 0.68), (0.22, 0.12), "Beamformed IQ\nComplex slow-time data", fontsize=8.0)
    add_box(
        ax,
        (0.40, 0.68),
        (0.24, 0.12),
        "Conventional clutter suppression\nClutter-filtered residual",
        fontsize=7.8,
    )
    add_box(
        ax,
        (0.68, 0.68),
        (0.30, 0.15),
        "Local tiles and band summaries\nFlow, guard, and alias-band energy\nin overlapping neighborhoods",
        fc="#f4f7fb",
        fontsize=7.6,
    )
    add_box(
        ax,
        (0.40, 0.33),
        (0.42, 0.24),
        "Detector family\n\nFixed: non-whitened score\nAdaptive: whiten only when guard-band clutter rises\nFully whitened: local covariance-adaptive variant",
        fc="#fbfcfe",
        fontsize=7.4,
    )
    add_box(ax, (0.82, 0.33), (0.22, 0.13), "Output map\nDetector score map", fontsize=8.0)

    add_arrow(ax, (0.26, 0.68), (0.28, 0.68))
    add_arrow(ax, (0.52, 0.68), (0.55, 0.68))
    add_arrow(ax, (0.68, 0.60), (0.50, 0.44))
    add_arrow(ax, (0.61, 0.33), (0.70, 0.33))

    fig.tight_layout(pad=0.4)
    fig.savefig(out, bbox_inches="tight")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the main method-overview figure.")
    parser.add_argument("--out", default="figs/paper/stap_pipeline_bayes_block.pdf", help="Output path.")
    args = parser.parse_args()
    build(Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
