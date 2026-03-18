#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figs" / "paper" / "variant_selection_flowchart.pdf"


def add_round_box(ax, center, size, text, *, fc, fontsize=11, weight="normal"):
    cx, cy = center
    w, h = size
    x = cx - w / 2
    y = cy - h / 2
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor=fc,
        edgecolor="#2b2b2b",
        linewidth=1.3,
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
        linespacing=1.2,
    )


def add_diamond(ax, center, size, text, *, fc, fontsize=10.5):
    cx, cy = center
    w, h = size
    verts = [
        (cx, cy + h / 2),
        (cx + w / 2, cy),
        (cx, cy - h / 2),
        (cx - w / 2, cy),
    ]
    patch = Polygon(verts, closed=True, facecolor=fc, edgecolor="#2b2b2b", linewidth=1.3)
    ax.add_patch(patch)
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.15,
    )


def add_arrow(ax, start, end, *, connectionstyle="arc3", label=None, label_xy=None):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.4,
        color="#2b2b2b",
        shrinkA=4,
        shrinkB=6,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(patch)
    if label and label_xy is not None:
        ax.text(label_xy[0], label_xy[1], label, fontsize=11, weight="bold", ha="center", va="center")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.0, 8.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_round_box(
        ax,
        (0.50, 0.90),
        (0.44, 0.10),
        "Start with the fixed matched-subspace statistic",
        fc="#e8f0fa",
        fontsize=12.5,
        weight="bold",
    )
    add_round_box(
        ax,
        (0.50, 0.73),
        (0.66, 0.13),
        "Measure label-free runtime telemetry on representative windows:\n"
        r"high-end guard contamination $Q_{0.90}(r_g)$, promotion fraction $p_{\mathrm{promote}}$,"
        "\nand replay budget",
        fc="#fdf3db",
        fontsize=11.2,
    )
    add_diamond(
        ax,
        (0.50, 0.57),
        (0.32, 0.14),
        "Guard telemetry low\nacross windows?\n"
        r"$Q_{0.90}(r_g) < \tau_g$ and $p_{\mathrm{promote}} \approx 0$",
        fc="#f4f4f4",
        fontsize=10.8,
    )
    add_diamond(
        ax,
        (0.76, 0.38),
        (0.28, 0.12),
        "Guard telemetry persistently\n elevated and latency budget\nsufficient for whitening?",
        fc="#f4f4f4",
        fontsize=9.6,
    )
    add_round_box(
        ax,
        (0.18, 0.12),
        (0.26, 0.11),
        "Deploy the fixed statistic\n"
        "Default supported by the held-out\nSIMUS benchmark",
        fc="#e8f6ef",
        fontsize=11,
    )
    add_round_box(
        ax,
        (0.50, 0.12),
        (0.26, 0.11),
        "Keep the fixed statistic\n"
        "No validated prospective switch\nfor mixed telemetry",
        fc="#f9ebea",
        fontsize=11,
    )
    add_round_box(
        ax,
        (0.82, 0.12),
        (0.28, 0.11),
        "Use the fully whitened variant\n"
        "when guard telemetry stays high,\n"
        "promotion is nonzero, and latency permits",
        fc="#e8f0fa",
        fontsize=10.8,
    )

    add_arrow(ax, (0.50, 0.85), (0.50, 0.795))
    add_arrow(ax, (0.50, 0.665), (0.50, 0.64))
    add_arrow(
        ax,
        (0.34, 0.53),
        (0.18, 0.18),
        connectionstyle="angle3,angleA=180,angleB=-90",
        label="yes",
        label_xy=(0.27, 0.36),
    )
    add_arrow(
        ax,
        (0.66, 0.57),
        (0.67, 0.42),
        connectionstyle="angle3,angleA=0,angleB=90",
        label="no",
        label_xy=(0.68, 0.50),
    )
    add_arrow(
        ax,
        (0.69, 0.35),
        (0.57, 0.17),
        connectionstyle="angle3,angleA=180,angleB=90",
        label="no",
        label_xy=(0.67, 0.31),
    )
    add_arrow(
        ax,
        (0.83, 0.32),
        (0.82, 0.17),
        connectionstyle="arc3",
        label="yes",
        label_xy=(0.88, 0.29),
    )

    fig.tight_layout(pad=0.4)
    fig.savefig(OUT, bbox_inches="tight")


if __name__ == "__main__":
    main()
