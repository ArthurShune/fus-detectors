#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figs" / "paper" / "variant_selection_flowchart.pdf"


def add_box(ax, xy, wh, text, *, fc="#f7f7f7", ec="#222222", fontsize=11, weight="normal"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        wrap=True,
    )


def add_arrow(ax, start, end, text=None, *, text_offset=(0.0, 0.0)):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.2,
        color="#222222",
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arrow)
    if text:
        mx = 0.5 * (start[0] + end[0]) + text_offset[0]
        my = 0.5 * (start[1] + end[1]) + text_offset[1]
        ax.text(mx, my, text, ha="center", va="center", fontsize=10, weight="bold")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(
        ax,
        (0.34, 0.82),
        (0.32, 0.11),
        "Start with the fixed matched-subspace head",
        fc="#eaf2f8",
        fontsize=12,
        weight="bold",
    )
    add_box(
        ax,
        (0.25, 0.59),
        (0.50, 0.15),
        "Measure label-free runtime telemetry on representative windows:\n"
        r"$Q_{0.90}(r_g)$, adaptive promotion fraction $p_{\mathrm{promote}}$, and replay budget",
        fc="#fdf6e3",
        fontsize=11,
    )
    add_box(
        ax,
        (0.24, 0.35),
        (0.52, 0.14),
        "Is guard-band clutter persistently low?\n"
        r"$Q_{0.90}(r_g) < \tau_g$ and $p_{\mathrm{promote}} \approx 0$ across windows",
        fc="#f4f4f4",
        fontsize=11,
    )
    add_box(
        ax,
        (0.04, 0.08),
        (0.27, 0.16),
        "Deploy the fixed head\n"
        "Default supported by SIMUS-Struct\n"
        "and by any regime with inactive adaptive telemetry",
        fc="#e8f6ef",
        fontsize=10.5,
    )
    add_box(
        ax,
        (0.37, 0.08),
        (0.26, 0.16),
        "Keep the fixed head\n"
        "No validated prospective switch\n"
        "is established for mixed telemetry",
        fc="#f9ebea",
        fontsize=10.5,
    )
    add_box(
        ax,
        (0.69, 0.08),
        (0.27, 0.16),
        "Use the fully whitened variant\n"
        "only when guard telemetry is repeatedly elevated,\n"
        "promotion is nonzero, and latency permits whitening",
        fc="#eaf2f8",
        fontsize=10.2,
    )
    add_box(
        ax,
        (0.67, 0.35),
        (0.29, 0.14),
        "Is the elevated guard telemetry persistent across windows,\n"
        "and does the replay budget tolerate whitening?",
        fc="#f4f4f4",
        fontsize=10.5,
    )

    add_arrow(ax, (0.50, 0.82), (0.50, 0.74))
    add_arrow(ax, (0.50, 0.59), (0.50, 0.49))
    add_arrow(ax, (0.24, 0.35), (0.18, 0.24), "yes", text_offset=(-0.02, 0.01))
    add_arrow(ax, (0.76, 0.42), (0.82, 0.24))
    add_arrow(ax, (0.50, 0.35), (0.74, 0.42), "no", text_offset=(0.02, 0.03))
    add_arrow(ax, (0.67, 0.35), (0.50, 0.24), "no", text_offset=(-0.02, 0.03))
    add_arrow(ax, (0.96, 0.35), (0.83, 0.24), "yes", text_offset=(0.02, 0.03))

    ax.text(
        0.50,
        0.01,
        "Current evidence supports a conservative runtime triage rule, not a universal automatic selector.",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT, bbox_inches="tight")


if __name__ == "__main__":
    main()
