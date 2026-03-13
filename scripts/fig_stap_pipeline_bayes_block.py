#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def add_box(ax, x, y, w, h, title, body, *, fc="#f7f7f7") -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=fc,
        edgecolor="black",
        linewidth=1.0,
    )
    ax.add_patch(patch)
    ax.text(x + 0.14, y + h - 0.14, title, fontsize=8.8, ha="left", va="top", weight="bold")
    ax.text(x + w / 2, y + h - 0.45, body, fontsize=8.0, ha="center", va="top", linespacing=1.08)


def add_arrow(ax, x0, y0, x1, y1, *, text=None, text_xy=None) -> None:
    arr = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.3,
        color="black",
    )
    ax.add_patch(arr)
    if text and text_xy is not None:
        ax.text(text_xy[0], text_xy[1], text, fontsize=7.8, ha="center", va="center")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the main method-overview figure.")
    parser.add_argument(
        "--out",
        default="figs/paper/stap_pipeline_bayes_block.pdf",
        help="Output PDF path.",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(8.4, 3.3))
    ax.set_xlim(0, 17.2)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    h = 0.95
    add_box(ax, 0.45, 2.0, 2.4, h, "Beamformed IQ", r"$X[t,y,x]$")
    add_box(ax, 3.2, 2.0, 2.9, h, "Residualization", "baseline clutter suppression\n(e.g. MC--SVD)")
    add_box(
        ax,
        6.45,
        2.0,
        3.2,
        h,
        "Tile extraction + band geometry",
        "slow-time tiles\nflow / guard / alias bands",
        fc="#f4f7fb",
    )

    add_box(
        ax,
        10.25,
        3.35,
        2.65,
        h,
        "Fixed detector",
        "band-limited matched-\nsubspace score",
        fc="#eef6ff",
    )
    add_box(
        ax,
        10.25,
        1.95,
        2.65,
        h,
        "Adaptive detector",
        "fixed score by default\nwhiten only in clutter-heavy tiles",
        fc="#eef6ff",
    )
    add_box(
        ax,
        10.25,
        0.55,
        2.65,
        h,
        "Fully whitened detector",
        "tile-local covariance\nwhitening everywhere",
        fc="#eef6ff",
    )

    add_box(
        ax,
        13.4,
        2.0,
        1.7,
        h,
        "Optional penalty layer",
        "shrink-only\nvessel-proxy protection",
        fc="#fff4e8",
    )
    add_box(ax, 15.45, 2.0, 1.25, h, "Output map", "score map")

    # Main linear flow.
    add_arrow(ax, 2.85, 2.48, 3.2, 2.48)
    add_arrow(ax, 6.1, 2.48, 6.45, 2.48)

    # Branching from tile/band block.
    add_arrow(ax, 9.65, 2.65, 10.25, 3.82, text="use everywhere", text_xy=(10.0, 3.2))
    add_arrow(ax, 9.65, 2.48, 10.25, 2.48, text="guard-triggered switch", text_xy=(10.0, 2.78))
    add_arrow(ax, 9.65, 2.31, 10.25, 1.02, text="force whitening", text_xy=(9.95, 1.72))

    # Variant outputs toward optional penalty layer.
    add_arrow(ax, 12.9, 3.82, 13.4, 2.62)
    add_arrow(ax, 12.9, 2.48, 13.4, 2.48)
    add_arrow(ax, 12.9, 1.02, 13.4, 2.34)
    add_arrow(ax, 15.1, 2.48, 15.45, 2.48)

    ax.text(
        11.55,
        4.55,
        "Three detector modes share the same\nflow-band score family; they differ only in\nwhen local whitening is used.",
        fontsize=8.0,
        ha="center",
        va="top",
    )

    fig.savefig(out, bbox_inches="tight", format="pdf")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
