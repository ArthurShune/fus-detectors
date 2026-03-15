#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def add_box(ax, x, y, w, h, title, body, *, fc="#f7f7f7", title_fs=8.8, body_fs=8.0) -> None:
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
    title_text = ax.text(x + 0.14, y + h - 0.14, title, fontsize=title_fs, ha="left", va="top", weight="bold")
    body_text = ax.text(x + w / 2, y + h - 0.48, body, fontsize=body_fs, ha="center", va="top", linespacing=1.12)
    title_text.set_clip_path(patch)
    body_text.set_clip_path(patch)


def add_arrow(ax, x0, y0, x1, y1, *, text=None, text_xy=None, text_fs=7.5) -> None:
    arr = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.15,
        color="black",
    )
    ax.add_patch(arr)
    if text and text_xy is not None:
        ax.text(text_xy[0], text_xy[1], text, fontsize=text_fs, ha="center", va="center")


def add_group(ax, x, y, w, h, title, *, fc="#fbfcfe") -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.09",
        facecolor=fc,
        edgecolor="black",
        linewidth=1.0,
        linestyle="-",
    )
    ax.add_patch(patch)
    title_text = ax.text(x + 0.16, y + h - 0.16, title, fontsize=8.5, ha="left", va="top", weight="bold")
    title_text.set_clip_path(patch)


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

    fig, ax = plt.subplots(figsize=(10.9, 4.9))
    ax.set_xlim(0, 20.6)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    h = 1.12
    add_box(ax, 0.55, 2.45, 2.55, h, "Beamformed IQ", r"$X[t,y,x]$", body_fs=8.2)
    add_box(ax, 3.45, 2.45, 3.05, h, "Residualization", "baseline clutter suppression\n(e.g. MC--SVD)", body_fs=8.15)
    add_box(
        ax,
        6.9,
        2.45,
        3.35,
        h,
        "Tile extraction +\nband geometry",
        "slow-time tiles\nflow / guard / alias bands",
        fc="#f4f7fb",
        title_fs=8.25,
        body_fs=8.1,
    )

    add_group(ax, 10.8, 0.7, 4.05, 5.15, "Detector family")
    add_box(
        ax,
        11.3,
        3.7,
        3.05,
        1.02,
        "Fixed detector",
        "band-limited matched-\nsubspace score",
        fc="#eef6ff",
        body_fs=8.0,
    )
    add_box(
        ax,
        11.3,
        2.23,
        3.05,
        1.02,
        "Adaptive detector",
        "fixed score by default\nwhiten only in clutter-heavy tiles",
        fc="#eef6ff",
        body_fs=7.8,
    )
    add_box(
        ax,
        11.3,
        0.76,
        3.05,
        1.02,
        "Fully whitened detector",
        "tile-local covariance\nwhitening everywhere",
        fc="#eef6ff",
        body_fs=7.95,
    )

    add_box(
        ax,
        15.15,
        2.45,
        2.25,
        h,
        "Optional penalty",
        "shrink-only\nvessel-proxy protection",
        fc="#fff4e8",
        title_fs=8.3,
        body_fs=7.55,
    )
    add_box(ax, 18.0, 2.45, 1.55, h, "Output map", "score map", title_fs=8.05, body_fs=7.8)

    # Main linear flow.
    add_arrow(ax, 3.1, 3.0, 3.45, 3.0)
    add_arrow(ax, 6.5, 3.0, 6.9, 3.0)

    # Detector family as a grouped stage.
    add_arrow(ax, 10.25, 3.0, 10.8, 3.0)
    add_arrow(ax, 14.85, 3.0, 15.15, 3.0)
    add_arrow(ax, 17.4, 3.0, 18.0, 3.0)

    fig.savefig(out, bbox_inches="tight", format="pdf")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
