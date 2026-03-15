#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def add_box(ax, x, y, w, h, title, body, *, fc="#f7f7f7", title_fs=8.7, body_fs=7.8) -> None:
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
    body_text = ax.text(x + w / 2, y + 0.40 * h, body, fontsize=body_fs, ha="center", va="center", linespacing=1.08)
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


def add_group(ax, x, y, w, h, title, subtitle=None, *, fc="#fbfcfe") -> None:
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
    title_text = ax.text(x + 0.16, y + h - 0.16, title, fontsize=8.45, ha="left", va="top", weight="bold")
    title_text.set_clip_path(patch)
    if subtitle:
        sub_text = ax.text(x + 0.16, y + h - 0.48, subtitle, fontsize=7.35, ha="left", va="top")
        sub_text.set_clip_path(patch)


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

    fig, ax = plt.subplots(figsize=(11.1, 4.6))
    ax.set_xlim(0, 21.0)
    ax.set_ylim(0, 5.8)
    ax.axis("off")

    h = 1.08
    y0 = 2.30
    add_box(ax, 0.55, y0, 2.55, h, "Beamformed IQ", "complex slow-time data", body_fs=7.75)
    add_box(ax, 3.45, y0, 3.05, h, "Residualization", "coarse clutter suppression", body_fs=7.75)
    add_box(
        ax,
        6.9,
        y0,
        3.35,
        h,
        "Local tiles",
        "overlapping tiles\nflow / guard / alias bands",
        fc="#f4f7fb",
        title_fs=8.45,
        body_fs=7.7,
    )

    add_group(ax, 10.8, 0.95, 4.5, 4.0, "Detector family", "matched-subspace scoring")
    add_box(
        ax,
        11.3,
        3.35,
        3.45,
        0.82,
        "Fixed",
        "no whitening",
        fc="#eef6ff",
        title_fs=7.75,
        body_fs=7.45,
    )
    add_box(
        ax,
        11.3,
        2.20,
        3.45,
        0.82,
        "Adaptive",
        "whiten on\nclutter cue",
        fc="#eef6ff",
        title_fs=7.75,
        body_fs=7.2,
    )
    add_box(
        ax,
        11.3,
        1.05,
        3.45,
        0.82,
        "Whitened",
        "local\ncovariance",
        fc="#eef6ff",
        title_fs=7.55,
        body_fs=7.1,
    )

    add_box(
        ax,
        15.95,
        y0,
        2.6,
        h,
        "Optional penalty",
        "shrink-only\nscore reduction",
        fc="#fff4e8",
        title_fs=8.15,
        body_fs=7.45,
    )
    add_box(ax, 18.95, y0, 1.45, h, "Output", "score map", title_fs=7.9, body_fs=7.4)

    # Main linear flow.
    add_arrow(ax, 3.1, y0 + 0.54, 3.45, y0 + 0.54)
    add_arrow(ax, 6.5, y0 + 0.54, 6.9, y0 + 0.54)

    # Detector family as a grouped stage.
    add_arrow(ax, 10.25, y0 + 0.54, 10.8, y0 + 0.54)
    add_arrow(ax, 15.3, y0 + 0.54, 15.95, y0 + 0.54)
    add_arrow(ax, 18.55, y0 + 0.54, 18.95, y0 + 0.54)

    fig.savefig(out, bbox_inches="tight", format="pdf")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
