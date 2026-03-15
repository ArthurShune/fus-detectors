#!/usr/bin/env python3
"""Generate patent-clean line drawings for the provisional package."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTDIR = Path(__file__).resolve().parent / "figs"


def setup_ax(figsize=(11, 6.5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fontsize=11, lw=1.5):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=lw,
        edgecolor="black",
        facecolor="white",
    )
    ax.add_patch(patch)
    wrapped = "\n".join(textwrap.wrap(text, width=max(12, int(w * 48))))
    ax.text(
        x + w / 2,
        y + h / 2,
        wrapped,
        ha="center",
        va="center",
        fontsize=fontsize,
        family="DejaVu Sans",
    )


def arrow(ax, start, end, lw=1.4):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=lw,
            color="black",
        )
    )


def save(fig, name):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def draw_overview():
    fig, ax = setup_ax()
    box(ax, 0.04, 0.36, 0.13, 0.22, "Beamformed IQ or residual slow-time data")
    box(ax, 0.22, 0.36, 0.13, 0.22, "Residualization or clutter suppression")
    box(ax, 0.40, 0.18, 0.18, 0.56, "Localized detector family\n\nFixed detector\nAdaptive detector\nFully whitened detector")
    box(ax, 0.63, 0.36, 0.12, 0.22, "Optional shrink-only penalty")
    box(ax, 0.80, 0.36, 0.15, 0.22, "Score map, detection map, or vascular display")
    arrow(ax, (0.17, 0.47), (0.22, 0.47))
    arrow(ax, (0.35, 0.47), (0.40, 0.47))
    arrow(ax, (0.58, 0.47), (0.63, 0.47))
    arrow(ax, (0.75, 0.47), (0.80, 0.47))
    save(fig, "figure_01_overview.pdf")


def draw_local_processing():
    fig, ax = setup_ax()
    box(ax, 0.05, 0.63, 0.22, 0.17, "Residual slow-time cube")
    box(ax, 0.05, 0.22, 0.22, 0.17, "Overlap-add or weighted aggregation")
    box(ax, 0.39, 0.63, 0.20, 0.17, "Overlapping spatial tiles")
    box(ax, 0.39, 0.22, 0.20, 0.17, "Localized scores")
    box(ax, 0.72, 0.63, 0.21, 0.17, "Slow-time vectors or embedded snapshots")
    box(ax, 0.72, 0.22, 0.21, 0.17, "Per-tile detector output")
    arrow(ax, (0.27, 0.71), (0.39, 0.71))
    arrow(ax, (0.59, 0.71), (0.72, 0.71))
    arrow(ax, (0.82, 0.63), (0.82, 0.39))
    arrow(ax, (0.72, 0.31), (0.59, 0.31))
    arrow(ax, (0.39, 0.31), (0.27, 0.31))
    save(fig, "figure_02_local_processing.pdf")


def draw_adaptive_switch():
    fig, ax = setup_ax()
    box(ax, 0.05, 0.38, 0.15, 0.18, "Localized support")
    box(ax, 0.29, 0.63, 0.22, 0.16, "Clutter-evidence feature\n(guard, alias, motion, condition)")
    box(ax, 0.29, 0.36, 0.22, 0.16, "Fixed detector branch")
    box(ax, 0.29, 0.09, 0.22, 0.16, "Fully whitened branch")
    box(ax, 0.62, 0.33, 0.16, 0.22, "Branch selector or blend")
    box(ax, 0.84, 0.33, 0.11, 0.22, "Output score")
    arrow(ax, (0.20, 0.47), (0.29, 0.71))
    arrow(ax, (0.20, 0.47), (0.29, 0.44))
    arrow(ax, (0.20, 0.47), (0.29, 0.17))
    arrow(ax, (0.51, 0.71), (0.62, 0.52))
    arrow(ax, (0.51, 0.44), (0.62, 0.44))
    arrow(ax, (0.51, 0.17), (0.62, 0.36))
    arrow(ax, (0.78, 0.44), (0.84, 0.44))
    save(fig, "figure_03_adaptive_switch.pdf")


def draw_penalty():
    fig, ax = setup_ax()
    box(ax, 0.05, 0.38, 0.16, 0.18, "Detector score map")
    box(ax, 0.30, 0.63, 0.20, 0.16, "Candidate-set logic")
    box(ax, 0.30, 0.38, 0.20, 0.16, "Protected-set logic")
    box(ax, 0.30, 0.13, 0.20, 0.16, "Side-information features")
    box(ax, 0.62, 0.38, 0.18, 0.18, "Shrink-only weight or monotone suppression")
    box(ax, 0.85, 0.38, 0.10, 0.18, "Post-penalty score")
    arrow(ax, (0.21, 0.47), (0.30, 0.71))
    arrow(ax, (0.21, 0.47), (0.30, 0.46))
    arrow(ax, (0.21, 0.47), (0.30, 0.21))
    arrow(ax, (0.50, 0.71), (0.62, 0.53))
    arrow(ax, (0.50, 0.46), (0.62, 0.47))
    arrow(ax, (0.50, 0.21), (0.62, 0.39))
    arrow(ax, (0.80, 0.47), (0.85, 0.47))
    save(fig, "figure_04_penalty_layer.pdf")


def draw_threshold_transfer():
    fig, ax = setup_ax()
    box(ax, 0.05, 0.36, 0.18, 0.20, "Calibration data\n(background or nuisance bank)")
    box(ax, 0.31, 0.36, 0.18, 0.20, "Threshold learning\nor operating-point calibration")
    box(ax, 0.57, 0.36, 0.16, 0.20, "Fixed threshold family")
    box(ax, 0.81, 0.36, 0.14, 0.20, "Held-out deployment data")
    arrow(ax, (0.23, 0.46), (0.31, 0.46))
    arrow(ax, (0.49, 0.46), (0.57, 0.46))
    arrow(ax, (0.73, 0.46), (0.81, 0.46))
    save(fig, "figure_05_threshold_transfer.pdf")


def draw_realtime():
    fig, ax = setup_ax()
    box(ax, 0.03, 0.38, 0.13, 0.20, "Batched tile extraction")
    box(ax, 0.21, 0.65, 0.18, 0.15, "Reusable localized operators\n(overlap, geometry, projectors)")
    box(ax, 0.21, 0.38, 0.18, 0.15, "Conditional branch execution")
    box(ax, 0.21, 0.11, 0.18, 0.15, "Fixed-batch or replay execution")
    box(ax, 0.48, 0.38, 0.15, 0.20, "Localized scoring")
    box(ax, 0.70, 0.38, 0.12, 0.20, "Overlap-add")
    box(ax, 0.86, 0.38, 0.11, 0.20, "Output map")
    arrow(ax, (0.16, 0.48), (0.21, 0.72))
    arrow(ax, (0.16, 0.48), (0.21, 0.46))
    arrow(ax, (0.16, 0.48), (0.21, 0.19))
    arrow(ax, (0.39, 0.72), (0.48, 0.53))
    arrow(ax, (0.39, 0.46), (0.48, 0.48))
    arrow(ax, (0.39, 0.19), (0.48, 0.43))
    arrow(ax, (0.63, 0.48), (0.70, 0.48))
    arrow(ax, (0.82, 0.48), (0.86, 0.48))
    save(fig, "figure_06_realtime_architecture.pdf")


def main():
    draw_overview()
    draw_local_processing()
    draw_adaptive_switch()
    draw_penalty()
    draw_threshold_transfer()
    draw_realtime()


if __name__ == "__main__":
    main()
