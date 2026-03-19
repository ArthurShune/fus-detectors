#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


@dataclass(frozen=True)
class BoxSpec:
    cx: float
    cy: float
    w: float
    h: float

    @property
    def left(self) -> float:
        return self.cx - self.w / 2.0

    @property
    def right(self) -> float:
        return self.cx + self.w / 2.0

    @property
    def bottom(self) -> float:
        return self.cy - self.h / 2.0

    @property
    def top(self) -> float:
        return self.cy + self.h / 2.0


def add_box(
    ax,
    spec: BoxSpec,
    text: str,
    *,
    fc: str = "#f7f7f7",
    fontsize: float = 9.0,
    weight: str = "normal",
) -> None:
    patch = FancyBboxPatch(
        (spec.left, spec.bottom),
        spec.w,
        spec.h,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        facecolor=fc,
        edgecolor="#2b2b2b",
        linewidth=1.35,
    )
    ax.add_patch(patch)
    ax.text(
        spec.cx,
        spec.cy,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        linespacing=1.18,
        multialignment="center",
    )


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.4,
            color="#2b2b2b",
            shrinkA=5,
            shrinkB=7,
            connectionstyle="arc3",
        )
    )


def build(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.2, 5.4))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    iq_box = BoxSpec(0.14, 0.70, 0.20, 0.16)
    clutter_box = BoxSpec(0.40, 0.70, 0.24, 0.16)
    bands_box = BoxSpec(0.71, 0.70, 0.31, 0.19)
    detector_box = BoxSpec(0.47, 0.29, 0.45, 0.27)
    output_box = BoxSpec(0.84, 0.29, 0.18, 0.16)

    add_box(
        ax,
        iq_box,
        "Beamformed IQ\ncomplex slow-time data",
        fc="#f7f7f7",
        fontsize=9.6,
    )
    add_box(
        ax,
        clutter_box,
        "Conventional clutter suppression\nclutter-filtered residual",
        fc="#f7f7f7",
        fontsize=9.3,
    )
    add_box(
        ax,
        bands_box,
        "Local tiles and band summaries\nflow, guard, and alias-band energy\nin overlapping neighborhoods",
        fc="#f3f7fc",
        fontsize=9.0,
    )
    add_box(
        ax,
        detector_box,
        "Detector family\n\n"
        "Fixed: non-whitened statistic\n"
        "Adaptive: whiten only when guard-band clutter rises\n"
        "Fully whitened: local covariance-adaptive variant",
        fc="#fbfcfe",
        fontsize=8.7,
    )
    add_box(
        ax,
        output_box,
        "Output map\nfinal detector readout",
        fc="#f7f7f7",
        fontsize=9.4,
    )

    add_arrow(
        ax,
        (iq_box.right, iq_box.cy),
        (clutter_box.left, clutter_box.cy),
    )
    add_arrow(
        ax,
        (clutter_box.right, clutter_box.cy),
        (bands_box.left, bands_box.cy),
    )
    add_arrow(
        ax,
        (bands_box.cx, bands_box.bottom),
        (detector_box.cx + 0.02, detector_box.top),
    )
    add_arrow(
        ax,
        (detector_box.right, detector_box.cy),
        (output_box.left, output_box.cy),
    )

    fig.tight_layout(pad=0.5)
    fig.savefig(out, bbox_inches="tight")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the main method-overview figure.")
    parser.add_argument("--out", default="figs/paper/stap_pipeline_bayes_block.pdf", help="Output path.")
    args = parser.parse_args()
    build(Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
