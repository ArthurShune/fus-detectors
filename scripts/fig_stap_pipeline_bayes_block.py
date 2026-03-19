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
            mutation_scale=18,
            linewidth=1.55,
            color="#2b2b2b",
            shrinkA=2,
            shrinkB=4,
            connectionstyle="arc3,rad=0.0",
        )
    )


def build(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.9, 5.2))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    iq_box = BoxSpec(0.12, 0.73, 0.16, 0.17)
    clutter_box = BoxSpec(0.41, 0.73, 0.21, 0.17)
    bands_box = BoxSpec(0.77, 0.73, 0.26, 0.20)
    detector_box = BoxSpec(0.42, 0.28, 0.43, 0.29)
    output_box = BoxSpec(0.84, 0.28, 0.16, 0.17)

    add_box(
        ax,
        iq_box,
        "Beamformed IQ\ncomplex slow-time data",
        fc="#f7f7f7",
        fontsize=11.4,
    )
    add_box(
        ax,
        clutter_box,
        "Conventional clutter suppression\nclutter-filtered residual",
        fc="#f7f7f7",
        fontsize=10.8,
    )
    add_box(
        ax,
        bands_box,
        "Local tiles and band summaries\nflow, guard, and alias-band energy\nin overlapping neighborhoods",
        fc="#f3f7fc",
        fontsize=10.4,
    )
    add_box(
        ax,
        detector_box,
        "Detector family\n\n"
        "Fixed: non-whitened statistic\n"
        "Adaptive: whiten only when guard-band clutter rises\n"
        "Fully whitened: local covariance-adaptive variant",
        fc="#fbfcfe",
        fontsize=10.3,
    )
    add_box(
        ax,
        output_box,
        "Output map\nfinal detector readout",
        fc="#f7f7f7",
        fontsize=11.0,
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
        (detector_box.cx + 0.10, detector_box.top),
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
