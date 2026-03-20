#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


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


def add_round_box(
    ax,
    spec: BoxSpec,
    *,
    facecolor: str,
    edgecolor: str = "#27313a",
    linewidth: float = 1.35,
    radius: float = 0.025,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (spec.left, spec.bottom),
            spec.w,
            spec.h,
            boxstyle=f"round,pad=0.015,rounding_size={radius}",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
    )


def add_box_text(
    ax,
    spec: BoxSpec,
    *,
    title: str,
    body: str,
    title_size: float = 11.5,
    body_size: float = 9.6,
) -> None:
    ax.text(
        spec.cx,
        spec.top - 0.055,
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color="#111111",
    )
    ax.text(
        spec.cx,
        spec.cy - 0.012,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color="#1a1a1a",
        linespacing=1.15,
        multialignment="center",
    )


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    rad: float = 0.0,
    lw: float = 1.55,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=lw,
            color="#27313a",
            shrinkA=2,
            shrinkB=4,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def add_band_bar(ax, spec: BoxSpec) -> None:
    bar_x = spec.left + 0.04
    bar_y = spec.bottom + 0.048
    bar_w = spec.w - 0.08
    bar_h = 0.04
    segments = [
        ("flow $P_f$", "#57b881", 0.42),
        ("guard $P_g$", "#f4c95d", 0.24),
        ("alias $P_a$", "#ee6c73", 0.34),
    ]
    offset = 0.0
    for label, color, frac in segments:
        seg_w = bar_w * frac
        ax.add_patch(
            Rectangle(
                (bar_x + offset, bar_y),
                seg_w,
                bar_h,
                facecolor=color,
                edgecolor="#27313a",
                linewidth=1.0,
            )
        )
        ax.text(
            bar_x + offset + seg_w / 2.0,
            bar_y + bar_h / 2.0,
            label,
            ha="center",
            va="center",
            fontsize=8.3,
            color="#182026",
            fontweight="bold",
        )
        offset += seg_w


def add_tile_icon(ax, spec: BoxSpec) -> None:
    icon_w = 0.07
    icon_h = 0.07
    x0 = spec.left + 0.04
    y0 = spec.bottom + 0.045
    cell_pad = 0.004
    cell_w = (icon_w - 2 * cell_pad) / 3.0
    cell_h = (icon_h - 2 * cell_pad) / 3.0
    for row in range(3):
        for col in range(3):
            ax.add_patch(
                Rectangle(
                    (
                        x0 + col * cell_w,
                        y0 + row * cell_h,
                    ),
                    cell_w - cell_pad,
                    cell_h - cell_pad,
                    facecolor="#d9e6f2" if (row + col) % 2 == 0 else "#f3f7fb",
                    edgecolor="#46627a",
                    linewidth=0.7,
                )
            )
    ax.text(
        x0 + icon_w + 0.012,
        y0 + icon_h / 2.0,
        "same residual,\noverlapping tiles",
        ha="left",
        va="center",
        fontsize=8.9,
        color="#26415a",
    )


def add_adaptive_inset(ax, spec: BoxSpec) -> None:
    pill_h = 0.06
    inset_w = spec.w - 0.09
    top_pill = BoxSpec(spec.cx, spec.cy + 0.02, inset_w, pill_h)
    left_pill = BoxSpec(spec.cx - 0.09, spec.bottom + 0.08, 0.16, pill_h)
    right_pill = BoxSpec(spec.cx + 0.09, spec.bottom + 0.08, 0.16, pill_h)

    for pill, fc in (
        (top_pill, "#fff8da"),
        (left_pill, "#eef6ff"),
        (right_pill, "#edf8f0"),
    ):
        add_round_box(ax, pill, facecolor=fc, linewidth=1.0, radius=0.02)

    ax.text(
        top_pill.cx,
        top_pill.cy,
        "tile-mean PSD -> guard fraction $r_g$",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#1c1c1c",
    )
    ax.text(
        left_pill.cx,
        left_pill.cy,
        "$r_g < \\tau_g$\nkeep fixed score",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#1c1c1c",
        linespacing=1.08,
    )
    ax.text(
        right_pill.cx,
        right_pill.cy,
        "$r_g \\geq \\tau_g$\nrerun whitened score",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#1c1c1c",
        linespacing=1.08,
    )

    add_arrow(
        ax,
        (top_pill.cx, top_pill.bottom),
        (left_pill.cx, left_pill.top),
        rad=0.08,
        lw=1.15,
    )
    add_arrow(
        ax,
        (top_pill.cx, top_pill.bottom),
        (right_pill.cx, right_pill.top),
        rad=-0.08,
        lw=1.15,
    )


def build(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.6, 6.15))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    input_box = BoxSpec(0.14, 0.79, 0.20, 0.19)
    tile_box = BoxSpec(0.38, 0.79, 0.20, 0.19)
    band_box = BoxSpec(0.63, 0.79, 0.24, 0.20)
    output_box = BoxSpec(0.50, 0.12, 0.46, 0.15)

    fixed_box = BoxSpec(0.18, 0.43, 0.25, 0.23)
    adaptive_box = BoxSpec(0.50, 0.43, 0.31, 0.28)
    whitened_box = BoxSpec(0.82, 0.43, 0.25, 0.23)

    add_round_box(ax, input_box, facecolor="#f6f7f9")
    add_round_box(ax, tile_box, facecolor="#f5f8fb")
    add_round_box(ax, band_box, facecolor="#eef5fb")
    add_round_box(ax, fixed_box, facecolor="#edf8f0")
    add_round_box(ax, adaptive_box, facecolor="#fff5d8")
    add_round_box(ax, whitened_box, facecolor="#edf3fd")
    add_round_box(ax, output_box, facecolor="#f7f7f7")

    add_box_text(
        ax,
        input_box,
        title="Input Residual",
        body="Same clutter-filtered\ncomplex slow-time cube\n$X_{\\mathrm{base}}(T,H,W)$",
    )
    add_box_text(
        ax,
        tile_box,
        title="Localized Supports",
        body="Extract overlapping spatial tiles\nand $L_t$ slow-time supports\nfor each neighborhood",
    )
    add_box_text(
        ax,
        band_box,
        title="Band Model",
        body="Per-tile slow-time summaries\nuse the same prespecified flow,\nguard, and alias bands",
    )
    add_box_text(
        ax,
        fixed_box,
        title="Fixed Statistic",
        body="Direct unwhitened\nband-limited ratio\n(no covariance estimate)",
    )
    add_box_text(
        ax,
        adaptive_box,
        title="Adaptive Statistic",
        body="Score every tile once with the fixed rule,\nthen selectively promote only\nclutter-heavy tiles",
    )
    add_box_text(
        ax,
        whitened_box,
        title="Fully Whitened",
        body="Estimate local covariance,\nload + whiten,\nthen score every tile",
    )
    add_box_text(
        ax,
        output_box,
        title="Reconstruction",
        body="Overlap-add tile outputs into the final\nscore map and detector-weighted readout map",
    )

    add_tile_icon(ax, tile_box)
    add_band_bar(ax, band_box)
    add_adaptive_inset(ax, adaptive_box)

    ax.text(
        0.50,
        0.62,
        "Same residual and tile geometry; only the final scoring stage changes.",
        ha="center",
        va="center",
        fontsize=10.0,
        color="#3b4752",
        fontweight="bold",
    )

    add_arrow(ax, (input_box.right, input_box.cy), (tile_box.left, tile_box.cy))
    add_arrow(ax, (tile_box.right, tile_box.cy), (band_box.left, band_box.cy))
    add_arrow(ax, (band_box.cx, band_box.bottom), (adaptive_box.cx, adaptive_box.top))
    add_arrow(ax, (band_box.left + 0.02, band_box.bottom), (fixed_box.cx, fixed_box.top), rad=0.12)
    add_arrow(ax, (band_box.right - 0.02, band_box.bottom), (whitened_box.cx, whitened_box.top), rad=-0.12)
    add_arrow(ax, (fixed_box.cx, fixed_box.bottom), (output_box.left + 0.09, output_box.top), rad=0.06)
    add_arrow(ax, (adaptive_box.cx, adaptive_box.bottom), (output_box.cx, output_box.top))
    add_arrow(ax, (whitened_box.cx, whitened_box.bottom), (output_box.right - 0.09, output_box.top), rad=-0.06)

    fig.tight_layout(pad=0.35)
    fig.savefig(out, bbox_inches="tight")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the main method-overview figure.")
    parser.add_argument("--out", default="figs/paper/stap_pipeline_bayes_block.pdf", help="Output path.")
    args = parser.parse_args()
    build(Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
