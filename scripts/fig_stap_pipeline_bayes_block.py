#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


def _add_text(
    ax: plt.Axes,
    x: float,
    y: float,
    s: str,
    *,
    size: float | None = None,
    ha: str = "center",
    va: str = "center",
    weight: str | None = None,
) -> None:
    ax.text(x, y, s, fontsize=size, ha=ha, va=va, weight=weight)


def _arrow(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    lw: float = 1.4,
    color: str = "black",
    text: str | None = None,
    text_offset: tuple[float, float] = (0.0, 0.0),
) -> None:
    a = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=lw,
        color=color,
    )
    ax.add_patch(a)
    if text:
        _add_text(
            ax,
            (x0 + x1) / 2 + text_offset[0],
            (y0 + y1) / 2 + text_offset[1],
            text,
            size=8,
            ha="center",
            va="center",
        )


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    *,
    fc: str = "#f6f6f6",
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor=fc,
        edgecolor="black",
        linewidth=1.0,
    )
    ax.add_patch(patch)
    ax.text(x + 0.16, y + h - 0.14, title, fontsize=8.5, ha="left", va="top", weight="bold")
    ax.text(
        x + w / 2,
        y + h - 0.46,
        body,
        fontsize=8.0,
        ha="center",
        va="top",
        linespacing=1.05,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a formal Bayesian/signal-processing block diagram for the STAP architecture "
            "(GLRT score + spatial-spectral prior penalty), as a vector PDF."
        )
    )
    parser.add_argument(
        "--out",
        default="figs/paper/stap_pipeline_bayes_block.pdf",
        help="Output PDF path (default: %(default)s).",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(8.2, 2.65))
    ax.set_xlim(0.0, 17.4)
    ax.set_ylim(0.0, 4.4)
    ax.axis("off")

    # Block geometry (x, y, w, h) in diagram units.
    h = 1.05
    input_blk = (0.55, 1.85, 2.65, h)
    base_blk = (3.45, 1.85, 3.40, h)
    split = (7.05, 2.375)
    stap_blk = (7.45, 3.05, 3.10, h)
    feat_blk = (7.45, 0.55, 3.10, h)
    pen_blk = (10.85, 0.55, 2.55, h)
    reg_blk = (13.65, 1.85, 3.10, h)

    # Draw blocks.
    _box(
        ax,
        *input_blk,
        title="Input IQ data",
        body=r"$X[t,y,x]\in\mathbb{C}$",
        fc="#ffffff",
    )
    _box(
        ax,
        *base_blk,
        title="Baseline clutter suppression",
        body="MC--SVD\n" + r"$\to\ X_{\mathrm{base}}$",
    )
    _box(
        ax,
        *stap_blk,
        title="STAP GLRT (uninformed)",
        body=r"$S_{\mathrm{pre}}(y,x)$",
        fc="#eef5ff",
    )
    _box(
        ax,
        *feat_blk,
        title="Feature vector",
        body=r"$(E_f,E_g,E_a),\ c_f(i)$" + "\n" + r"$\to\ \phi_i$",
        fc="#fff4e6",
    )
    _box(
        ax,
        *pen_blk,
        title="Penalty weight",
        body=r"$\gamma(\phi_i)\geq 1$",
        fc="#fff4e6",
    )
    _box(
        ax,
        *reg_blk,
        title="Post-regularization score",
        body="(shrink-only)\n" + r"$S_{\mathrm{post}}=S_{\mathrm{pre}}/\gamma(\phi_i)$",
        fc="#ffffff",
    )

    # Split node.
    ax.add_patch(Circle(split, radius=0.07, facecolor="black", edgecolor="black", linewidth=0))

    # Helper for centers/edges.
    def left_center(b: tuple[float, float, float, float]) -> tuple[float, float]:
        x, y, w, hh = b
        return (x, y + hh / 2)

    def right_center(b: tuple[float, float, float, float]) -> tuple[float, float]:
        x, y, w, hh = b
        return (x + w, y + hh / 2)

    def top_center(b: tuple[float, float, float, float]) -> tuple[float, float]:
        x, y, w, hh = b
        return (x + w / 2, y + hh)

    def bottom_center(b: tuple[float, float, float, float]) -> tuple[float, float]:
        x, y, w, hh = b
        return (x + w / 2, y)

    # Main flow arrows.
    _arrow(ax, *right_center(input_blk), *left_center(base_blk))
    _arrow(ax, *right_center(base_blk), split[0] - 0.10, split[1])

    # Branch arrows from split node.
    _arrow(ax, split[0] + 0.10, split[1], *left_center(stap_blk))
    _arrow(ax, split[0] + 0.10, split[1], *left_center(feat_blk))

    # Bottom branch (features -> penalty).
    _arrow(ax, *right_center(feat_blk), *left_center(pen_blk))

    # Recombination into shrink-only regularization.
    _arrow(ax, *right_center(stap_blk), *left_center(reg_blk))
    _arrow(ax, *top_center(pen_blk), *bottom_center(reg_blk))

    # Output.
    out_x0, out_y0 = right_center(reg_blk)
    _arrow(ax, out_x0, out_y0, 17.15, out_y0)

    fig.savefig(out_path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
