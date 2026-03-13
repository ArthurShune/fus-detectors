#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle


def _add_text(
    ax: plt.Axes,
    x: float,
    y: float,
    s: str,
    *,
    size: float | None = None,
    ha: str = "left",
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
    arrow = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=lw,
        color=color,
    )
    ax.add_patch(arrow)
    if text:
        _add_text(ax, (x0 + x1) / 2 + text_offset[0], (y0 + y1) / 2 + text_offset[1], text, size=8, ha="center")


def _cube_faces(front: tuple[float, float, float, float], dt: tuple[float, float]) -> dict[str, list[tuple[float, float]]]:
    fx, fy, fw, fh = front
    dx, dy = dt
    f0 = (fx, fy)
    f1 = (fx + fw, fy)
    f2 = (fx + fw, fy + fh)
    f3 = (fx, fy + fh)
    b0 = (fx + dx, fy + dy)
    b1 = (fx + fw + dx, fy + dy)
    b2 = (fx + fw + dx, fy + fh + dy)
    b3 = (fx + dx, fy + fh + dy)
    return {
        "front": [f0, f1, f2, f3],
        "back": [b0, b1, b2, b3],
        "side": [f1, b1, b2, f2],
        "top": [f3, b3, b2, f2],
        "edges": [f0, f1, f2, f3, b0, b1, b2, b3],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate conceptual Hankelization + local STAP diagram (vector PDF).")
    parser.add_argument(
        "--out",
        default="figs/paper/hankelization_local_stap.pdf",
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

    fig, ax = plt.subplots(figsize=(8.0, 3.1))
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 4.55)
    ax.axis("off")

    # --- Left: IQ cube (pseudo-3D)
    front = (0.7, 1.2, 3.2, 2.0)  # x, y, w, h
    dt = (0.95, 0.62)  # time offset
    faces = _cube_faces(front, dt)

    def poly(points: list[tuple[float, float]], *, fc: str, ec: str, lw: float = 1.2, z: int = 1) -> None:
        ax.add_patch(Polygon(points, closed=True, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))

    poly(faces["front"], fc="#f0f0f0", ec="black", lw=1.4, z=1)
    poly(faces["back"], fc="#f8f8f8", ec="black", lw=1.0, z=0)
    poly(faces["side"], fc="#e9e9e9", ec="black", lw=1.0, z=0)
    poly(faces["top"], fc="#ececec", ec="black", lw=1.0, z=0)

    # Re-draw key edges thicker (front + connecting)
    f0, f1, f2, f3 = faces["front"]
    b0, b1, b2, b3 = faces["back"]
    ax.plot([f0[0], f1[0], f2[0], f3[0], f0[0]], [f0[1], f1[1], f2[1], f3[1], f0[1]], color="black", lw=1.6)
    for (p, q) in [(f0, b0), (f1, b1), (f2, b2), (f3, b3)]:
        ax.plot([p[0], q[0]], [p[1], q[1]], color="black", lw=1.3)

    title_x = front[0] - 0.05
    _add_text(ax, title_x, 4.42, "Beamformed IQ cube", weight="bold", size=9.5, ha="left", va="top")
    _add_text(
        ax,
        title_x,
        4.18,
        r"$X[t,y,x]\in\mathbb{C}\quad (H\times W\times T)$",
        size=8.4,
        ha="left",
        va="top",
    )

    # Axes (x=W, y=H, t=T)
    _arrow(ax, front[0] - 0.2, front[1] - 0.35, front[0] + 0.9, front[1] - 0.35)
    _add_text(ax, front[0] + 0.95, front[1] - 0.35, r"$x\ (W)$", size=8, ha="left", va="center")
    _arrow(ax, front[0] - 0.2, front[1] - 0.35, front[0] - 0.2, front[1] + 0.55)
    _add_text(ax, front[0] - 0.25, front[1] + 0.6, r"$y\ (H)$", size=8, ha="right", va="bottom")
    _arrow(ax, f1[0] + 0.15, f1[1] + 0.1, f1[0] + 0.85, f1[1] + 0.55)
    _add_text(ax, f1[0] + 0.9, f1[1] + 0.6, r"$t\ (T)$", size=8, ha="left", va="bottom")

    # Highlight tile prism (Omega_i)
    tile_front = (1.85, 1.75, 0.9, 0.6)  # inside front face
    tx, ty, tw, th = tile_front
    dx, dy = dt
    t0 = (tx, ty)
    t1 = (tx + tw, ty)
    t2 = (tx + tw, ty + th)
    t3 = (tx, ty + th)
    u0 = (tx + dx, ty + dy)
    u1 = (tx + tw + dx, ty + dy)
    u2 = (tx + tw + dx, ty + th + dy)
    u3 = (tx + dx, ty + th + dy)
    poly([t0, t1, t2, t3], fc="#f7c7c7", ec="#7a0000", lw=1.2, z=3)
    poly([u0, u1, u2, u3], fc="#fbe2e2", ec="#7a0000", lw=1.0, z=2)
    poly([t1, u1, u2, t2], fc="#f4baba", ec="#7a0000", lw=1.0, z=2)
    poly([t3, u3, u2, t2], fc="#f4baba", ec="#7a0000", lw=1.0, z=2)
    for (p, q) in [(t0, u0), (t1, u1), (t2, u2), (t3, u3)]:
        ax.plot([p[0], q[0]], [p[1], q[1]], color="#7a0000", lw=1.0, zorder=3)
    _add_text(ax, t1[0] + 0.15, t1[1] + 0.25, r"tile $\Omega_i$", size=8.5, ha="left", va="center")

    # Inset: global Casorati stacking (MC-SVD)
    cas = Rectangle((0.85, 0.25), 2.45, 0.95, facecolor="white", edgecolor="black", linewidth=1.0)
    ax.add_patch(cas)
    for k in range(1, 6):
        x = 0.85 + 0.35 * k
        ax.plot([x, x], [0.28, 1.17], color="#808080", lw=0.8)
    _add_text(ax, 0.9, 1.25, "MC–SVD (global)", size=8.2, ha="left", va="bottom")
    _add_text(ax, 0.85 + 1.22, 0.25 + 0.48, r"$X\in\mathbb{C}^{T\times(HW)}$", size=8.2, ha="center")

    # --- Right: Local STAP branch boxes
    def box(x: float, y: float, w: float, h: float, title: str, body: str) -> Rectangle:
        r = Rectangle((x, y), w, h, facecolor="#f6f6f6", edgecolor="black", linewidth=1.0)
        ax.add_patch(r)
        _add_text(ax, x + 0.12, y + h - 0.18, title, size=9.0, ha="left", va="top", weight="bold")
        _add_text(ax, x + w / 2, y + h / 2 - 0.10, body, size=8.6, ha="center", va="center")
        return r

    x0 = 5.2
    bw = 5.2
    bh = 0.95
    b1 = box(x0, 2.95, bw, bh, "Extract tile", r"$X_{\Omega_i}\in\mathbb{C}^{T\times(hw)}$")

    # Hankel box (draw diagonal pattern)
    hank = Rectangle((x0, 1.75), bw, bh, facecolor="white", edgecolor="black", linewidth=1.0)
    ax.add_patch(hank)
    _add_text(ax, x0 + 0.12, 1.75 + bh - 0.18, "Hankelize / snapshot stack", size=9.0, ha="left", va="top", weight="bold")
    for k in range(0, 9):
        xs = x0 + 0.35 + 0.45 * k
        ax.plot([xs, xs - 0.55], [1.75 + bh - 0.25, 1.75 + 0.20], color="#888888", lw=0.9, clip_on=True)
    _add_text(ax, x0 + bw / 2, 1.75 + bh / 2 - 0.05, r"$X_i\in\mathbb{C}^{L_t\times K}$", size=8.6, ha="center", va="center")

    b3 = box(
        x0,
        0.55,
        bw,
        bh,
        "Local covariance",
        r"$\widehat R_i=\frac{1}{K}\sum_k x_{i,k}x_{i,k}^{\mathrm{H}}$",
    )

    # Arrows between boxes
    _arrow(ax, x0 + bw / 2, 2.95, x0 + bw / 2, 1.75 + bh)
    _arrow(ax, x0 + bw / 2, 1.75, x0 + bw / 2, 0.55 + bh)

    # Arrow from tile to extract (local STAP)
    tile_anchor = (u2[0] + 0.05, u2[1] - 0.1)
    extract_anchor = (x0 - 0.15, 2.95 + bh / 2)
    _arrow(ax, tile_anchor[0], tile_anchor[1], extract_anchor[0], extract_anchor[1], text="local STAP", text_offset=(-0.05, 0.34))

    fig.savefig(out_path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
