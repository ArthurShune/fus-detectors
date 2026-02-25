#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Render a publication-style B-mode ROI with structural mask overlays "
            "(flow/background) using contour lines."
        )
    )
    ap.add_argument("--bmode", type=Path, required=True, help="Input B-mode ROI reference image (grayscale PNG).")
    ap.add_argument("--mask-flow", type=Path, required=True, help="Binary flow-mask PNG (same HxW as --bmode).")
    ap.add_argument("--mask-bg", type=Path, required=True, help="Binary background-mask PNG (same HxW as --bmode).")
    ap.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    ap.add_argument(
        "--min-width",
        type=int,
        default=0,
        help="Minimum output width in pixels (sets figsize given --dpi).",
    )
    ap.add_argument("--dpi", type=int, default=300, help="Output DPI (used with --min-width).")
    ap.add_argument(
        "--line-width",
        type=float,
        default=1.8,
        help="Base contour linewidth (flow contour is slightly thicker).",
    )
    ap.add_argument(
        "--pad-frac",
        type=float,
        default=0.02,
        help="Fractional padding around the image to avoid tight cropping (default: %(default)s).",
    )
    return ap.parse_args()


def _load_luma(path: Path) -> np.ndarray:
    if not path.is_file():
        raise SystemExit(f"Missing input file: {path}")
    img = Image.open(path).convert("L")
    return np.asarray(img)


def _draw_contour(ax, mask: np.ndarray, *, color: str, linewidth: float, linestyle: str) -> None:
    # Draw a black outline under the colored line for readability on bright B-mode regions.
    ax.contour(
        mask.astype(float),
        levels=[0.5],
        colors=["black"],
        linewidths=[linewidth + 1.2],
        linestyles=[linestyle],
        origin="upper",
    )
    ax.contour(
        mask.astype(float),
        levels=[0.5],
        colors=[color],
        linewidths=[linewidth],
        linestyles=[linestyle],
        origin="upper",
    )


def main() -> int:
    args = parse_args()

    bmode = _load_luma(args.bmode)
    mask_flow = _load_luma(args.mask_flow) > 0
    mask_bg = _load_luma(args.mask_bg) > 0

    if bmode.shape != mask_flow.shape or bmode.shape != mask_bg.shape:
        raise SystemExit(
            f"Shape mismatch: bmode={bmode.shape}, flow={mask_flow.shape}, bg={mask_bg.shape} "
            f"(all must match)."
        )

    h, w = bmode.shape
    dpi = int(args.dpi)

    out_w_px = int(args.min_width) if args.min_width > 0 else w
    out_w_in = out_w_px / float(dpi)
    out_h_in = out_w_in * (h / float(w))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(out_w_in, out_h_in), dpi=dpi)
    ax.imshow(bmode, cmap="gray", vmin=0, vmax=255, interpolation="nearest", origin="upper")

    # Okabe-Ito palette: blue/orange are colorblind-friendly.
    flow_color = "#0072B2"  # blue
    bg_color = "#D55E00"  # vermillion

    lw_bg = float(args.line_width)
    lw_flow = float(args.line_width) + 0.6

    _draw_contour(ax, mask_bg, color=bg_color, linewidth=lw_bg, linestyle="--")
    _draw_contour(ax, mask_flow, color=flow_color, linewidth=lw_flow, linestyle="-")

    ax.set_axis_off()
    ax.set_aspect("equal")
    pad = float(args.pad_frac)
    pad = max(0.0, min(pad, 0.15))
    fig.subplots_adjust(left=pad, right=1.0 - pad, bottom=pad, top=1.0 - pad)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=dpi)
    plt.close(fig)

    print(
        f"[twinkling-mask-overlay] {args.bmode.name} ({w}x{h}) -> {args.out} "
        f"({out_w_px}x{int(round(out_w_px * (h / float(w))))}) dpi={dpi}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
