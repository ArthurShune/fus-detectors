#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Upscale a PNG using nearest-neighbor resampling (for crisp masks).")
    ap.add_argument("--in", dest="in_path", type=Path, required=True, help="Input PNG path.")
    ap.add_argument("--out", dest="out_path", type=Path, required=True, help="Output PNG path.")
    ap.add_argument(
        "--scale",
        type=int,
        default=0,
        help="Integer scale factor (overrides --min-width if provided).",
    )
    ap.add_argument(
        "--min-width",
        type=int,
        default=0,
        help="Minimum output width in pixels (chooses an integer scale factor >=1).",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    if not args.in_path.is_file():
        raise SystemExit(f"Missing input PNG: {args.in_path}")

    if args.scale <= 0 and args.min_width <= 0:
        raise SystemExit("Must provide either --scale or --min-width.")

    img = Image.open(args.in_path)
    w, h = img.size

    if args.scale > 0:
        scale = int(args.scale)
    else:
        scale = max(1, (int(args.min_width) + w - 1) // w)

    out_w, out_h = w * scale, h * scale

    img_up = img.resize((out_w, out_h), resample=Image.Resampling.NEAREST)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    img_up.save(args.out_path, optimize=True)

    print(f"[upscale-png] {args.in_path} ({w}x{h}) -> {args.out_path} ({out_w}x{out_h}) scale={scale}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

