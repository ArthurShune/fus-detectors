#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class FigureQC:
    path: str
    kind: str  # "pdf" | "png"
    rendered_png: str | None
    width: int
    height: int
    margins_px: dict[str, int]
    border_ink_frac: dict[str, float]
    warnings: list[str]


def _render_pdf_first_page(pdf_path: Path, *, out_prefix: Path, dpi: int) -> Path:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    # pdftoppm writes `<prefix>-1.png` when rendering a single page without -singlefile.
    cmd = [
        "pdftoppm",
        "-png",
        "-r",
        str(int(dpi)),
        "-f",
        "1",
        "-l",
        "1",
        str(pdf_path),
        str(out_prefix),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    out_path = out_prefix.parent / f"{out_prefix.name}-1.png"
    if not out_path.is_file():
        raise RuntimeError(f"pdftoppm did not produce expected output: {out_path}")
    return out_path


def _analyze_ink(image: Image.Image, *, ink_thresh: int) -> tuple[dict[str, int], dict[str, float], list[str]]:
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.uint8)

    # Treat near-white as background.
    ink = arr < int(ink_thresh)
    h, w = ink.shape

    border = {
        "left": float(np.mean(ink[:, 0])),
        "right": float(np.mean(ink[:, -1])),
        "top": float(np.mean(ink[0, :])),
        "bottom": float(np.mean(ink[-1, :])),
    }

    ys, xs = np.nonzero(ink)
    if ys.size == 0 or xs.size == 0:
        # All-white: define margins as full image.
        margins = {"left": w, "right": w, "top": h, "bottom": h}
        warnings = ["no_ink_detected(all_white_or_threshold_too_low)"]
        return margins, border, warnings

    x0 = int(xs.min())
    x1 = int(xs.max())
    y0 = int(ys.min())
    y1 = int(ys.max())

    margins = {
        "left": x0,
        "right": (w - 1) - x1,
        "top": y0,
        "bottom": (h - 1) - y1,
    }

    warnings: list[str] = []
    if min(margins.values()) <= 4:
        warnings.append("very_small_margin_px(tight_bbox_or_clipping)")
    if max(border.values()) >= 0.01:
        warnings.append("ink_on_image_border(possible_clipping_or_frame)")

    return margins, border, warnings


def _make_montage(
    items: list[FigureQC],
    *,
    out_path: Path,
    max_thumb_w: int,
    max_thumb_h: int,
    cols: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pick source image per item (rendered for PDFs, native for PNGs).
    src_paths: list[Path] = []
    labels: list[str] = []
    for it in items:
        src = Path(it.rendered_png) if it.rendered_png is not None else Path(it.path)
        src_paths.append(src)
        labels.append(Path(it.path).name)

    thumbs: list[Image.Image] = []
    for p in src_paths:
        img = Image.open(p).convert("RGB")
        img.thumbnail((int(max_thumb_w), int(max_thumb_h)), Image.Resampling.LANCZOS)
        thumbs.append(img)

    # Cell layout.
    cell_w = max(t.width for t in thumbs) + 28
    cell_h = max(t.height for t in thumbs) + 50
    rows = int(math.ceil(len(thumbs) / float(cols)))

    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for idx, (thumb, label) in enumerate(zip(thumbs, labels, strict=True)):
        r = idx // cols
        c = idx % cols
        x = c * cell_w
        y = r * cell_h

        # Paste centered.
        px = x + (cell_w - thumb.width) // 2
        py = y + 8
        canvas.paste(thumb, (px, py))

        # Label.
        tx = x + 8
        ty = y + cell_h - 20
        draw.text((tx, ty), label, fill=(20, 20, 20), font=font)

    canvas.save(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Lightweight QC for figures under figs/paper (margins + border ink).")
    ap.add_argument("--fig-dir", type=Path, default=Path("figs/paper"), help="Figure directory (default: %(default)s).")
    ap.add_argument("--out-json", type=Path, default=Path("reports/figure_qc/figure_qc.json"))
    ap.add_argument("--render-dir", type=Path, default=Path("reports/figure_qc/rendered"))
    ap.add_argument("--out-montage", type=Path, default=Path("reports/figure_qc/figure_montage.png"))
    ap.add_argument("--dpi", type=int, default=300, help="PDF render DPI (default: %(default)s).")
    ap.add_argument("--ink-thresh", type=int, default=250, help="Ink threshold in [0,255] (default: %(default)s).")
    ap.add_argument("--cols", type=int, default=3, help="Montage columns (default: %(default)s).")
    args = ap.parse_args()

    fig_dir = Path(args.fig_dir)
    if not fig_dir.is_dir():
        raise SystemExit(f"Figure dir not found: {fig_dir}")

    paths = sorted([p for p in fig_dir.iterdir() if p.is_file() and p.suffix.lower() in {'.pdf', '.png'}])
    if not paths:
        raise SystemExit(f"No .pdf/.png figures found under: {fig_dir}")

    items: list[FigureQC] = []
    for p in paths:
        kind = p.suffix.lower().lstrip(".")
        rendered: Path | None = None
        img_path = p
        if kind == "pdf":
            rendered = _render_pdf_first_page(
                p,
                out_prefix=Path(args.render_dir) / p.stem,
                dpi=int(args.dpi),
            )
            img_path = rendered

        img = Image.open(img_path)
        margins, border, warnings = _analyze_ink(img, ink_thresh=int(args.ink_thresh))

        items.append(
            FigureQC(
                path=str(p),
                kind=kind,
                rendered_png=str(rendered) if rendered is not None else None,
                width=int(img.width),
                height=int(img.height),
                margins_px=margins,
                border_ink_frac=border,
                warnings=warnings,
            )
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump([it.__dict__ for it in items], f, indent=2, sort_keys=False)
        f.write("\n")

    _make_montage(
        items,
        out_path=Path(args.out_montage),
        max_thumb_w=780,
        max_thumb_h=520,
        cols=int(args.cols),
    )

    warn_count = sum(1 for it in items if it.warnings)
    print(f"[figure-qc] checked {len(items)} figures; {warn_count} with warnings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

