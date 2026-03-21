#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figs" / "paper" / "variant_selection_flowchart.pdf"
README_PNG = ROOT / "docs" / "assets" / "readme_variant_selection.png"

DOT_SOURCE = r"""
digraph G {
  graph [
    rankdir=TB,
    bgcolor="white",
    splines=ortho,
    nodesep=0.40,
    ranksep=0.58,
    pad=0.12
  ];
  node [fontname="Helvetica", color="#2b2b2b", penwidth=1.4];
  edge [fontname="Helvetica", color="#2b2b2b", penwidth=1.4, arrowsize=0.8];

  start [
    shape=box,
    style="rounded,filled",
    fillcolor="#e8f0fa",
    margin="0.34,0.28",
    fontsize=15,
    fontname="Helvetica",
    label="Start with the default fixed detector"
  ];

  measure [
    shape=box,
    style="rounded,filled",
    fillcolor="#fdf3db",
    margin="0.28,0.22",
    fontsize=13,
    label="Check a few representative windows:\nhow much clutter remains, how often tiles would switch to whitening,\nand whether your latency budget allows whitening"
  ];

  low_guard [
    shape=diamond,
    style="filled",
    fillcolor="#f6f6f6",
    margin="0.20,0.12",
    fontsize=13,
    label="Is clutter low in most windows?\nVery little need to switch tiles to whitening"
  ];

  high_guard [
    shape=diamond,
    style="filled",
    fillcolor="#f6f6f6",
    margin="0.20,0.12",
    fontsize=12,
    label="Is clutter still high often enough\nto justify whitening,\nand does latency allow it?"
  ];

  deploy_fixed [
    shape=box,
    style="rounded,filled",
    fillcolor="#e8f6ef",
    margin="0.24,0.18",
    fontsize=13,
    label="Use the fixed detector\nDefault supported by the held-out\nSIMUS benchmark"
  ];

  keep_fixed [
    shape=box,
    style="rounded,filled",
    fillcolor="#f9ebea",
    margin="0.24,0.18",
    fontsize=13,
    label="Still use the fixed detector\nMixed windows are not enough\nto justify a switch"
  ];

  use_whitened [
    shape=box,
    style="rounded,filled",
    fillcolor="#e8f0fa",
    margin="0.24,0.18",
    fontsize=13,
    label="Use the fully whitened detector\nwhen clutter stays high often enough\nand latency allows it"
  ];

  { rank=same; deploy_fixed keep_fixed use_whitened }

  start -> measure;
  measure -> low_guard;
  low_guard -> deploy_fixed [label="yes", labeldistance=2.0];
  low_guard -> high_guard [label="no", labeldistance=2.0];
  high_guard -> keep_fixed [label="no", labeldistance=2.0];
  high_guard -> use_whitened [label="yes", labeldistance=2.0];

  deploy_fixed -> keep_fixed [style=invis];
  keep_fixed -> use_whitened [style=invis];
}
"""


def _find_browser() -> str:
    for name in ("chromium", "chromium-browser", "google-chrome"):
        path = shutil.which(name)
        if path:
            return path
    snap_path = Path("/snap/bin/chromium")
    if snap_path.exists():
        return str(snap_path)
    raise RuntimeError("Could not find a Chromium-compatible browser for export.")


def _svg_size(svg: str) -> tuple[str, str]:
    match = re.search(r'width="([^"]+)"\s+height="([^"]+)"', svg)
    if not match:
        raise RuntimeError("Could not parse rendered SVG size.")
    return match.group(1), match.group(2)


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    try:
        subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing required tool: {cmd[0]}") from exc


def build_pdf(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = ROOT / "tmp" / "graphviz_variant_selection"
    temp_dir.mkdir(parents=True, exist_ok=True)

    dot_path = temp_dir / "variant_selection.dot"
    svg_path = temp_dir / "variant_selection.svg"
    html_path = temp_dir / "variant_selection.html"
    raw_pdf_path = temp_dir / "variant_selection_raw.pdf"

    dot_path.write_text(DOT_SOURCE, encoding="utf-8")

    helper = ROOT / "scripts" / "render_graphviz_svg.mjs"
    if not helper.exists():
        raise RuntimeError(f"Missing renderer helper: {helper}")
    if not (ROOT / "node_modules" / "@viz-js" / "viz").exists():
        raise RuntimeError(
            "Missing @viz-js/viz. Run `npm install` from the repository root "
            "before regenerating this figure."
        )

    _run(["node", str(helper), str(dot_path), str(svg_path)], cwd=ROOT)
    svg = svg_path.read_text(encoding="utf-8")
    width, height = _svg_size(svg)
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>@page {{ size: {width} {height}; margin: 0; }}"
        "html,body{margin:0;padding:0;background:#fff;}"
        f"svg{{display:block;width:{width};height:{height};}}"
        "</style></head><body>"
        f"{svg}</body></html>"
    )
    html_path.write_text(html, encoding="utf-8")

    browser = _find_browser()
    _run(
        [
            browser,
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            f"--print-to-pdf={raw_pdf_path}",
            "--print-to-pdf-no-header",
            html_path.as_uri(),
        ]
    )
    _run(["pdfcrop", str(raw_pdf_path), str(out)])


def build_readme_png(pdf_path: Path, png_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    base = png_path.with_suffix("")
    _run(["pdftoppm", "-png", "-r", "220", "-singlefile", str(pdf_path), str(base)])


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the detector-variant selection flowchart.")
    parser.add_argument("--out", default=str(OUT), help="Output PDF path.")
    parser.add_argument("--readme-png", default=str(README_PNG), help="Output README PNG path.")
    args = parser.parse_args()

    out = Path(args.out)
    readme_png = Path(args.readme_png)
    build_pdf(out)
    build_readme_png(out, readme_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
