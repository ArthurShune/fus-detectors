#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path


DOT_SOURCE = r"""
digraph G {
  graph [
    rankdir=TB,
    bgcolor="white",
    splines=ortho,
    nodesep=0.42,
    ranksep=0.50,
    pad=0.12
  ];
  node [shape=plain, fontname="Helvetica"];
  edge [color="#27313a", penwidth=1.5, arrowsize=0.8];

  input [label=<
    <TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="10" STYLE="ROUNDED" COLOR="#27313a" BGCOLOR="#f6f7f9">
      <TR><TD><B>Clutter-filtered slow-time data</B></TD></TR>
      <TR><TD>Same clutter-filtered<BR/>complex slow-time cube</TD></TR>
    </TABLE>
  >];

  shared [label=<
    <TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="8" STYLE="ROUNDED" COLOR="#27313a" BGCOLOR="#eef5fb">
      <TR><TD><B>Common setup for all variants</B></TD></TR>
      <TR><TD>Same clutter-filtered residual, tiles, and slow-time bands</TD></TR>
      <TR><TD><FONT POINT-SIZE="11"><B>Only the downstream score differs</B></FONT></TD></TR>
      <TR><TD>
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6" COLOR="#46627a">
          <TR>
            <TD BGCOLOR="#d9e6f2">extract tiles</TD>
            <TD BGCOLOR="#d9e6f2">define local neighborhoods</TD>
            <TD BGCOLOR="#d9e6f2">use shared slow-time bands</TD>
          </TR>
        </TABLE>
      </TD></TR>
      <TR><TD>
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6" COLOR="#27313a">
          <TR>
            <TD BGCOLOR="#57b881"><B>flow band P<FONT POINT-SIZE="10"><SUB>f</SUB></FONT></B></TD>
            <TD BGCOLOR="#f4c95d"><B>guard band P<FONT POINT-SIZE="10"><SUB>g</SUB></FONT></B></TD>
            <TD BGCOLOR="#ee6c73"><B>alias band P<FONT POINT-SIZE="10"><SUB>a</SUB></FONT></B></TD>
          </TR>
        </TABLE>
      </TD></TR>
    </TABLE>
  >];

  fixed [label=<
    <TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="10" STYLE="ROUNDED" COLOR="#27313a" BGCOLOR="#edf8f0">
      <TR><TD><B>Fixed statistic (default)</B></TD></TR>
      <TR><TD>Unwhitened band-limited score<BR/>No covariance estimation</TD></TR>
    </TABLE>
  >];

  adaptive [label=<
    <TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="8" STYLE="ROUNDED" COLOR="#27313a" BGCOLOR="#fff5d8">
      <TR><TD><B>Adaptive statistic</B></TD></TR>
      <TR><TD>1. score all tiles with the fixed rule</TD></TR>
      <TR><TD>2. estimate guard-band clutter level</TD></TR>
      <TR><TD>3. flag clutter-heavy tiles</TD></TR>
      <TR><TD>4. whiten and rescore flagged tiles</TD></TR>
    </TABLE>
  >];

  whitened [label=<
    <TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="10" STYLE="ROUNDED" COLOR="#27313a" BGCOLOR="#edf3fd">
      <TR><TD><B>Fully whitened</B></TD></TR>
      <TR><TD>Estimate local covariance<BR/>Whiten every tile before scoring</TD></TR>
    </TABLE>
  >];

  recon [label=<
    <TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="10" STYLE="ROUNDED" COLOR="#27313a" BGCOLOR="#f7f7f7">
      <TR><TD><B>Combine overlapping tiles</B></TD></TR>
      <TR><TD>Combine tile scores into the final<BR/>score and readout maps</TD></TR>
    </TABLE>
  >];

  { rank=same; fixed adaptive whitened }

  input -> shared;
  shared -> fixed;
  shared -> adaptive;
  shared -> whitened;
  fixed -> recon;
  adaptive -> recon;
  whitened -> recon;
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
    raise RuntimeError("Could not find a Chromium-compatible browser for PDF export.")


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


def build(out: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    out.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = root / "tmp" / "graphviz_method_overview"
    temp_dir.mkdir(parents=True, exist_ok=True)

    dot_path = temp_dir / "method_overview.dot"
    svg_path = temp_dir / "method_overview.svg"
    html_path = temp_dir / "method_overview.html"
    raw_pdf_path = temp_dir / "method_overview_raw.pdf"

    dot_path.write_text(DOT_SOURCE, encoding="utf-8")

    helper = root / "scripts" / "render_graphviz_svg.mjs"
    if not helper.exists():
        raise RuntimeError(f"Missing renderer helper: {helper}")
    if not (root / "node_modules" / "@viz-js" / "viz").exists():
        raise RuntimeError(
            "Missing @viz-js/viz. Run `npm install` from the repository root "
            "before regenerating this figure."
        )

    _run(["node", str(helper), str(dot_path), str(svg_path)], cwd=root)
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the main method-overview figure.")
    parser.add_argument("--out", default="figs/paper/stap_pipeline_bayes_block.pdf", help="Output path.")
    args = parser.parse_args()
    build(Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
