#!/usr/bin/env python
"""
Generate a block diagram of the clinical STAP PD pipeline using graphviz.

This script produces a simple left-to-right block diagram that captures the
major stages:

  k-Wave / acquisition -> MC-SVD baseline -> baseline PD / flow masks
  -> temporal STAP core (Hankelization, robust covariance, batched band-energy)
  -> conditional STAP gating -> clinical STAP PD map -> ROC / metrics.

Usage
-----
    python scripts/diagram_stap_pipeline.py [out_prefix]

By default, the output prefix is 'stap_pipeline'. The script will emit both
PDF and PNG via graphviz (assuming 'dot' is available on the system PATH).

Dependencies
------------
  - Python 'graphviz' package (pip install graphviz)
  - Graphviz command-line tools (e.g., 'dot')
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from graphviz import Digraph
except ImportError as exc:  # pragma: no cover - convenience message
    raise SystemExit(
        "This script requires the 'graphviz' Python package. "
        "Install it with 'pip install graphviz' and ensure Graphviz "
        "is installed on your system."
    ) from exc


def build_diagram() -> Digraph:
    """Construct the STAP pipeline block diagram."""
    dot = Digraph("STAPPipeline", format="pdf")
    dot.attr(rankdir="LR", splines="ortho", nodesep="0.6", ranksep="0.8")
    dot.attr("node", shape="box", style="rounded,filled", fillcolor="white", fontsize="10")

    # Acquisition and baseline path
    dot.node(
        "sim",
        "k-Wave / acquisition\nRF → beamforming → compounded frames",
    )
    dot.node(
        "mcsvd",
        "MC–SVD clutter filter\n(spatiotemporal SVD)",
    )
    dot.node(
        "pd_base",
        "Baseline PD map\nMC–SVD residual power",
    )

    # Flow / background masks
    dot.node(
        "flowmask",
        "Flow / background masks\n(geometry + PD thresholding)",
    )

    # Temporal STAP core
    dot.node(
        "hankel",
        "Hankelization per tile\nslow-time stacks (Lt × N)",
    )
    dot.node(
        "cov",
        "Robust covariance per tile\n+ shrinkage / diagonal loading",
    )
    dot.node(
        "band",
        "Batched band-energy core\n(flow-band fraction, PD fast path)",
    )

    # Conditional STAP gating and output
    dot.node(
        "gate",
        "Conditional STAP gating\n(skip zero-flow tiles)",
    )
    dot.node(
        "pd_stap",
        "Clinical STAP PD map\n(baseline PD + STAP on flow tiles)",
    )
    dot.node(
        "roc",
        "ROC / metrics\n(score S = −PD)",
    )

    # Edges: baseline path
    dot.edge("sim", "mcsvd", label="", fontsize="8")
    dot.edge("mcsvd", "pd_base", label="", fontsize="8")
    dot.edge("pd_base", "flowmask", label="", fontsize="8")

    # Edges: STAP temporal core path
    dot.edge("sim", "hankel", label="", fontsize="8")
    dot.edge("hankel", "cov", label="", fontsize="8")
    dot.edge("cov", "band", label="", fontsize="8")

    # Edges: gating and output
    dot.edge("band", "gate", label="", fontsize="8")
    dot.edge("pd_base", "gate", label="baseline PD", fontsize="8")
    dot.edge("flowmask", "gate", label="flow mask", fontsize="8")
    dot.edge("gate", "pd_stap", label="", fontsize="8")
    dot.edge("pd_stap", "roc", label="", fontsize="8")

    return dot


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    out_prefix = argv[0] if argv else "stap_pipeline"
    out_prefix_path = Path(out_prefix)

    dot = build_diagram()
    # Render to PDF
    pdf_path = dot.render(filename=str(out_prefix_path), format="pdf", cleanup=True)
    # Also render PNG for quick viewing
    png_path = dot.render(filename=str(out_prefix_path), format="png", cleanup=False)

    print(f"Rendered STAP pipeline diagram to {pdf_path} and {png_path}")


if __name__ == "__main__":
    main()
