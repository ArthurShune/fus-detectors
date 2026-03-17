#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import textwrap
from pathlib import Path

from graphviz import Digraph


def wrap_lines(text: str, width: int) -> str:
    parts: list[str] = []
    for paragraph in text.split("\n"):
        wrapped = textwrap.wrap(paragraph, width=width) or [""]
        parts.extend(wrapped)
    return "<BR/>".join(html.escape(line) for line in parts)


def boxed_label(title: str, body: str, *, fill: str = "#f7f7f7", body_width: int = 24) -> str:
    title_html = html.escape(title)
    body_html = wrap_lines(body, body_width)
    return f"""<
    <TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="10" BGCOLOR="{fill}" COLOR="black">
      <TR>
        <TD ALIGN="CENTER">
          <FONT POINT-SIZE="15"><B>{title_html}</B></FONT><BR/>
          <FONT POINT-SIZE="12">{body_html}</FONT>
        </TD>
      </TR>
    </TABLE>
    >"""


def detector_label() -> str:
    return """<
    <TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="8" BGCOLOR="#fbfcfe" COLOR="black">
      <TR>
        <TD ALIGN="CENTER">
          <FONT POINT-SIZE="15"><B>Detector family</B></FONT><BR/>
          <FONT POINT-SIZE="12">localized matched-subspace scoring</FONT>
        </TD>
      </TR>
      <TR><TD>
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="8" CELLPADDING="8">
          <TR>
            <TD BGCOLOR="#eef6ff" WIDTH="220" ALIGN="CENTER">
              <FONT POINT-SIZE="13"><B>Fixed detector</B></FONT><BR/>
              <FONT POINT-SIZE="11">non-whitened score</FONT>
            </TD>
          </TR>
          <TR>
            <TD BGCOLOR="#eef6ff" WIDTH="220" ALIGN="CENTER">
              <FONT POINT-SIZE="13"><B>Adaptive detector</B></FONT><BR/>
              <FONT POINT-SIZE="11">whitening only when guard-band clutter rises</FONT>
            </TD>
          </TR>
          <TR>
            <TD BGCOLOR="#eef6ff" WIDTH="220" ALIGN="CENTER">
              <FONT POINT-SIZE="13"><B>Fully whitened detector</B></FONT><BR/>
              <FONT POINT-SIZE="11">local covariance-adaptive variant</FONT>
            </TD>
          </TR>
        </TABLE>
      </TD></TR>
    </TABLE>
    >"""


def build_graph() -> Digraph:
    graph = Digraph("stap_pipeline_bayes_block")
    graph.attr(
        rankdir="LR",
        splines="ortho",
        nodesep="0.45",
        ranksep="0.7",
        pad="0.1",
        bgcolor="white",
    )
    graph.attr("node", shape="plain", fontname="Times-Roman")
    graph.attr("edge", arrowhead="normal", arrowsize="0.8", penwidth="1.1")

    graph.node(
        "iq",
        boxed_label("Beamformed IQ", "Complex slow-time data", fill="#f7f7f7", body_width=22),
    )
    graph.node(
        "resid",
        boxed_label("Residualization", "Conventional clutter suppression", fill="#f7f7f7", body_width=24),
    )
    graph.node(
        "tiles",
        boxed_label(
            "Local tiles",
            "Overlapping tiles with flow, guard, and alias-band summaries",
            fill="#f4f7fb",
            body_width=24,
        ),
    )
    graph.node("detector", detector_label())
    graph.node(
        "output",
        boxed_label("Output map", "Detector score map", fill="#f7f7f7", body_width=18),
    )

    graph.edge("iq", "resid")
    graph.edge("resid", "tiles")
    graph.edge("tiles", "detector")
    graph.edge("detector", "output")
    return graph


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the main method-overview figure.")
    parser.add_argument("--out", default="figs/paper/stap_pipeline_bayes_block.pdf", help="Output path.")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    graph = build_graph()
    out.write_bytes(graph.pipe(format=out.suffix.lstrip(".")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
