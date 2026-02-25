#!/usr/bin/env python3
"""
Plot structural-label ROC curves on the Twinkling Gammex flow phantom views.

We read the pooled ROC summaries produced by scripts/twinkling_eval_structural.py
with a dense --fprs grid and optional frame-bootstrap CIs.

This figure is intended as credibility scaffolding for the main text's strict
low-FPR operating-point tables: it shows the full curve shape on a log-FPR axis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _method_curve(md: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    roc = md.get("roc") or []
    xs: list[float] = []
    ys: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    have_ci = True
    for pt in roc:
        try:
            xs.append(float(pt.get("fpr_realized")))
            ys.append(float(pt.get("tpr")))
            ci = pt.get("tpr_ci95")
            if not (isinstance(ci, list) and len(ci) == 2 and all(c is not None for c in ci)):
                have_ci = False
                lo.append(float("nan"))
                hi.append(float("nan"))
            else:
                lo.append(float(ci[0]))
                hi.append(float(ci[1]))
        except Exception:
            continue
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if have_ci:
        ylo = np.asarray(lo, dtype=np.float64)[order]
        yhi = np.asarray(hi, dtype=np.float64)[order]
        return x, y, ylo, yhi
    return x, y, None, None


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Gammex structural-label ROC curves from pooled summaries.")
    ap.add_argument(
        "--along-summary-json",
        type=Path,
        default=Path("reports/twinkling_gammex_alonglinear17_prf2500_str6_msd_ka_roc_curve_summary.json"),
        help="Along-view pooled summary JSON (default: %(default)s).",
    )
    ap.add_argument(
        "--across-summary-json",
        type=Path,
        default=Path("reports/twinkling_gammex_across17_prf2500_str4_msd_ka_roc_curve_summary.json"),
        help="Across-view pooled summary JSON (default: %(default)s).",
    )
    ap.add_argument(
        "--out-pdf",
        type=Path,
        default=Path("figs/paper/twinkling_gammex_structural_roc_curves.pdf"),
        help="Output PDF path (paper figure).",
    )
    ap.add_argument("--also-png", action="store_true", help="Also write a PNG next to the PDF.")
    args = ap.parse_args()

    along = _load_json(Path(args.along_summary_json))
    across = _load_json(Path(args.across_summary_json))

    def _get_methods(rep: dict[str, Any]) -> dict[str, Any]:
        pooled = rep.get("pooled_roc") or {}
        methods = pooled.get("methods") or {}
        if not isinstance(methods, dict) or not methods:
            raise SystemExit(f"No pooled_roc.methods found in {rep.get('root', '<unknown>')}")
        return methods  # type: ignore[return-value]

    along_methods = _get_methods(along)
    across_methods = _get_methods(across)

    # Plot order + labels. We plot STAP pre-KA (primary detector) and baseline comparators.
    order = [
        ("base", r"Baseline (power Doppler)"),
        ("base_pdlog", r"Baseline (log power Doppler)"),
        ("base_kasai", r"Baseline (Kasai lag-1)"),
        ("stap_preka", r"STAP (matched-subspace, pre-KA)"),
    ]

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"matplotlib required to plot: {exc}")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    styles = {
        "base": dict(color="#666666", linewidth=1.5, linestyle="-"),
        "base_pdlog": dict(color="#999999", linewidth=1.2, linestyle="--"),
        "base_kasai": dict(color="#aaaaaa", linewidth=1.2, linestyle=":"),
        "stap_preka": dict(color="#1f77b4", linewidth=1.9, linestyle="-"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.2), constrained_layout=True)
    axA, axB = axes

    def _plot(ax, methods: dict[str, Any], title: str) -> None:
        for key, label in order:
            md = methods.get(key)
            if not isinstance(md, dict):
                continue
            x, y, ylo, yhi = _method_curve(md)
            if x.size == 0:
                continue
            st = styles.get(key, dict(color="#000000", linewidth=1.2, linestyle="-"))
            ax.plot(x, y, label=label, **st)
            if ylo is not None and yhi is not None and key == "stap_preka":
                ax.fill_between(x, ylo, yhi, color=st["color"], alpha=0.14, linewidth=0)

        ax.set_xscale("log")
        ax.set_xlim(1e-6, 1.0)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel("FPR")

        for x in (1e-4, 3e-4, 1e-3):
            ax.axvline(x, color="#000000", alpha=0.10, linewidth=0.8)

    _plot(axA, along_methods, "Gammex (along view; structural labels)")
    _plot(axB, across_methods, "Gammex (across view; structural labels)")
    axA.set_ylabel("TPR")
    axB.set_yticklabels([])

    axA.legend(loc="lower right", frameon=False, ncol=1)

    out_pdf = Path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.06)
    if bool(args.also_png):
        fig.savefig(out_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    print(f"[gammex-roc-fig] wrote {out_pdf}")


if __name__ == "__main__":
    main()

