#!/usr/bin/env python3
"""
Plot strict low-FPR operating-point curves on the Gammex flow phantom views.

The current phantom story is a same-residual score decomposition rather than a
"full ROC" result. This figure therefore focuses on the strict low-FPR region
actually used in the main text and shows the reported operating points for the
baseline PD score, whitened power, unwhitened matched-subspace ratio, and the
full whitened matched-subspace STAP score.
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
        "--along-stap-summary-json",
        type=Path,
        default=Path("reports/twinkling_gammex_alonglinear17_prf2500_str6_msd_nomotion_ref_fast_f020_structural_summary.json"),
        help="Along-view STAP summary JSON (default: %(default)s).",
    )
    ap.add_argument(
        "--across-stap-summary-json",
        type=Path,
        default=Path("reports/twinkling_gammex_across17_prf2500_str4_msd_nomotion_ref_fast_f020_structural_summary.json"),
        help="Across-view STAP summary JSON (default: %(default)s).",
    )
    ap.add_argument(
        "--along-whitened-power-summary-json",
        type=Path,
        default=Path("reports/twinkling_gammex_alonglinear17_prf2500_str6_whitened_power_structural_summary.json"),
        help="Along-view whitened-power summary JSON (default: %(default)s).",
    )
    ap.add_argument(
        "--across-whitened-power-summary-json",
        type=Path,
        default=Path("reports/twinkling_gammex_across17_prf2500_str4_whitened_power_structural_summary.json"),
        help="Across-view whitened-power summary JSON (default: %(default)s).",
    )
    ap.add_argument(
        "--along-unwhitened-summary-json",
        type=Path,
        default=Path("reports/twinkling_gammex_alonglinear17_prf2500_str6_unwhitened_ratio_structural_summary.json"),
        help="Along-view unwhitened-ratio summary JSON (default: %(default)s).",
    )
    ap.add_argument(
        "--across-unwhitened-summary-json",
        type=Path,
        default=Path("reports/twinkling_gammex_across17_prf2500_str4_unwhitened_ratio_structural_summary.json"),
        help="Across-view unwhitened-ratio summary JSON (default: %(default)s).",
    )
    ap.add_argument(
        "--out-pdf",
        type=Path,
        default=Path("figs/paper/twinkling_gammex_structural_roc_curves.pdf"),
        help="Output PDF path (paper figure).",
    )
    ap.add_argument("--also-png", action="store_true", help="Also write a PNG next to the PDF.")
    args = ap.parse_args()

    along_stap = _load_json(Path(args.along_stap_summary_json))
    across_stap = _load_json(Path(args.across_stap_summary_json))
    along_wp = _load_json(Path(args.along_whitened_power_summary_json))
    across_wp = _load_json(Path(args.across_whitened_power_summary_json))
    along_uw = _load_json(Path(args.along_unwhitened_summary_json))
    across_uw = _load_json(Path(args.across_unwhitened_summary_json))

    def _get_methods(rep: dict[str, Any]) -> dict[str, Any]:
        pooled = rep.get("pooled_roc") or {}
        methods = pooled.get("methods") or {}
        if not isinstance(methods, dict) or not methods:
            raise SystemExit(f"No pooled_roc.methods found in {rep.get('root', '<unknown>')}")
        return methods  # type: ignore[return-value]

    along_methods = {
        "baseline_pd": _get_methods(along_stap)["base"],
        "whitened_power": _get_methods(along_wp)["stap_preka"],
        "unwhitened_ratio": _get_methods(along_uw)["stap_preka"],
        "stap_preka": _get_methods(along_stap)["stap_preka"],
    }
    across_methods = {
        "baseline_pd": _get_methods(across_stap)["base"],
        "whitened_power": _get_methods(across_wp)["stap_preka"],
        "unwhitened_ratio": _get_methods(across_uw)["stap_preka"],
        "stap_preka": _get_methods(across_stap)["stap_preka"],
    }

    order = [
        ("baseline_pd", r"Baseline (power Doppler)"),
        ("whitened_power", r"Whitened power"),
        ("unwhitened_ratio", r"Unwhitened ratio"),
        ("stap_preka", r"Fully whitened matched-subspace"),
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
        "baseline_pd": dict(color="#666666", linewidth=1.6, linestyle="-", marker="o", markersize=4.5),
        "whitened_power": dict(color="#ff7f0e", linewidth=1.9, linestyle="-", marker="o", markersize=4.5),
        "unwhitened_ratio": dict(color="#9467bd", linewidth=1.6, linestyle="-", marker="o", markersize=4.5),
        "stap_preka": dict(color="#1f77b4", linewidth=2.0, linestyle="-", marker="o", markersize=4.8),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.2), constrained_layout=True)
    axA, axB = axes

    def _plot(ax, methods: dict[str, Any], title: str) -> None:
        panel_xmins: list[float] = []
        panel_xmaxs: list[float] = []
        panel_ymax = 0.0
        for key, label in order:
            md = methods.get(key)
            if not isinstance(md, dict):
                continue
            x, y, ylo, yhi = _method_curve(md)
            if x.size == 0:
                continue
            st = styles.get(key, dict(color="#000000", linewidth=1.2, linestyle="-"))
            ax.plot(x, y, label=label, **st)
            panel_xmins.append(float(np.min(x)))
            panel_xmaxs.append(float(np.max(x)))
            panel_ymax = max(panel_ymax, float(np.max(y)))
            if yhi is not None:
                panel_ymax = max(panel_ymax, float(np.nanmax(yhi)))
            if ylo is not None and yhi is not None and key == "stap_preka":
                ax.fill_between(x, ylo, yhi, color=st["color"], alpha=0.14, linewidth=0)

        ax.set_xscale("log")
        if panel_xmins and panel_xmaxs:
            x_lo = max(5e-5, min(panel_xmins) * 0.8)
            x_hi = min(2e-3, max(panel_xmaxs) * 1.25)
            ax.set_xlim(x_lo, x_hi)
        upper = min(1.0, max(0.06, panel_ymax * 1.18))
        ax.set_ylim(-0.01 * upper, upper)
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel("FPR")
        ax.ticklabel_format(axis="y", style="plain")

        for x in (1e-4, 3e-4, 1e-3):
            ax.axvline(x, color="#000000", alpha=0.10, linewidth=0.8)

    _plot(axA, along_methods, "Along view")
    _plot(axB, across_methods, "Across view")
    axA.set_ylabel("TPR")

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
