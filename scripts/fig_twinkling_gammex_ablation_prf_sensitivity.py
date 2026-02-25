#!/usr/bin/env python3
"""
Plot two small paper-facing sanity checks on the Gammex flow phantom:

1) STAP-only ablation: run STAP directly on raw (registered) IQ without the
   baseline SVD band-pass prefilter, and compare to the standard baseline->STAP
   pipeline at fixed low-FPR operating points.
2) PRF override sensitivity: rerun the same along-view pipeline with a small
   PRF sweep to check that conclusions are not brittle to the frozen PRF
   assumption required by the Twinkling dataset.

Inputs are the pooled structural summaries written by scripts/twinkling_eval_structural.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _get_methods(rep: dict[str, Any]) -> dict[str, Any]:
    pooled = rep.get("pooled_roc") or {}
    methods = pooled.get("methods") or {}
    if not isinstance(methods, dict) or not methods:
        raise ValueError("Missing pooled_roc.methods")
    return methods  # type: ignore[return-value]


def _find_pt(roc: list[dict[str, Any]], fpr_target: float) -> dict[str, Any]:
    # Prefer exact match on fpr_target (these files are generated from explicit target lists).
    for pt in roc:
        try:
            if float(pt.get("fpr_target")) == float(fpr_target):
                return pt
        except Exception:
            continue
    # Fallback: nearest.
    best = None
    best_err = float("inf")
    for pt in roc:
        try:
            err = abs(float(pt.get("fpr_target")) - float(fpr_target))
        except Exception:
            continue
        if err < best_err:
            best = pt
            best_err = err
    if best is None:
        raise ValueError(f"Could not find any point near fpr_target={fpr_target}")
    return best


def _extract_tpr(
    rep: dict[str, Any],
    *,
    method_key: str,
    fpr_target: float,
) -> tuple[float, float | None, float | None, float]:
    methods = _get_methods(rep)
    md = methods.get(method_key)
    if not isinstance(md, dict):
        raise ValueError(f"Missing method_key={method_key}")
    roc = md.get("roc") or []
    if not isinstance(roc, list) or not roc:
        raise ValueError(f"Missing roc list for method_key={method_key}")
    pt = _find_pt(roc, fpr_target)
    tpr = float(pt.get("tpr"))
    fpr_realized = float(pt.get("fpr_realized"))
    ci = pt.get("tpr_ci95")
    if isinstance(ci, list) and len(ci) == 2 and all(c is not None for c in ci):
        return tpr, float(ci[0]), float(ci[1]), fpr_realized
    return tpr, None, None, fpr_realized


def _err_from_ci(lo: float | None, hi: float | None, y: float) -> tuple[float, float] | None:
    if lo is None or hi is None:
        return None
    return (max(y - lo, 0.0), max(hi - y, 0.0))


def _write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    import csv

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in cols:
                cols.append(k)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Gammex STAP-only ablation and PRF sensitivity checks.")
    ap.add_argument(
        "--along-ref-summary-json",
        type=Path,
        default=Path("reports/twinkling_gammex_alonglinear17_prf2500_str6_msd_nomotion_ref_fast_f020_structural_summary.json"),
        help="Along-view baseline->STAP pooled structural summary (default: %(default)s).",
    )
    ap.add_argument(
        "--along-raw-summary-json",
        type=Path,
        default=Path("reports/twinkling_gammex_alonglinear17_prf2500_str6_msd_raw_f020_structural_summary.json"),
        help="Along-view STAP-only (raw baseline) pooled structural summary (default: %(default)s).",
    )
    ap.add_argument(
        "--across-ref-summary-json",
        type=Path,
        default=Path("reports/twinkling_gammex_across17_prf2500_str4_msd_nomotion_ref_fast_f020_structural_summary.json"),
        help="Across-view baseline->STAP pooled structural summary (default: %(default)s).",
    )
    ap.add_argument(
        "--across-raw-summary-json",
        type=Path,
        default=Path("reports/twinkling_gammex_across17_prf2500_str4_msd_raw_f020_structural_summary.json"),
        help="Across-view STAP-only (raw baseline) pooled structural summary (default: %(default)s).",
    )
    ap.add_argument(
        "--prf2000-summary-json",
        type=Path,
        default=Path("reports/twinkling_gammex_alonglinear17_prf2000_str6_msd_nomotion_ref_fast_f020_structural_summary.json"),
        help="Along-view PRF=2000 pooled summary (default: %(default)s).",
    )
    ap.add_argument(
        "--prf3000-summary-json",
        type=Path,
        default=Path("reports/twinkling_gammex_alonglinear17_prf3000_str6_msd_nomotion_ref_fast_f020_structural_summary.json"),
        help="Along-view PRF=3000 pooled summary (default: %(default)s).",
    )
    ap.add_argument(
        "--fprs",
        type=str,
        default="1e-4,3e-4,1e-3",
        help="Comma-separated FPR targets to summarize (default: %(default)s).",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/twinkling_gammex_ablation_prf_sensitivity.csv"),
        help="Output CSV path (default: %(default)s).",
    )
    ap.add_argument(
        "--out-pdf",
        type=Path,
        default=Path("figs/paper/twinkling_gammex_ablation_prf_sensitivity.pdf"),
        help="Output PDF path (paper figure; default: %(default)s).",
    )
    ap.add_argument("--also-png", action="store_true", help="Also write a PNG next to the PDF.")
    args = ap.parse_args()

    fprs = [float(x) for x in args.fprs.split(",") if x.strip()]
    if not fprs:
        raise SystemExit("--fprs must be non-empty")

    along_ref = _load_json(Path(args.along_ref_summary_json))
    along_raw = _load_json(Path(args.along_raw_summary_json))
    across_ref = _load_json(Path(args.across_ref_summary_json))
    across_raw = _load_json(Path(args.across_raw_summary_json))
    prf2000 = _load_json(Path(args.prf2000_summary_json))
    prf3000 = _load_json(Path(args.prf3000_summary_json))

    rows: list[dict[str, Any]] = []

    def _add_rows(
        *,
        group: str,
        view: str,
        variant: str,
        prf_hz: float,
        rep: dict[str, Any],
        method_key: str,
        fprs: Iterable[float],
    ) -> None:
        for a in fprs:
            tpr, lo, hi, fpr_realized = _extract_tpr(rep, method_key=method_key, fpr_target=a)
            rows.append(
                {
                    "group": group,
                    "view": view,
                    "variant": variant,
                    "prf_hz": prf_hz,
                    "method_key": method_key,
                    "fpr_target": a,
                    "fpr_realized": fpr_realized,
                    "tpr": tpr,
                    "tpr_ci95_lo": lo,
                    "tpr_ci95_hi": hi,
                }
            )

    # STAP-only ablation (both views at PRF=2500). We record both the baseline and STAP pre-KA.
    _add_rows(
        group="ablation",
        view="along",
        variant="svd_bandpass+stap",
        prf_hz=2500.0,
        rep=along_ref,
        method_key="stap_preka",
        fprs=fprs,
    )
    _add_rows(
        group="ablation",
        view="along",
        variant="stap_only_raw",
        prf_hz=2500.0,
        rep=along_raw,
        method_key="stap_preka",
        fprs=fprs,
    )
    _add_rows(
        group="ablation",
        view="across",
        variant="svd_bandpass+stap",
        prf_hz=2500.0,
        rep=across_ref,
        method_key="stap_preka",
        fprs=fprs,
    )
    _add_rows(
        group="ablation",
        view="across",
        variant="stap_only_raw",
        prf_hz=2500.0,
        rep=across_raw,
        method_key="stap_preka",
        fprs=fprs,
    )

    # PRF override sensitivity (along view only).
    _add_rows(
        group="prf_sensitivity",
        view="along",
        variant="svd_bandpass+stap",
        prf_hz=2000.0,
        rep=prf2000,
        method_key="stap_preka",
        fprs=fprs,
    )
    _add_rows(
        group="prf_sensitivity",
        view="along",
        variant="svd_bandpass+stap",
        prf_hz=2500.0,
        rep=along_ref,
        method_key="stap_preka",
        fprs=fprs,
    )
    _add_rows(
        group="prf_sensitivity",
        view="along",
        variant="svd_bandpass+stap",
        prf_hz=3000.0,
        rep=prf3000,
        method_key="stap_preka",
        fprs=fprs,
    )

    _write_csv(rows, Path(args.out_csv))

    # Plot: one panel for ablation (TPR@1e-3), one panel for PRF sensitivity (TPR@1e-3).
    alpha_plot = 1e-3
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

    def _row_sel(group: str, view: str, variant: str, prf: float) -> dict[str, Any]:
        for r in rows:
            if (
                r["group"] == group
                and r["view"] == view
                and r["variant"] == variant
                and float(r["prf_hz"]) == float(prf)
                and float(r["fpr_target"]) == float(alpha_plot)
                and r["method_key"] == "stap_preka"
            ):
                return r
        raise KeyError(f"Missing row for {group=} {view=} {variant=} {prf=}")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.6), constrained_layout=True)
    axA, axB = axes

    # Panel A: STAP-only ablation (along + across).
    views = ["along", "across"]
    variants = ["svd_bandpass+stap", "stap_only_raw"]
    labels = ["SVD band-pass + STAP", "STAP-only (raw IQ)"]
    colors = ["#1f77b4", "#ff7f0e"]
    x = np.arange(len(views), dtype=float)
    width = 0.33
    for j, (v, lab, col) in enumerate(zip(variants, labels, colors, strict=True)):
        ys = []
        yerr_lo = []
        yerr_hi = []
        for view in views:
            r = _row_sel("ablation", view, v, 2500.0)
            ys.append(float(r["tpr"]))
            err = _err_from_ci(r.get("tpr_ci95_lo"), r.get("tpr_ci95_hi"), float(r["tpr"]))
            if err is None:
                yerr_lo.append(0.0)
                yerr_hi.append(0.0)
            else:
                yerr_lo.append(err[0])
                yerr_hi.append(err[1])
        axA.bar(x + (j - 0.5) * width, ys, width=width, color=col, label=lab)
        axA.errorbar(
            x + (j - 0.5) * width,
            ys,
            yerr=[yerr_lo, yerr_hi],
            fmt="none",
            ecolor="#000000",
            elinewidth=1.0,
            capsize=2.5,
            alpha=0.7,
        )
    axA.set_xticks(x)
    axA.set_xticklabels(["Along view", "Across view"])
    axA.set_ylim(0.0, 1.0)
    axA.set_ylabel(r"TPR at FPR $=10^{-3}$")
    axA.set_title("STAP-only ablation (Gammex; 20 frames)")
    axA.grid(True, axis="y", alpha=0.25)
    axA.legend(loc="lower right", frameon=False)

    # Panel B: PRF override sensitivity (along view only).
    prfs = [2000.0, 2500.0, 3000.0]
    ys = []
    yerr_lo = []
    yerr_hi = []
    for prf in prfs:
        if prf == 2000.0:
            r = _row_sel("prf_sensitivity", "along", "svd_bandpass+stap", 2000.0)
        elif prf == 2500.0:
            r = _row_sel("prf_sensitivity", "along", "svd_bandpass+stap", 2500.0)
        else:
            r = _row_sel("prf_sensitivity", "along", "svd_bandpass+stap", 3000.0)
        ys.append(float(r["tpr"]))
        err = _err_from_ci(r.get("tpr_ci95_lo"), r.get("tpr_ci95_hi"), float(r["tpr"]))
        if err is None:
            yerr_lo.append(0.0)
            yerr_hi.append(0.0)
        else:
            yerr_lo.append(err[0])
            yerr_hi.append(err[1])
    axB.plot(prfs, ys, marker="o", color="#1f77b4", linewidth=1.6)
    axB.errorbar(
        prfs,
        ys,
        yerr=[yerr_lo, yerr_hi],
        fmt="none",
        ecolor="#000000",
        elinewidth=1.0,
        capsize=2.5,
        alpha=0.7,
    )
    axB.set_xlabel("PRF override (Hz)")
    axB.set_ylabel(r"TPR at FPR $=10^{-3}$")
    axB.set_ylim(0.0, 1.0)
    axB.set_title("PRF override sensitivity (along; 20 frames)")
    axB.grid(True, axis="y", alpha=0.25)

    out_pdf = Path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.06)
    if bool(args.also_png):
        fig.savefig(out_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    print(f"[gammex-ablation-prf] wrote {out_pdf} and {Path(args.out_csv)}")


if __name__ == "__main__":
    main()

