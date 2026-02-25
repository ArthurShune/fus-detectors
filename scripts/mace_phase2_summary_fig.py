#!/usr/bin/env python3
"""
Plot the Macé Phase 2.1 summary figure (Pf occupancy vs alias separation).

This consumes the per-plane CSV produced by:
  PYTHONPATH=. python scripts/mace_phase2_sweep.py --out-csv reports/mace_phase2_summary.csv

and writes a publication-ready figure:
  figs/paper/mace_phase2_summary.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Macé Phase 2.1 summary figure from CSV.")
    ap.add_argument(
        "--in-csv",
        type=Path,
        default=Path("reports/mace_phase2_summary.csv"),
        help="Input CSV produced by scripts/mace_phase2_sweep.py.",
    )
    ap.add_argument(
        "--out-png",
        type=Path,
        default=Path("figs/paper/mace_phase2_summary.png"),
        help="Output PNG path (paper figure).",
    )
    ap.add_argument("--also-pdf", action="store_true", help="Also write a matching PDF next to the PNG.")
    ap.add_argument(
        "--offline-thresh-h1",
        type=float,
        default=0.4,
        help="Offline descriptive threshold line for Pf peak fraction in H1 tiles (x-axis).",
    )
    ap.add_argument(
        "--offline-thresh-h0",
        type=float,
        default=0.25,
        help="Offline descriptive threshold line for Pf peak fraction in H0 tiles (y-axis).",
    )
    ap.add_argument(
        "--max-labels",
        type=int,
        default=8,
        help="Maximum number of contract-positive points to label (by descending x).",
    )
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    if df.empty:
        raise SystemExit(f"Empty CSV: {in_csv}")

    required = ["scan_name", "plane_idx", "frac_pf_peak_pos", "frac_pf_peak_neg", "delta_log_alias"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in {in_csv}: {missing}")

    x = pd.to_numeric(df["frac_pf_peak_pos"], errors="coerce").astype(float)
    y = pd.to_numeric(df["frac_pf_peak_neg"], errors="coerce").astype(float)
    c = pd.to_numeric(df["delta_log_alias"], errors="coerce").astype(float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    if not bool(ok.any()):
        raise SystemExit(f"No finite rows found in {in_csv} for required columns.")

    x_ok = x[ok].to_numpy()
    y_ok = y[ok].to_numpy()
    c_ok = c[ok].to_numpy()
    scan_ok = df.loc[ok, "scan_name"].astype(str).to_numpy()
    plane_ok = pd.to_numeric(df.loc[ok, "plane_idx"], errors="coerce").fillna(-1).astype(int).to_numpy()

    x_thr = float(args.offline_thresh_h1)
    y_thr = float(args.offline_thresh_h0)
    subset = (x_ok > x_thr) & (y_ok < y_thr)

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    sc = ax.scatter(
        x_ok,
        y_ok,
        c=c_ok,
        cmap="coolwarm",
        s=60,
        alpha=0.85,
        edgecolors="k",
        linewidths=0.25,
    )

    if bool(subset.any()):
        ax.scatter(
            x_ok[subset],
            y_ok[subset],
            s=180,
            facecolors="none",
            edgecolors="#f2c200",
            linewidths=2.5,
            label="contract-positive subset",
            zorder=3,
        )

        # Label up to max-labels subset points (prefer higher x, then lower y).
        idx = np.where(subset)[0]
        order = np.lexsort((y_ok[idx], -x_ok[idx]))
        idx = idx[order][: max(0, int(args.max_labels))]
        for j in idx:
            label = f"{scan_ok[j]}/p{plane_ok[j]}"
            ax.annotate(
                label,
                (x_ok[j], y_ok[j]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
                ha="left",
                va="bottom",
            )

    ax.axvline(x_thr, color="k", linestyle="--", linewidth=1.2, alpha=0.6)
    ax.axhline(y_thr, color="k", linestyle="--", linewidth=1.2, alpha=0.6)

    ax.set_title("Mac\u00e9 Phase 2.1 sweep: Pf occupancy vs alias separation")
    ax.set_xlabel("Pf peak fraction in H1 tiles")
    ax.set_ylabel("Pf peak fraction in H0 tiles")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)

    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"$\Delta\,\log(E_a/E_f)=\mathrm{median}(H_1)-\mathrm{median}(H_0)$")

    if bool(subset.any()):
        ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    if args.also_pdf:
        out_pdf = out_png.with_suffix(".pdf")
        fig.savefig(out_pdf, bbox_inches="tight")

    print(f"[mace-phase2 fig] wrote {out_png}")
    if args.also_pdf:
        print(f"[mace-phase2 fig] wrote {out_pdf}")


if __name__ == "__main__":
    main()

