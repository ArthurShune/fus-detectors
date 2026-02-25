#!/usr/bin/env python3
"""
Plot proxy-based baseline/STAP comparison on Shin RatBrain Fig3.

Consumes the scenario-matrix CSV produced by:
  PYTHONPATH=. python -m scripts.shin_ratbrain_baseline_matrix ...

and writes a paper-ready 2-panel figure:
  (A) hit_flow vs alpha (median + IQR over scenarios)
  (B) AUC(flow-proxy vs bg-proxy) distribution (boxplot)
  (C) runtime distribution (boxplot)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _summarize(df: pd.DataFrame, col: str) -> tuple[float, float, float]:
    vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.quantile(vals, 0.5)),
        float(np.quantile(vals, 0.25)),
        float(np.quantile(vals, 0.75)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Shin baseline-matrix proxy metrics (hit_flow curves + AUC).")
    ap.add_argument(
        "--in-csv",
        type=Path,
        default=Path("reports/shin_ratbrain_baseline_matrix_shinU_e970_Lt64_nomaskunion_k80.csv"),
        help="Input CSV produced by scripts/shin_ratbrain_baseline_matrix.py.",
    )
    ap.add_argument(
        "--out-pdf",
        type=Path,
        default=Path("figs/paper/shin_baseline_matrix_proxy_metrics.pdf"),
        help="Output PDF path (paper figure).",
    )
    ap.add_argument("--also-png", action="store_true", help="Also write a matching PNG next to the PDF.")
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_pdf = Path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    if df.empty:
        raise SystemExit(f"Empty CSV: {in_csv}")

    # Method order + labels (paper-facing).
    methods = [
        ("mc_svd", r"MC--SVD"),
        ("rpca", r"RPCA"),
        ("hosvd", r"HOSVD"),
        ("stap", r"STAP"),
        ("stap_raw", r"STAP-only"),
    ]

    # Alpha grid is encoded in the CSV columns as hit_flow_{tag} where tag is 1e-03, 3e-04, etc.
    # Keep a fixed (descending) order for plotting.
    alpha_tags = ["1e-01", "1e-02", "3e-03", "1e-03", "3e-04", "1e-04"]
    alpha_vals = np.array([float(x) for x in alpha_tags], dtype=np.float64)

    missing_cols: list[str] = []
    for tag in alpha_tags:
        col = f"hit_flow_{tag}"
        if col not in df.columns:
            missing_cols.append(col)
    if "auc_flow_vs_bg" not in df.columns:
        missing_cols.append("auc_flow_vs_bg")
    if missing_cols:
        raise SystemExit(f"Missing required columns in {in_csv}: {missing_cols}")

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

    colors = {
        "mc_svd": "#666666",
        "rpca": "#2ca02c",
        "hosvd": "#ff7f0e",
        "stap": "#1f77b4",
        "stap_raw": "#9467bd",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.0), constrained_layout=True)
    ax0, ax1, ax2 = axes

    # (A) hit_flow vs alpha
    for key, label in methods:
        sub = df.loc[df["method_key"] == key].copy()
        if sub.empty:
            continue
        med: list[float] = []
        q25: list[float] = []
        q75: list[float] = []
        for tag in alpha_tags:
            m, lo, hi = _summarize(sub, f"hit_flow_{tag}")
            med.append(m)
            q25.append(lo)
            q75.append(hi)

        c = colors.get(key, "#000000")
        ax0.plot(alpha_vals, med, "o-", color=c, label=label, linewidth=1.6, markersize=4)
        ax0.fill_between(alpha_vals, q25, q75, color=c, alpha=0.16, linewidth=0)

    ax0.set_xscale("log")
    ax0.invert_xaxis()
    ax0.set_ylim(-0.02, 1.02)
    ax0.set_xlabel(r"background tail rate $\alpha$")
    ax0.set_ylabel(r"hit rate on flow proxy ($\Pr\{S\geq\tau_\alpha\mid\mathrm{flow\ proxy}\}$)")
    ax0.set_title("Proxy hit rate vs background tail rate")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="lower left", frameon=False, ncol=1)

    # (B) AUC boxplot
    auc_data: list[np.ndarray] = []
    auc_labels: list[str] = []
    for key, label in methods:
        sub = df.loc[df["method_key"] == key]
        vals = pd.to_numeric(sub["auc_flow_vs_bg"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        auc_data.append(vals)
        auc_labels.append(label)

    bp = ax1.boxplot(
        auc_data,
        tick_labels=auc_labels,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "k", "linewidth": 1.2},
        boxprops={"linewidth": 1.0},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    for patch, (key, _) in zip(bp["boxes"], methods, strict=False):
        patch.set_facecolor(colors.get(key, "#cccccc"))
        patch.set_alpha(0.55)

    ax1.set_ylim(0.50, 1.01)
    ax1.set_ylabel(r"AUC($S$; flow proxy vs bg proxy)")
    ax1.set_title("Threshold-free proxy separation")
    ax1.grid(True, axis="y", alpha=0.3)

    # (C) Runtime (total time per scenario for that method).
    if "baseline_ms" in df.columns and "stap_total_ms" in df.columns:
        rt_data: list[np.ndarray] = []
        rt_labels: list[str] = []
        for key, label in methods:
            sub = df.loc[df["method_key"] == key].copy()
            if sub.empty:
                continue
            b = pd.to_numeric(sub["baseline_ms"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            s = pd.to_numeric(sub["stap_total_ms"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            b = np.where(np.isfinite(b), b, 0.0)
            s = np.where(np.isfinite(s), s, 0.0)
            tot_s = (b + s) / 1000.0
            tot_s = tot_s[np.isfinite(tot_s)]
            if tot_s.size == 0:
                continue
            rt_data.append(tot_s)
            rt_labels.append(label)

        bp2 = ax2.boxplot(
            rt_data,
            tick_labels=rt_labels,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "k", "linewidth": 1.2},
            boxprops={"linewidth": 1.0},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
        )
        for patch, (key, _) in zip(bp2["boxes"], methods, strict=False):
            patch.set_facecolor(colors.get(key, "#cccccc"))
            patch.set_alpha(0.55)
        ax2.set_yscale("log")
        ax2.set_ylabel("Runtime per scenario (s)")
        ax2.set_title("Runtime (CPU; baseline + STAP)")
        ax2.grid(True, axis="y", alpha=0.3)

    # Small note: finite-sample tail quantization at strict alphas.
    if "n_bg" in df.columns:
        try:
            n_bg = pd.to_numeric(df.loc[df["method_key"] == methods[0][0], "n_bg"], errors="coerce").median()
            if np.isfinite(n_bg) and float(n_bg) > 0:
                ax0.text(
                    0.98,
                    0.03,
                    (
                        f"n_bg~{int(n_bg):d} per scenario\n"
                        f"(alpha=1e-4 => ~{max(1, int(round(float(n_bg) * 1e-4)))} bg px)"
                    ),
                    transform=ax0.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=8.5,
                    color="#444444",
                )
        except Exception:
            pass

    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.06)
    if bool(args.also_png):
        out_png = out_pdf.with_suffix(".png")
        fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    print(f"[shin-baseline-fig] wrote {out_pdf}")


if __name__ == "__main__":
    main()
