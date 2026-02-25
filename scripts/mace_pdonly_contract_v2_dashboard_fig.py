#!/usr/bin/env python3
"""
Phase 3: Plot a PD-only KA prior summary panel for Macé/Urban plane sweeps.

This consumes the per-plane CSV produced by:
  PYTHONPATH=. python scripts/mace_pdonly_contract_v2_sweep.py \
    --out-csv reports/mace_pdonly_contract_v2.csv

and writes a compact summary panel suitable for inclusion in the paper.

The panel explicitly separates:
  - runtime label-free prior outputs (regimes, reasons, coverage/strength)
  - offline atlas-labeled descriptive context (Pf peak H1 vs H0)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except Exception:
        pass
    return f"{float(value):.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Macé PD-only KA prior summary panel from CSV.")
    ap.add_argument(
        "--in-csv",
        type=Path,
        default=Path("reports/mace_pdonly_contract_v2.csv"),
        help="Input per-plane CSV (Phase 2 output).",
    )
    ap.add_argument(
        "--out-png",
        type=Path,
        default=Path("figs/paper/mace_pdonly_contract_v2_dashboard.png"),
        help="Output PNG path (paper figure).",
    )
    ap.add_argument("--also-pdf", action="store_true", help="Also write a matching PDF next to the PNG")
    ap.add_argument(
        "--show-source",
        action="store_true",
        help="Include the input CSV path in the figure title (debug/provenance; not for paper).",
    )
    ap.add_argument("--max-reasons", type=int, default=8, help="Maximum state/reason bars to plot")
    ap.add_argument(
        "--offline-thresh-h1",
        type=float,
        default=0.4,
        help="Offline Macé contract-positive selection line (Pf peak fraction H1).",
    )
    ap.add_argument(
        "--offline-thresh-h0",
        type=float,
        default=0.25,
        help="Offline Macé contract-positive selection line (Pf peak fraction H0).",
    )
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    if df.empty:
        raise SystemExit(f"Empty CSV: {in_csv}")

    required = [
        "ka_contract_v2_state",
        "ka_contract_v2_reason",
        "scan_group",
        "plane_idx",
        "ka_contract_v2_p_shrink",
        "ka_contract_v2_iqr_logw_gated",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in {in_csv}: {missing}")

    df["state"] = df["ka_contract_v2_state"].fillna("NA").astype(str)
    df["reason"] = df["ka_contract_v2_reason"].fillna("NA").astype(str)
    df["state_reason"] = df["state"] + "/" + df["reason"]

    state_order = ["C0_OFF", "C1_SAFETY", "C2_UPLIFT"]
    state_colors = {
        "C0_OFF": "#9e9e9e",
        "C1_SAFETY": "#1f77b4",
        "C2_UPLIFT": "#ff7f0e",
    }
    state_short = {
        "C0_OFF": r"$\mathcal{R}_0$",
        "C1_SAFETY": r"$\mathcal{R}_1$",
        "C2_UPLIFT": r"$\mathcal{R}_2$",
    }

    state_counts = df["state"].value_counts().reindex(state_order).fillna(0).astype(int)
    state_counts = state_counts[state_counts > 0]

    sr_counts = df["state_reason"].value_counts()
    sr_counts = sr_counts.sort_values(ascending=False).head(max(1, int(args.max_reasons)))

    # "Enabled" here means contract state ok (C1 or C2). Phase 2 is telemetry-only.
    enabled_mask = df["state"].isin(["C1_SAFETY", "C2_UPLIFT"]) & (df["reason"] == "ok")
    df_enabled = df[enabled_mask].copy()

    # Deferred imports (matplotlib can be slow).
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 220,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0))

    # Panel A: prior regime histogram (runtime / label-free).
    ax = axes[0, 0]
    xs_state = list(state_counts.index)
    xs = [state_short.get(x, x) for x in xs_state]
    ys = state_counts.values
    colors = [state_colors.get(x, "#cccccc") for x in xs_state]
    bars = ax.bar(xs, ys, color=colors)
    ax.set_title("PD-only prior regime (Runtime, Label-Free)")
    ax.set_ylabel("Plane count")
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylim(0, max(ys) + 2)
    for rect, y in zip(bars, ys):
        ax.text(rect.get_x() + rect.get_width() / 2.0, y + 0.1, f"{int(y)}", ha="center", va="bottom")

    # Panel B: top (state, reason) combos.
    ax = axes[0, 1]
    ax.set_title("Regime / Reason (Top)")

    def _pretty_state_reason(sr: str) -> str:
        if "/" not in sr:
            return sr
        st, rsn = sr.split("/", 1)
        st = state_short.get(st, st)
        rsn = rsn.replace("_", " ")
        return f"{st}/{rsn}"

    ylabels = [_pretty_state_reason(x) for x in list(sr_counts.index)[::-1]]
    yvals = list(sr_counts.values)[::-1]
    bars = ax.barh(ylabels, yvals, color="#4c78a8")
    ax.set_xlabel("Plane count")
    ax.set_xlim(0, max(yvals) + 2)
    for rect, y in zip(bars, yvals):
        ax.text(y + 0.1, rect.get_y() + rect.get_height() / 2.0, f"{int(y)}", va="center")

    # Panel C: coverage vs strength (runtime / label-free).
    ax = axes[1, 0]
    ax.set_title("Coverage vs Strength (Runtime, Label-Free)")
    x = df["ka_contract_v2_p_shrink"].astype(float)
    y = df["ka_contract_v2_iqr_logw_gated"].astype(float)
    # Color by state, alpha by "ok" vs other.
    for state in ["C0_OFF", "C1_SAFETY", "C2_UPLIFT"]:
        m_state = df["state"] == state
        if not m_state.any():
            continue
        alpha = np.where(df.loc[m_state, "reason"].astype(str).values == "ok", 0.8, 0.4)
        ax.scatter(
            x[m_state],
            y[m_state],
            s=18,
            c=state_colors.get(state, "#cccccc"),
            alpha=alpha,
            label=state_short.get(state, state),
            edgecolors="none",
        )
    # Invariance threshold (from config; plot median if present).
    if "ka_contract_v2_cfg_iqr_logw_min" in df.columns:
        thr = float(pd.to_numeric(df["ka_contract_v2_cfg_iqr_logw_min"], errors="coerce").median())
        if np.isfinite(thr):
            ax.axhline(thr, color="k", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.text(0.99, thr + 0.002, f"iqr_logw_min={thr:.3f}", ha="right", va="bottom", transform=ax.get_yaxis_transform())
    ax.set_xlabel(r"$p_{\mathrm{shrink}}$ (candidate gated fraction)")
    ax.set_ylabel(r"$\mathrm{IQR}(\log w)\ \mathrm{on\ gated}$")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=False)

    # Add a small text box summarizing "active penalty" stats.
    n_total = int(len(df))
    n_enabled = int(len(df_enabled))
    p50_p = float(np.nanmedian(df_enabled["ka_contract_v2_p_shrink"])) if n_enabled else float("nan")
    p50_iqr = float(np.nanmedian(df_enabled["ka_contract_v2_iqr_logw_gated"])) if n_enabled else float("nan")
    lines = [
        f"planes={n_total}",
        f"penalty active (R1/R2; ok)={n_enabled}",
        f"p_shrink p50={_format_float(p50_p)}",
        f"iqr_logw p50={_format_float(p50_iqr)}",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    # Panel D: offline atlas-labeled descriptive scatter (kept separate by design).
    ax = axes[1, 1]
    ax.set_title("Offline (Atlas-Labeled, Descriptive Only)")
    required_off = ["offline_pf_peak_frac_H1", "offline_pf_peak_frac_H0", "offline_delta_logEaEf"]
    if all(c in df.columns for c in required_off):
        xh1 = pd.to_numeric(df["offline_pf_peak_frac_H1"], errors="coerce")
        xh0 = pd.to_numeric(df["offline_pf_peak_frac_H0"], errors="coerce")
        cval = pd.to_numeric(df["offline_delta_logEaEf"], errors="coerce")
        ok = np.isfinite(xh1) & np.isfinite(xh0) & np.isfinite(cval)
        if ok.any():
            sc = ax.scatter(
                xh1[ok],
                xh0[ok],
                c=cval[ok],
                cmap="coolwarm",
                s=22,
                alpha=0.8,
                edgecolors="none",
            )
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(r"$\Delta\log(E_a/E_f)$ (H1 - H0)")
        else:
            ax.text(0.5, 0.5, "No labeled planes", ha="center", va="center")
        ax.set_xlabel("Pf peak frac (H1 tiles)")
        ax.set_ylabel("Pf peak frac (H0 tiles)")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        # Selection lines used in Phase 2.1 (descriptive).
        ax.axvline(float(args.offline_thresh_h1), color="k", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.axhline(float(args.offline_thresh_h0), color="k", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.grid(True, alpha=0.25)
    else:
        ax.axis("off")
        ax.text(0.0, 1.0, "Offline panel unavailable\n(missing offline_* columns)", va="top", ha="left")

    suptitle = "Macé PD-only KA prior summary panel"
    if args.show_source:
        suptitle = f"{suptitle} ({in_csv.as_posix()})"
    fig.suptitle(suptitle, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, bbox_inches="tight")
    if args.also_pdf:
        out_pdf = out_png.with_suffix(".pdf")
        fig.savefig(out_pdf, bbox_inches="tight")

    print(f"[mace-pdonly-v2 fig] wrote {out_png}")
    if args.also_pdf:
        print(f"[mace-pdonly-v2 fig] wrote {out_pdf}")


if __name__ == "__main__":
    main()
