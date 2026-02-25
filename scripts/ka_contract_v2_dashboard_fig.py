#!/usr/bin/env python
"""
Generate a small KA prior summary-panel figure from KA Contract v2 sweep CSV output.

Typical usage:
  PYTHONPATH=. python scripts/ka_contract_v2_dashboard_fig.py \
    --in-csv reports/ka_v2_sweep_T64.csv \
    --out-png reports/ka_v2_sweep_T64_dashboard.png \
    --also-pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _truthy_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["1", "true", "yes", "y", "t"])


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
    parser = argparse.ArgumentParser(description="Plot a KA prior summary-panel figure from a sweep CSV.")
    parser.add_argument("--in-csv", type=str, required=True, help="Input CSV, e.g. reports/ka_v2_sweep_T64.csv")
    parser.add_argument("--out-png", type=str, required=True, help="Output PNG path")
    parser.add_argument("--also-pdf", action="store_true", help="Also write a matching PDF next to the PNG")
    parser.add_argument(
        "--show-source",
        action="store_true",
        help="Include the input CSV path in the figure (debug/provenance; not for paper).",
    )
    parser.add_argument("--max-reasons", type=int, default=6, help="Maximum state/reason bars to plot")
    args = parser.parse_args()

    in_csv = Path(args.in_csv)
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    if df.empty:
        raise SystemExit(f"Empty CSV: {in_csv}")

    state_col = "ka_contract_v2_state"
    reason_col = "ka_contract_v2_reason"
    applied_col = "score_ka_v2_applied"

    for col in [state_col, reason_col, applied_col]:
        if col not in df.columns:
            raise SystemExit(f"Missing required column '{col}' in {in_csv}")

    df["state"] = df[state_col].fillna("NA").astype(str)
    df["reason"] = df[reason_col].fillna("NA").astype(str)
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

    applied_mask = _truthy_mask(df[applied_col])
    df_applied = df[applied_mask].copy()

    n_bundles = int(len(df))
    n_applied = int(len(df_applied))

    def _median(col: str) -> float | None:
        if col not in df_applied.columns or df_applied.empty:
            return None
        val = df_applied[col].median()
        try:
            if pd.isna(val):
                return None
        except Exception:
            pass
        return float(val)

    def _max(col: str) -> float | None:
        if col not in df_applied.columns or df_applied.empty:
            return None
        val = df_applied[col].max()
        try:
            if pd.isna(val):
                return None
        except Exception:
            pass
        return float(val)

    p_shrink_p50 = _median("ka_contract_v2_p_shrink")
    scaled_frac_p50 = _median("score_ka_v2_scaled_pixel_fraction")
    scale_p90_p50 = _median("score_ka_v2_scale_p90")
    scale_max = _max("score_ka_v2_scale_max")

    # Deferred imports (matplotlib can be slow; keep CLI help snappy).
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

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1.0, 1.2, 1.1]})

    # Panel A: prior regime histogram.
    ax = axes[0]
    xs_state = list(state_counts.index)
    xs = [state_short.get(x, x) for x in xs_state]
    ys = state_counts.values
    colors = [state_colors.get(x, "#cccccc") for x in xs_state]
    bars = ax.bar(xs, ys, color=colors)
    ax.set_title("KA prior regime")
    ax.set_ylabel("Bundle count")
    ax.set_ylim(0, max(ys) + 1)
    ax.tick_params(axis="x", rotation=30)
    for rect, y in zip(bars, ys):
        ax.text(rect.get_x() + rect.get_width() / 2.0, y + 0.05, f"{int(y)}", ha="center", va="bottom")

    # Panel B: (regime, reason) histogram.
    ax = axes[1]
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
    ax.set_xlabel("Bundle count")
    ax.set_xlim(0, max(yvals) + 1)
    for rect, y in zip(bars, yvals):
        ax.text(y + 0.05, rect.get_y() + rect.get_height() / 2.0, f"{int(y)}", va="center")

    # Panel C: applied stats.
    ax = axes[2]
    ax.axis("off")

    state_counts_all = df["state"].value_counts().to_dict()
    c0 = int(state_counts_all.get("C0_OFF", 0))
    c1 = int(state_counts_all.get("C1_SAFETY", 0))
    c2 = int(state_counts_all.get("C2_UPLIFT", 0))

    risk_modes = {}
    if n_applied > 0 and "score_ka_v2_risk_mode" in df_applied.columns:
        risk_modes = df_applied["score_ka_v2_risk_mode"].fillna("NA").astype(str).value_counts().to_dict()

    lines = [
        f"Bundles: {n_bundles}",
        f"KA applied: {n_applied}",
        f"Regimes: R0={c0}, R1={c1}, R2={c2}",
    ]
    if n_applied > 0:
        lines += [
            "",
            "Penalty stats (median unless noted):",
            f"  p_shrink = {_format_float(p_shrink_p50)}",
            f"  scaled_px_frac = {_format_float(scaled_frac_p50)}",
            f"  scale_p90 = {_format_float(scale_p90_p50)}",
            f"  scale_max (max) = {_format_float(scale_max)}",
        ]
        if risk_modes:
            risk_mode_str = ", ".join(f"{k}:{v}" for k, v in sorted(risk_modes.items()))
            lines += ["", f"risk_mode: {risk_mode_str}"]
    else:
        lines += ["", "Applied: none"]

    if args.show_source:
        lines += ["", f"Source: {in_csv.as_posix()}"]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    if args.also_pdf:
        out_pdf = out_png.with_suffix(".pdf")
        fig.savefig(out_pdf, bbox_inches="tight")

    print(f"[ka-v2 fig] wrote {out_png}")
    if args.also_pdf:
        print(f"[ka-v2 fig] wrote {out_pdf}")


if __name__ == "__main__":
    main()
