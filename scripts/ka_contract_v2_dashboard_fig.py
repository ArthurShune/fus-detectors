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
import csv
from pathlib import Path
from typing import Iterable


def _is_truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _load_rows(in_csv: Path) -> list[dict[str, object]]:
    with in_csv.open(newline="") as f:
        return list(csv.DictReader(f))


def _value_counts(rows: Iterable[dict[str, object]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "NA") or "NA")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _value_counts_pair(rows: Iterable[dict[str, object]], key_a: str, key_b: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        a = str(row.get(key_a, "NA") or "NA")
        b = str(row.get(key_b, "NA") or "NA")
        pair = f"{a}/{b}"
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if text == "" or text.lower() == "none":
            return None
        out = float(text)
    except Exception:
        return None
    if out != out:
        return None
    return out


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    n = len(values)
    mid = n // 2
    if n % 2:
        return float(values[mid])
    return float(0.5 * (values[mid - 1] + values[mid]))


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

    rows = _load_rows(in_csv)
    if not rows:
        raise SystemExit(f"Empty CSV: {in_csv}")

    state_col = "ka_contract_v2_state"
    reason_col = "ka_contract_v2_reason"
    applied_col = "score_ka_v2_applied"

    for col in [state_col, reason_col, applied_col]:
        if col not in rows[0]:
            raise SystemExit(f"Missing required column '{col}' in {in_csv}")

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

    state_counts_raw = _value_counts(rows, state_col)
    state_counts = {k: state_counts_raw.get(k, 0) for k in state_order if state_counts_raw.get(k, 0) > 0}

    sr_counts = _value_counts_pair(rows, state_col, reason_col)
    sr_items = sorted(sr_counts.items(), key=lambda kv: (-kv[1], kv[0]))[: max(1, int(args.max_reasons))]

    applied_rows = [row for row in rows if _is_truthy(row.get(applied_col))]

    n_bundles = int(len(rows))
    n_applied = int(len(applied_rows))

    p_shrink_vals = [
        val
        for val in (_safe_float(row.get("ka_contract_v2_p_shrink")) for row in applied_rows)
        if val is not None
    ]
    scaled_frac_vals = [
        val
        for val in (_safe_float(row.get("score_ka_v2_scaled_pixel_fraction")) for row in applied_rows)
        if val is not None
    ]
    scale_p90_vals = [
        val
        for val in (_safe_float(row.get("score_ka_v2_scale_p90")) for row in applied_rows)
        if val is not None
    ]
    scale_max_vals = [
        val
        for val in (_safe_float(row.get("score_ka_v2_scale_max")) for row in applied_rows)
        if val is not None
    ]

    p_shrink_p50 = _median(p_shrink_vals)
    scaled_frac_p50 = _median(scaled_frac_vals)
    scale_p90_p50 = _median(scale_p90_vals)
    scale_max = max(scale_max_vals) if scale_max_vals else None

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
    xs_state = list(state_counts.keys())
    xs = [state_short.get(x, x) for x in xs_state]
    ys = [state_counts[x] for x in xs_state]
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

    ylabels = [_pretty_state_reason(x) for x, _ in sr_items[::-1]]
    yvals = [count for _, count in sr_items[::-1]]
    bars = ax.barh(ylabels, yvals, color="#4c78a8")
    ax.set_xlabel("Bundle count")
    ax.set_xlim(0, max(yvals) + 1)
    for rect, y in zip(bars, yvals):
        ax.text(y + 0.05, rect.get_y() + rect.get_height() / 2.0, f"{int(y)}", va="center")

    # Panel C: applied stats.
    ax = axes[2]
    ax.axis("off")

    state_counts_all = state_counts_raw
    c0 = int(state_counts_all.get("C0_OFF", 0))
    c1 = int(state_counts_all.get("C1_SAFETY", 0))
    c2 = int(state_counts_all.get("C2_UPLIFT", 0))

    risk_modes = {}
    if n_applied > 0:
        risk_modes = _value_counts(applied_rows, "score_ka_v2_risk_mode")

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
