#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _summarize_by_amp(rows: list[dict[str, str]], col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    if not rows or col not in rows[0]:
        raise KeyError(f"Missing column {col!r}")
    amps = sorted({float(r["amp_px"]) for r in rows if r.get("amp_px", "").strip()})
    if not amps:
        raise ValueError("No finite amp_px values.")
    med: list[float] = []
    q25: list[float] = []
    q75: list[float] = []
    for a in amps:
        vals = np.asarray(
            [
                float(r[col])
                for r in rows
                if r.get("amp_px", "").strip() and float(r["amp_px"]) == a and r.get(col, "").strip()
            ],
            dtype=np.float64,
        )
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            med.append(float("nan"))
            q25.append(float("nan"))
            q75.append(float("nan"))
            continue
        med.append(float(np.quantile(vals, 0.5)))
        q25.append(float(np.quantile(vals, 0.25)))
        q75.append(float(np.quantile(vals, 0.75)))
    return np.asarray(med), np.asarray(q25), np.asarray(q75), amps


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot ULM 7883227 motion-sweep correlations from an existing CSV.")
    ap.add_argument("--in-csv", type=Path, required=True, help="Input CSV produced by ulm_zenodo_7883227_motion_sweep.py")
    ap.add_argument("--out-png", type=Path, required=True, help="Output PNG figure path")
    ap.add_argument("--also-pdf", action="store_true", help="Also write a matching PDF next to the PNG")
    args = ap.parse_args()

    rows = _read_rows(args.in_csv)
    if not rows:
        raise SystemExit(f"Empty CSV: {args.in_csv}")
    if "amp_px" not in rows[0]:
        raise SystemExit(f"Missing required column 'amp_px' in {args.in_csv}")

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

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    modes = [
        ("none", "No alignment"),
        ("bmode", "Common B-mode alignment"),
        ("gt", "GT alignment (injected)"),
    ]

    for ax, (mode, title) in zip(axes, modes, strict=True):
        base_col = f"corr_score_base_align_{mode}"
        stap_col = f"corr_score_stap_align_{mode}"
        base_med, base_q25, base_q75, amps = _summarize_by_amp(rows, base_col)
        stap_med, stap_q25, stap_q75, _ = _summarize_by_amp(rows, stap_col)

        ax.plot(amps, base_med, "o-", color="#666666", label=r"Baseline $S_{\mathrm{base}}$")
        ax.fill_between(amps, base_q25, base_q75, color="#666666", alpha=0.18, linewidth=0)

        ax.plot(amps, stap_med, "o-", color="#1f77b4", label=r"Matched-subspace detector $S_{\mathrm{det,pre}}$")
        ax.fill_between(amps, stap_q25, stap_q75, color="#1f77b4", alpha=0.18, linewidth=0)

        ax.set_title(title)
        ax.set_xlabel("motion amplitude (px)")
        ax.set_ylabel("corr vs no-motion")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", frameon=False)

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=300, bbox_inches="tight", pad_inches=0.06)
    if args.also_pdf:
        fig.savefig(args.out_png.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    print(f"[ulm-motion-plot] wrote {args.out_png}")


if __name__ == "__main__":
    main()
