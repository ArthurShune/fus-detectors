from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Curve:
    label: str
    path: Path


def _to_float(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def _read_agg_csv(path: Path) -> dict[str, np.ndarray]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")

    def col(name: str) -> np.ndarray:
        return np.asarray([_to_float(r.get(name)) for r in rows], dtype=np.float64)

    return {
        "amp_px": col("amp_px"),
        "corr_base_median": col("corr_base_median"),
        "corr_stap_median": col("corr_stap_median"),
        "corr_base_q25": col("corr_base_q25"),
        "corr_base_q75": col("corr_base_q75"),
        "corr_stap_q25": col("corr_stap_q25"),
        "corr_stap_q75": col("corr_stap_q75"),
        "tpr_base_median": col("tpr_base_median"),
        "tpr_stap_median": col("tpr_stap_median"),
        "tpr_base_q25": col("tpr_base_q25"),
        "tpr_base_q75": col("tpr_base_q75"),
        "tpr_stap_q25": col("tpr_stap_q25"),
        "tpr_stap_q75": col("tpr_stap_q75"),
        "corr_frac_stap_gt_base": col("corr_frac_stap_gt_base"),
        "tpr_frac_stap_gt_base": col("tpr_frac_stap_gt_base"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare multiple Shin motion aggregate CSVs (from scripts/shin_ratbrain_motion_aggregate.py) "
            "in a single figure. Each input CSV is summarized in its own column (baseline vs STAP)."
        )
    )
    parser.add_argument(
        "--curve",
        type=str,
        action="append",
        required=True,
        help="Curve spec as label=path/to/agg.csv (repeatable).",
    )
    parser.add_argument("--out-png", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Shin RatBrain IQ: brainlike motion sweep")
    args = parser.parse_args()

    curves: list[Curve] = []
    for spec in args.curve:
        if "=" not in spec:
            raise SystemExit(f"Invalid --curve {spec!r} (expected label=path)")
        label, path_s = spec.split("=", 1)
        curves.append(Curve(label=label.strip(), path=Path(path_s).expanduser()))

    for c in curves:
        if not c.path.is_file():
            raise SystemExit(f"Missing curve CSV: {c.path}")

    series = [(_read_agg_csv(c.path), c.label) for c in curves]
    n = len(series)

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise SystemExit(f"matplotlib required to plot: {e}")

    fig, axes = plt.subplots(2, n, figsize=(4.0 * n, 6.5), sharex=False, sharey=False)
    if n == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for j, (s, label) in enumerate(series):
        x = s["amp_px"]
        order = np.argsort(x)
        x = x[order]

        def plot_band(ax, med, q25, q75, color, name):
            ax.plot(x, med[order], color=color, lw=2, label=name)
            ax.fill_between(x, q25[order], q75[order], color=color, alpha=0.18, linewidth=0)

        ax0 = axes[0, j]
        plot_band(ax0, s["corr_base_median"], s["corr_base_q25"], s["corr_base_q75"], color="tab:gray", name="MC-SVD PD")
        plot_band(ax0, s["corr_stap_median"], s["corr_stap_q25"], s["corr_stap_q75"], color="tab:blue", name="STAP PD (pre-KA)")
        ax0.set_title(label)
        ax0.set_xlabel("Motion amplitude (px)")
        ax0.set_ylabel("Map corr vs no-motion (median ± IQR)")
        ax0.set_ylim(-0.05, 1.05)
        ax0.grid(True, alpha=0.3)

        ax1 = axes[1, j]
        plot_band(ax1, s["tpr_base_median"], s["tpr_base_q25"], s["tpr_base_q75"], color="tab:gray", name="MC-SVD PD")
        plot_band(ax1, s["tpr_stap_median"], s["tpr_stap_q25"], s["tpr_stap_q75"], color="tab:blue", name="STAP PD (pre-KA)")
        ax1.set_xlabel("Motion amplitude (px)")
        ax1.set_ylabel("Self-tail recall @ bg-FPR=1e-3 (median ± IQR)")
        ax1.set_ylim(-0.01, 1.01)
        ax1.grid(True, alpha=0.3)

        # Small textual summary.
        if np.isfinite(s["corr_frac_stap_gt_base"]).any():
            frac_corr = float(s["corr_frac_stap_gt_base"][-1])
            frac_tpr = float(s["tpr_frac_stap_gt_base"][-1])
            ax1.text(
                0.02,
                0.06,
                f"frac(STAP>base) @max amp:\n"
                f"corr={frac_corr:.2f}, recall={frac_tpr:.2f}",
                transform=ax1.transAxes,
                fontsize=9,
                ha="left",
                va="bottom",
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(args.title, y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=180)
    plt.close(fig)

    print(f"[shin-motion-compare] wrote {args.out_png}")


if __name__ == "__main__":
    main()
