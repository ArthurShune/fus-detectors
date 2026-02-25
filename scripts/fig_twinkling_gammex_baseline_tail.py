#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt


def _load_mask(path: Path) -> np.ndarray:
    m = np.load(path)
    if m.dtype != bool:
        m = m != 0
    return m


def _iter_frame_dirs(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(root)
    frame_dirs = sorted([p for p in root.glob("**/frame*") if p.is_dir()])
    if not frame_dirs:
        raise FileNotFoundError(f"No frame directories found under {root}.")
    return frame_dirs


def _pool_scores(
    root: Path,
    *,
    score_name: str,
    mask_bg_name: str,
    mask_flow_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    bg_parts: list[np.ndarray] = []
    flow_parts: list[np.ndarray] = []

    for d in _iter_frame_dirs(root):
        score_path = d / score_name
        bg_path = d / mask_bg_name
        flow_path = d / mask_flow_name

        if not (score_path.exists() and bg_path.exists() and flow_path.exists()):
            # Skip non-standard frame dirs (e.g., debug packs) rather than fail.
            continue

        s = np.load(score_path).astype(np.float64, copy=False)
        m_bg = _load_mask(bg_path)
        m_flow = _load_mask(flow_path)
        if s.shape != m_bg.shape or s.shape != m_flow.shape:
            raise ValueError(f"Shape mismatch in {d}: score={s.shape}, bg={m_bg.shape}, flow={m_flow.shape}.")

        bg_parts.append(s[m_bg])
        flow_parts.append(s[m_flow])

    if not bg_parts or not flow_parts:
        raise FileNotFoundError(
            f"Did not find any frames containing {score_name}, {mask_bg_name}, and {mask_flow_name} under {root}."
        )

    return np.concatenate(bg_parts), np.concatenate(flow_parts)


def _tau_for_fpr(bg_scores: np.ndarray, alpha: float) -> float:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1).")
    # Choose tau so that P{S >= tau | H0} ~= alpha.
    return float(np.quantile(bg_scores, 1.0 - float(alpha)))


def _plot_panel(
    ax: plt.Axes,
    *,
    bg: np.ndarray,
    flow: np.ndarray,
    title: str,
    alphas: list[float],
    bins: int,
) -> None:
    lo = float(min(np.quantile(bg, 0.001), np.quantile(flow, 0.001)))
    hi = float(max(np.quantile(bg, 0.9999), np.quantile(flow, 0.9999)))
    edges = np.linspace(lo, hi, int(bins) + 1)

    ax.hist(bg, bins=edges, density=True, histtype="step", linewidth=1.8, color="0.35", label=r"background ($H_0$)")
    ax.hist(flow, bins=edges, density=True, histtype="step", linewidth=1.8, color="#1f77b4", label=r"lumen ($H_1$)")

    for j, a in enumerate(alphas):
        tau = _tau_for_fpr(bg, a)
        tpr = float(np.mean(flow >= tau))
        ax.axvline(tau, color="k", linestyle="--", linewidth=1.0)
        ax.text(
            tau,
            0.06 + 0.10 * j,
            rf"$\tau_{{{a:g}}}$ (TPR$\approx${tpr:.3f})",
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=9,
            transform=ax.get_xaxis_transform(),
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.75, edgecolor="0.85"),
        )

    ax.set_title(title)
    ax.set_yscale("log")
    ax.set_ylabel("density (log scale)")
    ax.grid(True, which="both", alpha=0.25)

    max_flow = float(np.max(flow))
    ax.text(
        0.02,
        0.02,
        rf"$\max(H_1)$={max_flow:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.85"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Twinkling/Gammex baseline tail visual showing why conventional baselines can have TPR=0 "
            "at ultra-low FPR (thresholds computed on pooled background pixels)."
        )
    )
    parser.add_argument(
        "--along-summary",
        default="reports/twinkling_gammex_alonglinear17_prf2500_str6_msd_ka_with_baselines_summary.json",
        help="Summary JSON containing the along-view run root (default: %(default)s).",
    )
    parser.add_argument(
        "--across-summary",
        default="reports/twinkling_gammex_across17_prf2500_str4_msd_ka_with_baselines_summary.json",
        help="Summary JSON containing the across-view run root (default: %(default)s).",
    )
    parser.add_argument(
        "--score",
        default="score_base_pdlog.npy",
        help="Baseline score filename in each frame directory (default: %(default)s).",
    )
    parser.add_argument(
        "--mask-bg",
        default="mask_bg.npy",
        help="Background mask filename in each frame directory (default: %(default)s).",
    )
    parser.add_argument(
        "--mask-flow",
        default="mask_flow.npy",
        help="Flow/lumen mask filename in each frame directory (default: %(default)s).",
    )
    parser.add_argument(
        "--alphas",
        default="1e-2,1e-3",
        help="Comma-separated FPR targets to visualize (default: %(default)s).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=220,
        help="Histogram bins (default: %(default)s).",
    )
    parser.add_argument(
        "--out",
        default="figs/paper/twinkling_gammex_baseline_tail.pdf",
        help="Output figure path (default: %(default)s).",
    )

    args = parser.parse_args()
    along_root = Path(json.loads(Path(args.along_summary).read_text())["root"])
    across_root = Path(json.loads(Path(args.across_summary).read_text())["root"])
    alphas = [float(x) for x in str(args.alphas).split(",") if x.strip()]

    along_bg, along_flow = _pool_scores(
        along_root, score_name=args.score, mask_bg_name=args.mask_bg, mask_flow_name=args.mask_flow
    )
    across_bg, across_flow = _pool_scores(
        across_root, score_name=args.score, mask_bg_name=args.mask_bg, mask_flow_name=args.mask_flow
    )

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.6), constrained_layout=True)
    _plot_panel(
        axes[0],
        bg=along_bg,
        flow=along_flow,
        title="Along view: baseline log power Doppler (pooled over frames)",
        alphas=alphas,
        bins=int(args.bins),
    )
    _plot_panel(
        axes[1],
        bg=across_bg,
        flow=across_flow,
        title="Across view: baseline log power Doppler (pooled over frames)",
        alphas=alphas,
        bins=int(args.bins),
    )

    axes[1].set_xlabel("baseline score (log power Doppler)")
    axes[0].legend(loc="upper left", frameon=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
