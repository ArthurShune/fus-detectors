#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt


def _load_mask(path: Path) -> np.ndarray:
    m = np.load(path)
    if m.dtype != bool:
        m = m != 0
    return m


def _finite_flat(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return x[np.isfinite(x)]


def _right_tail_threshold(bg_scores: np.ndarray, alpha: float) -> tuple[float, float]:
    bg = _finite_flat(bg_scores)
    n = int(bg.size)
    if n <= 0:
        raise ValueError("Empty background score pool.")
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("alpha must be in (0,1).")
    k = int(math.ceil(a * n))
    k = max(1, min(k, n))
    tau = float(np.partition(bg, n - k)[n - k])
    realized = float(np.mean(bg >= tau))
    return tau, realized


def _ccdf_curve(scores: np.ndarray, *, max_points: int = 2400) -> tuple[np.ndarray, np.ndarray, int]:
    s = _finite_flat(scores)
    if s.size <= 0:
        raise ValueError("Empty score pool.")
    s = np.sort(s)
    n = int(s.size)
    if n <= max_points:
        idx = np.arange(n, dtype=np.int64)
    else:
        idx = np.unique(np.linspace(0, n - 1, int(max_points), dtype=np.int64))
    x = s[idx]
    y = (n - idx) / float(n)  # tail probability at threshold x (>= x)
    return x, y, n


def _plot_panel(
    ax: plt.Axes,
    *,
    bg: np.ndarray,
    flow: np.ndarray,
    title: str,
    alphas: list[float],
) -> None:
    x_bg, y_bg, n_bg = _ccdf_curve(bg)
    x_flow, y_flow, n_flow = _ccdf_curve(flow)

    eps_x = 1e-12
    x_bg = np.maximum(x_bg, eps_x)
    x_flow = np.maximum(x_flow, eps_x)

    # Prevent log(0) in y while still allowing annotation of true zeros.
    y_floor = 0.5 / float(max(n_bg, n_flow))
    y_bg_plot = np.maximum(y_bg, y_floor)
    y_flow_plot = np.maximum(y_flow, y_floor)

    ax.plot(x_bg, y_bg_plot, color="0.35", linewidth=1.8, label=r"background ($H_0$)")
    ax.plot(x_flow, y_flow_plot, color="#1f77b4", linewidth=1.8, label=r"flow ($H_1$)")

    flow_finite = _finite_flat(flow)
    max_flow = float(np.max(flow_finite)) if flow_finite.size else float("nan")
    ax.text(
        0.02,
        0.02,
        (f"n0={n_bg}, n1={n_flow}\nmax(H1)={max_flow:.3g}" if np.isfinite(max_flow) else f"n0={n_bg}, n1={n_flow}"),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.85"),
    )

    tau_lines: list[tuple[float, float, float]] = []
    summary_bits: list[str] = []
    for j, a in enumerate(alphas):
        tau, fpr_realized = _right_tail_threshold(bg, a)
        tau_plot = float(max(tau, eps_x))
        tpr = float(np.mean(flow_finite >= float(tau))) if flow_finite.size else 0.0

        ax.axvline(tau_plot, color="k", linestyle="--", linewidth=1.0)
        ax.plot([tau_plot], [max(tpr, y_floor)], marker="o", markersize=4.0, color="k", zorder=5)
        tau_lines.append((tau_plot, fpr_realized, tpr))
        summary_bits.append(rf"$\alpha={a:g}$: TPR$\approx${tpr:.3g}")

    for j, (tau_plot, _fpr_realized, _tpr) in enumerate(tau_lines):
        ax.text(
            tau_plot,
            0.94 - 0.10 * j,
            rf"$\tau_{{{alphas[j]:g}}}$",
            rotation=90,
            va="top",
            ha="center",
            fontsize=9,
            transform=ax.get_xaxis_transform(),
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.75, edgecolor="0.85"),
        )

    ax.text(
        0.02,
        0.98,
        "\n".join(summary_bits),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.85, edgecolor="0.85"),
    )

    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"score threshold $\tau$")
    ax.set_ylabel(r"tail probability $\mathbb{P}\{S\geq \tau\}$")
    ax.grid(True, which="both", alpha=0.25)

    positive_scores = np.concatenate([x_bg, x_flow, np.asarray([t[0] for t in tau_lines], dtype=np.float64)])
    positive_scores = positive_scores[np.isfinite(positive_scores) & (positive_scores > eps_x)]
    if positive_scores.size > 0:
        x_lo = float(max(eps_x, np.min(positive_scores) * 0.3))
        x_hi = float(np.max(positive_scores) * 1.35)
        ax.set_xlim(x_lo, x_hi)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Generate a representative labeled-brain tail-collapse visual showing how PD-after-filter baselines can "
            "separate at relaxed operating points while collapsing at strict-tail thresholds due to an extreme "
            "background right tail."
        )
    )
    ap.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path(
            "runs/pilot/fair_filter_matrix_pd_r3_localbaselines/"
            "open_seed1_mcsvd_full/pw_7.5MHz_5ang_5ens_64T_seed1_win0_off0"
        ),
        help="Acceptance bundle directory containing masks and score maps (default: %(default)s).",
    )
    ap.add_argument(
        "--alphas",
        type=str,
        default="0.1,0.01,1e-3",
        help="Comma-separated FPR targets to visualize (default: %(default)s).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("figs/paper/brain_tail_collapse_visual.pdf"),
        help="Output figure path (default: %(default)s).",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    bundle = Path(args.bundle_dir)
    if not bundle.exists():
        raise FileNotFoundError(bundle)

    alphas = [float(x.strip()) for x in str(args.alphas).split(",") if x.strip()]
    if not alphas:
        raise SystemExit("--alphas must be non-empty.")

    m_bg = _load_mask(bundle / "mask_bg.npy")
    m_flow = _load_mask(bundle / "mask_flow.npy")

    s_base = np.load(bundle / "score_base.npy").astype(np.float64, copy=False)
    if s_base.shape != m_bg.shape or s_base.shape != m_flow.shape:
        raise ValueError(f"Shape mismatch in {bundle}: score_base={s_base.shape} masks={m_bg.shape}")

    bg_base = s_base[m_bg]
    flow_base = s_base[m_flow]

    stap_path = bundle / "score_stap_preka.npy"
    have_stap = stap_path.exists()
    if have_stap:
        s_stap = np.load(stap_path).astype(np.float64, copy=False)
        if s_stap.shape != s_base.shape:
            raise ValueError(f"Shape mismatch in {bundle}: score_stap_preka={s_stap.shape} score_base={s_base.shape}")
        bg_stap = s_stap[m_bg]
        flow_stap = s_stap[m_flow]

    if have_stap:
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), constrained_layout=True)
        _plot_panel(
            axes[0],
            bg=bg_base,
            flow=flow_base,
            title="MC--SVD residual + power Doppler score",
            alphas=alphas,
        )
        _plot_panel(
            axes[1],
            bg=bg_stap,
            flow=flow_stap,
            title="MC--SVD residual + STAP (pre-KA) score",
            alphas=alphas,
        )
        axes[0].legend(loc="upper right", frameon=True)
        axes[1].set_ylabel("")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7.6, 4.2), constrained_layout=True)
        _plot_panel(
            ax,
            bg=bg_base,
            flow=flow_base,
            title="MC--SVD residual + power Doppler score",
            alphas=alphas,
        )
        ax.legend(loc="upper right", frameon=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    print(f"[brain-tail-visual] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
