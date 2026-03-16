#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _ccdf(samples: np.ndarray, n_points: int = 800) -> tuple[np.ndarray, np.ndarray]:
    s = np.sort(np.asarray(samples, dtype=np.float64).ravel())
    idx = np.unique(np.linspace(0, s.size - 1, n_points, dtype=np.int64))
    x = s[idx]
    y = (s.size - idx) / float(s.size)
    return x, y


def _threshold_for_alpha(bg: np.ndarray, alpha: float) -> float:
    bg = np.sort(np.asarray(bg, dtype=np.float64).ravel())
    k = int(np.ceil(alpha * bg.size))
    k = max(1, min(k, bg.size))
    return float(bg[bg.size - k])


def _panel(
    ax: plt.Axes,
    *,
    bg: np.ndarray,
    flow: np.ndarray,
    title: str,
    color_flow: str = "#1f77b4",
) -> None:
    x_bg, y_bg = _ccdf(bg)
    x_flow, y_flow = _ccdf(flow)
    ax.plot(x_bg, y_bg, color="0.35", linewidth=2.0, label=r"background ($H_0$)")
    ax.plot(x_flow, y_flow, color=color_flow, linewidth=2.0, label=r"flow ($H_1$)")

    tau_2 = _threshold_for_alpha(bg, 1e-2)
    tau_3 = _threshold_for_alpha(bg, 1e-3)
    tpr_2 = float(np.mean(flow >= tau_2))
    tpr_3 = float(np.mean(flow >= tau_3))

    for tau, alpha, tpr, ypos in (
        (tau_2, r"10^{-2}", tpr_2, 0.78),
        (tau_3, r"10^{-3}", tpr_3, 0.52),
    ):
        ax.axvline(tau, color="k", linestyle="--", linewidth=1.1)
        ax.text(
            tau,
            ypos,
            rf"$\tau_{{{alpha}}}$" + "\n" + rf"TPR$\approx${tpr:.2f}",
            rotation=90,
            va="center",
            ha="right",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="0.85", alpha=0.9),
            transform=ax.get_xaxis_transform(),
        )

    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"score threshold $\tau$")
    ax.grid(True, which="both", alpha=0.22)


def main() -> int:
    rng = np.random.default_rng(7)
    n = 250_000

    # Baseline: moderate separation overall, but a heavy nuisance right tail.
    bg_baseline = np.exp(rng.normal(loc=-0.05, scale=0.95, size=n))
    flow_baseline = np.exp(rng.normal(loc=-0.65, scale=0.42, size=n // 4))

    # Tail-controlled detector: lower nuisance tail with retained flow support.
    bg_tailstable = np.exp(rng.normal(loc=-0.85, scale=0.48, size=n))
    flow_tailstable = np.exp(rng.normal(loc=-0.32, scale=0.40, size=n // 4))

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), constrained_layout=True)
    _panel(
        axes[0],
        bg=bg_baseline,
        flow=flow_baseline,
        title="Conventional score: heavy nuisance tail",
    )
    _panel(
        axes[1],
        bg=bg_tailstable,
        flow=flow_tailstable,
        title="Tail-stable detector: lower nuisance tail",
    )
    axes[0].set_ylabel(r"tail probability $\mathbb{P}\{S\geq\tau\}$")
    axes[1].set_ylabel("")
    axes[0].legend(loc="lower left", frameon=True)

    fig.suptitle(
        "Conceptual mechanism behind strict low-FPR collapse",
        fontsize=12,
        y=1.02,
    )

    out = Path("figs/paper/strict_tail_collapse_concept.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"[strict-tail-concept] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
