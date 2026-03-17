#!/usr/bin/env python3
"""
Make a compact qualitative baseline-vs-STAP comparison panel for Shin RatBrain Fig3.

This is intended to be used on an existing baseline-matrix run output directory.
It visualizes:
  - MC-SVD baseline PD score map
  - STAP score map (on MC-SVD residual)
  - STAP-only score map (raw/registered IQ)
with the shared flow-proxy mask overlaid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load(path: Path) -> np.ndarray:
    return np.load(str(path), allow_pickle=False)


def _log_view(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.log10(np.maximum(x, 0.0) + float(eps))


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot a 3-panel Shin baseline-vs-STAP qualitative example.")
    ap.add_argument("--bundle-stap", type=Path, required=True, help="MC-SVD+STAP bundle dir (contains score_base.npy).")
    ap.add_argument("--bundle-stap-raw", type=Path, required=True, help="STAP-only bundle dir (contains score_stap_preka.npy).")
    ap.add_argument("--out-png", type=Path, default=Path("figs/paper/shin_baseline_matrix_example.png"))
    ap.add_argument("--clip", type=str, default="1,99", help="Percentile clip for display (default: %(default)s).")
    ap.add_argument("--eps", type=float, default=1e-12, help="Epsilon for log view (default: %(default)s).")
    args = ap.parse_args()

    bundle_stap = Path(args.bundle_stap)
    bundle_raw = Path(args.bundle_stap_raw)
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    score_base = _load(bundle_stap / "score_base.npy")
    score_stap = _load(bundle_stap / "score_stap_preka.npy")
    score_stap_raw = _load(bundle_raw / "score_stap_preka.npy")

    mask_flow = _load(bundle_stap / "mask_flow.npy").astype(bool, copy=False)
    if mask_flow.shape != score_base.shape:
        raise SystemExit("mask_flow shape mismatch vs score maps")

    # Shared log view and shared color limits.
    view_base = _log_view(score_base, eps=float(args.eps))
    view_stap = _log_view(score_stap, eps=float(args.eps))
    view_raw = _log_view(score_stap_raw, eps=float(args.eps))

    clip_lo, clip_hi = [float(x) for x in str(args.clip).split(",")]
    pooled = np.concatenate([view_base.ravel(), view_stap.ravel(), view_raw.ravel()])
    pooled = pooled[np.isfinite(pooled)]
    vmin = float(np.quantile(pooled, clip_lo / 100.0))
    vmax = float(np.quantile(pooled, clip_hi / 100.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(pooled)), float(np.nanmax(pooled))

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
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0), constrained_layout=True)
    ims = []
    panels = [
        (view_base, r"MC--SVD (baseline PD)"),
        (view_stap, r"Matched-subspace detector on MC--SVD residual"),
        (view_raw, r"Matched-subspace detector on raw IQ"),
    ]
    for ax, (img, title) in zip(axes, panels, strict=True):
        im = ax.imshow(img, cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest")
        ims.append(im)
        ax.contour(mask_flow.astype(float), levels=[0.5], colors=["#00d0ff"], linewidths=1.2)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(ims[-1], ax=axes, fraction=0.035, pad=0.02)
    cbar.set_label(r"$\log_{10}(S+\epsilon)$", rotation=90)

    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    print(f"[shin-example-fig] wrote {out_png}")


if __name__ == "__main__":
    main()
