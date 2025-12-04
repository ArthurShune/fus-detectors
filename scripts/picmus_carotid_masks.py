#!/usr/bin/env python3
"""
Heuristic vessel/background masks for PICMUS carotid PD cube.

This script loads a PD cube saved by picmus_carotid_pd.py, computes a
time-averaged PD image, and derives lumen (vessel) and tissue (background)
masks by simple intensity quantiles. It saves masks to an .npz for use in
tile labeling.

Usage
-----
    PYTHONPATH=. python scripts/picmus_carotid_masks.py \
        --pd data/picmus/pd_cube.npy \
        --vessel-quantile 0.25 \
        --bg-quantile 0.6 \
        --out data/picmus/picmus_masks.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Derive heuristic masks for PICMUS carotid PD cube.")
    ap.add_argument(
        "--pd",
        type=Path,
        default=Path("data/picmus/pd_cube.npy"),
        help="Path to PD cube (T,H,W) saved as .npy.",
    )
    ap.add_argument(
        "--vessel-quantile",
        type=float,
        default=0.25,
        help="Quantile threshold for vessel (lumen) mask (default: 0.25). Lower selects darker lumen.",
    )
    ap.add_argument(
        "--bg-quantile",
        type=float,
        default=0.6,
        help="Quantile threshold for background tissue (default: 0.6). Higher selects brighter tissue.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/picmus/picmus_masks.npz"),
        help="Output npz with mask_vessel and mask_bg.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    pd = np.load(args.pd)  # (T,H,W)
    mean_pd = pd.mean(axis=0)

    v_q = float(args.vessel_quantile)
    b_q = float(args.bg_quantile)
    if not (0.0 < v_q < 1.0 and 0.0 < b_q < 1.0):
        raise ValueError("Quantiles must be in (0,1).")

    v_thr = np.quantile(mean_pd, v_q)
    b_thr = np.quantile(mean_pd, b_q)
    mask_vessel = mean_pd <= v_thr
    mask_bg = (mean_pd >= b_thr) & (~mask_vessel)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, mask_vessel=mask_vessel.astype(bool), mask_bg=mask_bg.astype(bool))

    print(f"[picmus-masks] mean_pd shape {mean_pd.shape}, dtype {mean_pd.dtype}")
    print(
        f"[picmus-masks] vessel_q={v_q:.3f} thr={v_thr:.3e}, "
        f"bg_q={b_q:.3f} thr={b_thr:.3e}"
    )
    print(
        f"[picmus-masks] vessel pixels={mask_vessel.sum()}, "
        f"background pixels={mask_bg.sum()}"
    )
    print(f"[picmus-masks] masks saved to {args.out}")


if __name__ == "__main__":
    main()
