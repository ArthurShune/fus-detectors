#!/usr/bin/env python3
"""
Tile-based PD baseline summary for the Macé/Urban whole-brain fUS dataset.

This script is a wiring check for:
  - the Macé scan loader,
  - 2D tiling with the clinical STAP tile geometry,
  - simple PD-based tile scores and ROC utilities.

For now it uses a synthetic ROI definition on a single plane:
  - "positive" tiles are those whose centers lie in a central box,
  - "negative" tiles lie outside that box.

This is intended as a smoke test for the real-data pipeline rather than a
scientific analysis; atlas-based ROIs can be added in a subsequent step.

Usage
-----
    PYTHONPATH=. python scripts/mace_pd_tiles_roc.py \
        --scan-index 0 \
        --plane-index 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from eval.metrics import partial_auc, roc_curve, tpr_at_fpr_target
from pipeline.realdata import extract_tile_stack, iter_pd_slices, tile_iter
from pipeline.realdata.mace_wholebrain import load_all_mace_scans


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Tile-based PD baseline ROC on Macé whole-brain fUS")
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root directory containing whole-brain-fUS (defaults to data/whole-brain-fUS).",
    )
    ap.add_argument(
        "--scan-index",
        type=int,
        default=0,
        help="Index of scan to analyze (0-based).",
    )
    ap.add_argument(
        "--plane-index",
        type=int,
        default=0,
        help="Coronal plane index (0-based) to analyze.",
    )
    ap.add_argument(
        "--tile-h",
        type=int,
        default=8,
        help="Tile height (pixels).",
    )
    ap.add_argument(
        "--tile-w",
        type=int,
        default=8,
        help="Tile width (pixels).",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=3,
        help="Tile stride (pixels).",
    )
    ap.add_argument(
        "--central-frac",
        type=float,
        default=0.5,
        help="Fraction of the FOV (per dimension) defining the central positive box.",
    )
    return ap.parse_args()


def _central_box(H: int, W: int, frac: float) -> Tuple[float, float, float, float]:
    frac = float(np.clip(frac, 0.0, 1.0))
    cy = 0.5 * H
    cx = 0.5 * W
    hy = 0.5 * frac * H
    hx = 0.5 * frac * W
    y_min = cy - hy
    y_max = cy + hy
    x_min = cx - hx
    x_max = cx + hx
    return y_min, y_max, x_min, x_max


def main() -> None:
    args = parse_args()
    scans = load_all_mace_scans(args.data_root)
    if not scans:
        raise RuntimeError("No Macé scans found.")
    if args.scan_index < 0 or args.scan_index >= len(scans):
        raise ValueError(f"scan_index {args.scan_index} out of range for {len(scans)} scans")
    scan = scans[args.scan_index]
    if args.plane_index < 0 or args.plane_index >= scan.n_planes:
        raise ValueError(f"plane_index {args.plane_index} out of range for {scan.n_planes} planes")

    # Select plane (T, H, W)
    _, pd_T_HW = list(iter_pd_slices(scan))[args.plane_index]
    T, H, W = pd_T_HW.shape
    tile_hw = (int(args.tile_h), int(args.tile_w))
    stride = int(args.stride)

    # Simple PD baseline score: mean PD over time and space in each tile.
    scores_pos = []
    scores_neg = []
    y_min, y_max, x_min, x_max = _central_box(H, W, args.central_frac)

    for y0, x0 in tile_iter((H, W), tile_hw, stride):
        tile = extract_tile_stack(pd_T_HW, y0, x0, tile_hw)  # (T, th, tw)
        score = float(tile.mean())
        # Tile center in image coordinates
        th, tw = tile_hw
        cy = y0 + 0.5 * th
        cx = x0 + 0.5 * tw
        inside_central = (y_min <= cy <= y_max) and (x_min <= cx <= x_max)
        if inside_central:
            scores_pos.append(score)
        else:
            scores_neg.append(score)

    scores_pos = np.asarray(scores_pos, dtype=np.float64)
    scores_neg = np.asarray(scores_neg, dtype=np.float64)
    if scores_pos.size == 0 or scores_neg.size == 0:
        raise RuntimeError(
            "No positive or negative tiles selected; adjust central-frac or tile/stride."
        )

    fpr, tpr, _ = roc_curve(scores_pos, scores_neg, num_thresh=4096)
    auc = partial_auc(fpr, tpr, fpr_max=1e-3)
    n_neg = scores_neg.size
    fpr_min = 1.0 / float(n_neg)
    fpr_min = float(np.clip(fpr_min, 1e-8, 1.0))
    tpr_emp = tpr_at_fpr_target(fpr, tpr, target_fpr=fpr_min)

    print(f"Macé scan '{scan.name}', plane {args.plane_index}:")
    print(f"  PD shape T,H,W = ({T}, {H}, {W}), tile_hw={tile_hw}, stride={stride}")
    print(f"  positives={scores_pos.size}, negatives={scores_neg.size}")
    print(f"  partial AUC (FPR<=1e-3): {auc:.4f}")
    print(f"  TPR at empirical FPR_min={fpr_min:.3e}: {tpr_emp:.4f}")


if __name__ == "__main__":
    main()
