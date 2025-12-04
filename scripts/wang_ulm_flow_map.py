#!/usr/bin/env python3
"""
Compute a minimal ULM-like vessel/flow map on the Wang awake-mouse IQ dataset.

This script:
  - Loads a single IQDataXXX.dat file (default: IQData001.dat).
  - Uses a subset of frames (default: 0–255, inclusive).
  - Applies lightweight temporal clutter suppression and magnitude thresholding.
  - Accumulates detections into a (H, W) float32 flow map aligned with the IQ grid.
  - Saves the result to data/wang_ulm/flow_map.npy (by default).

Example
-------
    PYTHONPATH=. python scripts/wang_ulm_flow_map.py \
        --file-index 1 \
        --frames 0 255 \
        --quantile 0.999 \
        --out data/wang_ulm/flow_map.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from pipeline.realdata.wang_ulm_flow import (
    ULMFlowConfig,
    compute_and_save_flow_map,
)


def _parse_frames(frames_arg: Optional[Sequence[int]]) -> Optional[Tuple[int, int]]:
    if frames_arg is None:
        return None
    if len(frames_arg) != 2:
        raise ValueError("frames must be two integers: START STOP")
    start, stop = int(frames_arg[0]), int(frames_arg[1])
    return (start, stop)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute a minimal Wang ULM-like flow map.")
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root containing IQDataXXX.dat and IQSizeInfo.mat (default: data/wang_ulm).",
    )
    ap.add_argument(
        "--file-index",
        type=int,
        default=1,
        help="1-based index of IQDataXXX.dat to load (default: 1).",
    )
    ap.add_argument(
        "--frames",
        type=int,
        nargs=2,
        default=[0, 255],
        metavar=("START", "STOP"),
        help="Inclusive frame range to use for ULM processing (default: 0 255).",
    )
    ap.add_argument(
        "--quantile",
        type=float,
        default=0.999,
        help="Global power quantile for bubble detection threshold (default: 0.999).",
    )
    ap.add_argument(
        "--highpass",
        type=str,
        default="diff",
        choices=["diff", "svd"],
        help="High-pass mode: 'diff' (2nd-order temporal difference) or 'svd' (remove leading k modes).",
    )
    ap.add_argument(
        "--svd-k",
        type=int,
        default=1,
        help="Number of leading singular modes to remove when highpass='svd' (default: 1).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .npy path for the flow map (default: data/wang_ulm/flow_map.npy).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    frames = _parse_frames(args.frames)

    cfg = ULMFlowConfig(
        file_index=int(args.file_index),
        frames=frames,
        quantile=float(args.quantile),
        highpass=str(args.highpass),
        svd_k=int(args.svd_k),
    )

    flow = compute_and_save_flow_map(
        cfg,
        out_path=args.out,
        data_root=args.data_root,
    )

    H, W = flow.shape
    print(
        f"[wang-ulm-flow] flow map computed: shape=(H={H}, W={W}), "
        f"dtype={flow.dtype}, min={flow.min():.3e}, max={flow.max():.3e}"
    )


if __name__ == "__main__":
    main()
