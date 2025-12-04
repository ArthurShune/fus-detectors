#!/usr/bin/env python3
"""
Lightweight sanity check for the Wang ULM IQ dataset.

This script loads the first (or specified) IQDataXXX.dat file, optionally
downselects frames, and reports basic metadata and simple PSD telemetry on a
single tile to confirm that the loader and reshape logic are correct without
materializing the full dataset.

Usage
-----
    PYTHONPATH=. python scripts/wang_iq_sanity.py \
        --file-index 1 \
        --frames 0 63 \
        --tile-h 16 --tile-w 16
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from pipeline.realdata.wang_ulm import (
    iter_wang_frames,
    list_wang_files,
    load_wang_iq,
    load_wang_metadata,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sanity check Wang ULM IQ loader.")
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root directory containing IQDataXXX.dat and IQSizeInfo.mat (defaults to data/wang_ulm).",
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
        default=None,
        metavar=("START", "STOP"),
        help="Optional frame range [start, stop] (inclusive) to load; default loads all frames.",
    )
    ap.add_argument(
        "--tile-h",
        type=int,
        default=16,
        help="Tile height for PSD telemetry.",
    )
    ap.add_argument(
        "--tile-w",
        type=int,
        default=16,
        help="Tile width for PSD telemetry.",
    )
    ap.add_argument(
        "--tile-y",
        type=int,
        default=0,
        help="Tile origin Y (row index).",
    )
    ap.add_argument(
        "--tile-x",
        type=int,
        default=0,
        help="Tile origin X (col index).",
    )
    return ap.parse_args()


def _frame_indices(frames_arg: Sequence[int] | None) -> np.ndarray | None:
    if frames_arg is None:
        return None
    if len(frames_arg) != 2:
        raise ValueError("frames must be two integers: START STOP")
    start, stop = frames_arg
    if stop < start:
        raise ValueError("frames STOP must be >= START")
    return np.arange(start, stop + 1, dtype=int)


def main() -> None:
    args = parse_args()

    info = load_wang_metadata(args.data_root)
    files = list_wang_files(args.data_root)
    if not files:
        raise RuntimeError("No IQDataXXX.dat files found.")
    if args.file_index < 1 or args.file_index > len(files):
        raise ValueError(f"file_index {args.file_index} out of range 1..{len(files)}")
    path = files[args.file_index - 1]
    idx = _frame_indices(args.frames)

    print(f"Wang IQ metadata: row={info.row}, col={info.col}, frames={info.frames}")
    print(f"Using file: {path.name}")
    if idx is None:
        print("Loading all frames (this may use ~0.6 GB for one file).")
    else:
        print(f"Loading frames {idx[0]}..{idx[-1]} (count={idx.size}).")

    iq = load_wang_iq(path, info, frames=idx, dtype=np.complex64)
    print(f"IQ shape: {iq.shape}, dtype: {iq.dtype}, abs mean={np.abs(iq).mean():.3g}")

    # Simple PSD on a single tile to sanity check frequency content without huge allocations.
    th, tw = int(args.tile_h), int(args.tile_w)
    y0, x0 = int(args.tile_y), int(args.tile_x)
    T, H, W = iq.shape
    if y0 + th > H or x0 + tw > W:
        raise ValueError(f"Tile ({th},{tw}) at ({y0},{x0}) exceeds IQ dims ({H},{W})")

    tile = iq[:, y0 : y0 + th, x0 : x0 + tw]  # (T, th, tw)
    # Collapse spatially for a quick PSD.
    ts = tile.reshape(T, -1).mean(axis=1)
    freqs = np.fft.rfftfreq(T, d=1.0)  # PRF not known here; use 1.0 step as placeholder.
    psd = np.abs(np.fft.rfft(ts)) ** 2
    top_idx = np.argsort(psd)[-5:][::-1]
    print("Top 5 PSD bins (index, freq, power):")
    for i in top_idx:
        print(f"  k={int(i):4d}  f={freqs[i]:7.3f}  |X|^2={psd[i]:.3e}")


if __name__ == "__main__":
    main()
