#!/usr/bin/env python3
"""
Tile-level doppler-style STAP sanity check on a single Wang ULM IQ file.

Loads a small frame range from one IQDataXXX.dat file, converts IQ -> power
time-series per tile, and runs the miniature STAP core (Hankel + covariance +
Pf/Pa projectors) with Doppler-band settings to sanity check the pipeline
without materializing the full dataset.

Usage
-----
    PYTHONPATH=. python scripts/wang_tiles_doppler.py \
        --file-index 1 \
        --frames 0 127 \
        --tile-h 8 --tile-w 8 --stride 3
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from pipeline.realdata import tile_iter
from pipeline.realdata.wang_ulm import (
    load_wang_iq,
    load_wang_metadata,
    list_wang_files,
)
from pipeline.stap.hemo import HemoStapConfig, hemo_stap_scores_for_tiles


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Doppler-style STAP sanity on Wang IQ tiles.")
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root directory containing IQDataXXX.dat and IQSizeInfo.mat (default: data/wang_ulm).",
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
        help="Optional frame range [start, stop] inclusive; default loads all frames from the file.",
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
        "--lt",
        type=int,
        default=8,
        help="Hankel length L_t for the STAP core.",
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

    print(f"[wang] metadata row={info.row}, col={info.col}, frames={info.frames}")
    print(f"[wang] using file: {path.name}")
    if idx is None:
        print("[wang] loading all frames (may be ~0.6 GB for one file).")
    else:
        print(f"[wang] loading frames {idx[0]}..{idx[-1]} (count={idx.size}).")

    # Load IQ subset and form power time-series.
    iq = load_wang_iq(path, info, frames=idx, dtype=np.complex64)  # (T, H, W)
    pd_T_HW = (np.abs(iq) ** 2).astype(np.float32, copy=False)
    T, H, W = pd_T_HW.shape
    print(f"[wang] PD cube shape (T,H,W) = ({T}, {H}, {W}), dtype={pd_T_HW.dtype}")

    tile_hw = (int(args.tile_h), int(args.tile_w))
    stride = int(args.stride)

    tile_series = []
    for y0, x0 in tile_iter((H, W), tile_hw, stride):
        tile = pd_T_HW[:, y0 : y0 + tile_hw[0], x0 : x0 + tile_hw[1]]
        p_t = tile.mean(axis=(1, 2))
        tile_series.append(p_t.astype(np.float32))

    if not tile_series:
        raise RuntimeError("No tiles extracted; adjust tile size/stride.")

    p_tiles = np.stack(tile_series, axis=0)  # (N_tiles, T)
    print(f"[wang] collected {p_tiles.shape[0]} tiles, each of length T={T}")

    # Doppler-band STAP config (PRF ~ 1000 Hz assumed).
    stap_cfg = HemoStapConfig(
        dt=0.001,  # 1 kHz frame rate
        L_t=int(args.lt),
        pf_band=(30.0, 200.0),
        pg_band=(200.0, 350.0),
        pa_band=(350.0, 500.0),
        bg_beta=0.0,
    )
    scores = hemo_stap_scores_for_tiles(p_tiles, stap_cfg)

    hemo_br = scores["hemo_br"]
    hemo_glrt = scores["hemo_glrt"]
    Ef = scores["Ef"]
    Ea = scores["Ea"]

    print(f"[wang] BR stats: mean={hemo_br.mean():.3e}, median={np.median(hemo_br):.3e}")
    print(f"[wang] GLRT stats: mean={hemo_glrt.mean():.3e}, median={np.median(hemo_glrt):.3e}")
    alias_ratio = (Ea + 1e-8) / (Ef + 1e-8)
    print(f"[wang] alias ratio log10 stats: mean={np.log10(alias_ratio).mean():.3f}, "
          f"median={np.median(np.log10(alias_ratio)):.3f}")


if __name__ == "__main__":
    main()
