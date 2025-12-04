#!/usr/bin/env python3
"""
Compute a simple power cube from the PICMUS carotid UFF and save to disk.

This is a minimal Step 1: no beamforming is applied. Each emission's
samples x channels matrix is treated as a 2-D frame, and power is
computed as |rf|^2, yielding a PD cube shaped (T, H, W) = (pulses,
samples, channels).

Usage
-----
    PYTHONPATH=. python scripts/picmus_carotid_pd.py \
        --uff data/picmus/PICMUS_carotid_cross.uff \
        --out data/picmus/pd_cube.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pipeline.realdata.picmus_carotid import load_picmus_carotid_uff, rf_to_pd_cube


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute PD cube from PICMUS carotid UFF.")
    ap.add_argument(
        "--uff",
        type=Path,
        default=Path("data/picmus/PICMUS_carotid_cross.uff"),
        help="Path to PICMUS_carotid_cross.uff",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/picmus/pd_cube.npy"),
        help="Output .npy path for the PD cube (T,H,W).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    scan = load_picmus_carotid_uff(args.uff)
    pd = rf_to_pd_cube(scan.rf)  # (T, H, W) where H=samples, W=channels

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, pd)

    T, H, W = pd.shape
    print(f"[picmus-pd] PD cube saved to {args.out} with shape (T,H,W)=({T},{H},{W}), "
          f"dtype={pd.dtype}, mean={pd.mean():.3e}, std={pd.std():.3e}")
    print(f"[picmus-pd] fs={scan.fs} Hz, prf={scan.prf} Hz")


if __name__ == "__main__":
    main()
