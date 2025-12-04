#!/usr/bin/env python3
"""
Compute a simple analytic IQ and PD cube from the PICMUS carotid UFF.

This uses a per-channel Hilbert transform along the fast-time axis to form
an analytic IQ cube shaped (T, H, W) = (pulses, samples, channels), and
saves both IQ and PD (|IQ|^2) to disk.

Usage
-----
    PYTHONPATH=. python scripts/picmus_carotid_beamform.py \
        --uff data/picmus/PICMUS_carotid_cross.uff \
        --iq-out data/picmus/iq_cube.npy \
        --pd-out data/picmus/pd_cube_hilbert.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pipeline.realdata.picmus_carotid import (
    load_picmus_carotid_uff,
    rf_to_hilbert_iq,
    rf_to_pd_cube,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Hilbert IQ/PD cube from PICMUS carotid UFF.")
    ap.add_argument(
        "--uff",
        type=Path,
        default=Path("data/picmus/PICMUS_carotid_cross.uff"),
        help="Path to PICMUS_carotid_cross.uff",
    )
    ap.add_argument(
        "--iq-out",
        type=Path,
        default=Path("data/picmus/iq_cube.npy"),
        help="Output path for IQ cube (.npy).",
    )
    ap.add_argument(
        "--pd-out",
        type=Path,
        default=Path("data/picmus/pd_cube_hilbert.npy"),
        help="Output path for PD cube (.npy).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    scan = load_picmus_carotid_uff(args.uff)
    iq = rf_to_hilbert_iq(scan.rf)  # (T,H,W)
    pd = rf_to_pd_cube(iq)

    args.iq_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.iq_out, iq)
    np.save(args.pd_out, pd)

    T, H, W = iq.shape
    print(f"[picmus-beam] IQ cube saved to {args.iq_out} with shape (T,H,W)=({T},{H},{W}), "
          f"dtype={iq.dtype}, abs mean/std={np.abs(iq).mean():.3e}/{np.abs(iq).std():.3e}")
    print(f"[picmus-beam] PD cube saved to {args.pd_out} with mean={pd.mean():.3e}, std={pd.std():.3e}")
    print(f"[picmus-beam] fs={scan.fs} Hz, prf={scan.prf} Hz")


if __name__ == "__main__":
    main()
