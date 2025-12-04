#!/usr/bin/env python3
"""
Delay-and-sum beamform PICMUS carotid RF to an IQ/PD cube on a small grid.

Uses a normal-incidence plane-wave model and the provided element geometry.
This is a lightweight front end to produce (T,H,W) IQ/PD cubes suitable
for tiling and Doppler STAP telemetry.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pipeline.realdata.picmus_carotid import (
    picmus_data_root,
    load_picmus_rf,
    make_picmus_grid,
    precompute_das_delays,
    das_beamform_picmus,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Beamform PICMUS carotid RF (DAS).")
    ap.add_argument(
        "--uff",
        type=Path,
        default=None,
        help="Path to PICMUS_carotid_cross.uff (default: data/picmus/PICMUS_carotid_cross.uff)",
    )
    ap.add_argument(
        "--z-min-mm",
        type=float,
        default=1.0,
        help="Minimum depth in mm (default: 1.0).",
    )
    ap.add_argument(
        "--z-max-mm",
        type=float,
        default=6.0,
        help="Maximum depth in mm (default: 6.0).",
    )
    ap.add_argument(
        "--nz",
        type=int,
        default=128,
        help="Number of depth samples (default: 128).",
    )
    ap.add_argument(
        "--nx",
        type=int,
        default=64,
        help="Number of lateral samples (default: 64).",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output directory (default: data/picmus).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = picmus_data_root()
    uff_path = args.uff if args.uff is not None else (root / "PICMUS_carotid_cross.uff")
    out_root = args.out_root if args.out_root is not None else root

    rf, meta = load_picmus_rf(uff_path)
    print(f"[picmus-das] rf shape (samples, tx, el) = {rf.shape}")
    print(
        f"[picmus-das] fs={meta.fs:.3e} Hz, prf={meta.prf:.3e} Hz, "
        f"c={meta.c:.1f} m/s, pitch={meta.pitch*1e3:.3f} mm"
    )

    z_grid, x_grid = make_picmus_grid(
        meta,
        z_min_m=args.z_min_mm * 1e-3,
        z_max_m=args.z_max_mm * 1e-3,
        n_z=args.nz,
        n_x=args.nx,
    )
    print(
        f"[picmus-das] z_grid: {z_grid[0]*1e3:.2f}..{z_grid[-1]*1e3:.2f} mm (n_z={len(z_grid)})"
    )
    print(
        f"[picmus-das] x_grid: {x_grid[0]*1e3:.2f}..{x_grid[-1]*1e3:.2f} mm (n_x={len(x_grid)})"
    )

    sample_idx, valid = precompute_das_delays(meta, z_grid, x_grid)
    iq_cube, pd_cube = das_beamform_picmus(rf, meta, z_grid, x_grid, sample_idx, valid)
    print(
        f"[picmus-das] iq_cube shape (T,H,W) = {iq_cube.shape}, dtype={iq_cube.dtype}, "
        f"abs mean/std={np.abs(iq_cube).mean():.3e}/{np.abs(iq_cube).std():.3e}"
    )
    print(f"[picmus-das] pd_cube mean/std = {pd_cube.mean():.3e} / {pd_cube.std():.3e}")

    iq_path = out_root / "iq_cube_das.npy"
    pd_path = out_root / "pd_cube_das.npy"
    np.save(iq_path, iq_cube)
    np.save(pd_path, pd_cube)
    print(f"[picmus-das] saved IQ to {iq_path}")
    print(f"[picmus-das] saved PD to {pd_path}")


if __name__ == "__main__":
    main()
