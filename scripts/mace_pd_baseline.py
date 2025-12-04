#!/usr/bin/env python3
"""
Simple PD-only baseline analysis for the Macé/Urban whole-brain fUS dataset.

This script exercises the real-data loader by:
  - Loading one or more Macé scans,
  - Selecting a single coronal plane,
  - Computing per-voxel PD z-scores over time,
  - Printing basic summary statistics.

This is intentionally lightweight and does not yet perform full ROC analysis
or atlas-based ROI labeling; it serves as a sanity check that the wiring to
`data/whole-brain-fUS` is correct.

Usage
-----
    PYTHONPATH=. python scripts/mace_pd_baseline.py \
        --data-root data/whole-brain-fUS \
        --scan-index 0 \
        --plane-index 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pipeline.realdata import iter_pd_slices, mace_data_root
from pipeline.realdata.mace_wholebrain import load_all_mace_scans


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="PD-only sanity check for Macé whole-brain fUS")
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
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = mace_data_root() if args.data_root is None else args.data_root
    scans = load_all_mace_scans(root)
    if not scans:
        raise RuntimeError(f"No Macé scans found under {root}")
    if args.scan_index < 0 or args.scan_index >= len(scans):
        raise ValueError(f"scan_index {args.scan_index} out of range for {len(scans)} scans")
    scan = scans[args.scan_index]
    if args.plane_index < 0 or args.plane_index >= scan.n_planes:
        raise ValueError(f"plane_index {args.plane_index} out of range for {scan.n_planes} planes")

    # Extract the requested plane (T, H, W)
    _, pd_T_HW = list(iter_pd_slices(scan))[args.plane_index]
    T, H, W = pd_T_HW.shape
    pd_flat = pd_T_HW.reshape(T, -1)
    mu = pd_flat.mean(axis=0)
    sigma = pd_flat.std(axis=0)
    sigma_safe = np.where(sigma > 0.0, sigma, 1.0)
    z = (pd_flat - mu) / sigma_safe

    z_abs = np.abs(z)
    # Basic summary: how heavy are the tails?
    p95 = float(np.percentile(z_abs, 95.0))
    p99 = float(np.percentile(z_abs, 99.0))
    p999 = float(np.percentile(z_abs, 99.9))

    print(f"Loaded Macé scan '{scan.name}' with shape T,H,W,Z={scan.shape}")
    print(f"Analyzing plane {args.plane_index} (T={T}, H={H}, W={W})")
    print(f"dt={scan.dt:.3f} s, voxel_size_um={scan.voxel_size_um}")
    print(
        f"plane_mm={scan.planes_mm[args.plane_index]:.3f}, "
        f"|z|-score percentiles (per-voxel over time): "
        f"p95={p95:.3f}, p99={p99:.3f}, p99.9={p999:.3f}"
    )


if __name__ == "__main__":
    main()
