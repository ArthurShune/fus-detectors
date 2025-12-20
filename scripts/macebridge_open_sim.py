#!/usr/bin/env python3
"""
MaceBridge-OpenSkull: k-Wave plane-wave simulations on Macé-aligned grids.

This script is a minimal Phase-2 scaffold for the Brain-MaceBridgeOpen profile:

  - It loads a MaceTemplate (built from the Macé + Allen atlas bundle) which
    defines, for a small set of (scan_idx, plane_idx) pairs, the native PD
    grid size (H,W), voxel size, ROI masks (H1/H0), and a tile layout that
    matches the clinical STAP profile (8x8 tiles, stride 3).
  - For each selected slice it instantiates a simple 2D k-Wave SimGeom whose
    spatial grid matches the Macé slice (Nx=W, Ny=H, dx,dy from voxel_size),
    and runs a homogeneous-medium plane-wave simulation across a small set of
    angles using sim.kwave.common.run_angle_once.
  - The resulting RF stacks and basic metadata (including the H1/H0 ROI masks)
    are written under a per-slice output directory, ready to be consumed by
    the existing replay / STAP pipeline in later phases.

Physics here is deliberately simple ("open-skull" homogeneous medium); later
Brain-MaceBridge* profiles can replace the medium construction with a richer
vessel / flow model while reusing the same anatomical scaffold and RF format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

from mace_bridge import (
    alias_vessels_to_array,
    generate_alias_vessels,
    generate_microvascular_vessels,
    load_mace_template,
    micro_vessels_to_array,
)
from sim.kwave.common import AngleData, SimGeom, run_angle_once


def _iter_slices(template: MaceTemplate) -> Iterable[Tuple[int, int, int]]:
    """
    Yield (idx, scan_idx, plane_idx) for each slice in the template.
    """

    for idx, s in enumerate(template.slices):
        yield idx, int(s.scan_idx), int(s.plane_idx)


def _parse_angles(angles_str: str) -> List[float]:
    out: List[float] = []
    for part in angles_str.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _max_prime_factor(n: int) -> int:
    """Return the largest prime factor of n (n>=2)."""

    n = int(n)
    if n <= 1:
        return n
    max_p = 1
    # Factor out 2s
    while n % 2 == 0:
        max_p = 2
        n //= 2
    p = 3
    while p * p <= n:
        while n % p == 0:
            max_p = p
            n //= p
        p += 2
    if n > 1:
        max_p = n
    return max_p


def _adjust_dim_for_kwave(n: int, max_prime: int = 7, search_span: int = 16) -> int:
    """
    Adjust a grid dimension to keep k-Wave prime factors small for speed.

    We search upward from n to n+search_span and return the first m whose
    largest prime factor is <= max_prime. If none is found, we fall back to n.
    """

    n_int = int(n)
    for m in range(n_int, n_int + search_span + 1):
        if _max_prime_factor(m) <= max_prime:
            return m
    return n_int


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "MaceBridge-OpenSkull k-Wave simulations on Macé-aligned grids. "
            "This is a geometry/IO scaffold; physics is a simple homogeneous medium."
        )
    )
    ap.add_argument(
        "--template",
        type=Path,
        default=Path("data/mace_template/mace_template_v1.npz"),
        help="Path to a MaceTemplate .npz file (see mace_bridge.template).",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("runs/macebridge_open"),
        help="Root directory for simulated RF stacks and metadata.",
    )
    ap.add_argument(
        "--scan-idx",
        type=int,
        default=None,
        help="Optional scan index filter (0-based) to restrict which slices to simulate.",
    )
    ap.add_argument(
        "--plane-idx",
        type=int,
        default=None,
        help="Optional plane index filter (0-based) to restrict which slices to simulate.",
    )
    ap.add_argument(
        "--angles",
        type=str,
        default="-12,-6,0,6,12",
        help="Comma-separated steering angles in degrees (default: '-12,-6,0,6,12').",
    )
    ap.add_argument(
        "--f0-mhz",
        type=float,
        default=7.5,
        help="Center frequency in MHz (default: 7.5).",
    )
    ap.add_argument(
        "--ncycles",
        type=int,
        default=3,
        help="Tone-burst cycles for the transmit pulse (default: 3).",
    )
    ap.add_argument(
        "--prf-hz",
        type=float,
        default=1500.0,
        help="Slow-time PRF in Hz (recorded to metadata; k-Wave itself uses makeTime).",
    )
    ap.add_argument(
        "--pulses",
        type=int,
        default=64,
        help="Synthetic slow-time pulses per angle set (recorded to metadata).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for any downstream stochastic components (recorded only).",
    )
    ap.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU execution for k-Wave even if GPU binaries are available.",
    )
    args = ap.parse_args()

    template = load_mace_template(args.template)
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    angles = _parse_angles(args.angles)
    if not angles:
        raise SystemExit("At least one steering angle must be provided via --angles.")

    for idx, scan_idx, plane_idx in _iter_slices(template):
        slice_tpl = template.slices[idx]
        if args.scan_idx is not None and scan_idx != args.scan_idx:
            continue
        if args.plane_idx is not None and plane_idx != args.plane_idx:
            continue

        H_native = int(slice_tpl.height)
        W_native = int(slice_tpl.width)
        dy_um, dx_um, dz_um = slice_tpl.voxel_size_um
        dx = float(dx_um) * 1e-6
        dy = float(dy_um) * 1e-6

        # k-Wave prefers grid dimensions whose prime factors are small; adjust
        # the depth dimension upward slightly when necessary to avoid large
        # primes (e.g., 61) that trigger performance warnings. Lateral size
        # W_native is already typically power-of-two; we keep it unchanged.
        Ny = _adjust_dim_for_kwave(H_native)
        Nx = W_native

        g = SimGeom(
            Nx=Nx,
            Ny=Ny,
            dx=dx,
            dy=dy,
            c0=1540.0,
            rho0=1000.0,
            f0=float(args.f0_mhz) * 1e6,
            ncycles=int(args.ncycles),
        )

        slice_dir = out_root / f"scan{scan_idx}_plane{plane_idx}"
        slice_dir.mkdir(parents=True, exist_ok=True)

        angle_data: List[AngleData] = []
        for ang_deg in angles:
            ang_dir = slice_dir / f"angle_{int(round(ang_deg))}"
            res = run_angle_once(ang_dir, ang_deg, g, use_gpu=not args.force_cpu)
            # Save RF and dt per angle for convenience.
            np.save(ang_dir / "rf.npy", res.rf.astype(np.float32), allow_pickle=False)
            np.save(ang_dir / "dt.npy", np.array(res.dt, dtype=np.float32), allow_pickle=False)
            angle_data.append(res)

        # Stack angles into a single RF cube for downstream beamforming/replay.
        if angle_data:
            rf_stack = np.stack([ad.rf for ad in angle_data], axis=0).astype(np.float32)
        else:
            rf_stack = np.empty((0,), dtype=np.float32)
        np.save(slice_dir / "rf_stack.npy", rf_stack, allow_pickle=False)

        # Save ROI masks and basic metadata so replay/STAP can align to Macé ROIs.
        roi_H1 = slice_tpl.roi_masks["H1"].astype(bool, copy=False)
        roi_H0 = slice_tpl.roi_masks["H0"].astype(bool, copy=False)
        np.save(slice_dir / "roi_H1.npy", roi_H1, allow_pickle=False)
        np.save(slice_dir / "roi_H0.npy", roi_H0, allow_pickle=False)

        # Phase A: precompute simple ROI-conditioned vessel fields on the
        # native Macé grid so that downstream replay can inject Pf/Pa
        # physics in a way that respects Allen anatomy without modifying
        # the k-Wave front-end. These arrays are internal to MaceBridge
        # and are stored in a numeric (N, D) layout for ease of use.
        rng = np.random.default_rng(int(args.seed) + 31 * scan_idx + 7 * plane_idx)
        micro_vessels = generate_microvascular_vessels(roi_H1, roi_H0, rng=rng)
        alias_vessels = generate_alias_vessels(roi_H1, roi_H0, rng=rng)
        if micro_vessels:
            mv_arr = micro_vessels_to_array(micro_vessels)
            np.save(slice_dir / "micro_vessels.npy", mv_arr, allow_pickle=False)
        if alias_vessels:
            av_arr = alias_vessels_to_array(alias_vessels)
            np.save(slice_dir / "alias_vessels.npy", av_arr, allow_pickle=False)

        meta = {
            "sim_geom": {
                "Nx": g.Nx,
                "Ny": g.Ny,
                "dx": g.dx,
                "dy": g.dy,
                "c0": g.c0,
                "rho0": g.rho0,
                "pml_size": g.pml_size,
                "cfl": g.cfl,
                "f0_hz": g.f0,
                "ncycles": g.ncycles,
            },
            "angles_deg": angles,
            "seed": int(args.seed),
            "prf_hz": float(args.prf_hz),
            "pulses": int(args.pulses),
            "scan_idx": scan_idx,
            "scan_name": slice_tpl.scan_name,
            "plane_idx": plane_idx,
            "voxel_size_um": {
                "dy": float(dy_um),
                "dx": float(dx_um),
                "dz": float(dz_um),
            },
            "roi_pixel_counts": {
                "H1": int(roi_H1.sum()),
                "H0": int(roi_H0.sum()),
            },
            "tile": {
                "tile_h": slice_tpl.tile_h,
                "tile_w": slice_tpl.tile_w,
                "stride": slice_tpl.stride,
                "n_tiles": len(slice_tpl.tile_origins),
            },
            "source": "macebridge_open_sim",
            "force_cpu": bool(args.force_cpu),
        }
        with open(slice_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(
            f"[macebridge_open_sim] slice (scan_idx={scan_idx}, plane_idx={plane_idx}) -> "
            f"{slice_dir} (angles={len(angles)}, Nx={g.Nx}, Ny={g.Ny})"
        )


if __name__ == "__main__":
    main()
