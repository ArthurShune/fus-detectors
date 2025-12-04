#!/usr/bin/env python3
"""
Quantify Doppler aliasing and motion conditions for the current pilots.

Outputs:
- Lambda, Nyquist, and minimum flow speed to induce aliasing
- Required PRF for alias at given flow speeds
- Per-frame motion shifts vs grid spacing to justify registration stress

Usage:
  PYTHONPATH=. python scripts/alias_math.py \
    --f0 7.5e6 --prf 1500 --angles -12,-6,0,6,12 \
    --flow-mm-s 8,20,40,80 --motion-um 40 --motion-hz 0.5 \
    --dx 9.0e-5 --dy 9.0e-5
"""
from __future__ import annotations

import argparse
import math
from typing import List


def doppler_hz(v_m_s: float, f0_hz: float, c0: float, theta_deg: float) -> float:
    lam = c0 / f0_hz
    return 2.0 * v_m_s * math.cos(math.radians(theta_deg)) / lam


def min_speed_for_alias(prf_hz: float, f0_hz: float, c0: float, theta_deg: float = 0.0) -> float:
    lam = c0 / f0_hz
    # alias if |f_d| > PRF/2 => 2 v cos / lam > PRF/2
    cos = abs(math.cos(math.radians(theta_deg))) or 1e-12
    return (prf_hz * lam) / (4.0 * cos)


def prf_for_alias(v_m_s: float, f0_hz: float, c0: float, theta_deg: float = 0.0) -> float:
    lam = c0 / f0_hz
    cos = abs(math.cos(math.radians(theta_deg))) or 1e-12
    # rearrange: PRF_min = 4 v cos / lam to place f_d at Nyquist
    return 4.0 * v_m_s * cos / lam


def motion_peak_velocity(motion_um: float, motion_hz: float) -> float:
    # sinusoidal displacement with amplitude A (meters) and freq f (Hz): v_max = 2π f A
    return 2.0 * math.pi * motion_hz * (motion_um * 1e-6)


def main() -> None:
    ap = argparse.ArgumentParser(description="Alias/motion math for r4")
    ap.add_argument("--f0", type=float, default=7.5e6)
    ap.add_argument("--prf", type=float, default=1500.0)
    ap.add_argument("--c0", type=float, default=1540.0)
    ap.add_argument("--angles", type=str, default="-12,-6,0,6,12")
    ap.add_argument("--flow-mm-s", type=str, default="8,20,40,80")
    ap.add_argument("--motion-um", type=float, default=40.0)
    ap.add_argument("--motion-hz", type=float, default=0.5)
    ap.add_argument("--dx", type=float, default=90e-6)
    ap.add_argument("--dy", type=float, default=90e-6)
    args = ap.parse_args()

    angles = [float(s) for s in args.angles.split(",") if s.strip()]
    flows = [float(s) * 1e-3 for s in args.flow_mm_s.split(",") if s.strip()]  # m/s

    lam = args.c0 / args.f0
    nyq = args.prf / 2.0
    print(f"lambda = {lam*1e3:.3f} mm, PRF = {args.prf:.1f} Hz, Nyquist = {nyq:.1f} Hz")

    print("\nFlow Doppler (no alias unless > Nyquist):")
    for v in flows:
        vals = [doppler_hz(v, args.f0, args.c0, a) for a in angles]
        print(f"  v={v*1e3:6.1f} mm/s => f_d in [{min(vals):.1f}, {max(vals):.1f}] Hz")

    print("\nMin speed for alias at PRF, per angle:")
    for a in angles:
        vmin = min_speed_for_alias(args.prf, args.f0, args.c0, a)
        print(f"  theta={a:>5.1f}° => v_alias_min = {vmin*1e3:6.1f} mm/s")

    print("\nPRF needed for alias at given speeds (theta=0°):")
    for v in flows:
        prf_need = prf_for_alias(v, args.f0, args.c0, 0.0)
        print(f"  v={v*1e3:6.1f} mm/s => PRF_min for alias ~ {prf_need:.0f} Hz")

    v_motion = motion_peak_velocity(args.motion_um, args.motion_hz)
    fdm = doppler_hz(v_motion, args.f0, args.c0, 0.0)
    print("\nMotion (sinusoidal) peak velocity:")
    print(
        f"  A={args.motion_um:.1f} µm, f={args.motion_hz:.2f} Hz => v_max={v_motion*1e3:.3f} mm/s, f_d≈{fdm:.2f} Hz"
    )

    # Per-frame pixel shift amplitude (heuristic): compare displacement amplitude vs grid spacing
    shift_x = args.motion_um * 1e-6 / max(args.dx, 1e-12)
    shift_z = 0.5 * args.motion_um * 1e-6 / max(args.dy, 1e-12)
    print("\nRegistration stress (amplitude vs grid spacing):")
    print(
        f"  Grid dx={args.dx*1e6:.1f} µm, dy={args.dy*1e6:.1f} µm => peak shift ~ {shift_x:.2f} px (x), {shift_z:.2f} px (z)"
    )


if __name__ == "__main__":
    main()
