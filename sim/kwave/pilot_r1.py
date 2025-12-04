# sim/kwave/pilot_r1.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from sim.kwave.common import (  # noqa: E402
    AngleData,
    SimGeom,
    run_angle_once,
    write_acceptance_bundle,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "k-Wave R1 pilot: emit acceptance artifacts via simulation or synthetic mode."
        )
    )
    ap.add_argument("--Nx", type=int, default=256)
    ap.add_argument("--Ny", type=int, default=256)
    ap.add_argument("--dx", type=float, default=90e-6)
    ap.add_argument("--dy", type=float, default=90e-6)
    ap.add_argument("--c0", type=float, default=1540.0)
    ap.add_argument("--rho0", type=float, default=1000.0)
    ap.add_argument(
        "--angles",
        type=str,
        default="-10,0,10",
        help="Comma-separated steering angles in degrees.",
    )
    ap.add_argument("--f0", type=float, default=7.5e6)
    ap.add_argument("--ncycles", type=int, default=3)
    ap.add_argument(
        "--pulses",
        type=int,
        default=64,
        help="Synthetic slow-time pulses to synthesize per angle set.",
    )
    ap.add_argument("--prf", type=float, default=3000.0, help="Slow-time PRF (Hz).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="Skip k-Wave and emit synthetic RF for fast smoke tests.",
    )
    ap.add_argument("--tile-h", type=int, default=12)
    ap.add_argument("--tile-w", type=int, default=12)
    ap.add_argument("--tile-stride", type=int, default=6)
    ap.add_argument("--lt", type=int, default=4)
    ap.add_argument("--diag-load", type=float, default=1e-2)
    ap.add_argument("--cov-estimator", type=str, default="tyler_pca")
    ap.add_argument("--huber-c", type=float, default=5.0)
    ap.add_argument("--stap-debug-samples", type=int, default=0)
    ap.add_argument(
        "--stap-debug-coord",
        action="append",
        default=[],
        help=(
            "Tile origin (y,x) to force into debug capture; repeat for multiple tiles. "
            "Coordinates should align with the chosen tile stride."
        ),
    )
    ap.add_argument(
        "--fd-span-mode",
        type=str,
        default="psd",
        choices=["psd", "auto", "fixed"],
        help="Strategy for Doppler span selection.",
    )
    ap.add_argument(
        "--fd-span-rel",
        type=str,
        default="0.30,1.10",
        help="Min,max relative span to PRF/Lt (used for psd/auto modes).",
    )
    ap.add_argument(
        "--fd-fixed-span-hz",
        type=float,
        default=None,
        help="Override absolute span (Hz) when fd-span-mode=fixed.",
    )
    ap.add_argument(
        "--constraint-mode",
        type=str,
        default="exp+deriv",
        choices=["exp", "exp+deriv"],
        help="Constraint mode for temporal bandpass projector.",
    )
    ap.add_argument(
        "--grid-step-rel",
        type=float,
        default=0.12,
        help="Relative frequency step for Doppler grid.",
    )
    ap.add_argument(
        "--max-pts",
        type=int,
        default=5,
        help="Maximum Doppler tones (odd) for the LCMV constraint grid.",
    )
    ap.add_argument(
        "--fd-min-pts",
        type=int,
        default=3,
        help="Minimum Doppler tones (odd) to keep after rank-based pruning.",
    )
    ap.add_argument(
        "--fd-min-abs-hz",
        type=float,
        default=0.0,
        help=(
            "Discard Doppler tones with |f| below this threshold when building the flow"
            " subspace."
        ),
    )
    ap.add_argument(
        "--mvdr-load-mode",
        type=str,
        default="auto",
        choices=["auto", "absolute"],
        help="Diagonal loading mode for the LCMV solver.",
    )
    ap.add_argument(
        "--mvdr-auto-kappa",
        type=float,
        default=50.0,
        help="Target condition number when mvdr-load-mode=auto.",
    )
    ap.add_argument(
        "--constraint-ridge",
        type=float,
        default=0.10,
        help="Constraint ridge (δ) added to the Gram matrix.",
    )
    ap.add_argument(
        "--msd-agg",
        type=str,
        default="median",
        choices=["mean", "median", "trim10"],
        help="Aggregation mode for temporal MSD ratio (default median).",
    )
    ap.add_argument(
        "--msd-lambda",
        type=float,
        default=5e-2,
        help="Absolute diagonal loading (λ) for MSD ratio scoring (defaults to 0.05).",
    )
    ap.add_argument(
        "--msd-ridge",
        type=float,
        default=0.12,
        help="Ridge term added to the constraint Gram matrix for MSD scoring.",
    )
    ap.add_argument(
        "--msd-ratio-rho",
        type=float,
        default=0.05,
        help="Shrinkage factor ρ applied to the MSD ratio denominator to tame heavy tails.",
    )
    ap.add_argument(
        "--motion-half-span-rel",
        type=float,
        default=0.15,
        help=(
            "Relative half-span (w.r.t. PRF/Lt) treated as motion band; set <=0 to disable"
            " motion contrast."
        ),
    )
    ap.add_argument(
        "--msd-contrast-alpha",
        type=float,
        default=0.7,
        help="Contrast weight between flow and motion bands (<=0 disables motion contrast).",
    )
    ap.add_argument(
        "--ka-mode",
        type=str,
        default="none",
        choices=["none", "analytic", "library"],
        help="KA prior mode for temporal whitening (default none).",
    )
    ap.add_argument(
        "--ka-prior-path",
        type=str,
        default=None,
        help="Path to a saved KA prior matrix (required when --ka-mode=library).",
    )
    ap.add_argument(
        "--ka-beta-bounds",
        type=str,
        default="0.05,0.50",
        help="Comma-separated min,max bounds for KA beta blending.",
    )
    ap.add_argument(
        "--ka-kappa",
        type=float,
        default=40.0,
        help="Target condition number for KA whitened covariance.",
    )
    ap.add_argument(
        "--ka-alpha",
        type=float,
        default=None,
        help="Optional LW-style alpha shrink toward identity before KA blend.",
    )
    ap.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU execution even if GPU binaries are available.",
    )
    ap.add_argument(
        "--stap-device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run STAP processing on (default auto).",
    )
    ap.add_argument(
        "--feasibility-mode",
        type=str,
        default="legacy",
        choices=["legacy", "updated"],
        help="Feasibility configuration applied to STAP/KA processing.",
    )
    ap.add_argument(
        "--guard-target-med",
        type=float,
        default=0.45,
        help="Target tile median flow ratio for STAP guard scaling (default 0.45).",
    )
    ap.add_argument(
        "--guard-target-low",
        type=float,
        default=0.16,
        help="Target tile low-percentile flow ratio for guard scaling (default 0.16).",
    )
    ap.add_argument(
        "--guard-percentile-low",
        type=float,
        default=0.10,
        help="Percentile (0-1) used for the low-ratio guard metric (default 0.10).",
    )
    ap.add_argument(
        "--guard-tile-coverage-min",
        type=float,
        default=0.10,
        help="Minimum flow-mask coverage fraction for a tile to count toward guard metrics.",
    )
    ap.add_argument(
        "--guard-max-scale",
        type=float,
        default=1.30,
        help="Maximum flow guard scaling factor (default 1.30).",
    )
    # Background variance guard controls
    ap.add_argument(
        "--bg-guard-enabled",
        action="store_true",
        help="Enable background variance guard (tile p90 cap).",
    )
    ap.add_argument(
        "--bg-guard-target-p90",
        type=float,
        default=1.15,
        help="Target p90 tile BG variance inflation ratio (default 1.15).",
    )
    ap.add_argument(
        "--bg-guard-min-alpha",
        type=float,
        default=0.6,
        help="Minimum shrink factor alpha used by BG guard (default 0.6).",
    )
    ap.add_argument(
        "--bg-guard-metric",
        type=str,
        default="tile_p90",
        choices=["tile_p90", "global"],
        help="Metric for BG guard trigger: tile p90 or global variance ratio.",
    )
    ap.add_argument("--out", type=str, default="runs/pilot/r1")
    args = ap.parse_args()

    try:
        ka_beta_bounds = tuple(float(v) for v in args.ka_beta_bounds.split(","))
    except ValueError as exc:  # pragma: no cover - CLI guard
        raise SystemExit(f"Invalid --ka-beta-bounds value {args.ka_beta_bounds!r}") from exc
    if len(ka_beta_bounds) != 2:
        raise SystemExit("--ka-beta-bounds must provide two comma-separated floats (min,max).")
    if ka_beta_bounds[0] < 0 or ka_beta_bounds[1] <= ka_beta_bounds[0]:
        raise SystemExit("Require 0 <= min_beta < max_beta for --ka-beta-bounds.")

    debug_coords: list[tuple[int, int]] = []
    for coord_str in args.stap_debug_coord:
        coord_clean = coord_str.strip().replace(":", ",")
        parts = [p.strip() for p in coord_clean.split(",") if p.strip()]
        if len(parts) != 2:
            raise SystemExit(f"Invalid --stap-debug-coord value {coord_str!r}; expected 'y,x'.")
        try:
            y0 = int(parts[0])
            x0 = int(parts[1])
        except ValueError as exc:  # pragma: no cover - CLI guard
            raise SystemExit(
                f"Invalid --stap-debug-coord value {coord_str!r}; expected integers."
            ) from exc
        if y0 < 0 or x0 < 0:
            raise SystemExit("--stap-debug-coord requires non-negative y,x indices.")
        if args.tile_stride > 0 and (y0 % args.tile_stride != 0 or x0 % args.tile_stride != 0):
            print(
                f"[pilot_r1] warning: debug coord ({y0},{x0}) is not aligned with stride "
                f"{args.tile_stride}; request may not match any tile.",
                file=sys.stderr,
            )
        debug_coords.append((y0, x0))

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    guard_opts = {
        "guard_target_med": float(args.guard_target_med),
        "guard_target_low": float(args.guard_target_low),
        "guard_percentile_low": float(args.guard_percentile_low),
        "guard_tile_coverage_min": float(args.guard_tile_coverage_min),
        "guard_max_scale": float(args.guard_max_scale),
    }
    # Merge BG guard controls
    guard_opts.update(
        {
            "bg_guard_enabled": bool(args.bg_guard_enabled),
            "bg_guard_target_p90": float(args.bg_guard_target_p90),
            "bg_guard_min_alpha": float(args.bg_guard_min_alpha),
            "bg_guard_metric": str(args.bg_guard_metric),
        }
    )

    geom = SimGeom(
        Nx=args.Nx,
        Ny=args.Ny,
        dx=args.dx,
        dy=args.dy,
        c0=args.c0,
        rho0=args.rho0,
        f0=args.f0,
        ncycles=args.ncycles,
    )

    angles: List[float] = [float(s) for s in args.angles.split(",") if s.strip()]
    angle_data: List[AngleData] = []
    rng = np.random.default_rng(args.seed)

    for idx, angle_deg in enumerate(angles):
        angle_dir = out_root / f"angle_{int(round(angle_deg))}"
        angle_dir.mkdir(parents=True, exist_ok=True)
        if args.synthetic:
            Nt = max(128, geom.Ny * 2)
            dt = geom.cfl * min(geom.dx, geom.dy) / max(geom.c0, 1.0)
            rf = rng.standard_normal((Nt, geom.Nx)).astype(np.float32)
            rf += 0.2 * rng.standard_normal((Nt, geom.Nx)).astype(np.float32)
            t = np.arange(Nt, dtype=np.float32) * dt
            rf += 0.1 * np.sin(2.0 * np.pi * 0.05 * (idx + 1) * t)[:, None]
            res = AngleData(angle_deg=angle_deg, rf=rf, dt=float(dt))
        else:
            res = run_angle_once(angle_dir, angle_deg, geom, use_gpu=not args.force_cpu)
        np.save(angle_dir / "rf.npy", res.rf.astype(np.float32), allow_pickle=False)
        np.save(angle_dir / "dt.npy", np.array(res.dt, dtype=np.float32), allow_pickle=False)
        angle_data.append(res)

    if angle_data:
        stack = np.stack([ad.rf for ad in angle_data], axis=0).astype(np.float32)
    else:
        stack = np.empty((0,), dtype=np.float32)
    np.save(out_root / "rf_stack.npy", stack, allow_pickle=False)

    meta = {
        "geometry": {
            "Nx": geom.Nx,
            "Ny": geom.Ny,
            "dx": geom.dx,
            "dy": geom.dy,
            "c0": geom.c0,
            "rho0": geom.rho0,
            "pml": geom.pml_size,
            "cfl": geom.cfl,
        },
        "angles_deg": angles,
        "f0_hz": geom.f0,
        "ncycles": geom.ncycles,
        "seed": args.seed,
        "pulses": args.pulses,
        "prf_hz": args.prf,
        "synthetic": bool(args.synthetic),
        "dt": [ad.dt for ad in angle_data],
    }
    with open(out_root / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    span_bounds = tuple(float(x.strip()) for x in args.fd_span_rel.split(",") if x.strip())
    if len(span_bounds) != 2:
        raise ValueError("--fd-span-rel must be 'min,max'")
    ratio_rho = max(0.0, float(args.msd_ratio_rho))
    motion_half_span_rel = None
    if args.motion_half_span_rel is not None and args.motion_half_span_rel > 0.0:
        motion_half_span_rel = float(args.motion_half_span_rel)
    contrast_alpha = None
    if args.msd_contrast_alpha is not None and args.msd_contrast_alpha > 0.0:
        contrast_alpha = float(args.msd_contrast_alpha)
    bundle_paths = write_acceptance_bundle(
        out_root=out_root,
        g=geom,
        angle_sets=[angle_data],
        pulses_per_set=args.pulses,
        prf_hz=args.prf,
        seed=args.seed,
        tile_hw=(int(args.tile_h), int(args.tile_w)),
        tile_stride=int(args.tile_stride),
        Lt=int(args.lt),
        diag_load=float(args.diag_load),
        cov_estimator=str(args.cov_estimator).lower(),
        huber_c=float(args.huber_c),
        stap_debug_samples=int(args.stap_debug_samples),
        fd_span_mode=str(args.fd_span_mode).lower(),
        fd_span_rel=span_bounds,
        fd_fixed_span_hz=args.fd_fixed_span_hz,
        constraint_mode=str(args.constraint_mode).lower(),
        grid_step_rel=float(args.grid_step_rel),
        fd_min_pts=int(args.fd_min_pts),
        fd_max_pts=int(args.max_pts),
        fd_min_abs_hz=float(args.fd_min_abs_hz),
        msd_lambda=args.msd_lambda,
        msd_ridge=float(args.msd_ridge),
        msd_agg_mode=str(args.msd_agg).lower(),
        msd_ratio_rho=ratio_rho,
        motion_half_span_rel=motion_half_span_rel,
        msd_contrast_alpha=contrast_alpha,
        stap_debug_tile_coords=debug_coords,
        ka_mode=str(args.ka_mode).lower(),
        ka_prior_path=args.ka_prior_path,
        ka_beta_bounds=ka_beta_bounds,
        ka_kappa=float(args.ka_kappa),
        ka_alpha=args.ka_alpha,
        ka_opts_extra=guard_opts,
        mvdr_load_mode=str(args.mvdr_load_mode).lower(),
        mvdr_auto_kappa=float(args.mvdr_auto_kappa),
        constraint_ridge=float(args.constraint_ridge),
        stap_device=args.stap_device,
        meta_extra={
            "source": "pilot_r1",
            "synthetic": bool(args.synthetic),
            "force_cpu": bool(args.force_cpu),
            "stap_device": args.stap_device,
            "msd_ratio_rho": ratio_rho,
            "motion_half_span_rel": motion_half_span_rel,
            "msd_contrast_alpha": contrast_alpha,
            "msd_agg": str(args.msd_agg).lower(),
        },
        feasibility_mode=str(args.feasibility_mode),
    )

    print(
        f"[pilot_r1] wrote {out_root} with {len(angles)} angles. "
        f"Acceptance bundle at {bundle_paths['meta']}"
    )


if __name__ == "__main__":
    main()
