#!/usr/bin/env python3
"""
Helper to launch the standard k-Wave pilot runs (R1 + R2) in sequence.

Edit ``DEFAULT_R1_ARGS`` / ``DEFAULT_R2_ARGS`` below if you need to tune
the configuration later; the CLI just exposes a few convenience toggles.
Run via:

    conda run -n fus-detectors python scripts/run_pilots.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import List


DEFAULT_R1_ARGS: List[str] = [
    "--out",
    "runs/pilot/r1_real",
    "--angles=-10,0,10",
    "--pulses",
    "64",
    "--prf",
    "3000",
    "--tile-h",
    "12",
    "--tile-w",
    "12",
    "--tile-stride",
    "6",
    "--lt",
    "4",
    "--diag-load",
    "1e-2",
    "--cov-estimator",
    "tyler_pca",
    "--mvdr-load-mode",
    "absolute",
    "--constraint-ridge",
    "0.12",
    "--msd-lambda",
    "5e-2",
    "--msd-ridge",
    "0.12",
    "--msd-agg",
    "median",
    "--msd-ratio-rho",
    "0.05",
    "--motion-half-span-rel",
    "0.15",
    "--msd-contrast-alpha",
    "0.7",
    "--stap-device",
    "cuda",
    "--fd-span-mode",
    "psd",
    "--fd-span-rel",
    "0.30,1.10",
    "--grid-step-rel",
    "0.12",
    "--max-pts",
    "5",
    "--stap-debug-samples",
    "3",
]


DEFAULT_R2_ARGS: List[str] = [
    "--out",
    "runs/pilot/r2_real",
    "--angles=-12,-6,0,6,12",
    "--ensembles",
    "5",
    "--pulses",
    "48",
    "--prf",
    "3000",
    "--tile-h",
    "12",
    "--tile-w",
    "12",
    "--tile-stride",
    "6",
    "--lt",
    "4",
    "--diag-load",
    "1e-2",
    "--cov-estimator",
    "tyler_pca",
    "--mvdr-load-mode",
    "auto",
    "--constraint-ridge",
    "0.15",
    "--msd-lambda",
    "6e-2",
    "--msd-ridge",
    "0.15",
    "--msd-agg",
    "median",
    "--msd-ratio-rho",
    "0.05",
    "--motion-half-span-rel",
    "0.20",
    "--msd-contrast-alpha",
    "0.8",
    "--stap-device",
    "cuda",
    "--fd-span-mode",
    "psd",
    "--fd-span-rel",
    "0.30,1.20",
    "--grid-step-rel",
    "0.12",
    "--max-pts",
    "5",
    "--stap-debug-samples",
    "3",
]


def _set_arg(args_list: List[str], flag: str, value: str) -> None:
    """Replace or append ``flag`` with ``value`` in-place."""
    for idx, token in enumerate(args_list):
        if token == flag:
            if idx + 1 >= len(args_list):
                raise ValueError(f"{flag!r} missing value in args list")
            args_list[idx + 1] = value
            return
        if token.startswith(f"{flag}="):
            args_list[idx] = f"{flag}={value}"
            return
    # Not present: append flag/value pair.
    if flag.endswith("="):
        args_list.append(f"{flag}{value}")
    else:
        args_list.extend([flag, value])


def _run(label: str, cmd: List[str], dry_run: bool) -> None:
    prefix = "[dry-run]" if dry_run else "[run]"
    print(f"{prefix} {' '.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the canonical R1 + R2 pilot workflows.")
    parser.add_argument("--r1-out", type=str, default=None, help="Override R1 output directory.")
    parser.add_argument("--r2-out", type=str, default=None, help="Override R2 output directory.")
    parser.add_argument("--skip-r1", action="store_true", help="Skip the R1 pilot run.")
    parser.add_argument("--skip-r2", action="store_true", help="Skip the R2 pilot run.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing them.",
    )
    args = parser.parse_args()

    if args.skip_r1 and args.skip_r2:
        print("Nothing to do (both pilots skipped).")
        return

    r1_args = list(DEFAULT_R1_ARGS)
    r2_args = list(DEFAULT_R2_ARGS)
    if args.r1_out:
        _set_arg(r1_args, "--out", args.r1_out)
    if args.r2_out:
        _set_arg(r2_args, "--out", args.r2_out)

    if not args.skip_r1:
        _run("pilot_r1", [sys.executable, "sim/kwave/pilot_r1.py", *r1_args], args.dry_run)
    if not args.skip_r2:
        _run("pilot_motion", [sys.executable, "sim/kwave/pilot_motion.py", *r2_args], args.dry_run)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        cmd_str = " ".join(exc.cmd) if isinstance(exc.cmd, list) else str(exc.cmd)
        print(
            f"[error] Command failed with exit code {exc.returncode}: {cmd_str}", file=sys.stderr
        )
        sys.exit(exc.returncode)
