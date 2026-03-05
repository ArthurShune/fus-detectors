#!/usr/bin/env python3
"""
Derive a standard acceptance bundle from an existing canonical dataset/ directory.

This keeps the repo's "dataset is canonical, bundle is derived" contract:
  <run>/dataset/{icube.npy,mask_*.npy,meta.json}  ->  <run>/bundle/<dataset_name>/*

Unlike the backend-specific pilots (k-Wave / SIMUS), this script does not
regenerate simulation outputs; it only reads the canonical artifacts and
executes the existing baseline+STAP pipeline on that Icube.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from sim.kwave.icube_bundle import write_acceptance_bundle_from_icube


def _slugify(text: str) -> str:
    s = re.sub(r"[\s/]+", "_", str(text).strip())
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s.strip("._-") or "run"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Derive an acceptance bundle from an existing dataset/ directory.")
    ap.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Run directory containing dataset/icube.npy + dataset/meta.json + masks.",
    )
    ap.add_argument(
        "--bundle-root",
        type=Path,
        default=None,
        help="Bundle root directory (default: <run>/bundle).",
    )
    ap.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Bundle dataset name under bundle-root (default: run directory name).",
    )
    ap.add_argument(
        "--stap-device",
        type=str,
        default=None,
        help="STAP device: cpu, cuda, cuda:0, ... (default: auto).",
    )

    # Bundle-profile knobs. Defaults match the physical-doppler / SIMUS pilots.
    ap.add_argument("--tile-hw", type=int, nargs=2, default=(8, 8), metavar=("H", "W"))
    ap.add_argument("--tile-stride", type=int, default=3)
    ap.add_argument("--Lt", type=int, default=8)
    ap.add_argument("--diag-load", type=float, default=1e-2)
    ap.add_argument("--cov-estimator", type=str, default="tyler_pca")
    ap.add_argument("--baseline-type", type=str, default="mc_svd")
    ap.add_argument("--svd-energy-frac", type=float, default=0.90)

    ap.add_argument("--stap-conditional-enable", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--feasibility-mode", type=str, default="updated", choices=["legacy", "updated"])
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run)
    dataset_dir = run_dir / "dataset"
    meta_path = dataset_dir / "meta.json"
    icube_path = dataset_dir / "icube.npy"
    mask_flow_path = dataset_dir / "mask_flow.npy"
    mask_bg_path = dataset_dir / "mask_bg.npy"
    for p in (meta_path, icube_path, mask_flow_path, mask_bg_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing required dataset artifact: {p}")

    meta = _load_json(meta_path)
    prf = float(meta.get("acquisition", {}).get("prf_hz", meta.get("config", {}).get("prf_hz", 0.0)))
    if prf <= 0.0:
        raise ValueError(f"dataset meta missing valid prf_hz: {prf}")

    Icube = np.load(icube_path).astype(np.complex64, copy=False)
    mask_flow = np.load(mask_flow_path).astype(bool, copy=False)
    mask_bg = np.load(mask_bg_path).astype(bool, copy=False)

    bundle_root = Path(args.bundle_root) if args.bundle_root is not None else (run_dir / "bundle")
    bundle_root.mkdir(parents=True, exist_ok=True)

    dataset_name = _slugify(args.dataset_name or run_dir.name)

    write_acceptance_bundle_from_icube(
        out_root=bundle_root,
        dataset_name=dataset_name,
        Icube=Icube,
        prf_hz=prf,
        seed=int(meta.get("config", {}).get("seed", meta.get("seed", 0)) or 0),
        tile_hw=(int(args.tile_hw[0]), int(args.tile_hw[1])),
        tile_stride=int(args.tile_stride),
        Lt=int(args.Lt),
        diag_load=float(args.diag_load),
        cov_estimator=str(args.cov_estimator),
        baseline_type=str(args.baseline_type),
        reg_enable=True,
        reg_method="phasecorr",
        reg_subpixel=4,
        reg_reference="median",
        svd_energy_frac=float(args.svd_energy_frac),
        mask_flow_override=mask_flow,
        mask_bg_override=mask_bg,
        stap_conditional_enable=bool(args.stap_conditional_enable),
        feasibility_mode=str(args.feasibility_mode),
        stap_device=args.stap_device,
        meta_extra={
            "bundle_from_dataset": True,
            "dataset_rel": str(dataset_dir.relative_to(run_dir)),
            "source_meta_rel": str(meta_path.relative_to(run_dir)),
        },
    )

    print(f"[icube-make-bundle] wrote bundle -> {bundle_root / dataset_name}", flush=True)


if __name__ == "__main__":
    main()

