#!/usr/bin/env python3
"""
Import an externally generated IQ cube (e.g., Field II) into the repo's
canonical dataset layout, then derive a standard acceptance bundle.

This is the "plumbing" needed for the simulation_spec.txt Phase 4/Field II
backend without requiring MATLAB/Field II inside this repo.

Inputs
------
- IQ cube Icube with shape (T, H, W), complex64/complex128:
    - .npy via --icube-npy
    - .mat via --icube-mat (set --icube-key if needed)
- Optional masks (H, W): --mask-flow-npy / --mask-bg-npy / --mask-alias-expected-npy

Outputs
-------
<OUT>/
  dataset/
    icube.npy
    mask_flow.npy (if provided)
    mask_bg.npy   (if provided)
    mask_alias_expected.npy (if provided)
    config.json
    meta.json
    hashes.json
  bundle/
    <dataset_name>/meta.json ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from sim.kwave.icube_bundle import write_acceptance_bundle_from_icube


REPO = Path(__file__).resolve().parents[1]


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_info() -> dict[str, Any]:
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO))
            .decode("utf-8")
            .strip()
        )
        dirty = subprocess.call(
            ["git", "diff", "--quiet"],
            cwd=str(REPO),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        dirty_cached = subprocess.call(
            ["git", "diff", "--quiet", "--cached"],
            cwd=str(REPO),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return {"commit": commit, "dirty": bool(dirty != 0 or dirty_cached != 0)}
    except Exception:
        return {"commit": None, "dirty": None}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _load_mat_array(path: Path, key: str | None) -> np.ndarray:
    import scipy.io as sio

    mat = sio.loadmat(path)
    if key is not None:
        if key not in mat:
            raise KeyError(f"Key {key!r} not found in {path}, keys={sorted(mat.keys())}")
        arr = mat[key]
        return np.asarray(arr)
    # Pick the first non-metadata ndarray.
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray):
            return np.asarray(v)
    raise KeyError(f"No ndarray found in {path}, keys={sorted(mat.keys())}")


def _load_icube(args: argparse.Namespace) -> np.ndarray:
    if args.icube_npy is not None:
        return np.load(Path(args.icube_npy))
    if args.icube_mat is not None:
        arr = _load_mat_array(Path(args.icube_mat), args.icube_key)
        return arr
    raise ValueError("Provide --icube-npy or --icube-mat")


def _maybe_load_mask(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    return np.load(Path(path)).astype(bool, copy=False)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Import external Icube + write canonical dataset + acceptance bundle.")
    ap.add_argument("--out", type=Path, required=True, help="Output run directory (created).")
    ap.add_argument("--dataset-name", type=str, required=True, help="Dataset name used under bundle/.")

    ap.add_argument("--icube-npy", type=str, default=None, help="Path to complex Icube.npy (T,H,W).")
    ap.add_argument("--icube-mat", type=str, default=None, help="Path to .mat containing Icube array.")
    ap.add_argument("--icube-key", type=str, default=None, help="Optional .mat key for Icube.")

    ap.add_argument("--mask-flow-npy", type=str, default=None, help="Optional mask_flow.npy (H,W) bool.")
    ap.add_argument("--mask-bg-npy", type=str, default=None, help="Optional mask_bg.npy (H,W) bool.")
    ap.add_argument(
        "--mask-alias-expected-npy", type=str, default=None, help="Optional mask_alias_expected.npy (H,W) bool."
    )

    ap.add_argument("--prf-hz", type=float, required=True, help="Slow-time PRF (Hz) for Icube.")
    ap.add_argument("--tile-hw", type=int, nargs=2, default=(8, 8), metavar=("H", "W"))
    ap.add_argument("--tile-stride", type=int, default=3)
    ap.add_argument("--Lt", type=int, default=64)
    ap.add_argument("--diag-load", type=float, default=0.07)
    ap.add_argument("--cov-estimator", type=str, default="tyler_pca")
    ap.add_argument("--score-mode", type=str, default="pd")
    ap.add_argument("--baseline-type", type=str, default="mc_svd")
    ap.add_argument("--svd-keep-min", type=int, default=2)
    ap.add_argument("--svd-keep-max", type=int, default=None)
    ap.add_argument("--flow-low-hz", type=float, default=30.0)
    ap.add_argument("--flow-high-hz", type=float, default=250.0)
    ap.add_argument("--alias-center-hz", type=float, default=650.0)
    ap.add_argument("--alias-width-hz", type=float, default=100.0)
    ap.add_argument("--stap-device", type=str, default="cpu")

    ap.add_argument("--write-bundle", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out)
    dataset_dir = out_root / "dataset"
    bundle_root = out_root / "bundle"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    Icube = _load_icube(args)
    Icube = np.asarray(Icube)
    if Icube.ndim != 3:
        raise ValueError(f"Icube must have shape (T,H,W), got {Icube.shape}")
    if not np.iscomplexobj(Icube):
        raise ValueError("Icube must be complex (IQ).")
    Icube = Icube.astype(np.complex64, copy=False)
    T, H, W = (int(Icube.shape[0]), int(Icube.shape[1]), int(Icube.shape[2]))

    mask_flow = _maybe_load_mask(args.mask_flow_npy)
    mask_bg = _maybe_load_mask(args.mask_bg_npy)
    mask_alias = _maybe_load_mask(args.mask_alias_expected_npy)
    for name, m in (("mask_flow", mask_flow), ("mask_bg", mask_bg), ("mask_alias_expected", mask_alias)):
        if m is None:
            continue
        if m.shape != (H, W):
            raise ValueError(f"{name} shape {m.shape} does not match Icube spatial {(H, W)}")

    # Write canonical dataset files.
    icube_path = dataset_dir / "icube.npy"
    np.save(icube_path, Icube)
    files: dict[str, Any] = {
        "icube": {"path": str(icube_path.relative_to(out_root)), "dtype": "complex64", "shape": [T, H, W]},
    }
    if mask_flow is not None:
        p = dataset_dir / "mask_flow.npy"
        np.save(p, mask_flow.astype(bool, copy=False))
        files["mask_flow"] = {"path": str(p.relative_to(out_root))}
    if mask_bg is not None:
        p = dataset_dir / "mask_bg.npy"
        np.save(p, mask_bg.astype(bool, copy=False))
        files["mask_bg"] = {"path": str(p.relative_to(out_root))}
    if mask_alias is not None:
        p = dataset_dir / "mask_alias_expected.npy"
        np.save(p, mask_alias.astype(bool, copy=False))
        files["mask_alias_expected"] = {"path": str(p.relative_to(out_root))}

    cfg = {
        "dataset_name": str(args.dataset_name),
        "source": {
            "icube_npy": str(args.icube_npy) if args.icube_npy else None,
            "icube_mat": str(args.icube_mat) if args.icube_mat else None,
            "icube_key": str(args.icube_key) if args.icube_key else None,
            "mask_flow_npy": str(args.mask_flow_npy) if args.mask_flow_npy else None,
            "mask_bg_npy": str(args.mask_bg_npy) if args.mask_bg_npy else None,
            "mask_alias_expected_npy": str(args.mask_alias_expected_npy) if args.mask_alias_expected_npy else None,
        },
        "acquisition": {"prf_hz": float(args.prf_hz)},
        "bundle_profile": {
            "tile_hw": [int(args.tile_hw[0]), int(args.tile_hw[1])],
            "tile_stride": int(args.tile_stride),
            "Lt": int(args.Lt),
            "diag_load": float(args.diag_load),
            "cov_estimator": str(args.cov_estimator),
            "score_mode": str(args.score_mode),
            "baseline_type": str(args.baseline_type),
            "svd_keep_min": int(args.svd_keep_min),
            "svd_keep_max": int(args.svd_keep_max) if args.svd_keep_max is not None else None,
            "band_ratio_flow_low_hz": float(args.flow_low_hz),
            "band_ratio_flow_high_hz": float(args.flow_high_hz),
            "band_ratio_alias_center_hz": float(args.alias_center_hz),
            "band_ratio_alias_width_hz": float(args.alias_width_hz),
            "stap_device": str(args.stap_device),
        },
    }
    cfg_path = dataset_dir / "config.json"
    _write_json(cfg_path, cfg)
    files["config"] = {"path": str(cfg_path.relative_to(out_root))}

    # Hashes.
    hashes: dict[str, str] = {}
    for k, v in files.items():
        rel = v.get("path")
        if not rel:
            continue
        p = out_root / rel
        hashes[k] = _sha256_file(p)
        v["sha256"] = hashes[k]
    hashes_path = dataset_dir / "hashes.json"
    _write_json(hashes_path, hashes)

    meta = {
        "schema_version": "external_icube_import.v1",
        "created_utc": _utc_now_iso(),
        "provenance": {
            "command": " ".join([Path(__file__).name] + sys.argv[1:]),
            "cwd": str(REPO),
            "git": _git_info(),
            "python": sys.version.splitlines()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "axes": {"order": ["t", "z", "x"], "units": {"t": "s", "z": "px", "x": "px"}},
        "acquisition": {"prf_hz": float(args.prf_hz), "nyquist_hz": float(0.5 * float(args.prf_hz))},
        "shape": {"T": T, "H": H, "W": W},
        "files": files,
    }
    meta_path = dataset_dir / "meta.json"
    _write_json(meta_path, meta)

    if args.write_bundle:
        tile_hw = (int(args.tile_hw[0]), int(args.tile_hw[1]))
        tile_stride = int(args.tile_stride)
        Lt_req = int(args.Lt)
        Lt = max(2, min(Lt_req, T - 1))
        if Lt != Lt_req:
            raise ValueError(f"Requested Lt={Lt_req} invalid for T={T}; choose 2 <= Lt < T.")
        write_acceptance_bundle_from_icube(
            out_root=bundle_root,
            dataset_name=str(args.dataset_name),
            Icube=Icube,
            prf_hz=float(args.prf_hz),
            tile_hw=tile_hw,
            tile_stride=tile_stride,
            Lt=Lt,
            diag_load=float(args.diag_load),
            cov_estimator=str(args.cov_estimator),
            score_mode=str(args.score_mode),
            baseline_type=str(args.baseline_type),
            svd_keep_min=int(args.svd_keep_min),
            svd_keep_max=int(args.svd_keep_max) if args.svd_keep_max is not None else None,
            band_ratio_flow_low_hz=float(args.flow_low_hz),
            band_ratio_flow_high_hz=float(args.flow_high_hz),
            band_ratio_alias_center_hz=float(args.alias_center_hz),
            band_ratio_alias_width_hz=float(args.alias_width_hz),
            mask_flow_override=mask_flow,
            mask_bg_override=mask_bg,
            stap_conditional_enable=False,
            stap_device=str(args.stap_device),
            run_stap=True,
            score_ka_v2_enable=False,
            meta_extra={"import": {"schema_version": "external_icube_import.v1", "config": cfg}},
        )

    print(f"[fieldii-import] wrote dataset: {dataset_dir}")
    if args.write_bundle:
        print(f"[fieldii-import] wrote bundle root: {bundle_root}")


if __name__ == "__main__":
    main()

