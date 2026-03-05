from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from sim.kwave.icube_bundle import write_acceptance_bundle_from_icube
from sim.simus.pymust_smoke import SimusConfig, default_config, dataset_meta, generate_icube


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_info(repo_root: Path) -> dict[str, Any]:
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root))
            .decode("utf-8")
            .strip()
        )
        dirty = subprocess.call(
            ["git", "diff", "--quiet"],
            cwd=str(repo_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        dirty_cached = subprocess.call(
            ["git", "diff", "--quiet", "--cached"],
            cwd=str(repo_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        is_dirty = bool(dirty != 0 or dirty_cached != 0)
        return {"commit": commit, "dirty": is_dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="PyMUST/SIMUS generator: write canonical dataset + optional bundle."
    )
    ap.add_argument("--out", type=Path, required=True, help="Output run directory (created).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--preset",
        type=str,
        default="microvascular_like",
        choices=["microvascular_like", "alias_stress"],
        help="Named SIMUS regime (geometry + flow speed).",
    )
    ap.add_argument(
        "--tier",
        type=str,
        default="smoke",
        choices=["smoke", "paper"],
        help="Runtime tier: small fast config vs moderate paper-scale cross-check.",
    )

    # Optional overrides (rarely needed; keep defaults preset-driven).
    ap.add_argument("--probe", type=str, default=None)
    ap.add_argument("--prf-hz", type=float, default=None)
    ap.add_argument("--T", type=int, default=None)
    ap.add_argument("--H", type=int, default=None)
    ap.add_argument("--W", type=int, default=None)
    ap.add_argument("--x-min-m", type=float, default=None)
    ap.add_argument("--x-max-m", type=float, default=None)
    ap.add_argument("--z-min-m", type=float, default=None)
    ap.add_argument("--z-max-m", type=float, default=None)
    ap.add_argument("--blood-vmax-mps", type=float, default=None)
    ap.add_argument("--blood-profile", type=str, default=None, choices=["plug", "poiseuille"])
    ap.add_argument("--vessel-radius-m", type=float, default=None)
    ap.add_argument("--tissue-count", type=int, default=None)
    ap.add_argument("--blood-count", type=int, default=None)
    ap.add_argument("--skip-bundle", action="store_true", help="Only write dataset/, skip derived bundle/.")
    ap.add_argument("--dataset-name", type=str, default=None, help="Bundle dataset name (default derived).")
    return ap.parse_args()


def _sanitize_name(s: str) -> str:
    allowed = []
    for ch in str(s):
        if ch.isalnum() or ch in {"_", "-", "."}:
            allowed.append(ch)
        else:
            allowed.append("_")
    out = "".join(allowed).strip("_")
    return out or "run"


def main() -> None:
    args = parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = out_root / "dataset"
    debug_dir = dataset_dir / "debug"
    bundle_root = out_root / "bundle"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    bundle_root.mkdir(parents=True, exist_ok=True)

    cfg = default_config(
        preset=str(args.preset),  # type: ignore[arg-type]
        tier=str(args.tier),  # type: ignore[arg-type]
        seed=int(args.seed),
    )
    overrides: dict[str, object] = {}
    for k_cfg, k_arg in [
        ("probe", "probe"),
        ("prf_hz", "prf_hz"),
        ("T", "T"),
        ("H", "H"),
        ("W", "W"),
        ("x_min_m", "x_min_m"),
        ("x_max_m", "x_max_m"),
        ("z_min_m", "z_min_m"),
        ("z_max_m", "z_max_m"),
        ("blood_vmax_mps", "blood_vmax_mps"),
        ("blood_profile", "blood_profile"),
        ("vessel_radius_m", "vessel_radius_m"),
        ("tissue_count", "tissue_count"),
        ("blood_count", "blood_count"),
    ]:
        v = getattr(args, k_arg)
        if v is not None:
            overrides[k_cfg] = v
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)

    result = generate_icube(cfg)
    icube = result["Icube"]
    mask_flow = result["mask_flow"]
    mask_bg = result["mask_bg"]
    mask_alias = result["mask_alias_expected"]
    debug = result["debug"]
    param = result["param"]

    # ---- Write canonical dataset artifacts ----
    paths: Dict[str, Path] = {}

    def _save(dst_dir: Path, name: str, arr: np.ndarray) -> None:
        p = dst_dir / f"{name}.npy"
        np.save(p, arr, allow_pickle=False)
        paths[name] = p

    _save(dataset_dir, "icube", np.asarray(icube, dtype=np.complex64))
    _save(dataset_dir, "mask_flow", mask_flow.astype(bool))
    _save(dataset_dir, "mask_bg", mask_bg.astype(bool))
    _save(dataset_dir, "mask_alias_expected", mask_alias.astype(bool))

    _save(debug_dir, "expected_fd_hz", np.asarray(debug.get("expected_fd_hz"), dtype=np.float32))
    _save(debug_dir, "expected_vz_mps", np.asarray(debug.get("expected_vz_mps"), dtype=np.float32))
    np.savez_compressed(debug_dir / "scatterers_init.npz", **debug.get("scatterers_init", {}))
    paths["scatterers_init_npz"] = debug_dir / "scatterers_init.npz"
    _save(debug_dir, "txdel_s", np.asarray(debug.get("txdel_s"), dtype=np.float32))
    _save(debug_dir, "grid_x_m", np.asarray(debug.get("grid_x_m"), dtype=np.float32))
    _save(debug_dir, "grid_z_m", np.asarray(debug.get("grid_z_m"), dtype=np.float32))

    config_path = dataset_dir / "config.json"
    _write_json(config_path, dataset_meta(cfg))

    hashes: dict[str, Any] = {}
    for k, p in sorted(paths.items()):
        info = {"sha256": _sha256_file(p), "path": str(p.relative_to(out_root))}
        if k == "icube":
            info.update({"dtype": str(icube.dtype), "shape": list(icube.shape)})
        hashes[k] = info
    hashes["config"] = {"sha256": _sha256_file(config_path), "path": str(config_path.relative_to(out_root))}
    _write_json(dataset_dir / "hashes.json", hashes)

    repo_root = Path(__file__).resolve().parents[2]
    prov: dict[str, Any] = {
        "command": " ".join(sys.argv),
        "cwd": str(Path.cwd()),
        "git": _git_info(repo_root),
        "python": sys.version.splitlines()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import importlib.metadata as md  # py311

        prov["pymust"] = md.version("PyMUST")
    except Exception:
        prov["pymust"] = None

    meta: dict[str, Any] = {
        "schema_version": "simus_pymust.v1",
        "created_utc": _utc_now_iso(),
        "provenance": prov,
        "axes": {"order": ["t", "z", "x"], "units": {"x": "m", "z": "m", "t": "s"}},
        "grid": {
            "H": int(cfg.H),
            "W": int(cfg.W),
            "x_min_m": float(cfg.x_min_m),
            "x_max_m": float(cfg.x_max_m),
            "z_min_m": float(cfg.z_min_m),
            "z_max_m": float(cfg.z_max_m),
            "dx_m": float((float(cfg.x_max_m) - float(cfg.x_min_m)) / max(int(cfg.W) - 1, 1)),
            "dz_m": float((float(cfg.z_max_m) - float(cfg.z_min_m)) / max(int(cfg.H) - 1, 1)),
        },
        "acquisition": {
            "prf_hz": float(cfg.prf_hz),
            "dt_s": float(1.0 / cfg.prf_hz),
            "nyquist_hz": float(0.5 * cfg.prf_hz),
            "f0_hz": float(param.get("fc_hz", 0.0)),
            "fs_hz": float(param.get("fs_hz", 0.0)),
            "c0_mps": float(param.get("c_mps", 0.0)),
        },
        "simus": {"probe": str(cfg.probe), "preset": str(cfg.preset), "tier": str(cfg.tier)},
        "slow_time": {"T": int(cfg.T)},
        "config": dataset_meta(cfg),
        "files": hashes,
    }
    _write_json(dataset_dir / "meta.json", meta)

    if args.skip_bundle:
        print(f"[pilot_pymust_simus] wrote dataset -> {dataset_dir}", flush=True)
        return

    dataset_name_default = f"simus_{cfg.preset}_{cfg.tier}_{cfg.probe}_{cfg.T}T_seed{cfg.seed}"
    dataset_name = _sanitize_name(args.dataset_name or dataset_name_default)

    write_acceptance_bundle_from_icube(
        out_root=bundle_root,
        dataset_name=dataset_name,
        Icube=icube,
        prf_hz=float(cfg.prf_hz),
        seed=int(cfg.seed),
        tile_hw=(8, 8),
        tile_stride=3,
        Lt=int(min(8, max(2, int(cfg.T) - 1))),
        diag_load=1e-2,
        cov_estimator="tyler_pca",
        huber_c=5.0,
        mvdr_load_mode="auto",
        mvdr_auto_kappa=30.0,
        constraint_ridge=0.15,
        msd_lambda=0.05,
        msd_ridge=0.06,
        msd_agg_mode="median",
        msd_ratio_rho=0.05,
        motion_half_span_rel=0.25,
        msd_contrast_alpha=0.8,
        baseline_type="mc_svd",
        reg_enable=True,
        reg_method="phasecorr",
        reg_subpixel=4,
        reg_reference="median",
        svd_energy_frac=0.90,
        mask_flow_override=mask_flow.astype(bool),
        mask_bg_override=mask_bg.astype(bool),
        stap_conditional_enable=False,
        feasibility_mode="updated",
        meta_extra={
            "simus_pymust_dataset": True,
            "dataset_rel": str(dataset_dir.relative_to(out_root)),
        },
    )

    print(f"[pilot_pymust_simus] wrote dataset -> {dataset_dir}", flush=True)
    print(f"[pilot_pymust_simus] wrote bundle -> {bundle_root / dataset_name}", flush=True)


if __name__ == "__main__":
    main()
