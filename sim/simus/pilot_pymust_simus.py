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
from sim.simus.config import default_profile_config
from sim.simus.pymust_smoke import SimusConfig, default_config, dataset_meta, generate_icube

SUPPORTED_SIMUS_PROFILES = ("ClinIntraOp-Pf-v1", "ClinMobile-Pf-v1", "ClinIntraOp-Pf-Struct-v2", "ClinIntraOp-Pf-v2")


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
        "--profile",
        type=str,
        default=None,
        choices=list(SUPPORTED_SIMUS_PROFILES),
        help="Clinically aligned SIMUS profile. When set, this overrides --preset defaults.",
    )
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


def resolve_config_from_args(args: argparse.Namespace) -> SimusConfig:
    if args.profile:
        cfg = default_profile_config(
            profile=str(args.profile),  # type: ignore[arg-type]
            tier=str(args.tier),  # type: ignore[arg-type]
            seed=int(args.seed),
        )
    else:
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
    if args.profile:
        unsupported = [
            key
            for key in ("blood_vmax_mps", "blood_profile", "vessel_radius_m", "blood_count")
            if key in overrides
        ]
        if unsupported:
            names = ", ".join(f"--{name.replace('_', '-')}" for name in unsupported)
            raise ValueError(
                f"profile-driven runs manage per-vessel flow internally; unsupported override(s): {names}"
            )
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    return cfg


def write_simus_run(
    *,
    out_root: Path,
    cfg: SimusConfig,
    skip_bundle: bool = False,
    dataset_name: str | None = None,
) -> dict[str, Path]:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = out_root / "dataset"
    debug_dir = dataset_dir / "debug"
    bundle_root = out_root / "bundle"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    bundle_root.mkdir(parents=True, exist_ok=True)

    result = generate_icube(cfg)
    icube = result["Icube"]
    mask_flow = result["mask_flow"]
    mask_bg = result["mask_bg"]
    mask_alias = result["mask_alias_expected"]
    mask_microvascular = result.get("mask_microvascular")
    mask_nuisance_pa = result.get("mask_nuisance_pa")
    mask_guard = result.get("mask_guard")
    mask_h1_pf_main = result.get("mask_h1_pf_main")
    mask_h1_alias_qc = result.get("mask_h1_alias_qc")
    mask_h0_bg = result.get("mask_h0_bg")
    mask_h0_nuisance_pa = result.get("mask_h0_nuisance_pa")
    debug = result["debug"]
    param = result["param"]

    paths: Dict[str, Path] = {}

    def _save(dst_dir: Path, name: str, arr: np.ndarray) -> None:
        p = dst_dir / f"{name}.npy"
        np.save(p, arr, allow_pickle=False)
        paths[name] = p

    _save(dataset_dir, "icube", np.asarray(icube, dtype=np.complex64))
    _save(dataset_dir, "mask_flow", mask_flow.astype(bool))
    _save(dataset_dir, "mask_bg", mask_bg.astype(bool))
    _save(dataset_dir, "mask_alias_expected", mask_alias.astype(bool))
    if mask_microvascular is not None:
        _save(dataset_dir, "mask_microvascular", np.asarray(mask_microvascular, dtype=bool))
    if mask_nuisance_pa is not None:
        _save(dataset_dir, "mask_nuisance_pa", np.asarray(mask_nuisance_pa, dtype=bool))
    if mask_guard is not None:
        _save(dataset_dir, "mask_guard", np.asarray(mask_guard, dtype=bool))
    if mask_h1_pf_main is not None:
        _save(dataset_dir, "mask_h1_pf_main", np.asarray(mask_h1_pf_main, dtype=bool))
    if mask_h1_alias_qc is not None:
        _save(dataset_dir, "mask_h1_alias_qc", np.asarray(mask_h1_alias_qc, dtype=bool))
    if mask_h0_bg is not None:
        _save(dataset_dir, "mask_h0_bg", np.asarray(mask_h0_bg, dtype=bool))
    if mask_h0_nuisance_pa is not None:
        _save(dataset_dir, "mask_h0_nuisance_pa", np.asarray(mask_h0_nuisance_pa, dtype=bool))
    if result.get("mask_h0_specular_struct") is not None:
        _save(dataset_dir, "mask_h0_specular_struct", np.asarray(result.get("mask_h0_specular_struct"), dtype=bool))

    _save(debug_dir, "expected_fd_hz", np.asarray(debug.get("expected_fd_hz"), dtype=np.float32))
    if debug.get("expected_fd_true_hz") is not None:
        _save(debug_dir, "expected_fd_true_hz", np.asarray(debug.get("expected_fd_true_hz"), dtype=np.float32))
    if debug.get("expected_fd_sampled_hz") is not None:
        _save(debug_dir, "expected_fd_sampled_hz", np.asarray(debug.get("expected_fd_sampled_hz"), dtype=np.float32))
    _save(debug_dir, "expected_vz_mps", np.asarray(debug.get("expected_vz_mps"), dtype=np.float32))
    if debug.get("vessel_role_map") is not None:
        _save(debug_dir, "vessel_role_map", np.asarray(debug.get("vessel_role_map"), dtype=np.int16))
    if debug.get("motion_dx_px") is not None:
        _save(debug_dir, "motion_dx_px", np.asarray(debug.get("motion_dx_px"), dtype=np.float32))
    if debug.get("motion_dz_px") is not None:
        _save(debug_dir, "motion_dz_px", np.asarray(debug.get("motion_dz_px"), dtype=np.float32))
    if debug.get("motion_rigid_dx_px") is not None:
        _save(debug_dir, "motion_rigid_dx_px", np.asarray(debug.get("motion_rigid_dx_px"), dtype=np.float32))
    if debug.get("motion_rigid_dz_px") is not None:
        _save(debug_dir, "motion_rigid_dz_px", np.asarray(debug.get("motion_rigid_dz_px"), dtype=np.float32))
    if debug.get("motion_elastic_base_dx_px") is not None:
        _save(debug_dir, "motion_elastic_base_dx_px", np.asarray(debug.get("motion_elastic_base_dx_px"), dtype=np.float32))
    if debug.get("motion_elastic_base_dz_px") is not None:
        _save(debug_dir, "motion_elastic_base_dz_px", np.asarray(debug.get("motion_elastic_base_dz_px"), dtype=np.float32))
    if debug.get("motion_elastic_coef_x") is not None:
        _save(debug_dir, "motion_elastic_coef_x", np.asarray(debug.get("motion_elastic_coef_x"), dtype=np.float32))
    if debug.get("motion_elastic_coef_z") is not None:
        _save(debug_dir, "motion_elastic_coef_z", np.asarray(debug.get("motion_elastic_coef_z"), dtype=np.float32))
    if debug.get("phase_screen_rad") is not None:
        _save(debug_dir, "phase_screen_rad", np.asarray(debug.get("phase_screen_rad"), dtype=np.float32))
    if debug.get("mask_h0_specular_struct") is not None:
        _save(debug_dir, "mask_h0_specular_struct", np.asarray(debug.get("mask_h0_specular_struct"), dtype=bool))
    np.savez_compressed(debug_dir / "scatterers_init.npz", **debug.get("scatterers_init", {}))
    paths["scatterers_init_npz"] = debug_dir / "scatterers_init.npz"
    _save(debug_dir, "txdel_s", np.asarray(debug.get("txdel_s"), dtype=np.float32))
    _save(debug_dir, "grid_x_m", np.asarray(debug.get("grid_x_m"), dtype=np.float32))
    _save(debug_dir, "grid_z_m", np.asarray(debug.get("grid_z_m"), dtype=np.float32))
    scene_path = debug_dir / "scene_telemetry.json"
    _write_json(scene_path, dict(debug.get("scene_telemetry", {})))
    paths["scene_telemetry_json"] = scene_path

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
        import importlib.metadata as md

        prov["pymust"] = md.version("PyMUST")
    except Exception:
        prov["pymust"] = None

    meta: dict[str, Any] = {
        "schema_version": "simus_pymust.v3",
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
        "simus": {
            "probe": str(cfg.probe),
            "preset": str(cfg.preset),
            "profile": str(cfg.profile) if cfg.profile else None,
            "tier": str(cfg.tier),
        },
        "bands_hz": dataclasses.asdict(cfg.bands),
        "motion": {
            "config": dataclasses.asdict(cfg.motion),
            "telemetry": dict(debug.get("motion_telemetry", {})),
        },
        "phase_screen": {
            "config": dataclasses.asdict(cfg.phase_screen),
            "telemetry": dict(debug.get("phase_screen_telemetry", {})),
        },
        "noise": {
            "config": dataclasses.asdict(cfg.noise),
            "telemetry": dict(debug.get("noise_telemetry", {})),
        },
        "slow_time": {"T": int(cfg.T)},
        "labels": {
            "mask_h1_pf_main": "microvascular & sampled_fd in Pf_core & unaliased & ~guard",
            "mask_h1_alias_qc": "flow & (expected alias or sampled_fd in Pa) & ~guard",
            "mask_h0_bg": "background excluding flow and guard",
            "mask_h0_nuisance_pa": "nuisance_vessel & sampled_fd in Pa & ~guard",
            "mask_h0_specular_struct": "structured clutter / specular-support diagnostic negative",
        },
        "scene": dict(debug.get("scene_telemetry", {})),
        "config": dataset_meta(cfg),
        "files": hashes,
    }
    _write_json(dataset_dir / "meta.json", meta)

    bundle_dir = None
    if not skip_bundle:
        dataset_name_default = (
            f"simus_{_sanitize_name(cfg.profile)}_{cfg.tier}_seed{cfg.seed}"
            if cfg.profile
            else f"simus_{cfg.preset}_{cfg.tier}_{cfg.probe}_{cfg.T}T_seed{cfg.seed}"
        )
        dataset_slug = _sanitize_name(dataset_name or dataset_name_default)
        write_acceptance_bundle_from_icube(
            out_root=bundle_root,
            dataset_name=dataset_slug,
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
        bundle_dir = bundle_root / dataset_slug

    return {
        "run_root": out_root,
        "dataset_dir": dataset_dir,
        "bundle_root": bundle_root,
        "bundle_dir": bundle_dir,
    }


def main() -> None:
    args = parse_args()
    cfg = resolve_config_from_args(args)
    outputs = write_simus_run(
        out_root=Path(args.out),
        cfg=cfg,
        skip_bundle=bool(args.skip_bundle),
        dataset_name=args.dataset_name,
    )
    print(f"[pilot_pymust_simus] wrote dataset -> {outputs['dataset_dir']}", flush=True)
    if outputs["bundle_dir"] is not None:
        print(f"[pilot_pymust_simus] wrote bundle -> {outputs['bundle_dir']}", flush=True)


if __name__ == "__main__":
    main()
