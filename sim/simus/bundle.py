from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from sim.kwave.icube_bundle import write_acceptance_bundle_from_icube

SUPPORTED_SIMUS_STAP_PROFILES = (
    "Brain-SIMUS-Clin",
    "Brain-SIMUS-Clin-MotionWide-v0",
    "Brain-SIMUS-Clin-MotionShort-v0",
    "Brain-SIMUS-Clin-MotionLong-v0",
    "Brain-SIMUS-Clin-MotionRobust-v0",
)


def slugify(text: str) -> str:
    s = re.sub(r"[\s/]+", "_", str(text).strip())
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s.strip("._-") or "run"


def load_canonical_run(run_dir: Path) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    ds = Path(run_dir) / "dataset"
    meta_path = ds / "meta.json"
    icube_path = ds / "icube.npy"
    if not meta_path.is_file() or not icube_path.is_file():
        raise FileNotFoundError(f"{run_dir}: missing dataset/meta.json or dataset/icube.npy")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    icube = np.load(icube_path).astype(np.complex64, copy=False)
    masks: dict[str, np.ndarray] = {}
    for name in (
        "mask_flow",
        "mask_bg",
        "mask_alias_expected",
        "mask_microvascular",
        "mask_nuisance_pa",
        "mask_guard",
        "mask_h1_pf_main",
        "mask_h1_alias_qc",
        "mask_h0_bg",
        "mask_h0_nuisance_pa",
    ):
        path = ds / f"{name}.npy"
        if path.is_file():
            masks[name] = np.load(path).astype(bool, copy=False)
    return icube, masks, meta


def _prf_from_meta(meta: dict[str, Any]) -> float:
    prf = float(meta.get("acquisition", {}).get("prf_hz", meta.get("config", {}).get("prf_hz", 0.0)))
    if prf <= 0.0:
        raise ValueError(f"dataset meta missing valid prf_hz: {prf}")
    return prf


def _seed_from_meta(meta: dict[str, Any]) -> int:
    return int(meta.get("config", {}).get("seed", meta.get("seed", 0)) or 0)


def bundle_profile_kwargs(profile: str, *, T: int, baseline_type: str) -> dict[str, Any]:
    name = str(profile).strip()
    if name not in SUPPORTED_SIMUS_STAP_PROFILES:
        raise ValueError(f"Unsupported STAP profile {profile!r}")
    base = {
        "tile_hw": (8, 8),
        "tile_stride": 3,
        "Lt": int(min(8, max(2, int(T) - 1))),
        "diag_load": 0.07,
        "stap_cov_train_trim_q": 0.0,
        "cov_estimator": "tyler_pca",
        "huber_c": 5.0,
        "mvdr_load_mode": "auto",
        "mvdr_auto_kappa": 120.0,
        "constraint_ridge": 0.18,
        "fd_span_mode": "fixed",
        "fd_fixed_span_hz": 250.0,
        "fd_span_rel": (0.30, 1.10),
        "constraint_mode": "exp+deriv",
        "grid_step_rel": 0.20,
        "fd_min_pts": 9,
        "fd_max_pts": 15,
        "msd_lambda": 0.05,
        "msd_ridge": 0.10,
        "msd_agg_mode": "median",
        "msd_ratio_rho": 0.05,
        "msd_contrast_alpha": 0.6,
        "motion_half_span_rel": 0.25,
        "baseline_type": str(baseline_type),
        "reg_enable": True,
        "reg_method": "phasecorr",
        "reg_subpixel": 4,
        "reg_reference": "median",
        "svd_energy_frac": 0.90,
        "stap_conditional_enable": False,
        "feasibility_mode": "updated",
    }
    if name == "Brain-SIMUS-Clin":
        return base
    if name == "Brain-SIMUS-Clin-MotionWide-v0":
        out = dict(base)
        out["motion_half_span_rel"] = 0.50
        return out
    if name == "Brain-SIMUS-Clin-MotionShort-v0":
        out = dict(base)
        out["Lt"] = int(min(6, max(2, int(T) - 1)))
        out["motion_half_span_rel"] = 0.50
        return out
    if name == "Brain-SIMUS-Clin-MotionLong-v0":
        out = dict(base)
        out["Lt"] = int(min(12, max(2, int(T) - 1)))
        out["motion_half_span_rel"] = 0.50
        return out
    if name == "Brain-SIMUS-Clin-MotionRobust-v0":
        out = dict(base)
        out.update(
            {
                "Lt": int(min(6, max(2, int(T) - 1))),
                "diag_load": 0.10,
                "stap_cov_train_trim_q": 0.05,
                "mvdr_auto_kappa": 200.0,
                "constraint_ridge": 0.25,
                "msd_lambda": 0.08,
                "motion_half_span_rel": 0.50,
                "msd_contrast_alpha": 0.8,
            }
        )
        return out
    raise ValueError(f"Unsupported STAP profile {profile!r}")


def derive_bundle_from_run(
    *,
    run_dir: Path,
    out_root: Path,
    dataset_name: str | None = None,
    stap_profile: str = "Brain-SIMUS-Clin",
    baseline_type: str = "mc_svd",
    run_stap: bool = True,
    stap_device: str | None = None,
    mask_flow_override: np.ndarray | None = None,
    mask_bg_override: np.ndarray | None = None,
    meta_extra: dict[str, Any] | None = None,
) -> Path:
    icube, masks, meta = load_canonical_run(run_dir)
    prf = _prf_from_meta(meta)
    seed = _seed_from_meta(meta)
    profile_kwargs = bundle_profile_kwargs(stap_profile, T=int(icube.shape[0]), baseline_type=baseline_type)
    dataset_slug = slugify(dataset_name or Path(run_dir).name)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    paths = write_acceptance_bundle_from_icube(
        out_root=out_root,
        dataset_name=dataset_slug,
        Icube=icube,
        prf_hz=prf,
        seed=seed,
        mask_flow_override=np.asarray(mask_flow_override if mask_flow_override is not None else masks.get("mask_flow"), dtype=bool),
        mask_bg_override=np.asarray(mask_bg_override if mask_bg_override is not None else masks.get("mask_bg"), dtype=bool),
        run_stap=bool(run_stap),
        stap_device=stap_device,
        meta_extra={
            "simus_bundle_from_dataset": True,
            "simus_stap_profile": str(stap_profile),
            "dataset_rel": "dataset",
            **(meta_extra or {}),
        },
        **profile_kwargs,
    )
    return Path(paths["meta"]).parent
