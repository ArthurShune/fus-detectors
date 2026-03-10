from __future__ import annotations

import csv
import dataclasses
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sim.simus.config import SimusConfig, default_profile_config
from sim.simus.pilot_pymust_simus import write_simus_run


@dataclass(frozen=True)
class FunctionalDesignSpec:
    ensemble_count: int = 16
    ensemble_dt_s: float = 0.75
    off_length: int = 4
    on_length: int = 4
    activation_gain: float = 0.18
    activation_vessel_names: tuple[str, ...] = ("micro_mid", "micro_right")
    kernel_tau_ensembles: float = 2.0
    kernel_len: int = 8


@dataclass(frozen=True)
class FunctionalEnsembleSpec:
    index: int
    scene_seed: int
    realization_seed: int
    activation_gain: float
    activation_vessel_names: tuple[str, ...]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_boxcar_design(*, ensemble_count: int, off_length: int, on_length: int) -> np.ndarray:
    if ensemble_count <= 0:
        raise ValueError("ensemble_count must be positive")
    if off_length <= 0 or on_length <= 0:
        raise ValueError("off_length and on_length must be positive")
    pattern = np.concatenate(
        [
            np.zeros((int(off_length),), dtype=np.float32),
            np.ones((int(on_length),), dtype=np.float32),
        ]
    )
    reps = int(np.ceil(float(ensemble_count) / float(pattern.size)))
    design = np.tile(pattern, reps)[: int(ensemble_count)].astype(np.float32, copy=False)
    return design


def build_hemo_regressor(
    design: np.ndarray,
    *,
    kernel_tau_ensembles: float = 2.0,
    kernel_len: int = 8,
) -> np.ndarray:
    x = np.asarray(design, dtype=np.float32).reshape(-1)
    t = np.arange(max(1, int(kernel_len)), dtype=np.float32)
    tau = max(float(kernel_tau_ensembles), 1e-3)
    kernel = (t * t) * np.exp(-t / np.float32(tau))
    if float(np.max(kernel)) <= 0.0:
        kernel = np.ones_like(t, dtype=np.float32)
    kernel = (kernel / np.max(kernel)).astype(np.float32, copy=False)
    reg = np.convolve(x, kernel, mode="full")[: x.size].astype(np.float32, copy=False)
    mx = float(np.max(reg))
    if mx > 0.0:
        reg = (reg / mx).astype(np.float32, copy=False)
    return reg


def default_functional_design(base_profile: str, *, ensemble_count: int = 16) -> FunctionalDesignSpec:
    if int(ensemble_count) >= 8:
        off_on = 4
    elif int(ensemble_count) >= 6:
        off_on = 2
    else:
        off_on = 1
    profile = str(base_profile)
    if profile == "ClinIntraOpParenchyma-Pf-v3":
        return FunctionalDesignSpec(
            ensemble_count=int(ensemble_count),
            ensemble_dt_s=0.75,
            off_length=off_on,
            on_length=off_on,
            activation_gain=0.16,
            activation_vessel_names=("micro_left", "micro_mid"),
            kernel_tau_ensembles=2.0,
            kernel_len=8,
        )
    if profile == "ClinMobile-Pf-v2":
        return FunctionalDesignSpec(
            ensemble_count=int(ensemble_count),
            ensemble_dt_s=0.75,
            off_length=off_on,
            on_length=off_on,
            activation_gain=0.20,
            activation_vessel_names=("micro_mid", "micro_right"),
            kernel_tau_ensembles=2.0,
            kernel_len=8,
        )
    raise ValueError(f"Unsupported functional base profile {base_profile!r}")


def build_functional_ensemble_specs(
    *,
    base_profile: str,
    seed: int,
    design_spec: FunctionalDesignSpec,
    null_run: bool,
) -> tuple[np.ndarray, np.ndarray, list[FunctionalEnsembleSpec]]:
    design = build_boxcar_design(
        ensemble_count=int(design_spec.ensemble_count),
        off_length=int(design_spec.off_length),
        on_length=int(design_spec.on_length),
    )
    hemo = build_hemo_regressor(
        design,
        kernel_tau_ensembles=float(design_spec.kernel_tau_ensembles),
        kernel_len=int(design_spec.kernel_len),
    )
    scene_seed = int(seed) * 10007 + 17
    out: list[FunctionalEnsembleSpec] = []
    for idx in range(int(design_spec.ensemble_count)):
        realization_seed = int(seed) * 10007 + 1000 + idx
        gain = 0.0 if bool(null_run) else float(design_spec.activation_gain) * float(hemo[idx])
        out.append(
            FunctionalEnsembleSpec(
                index=int(idx),
                scene_seed=scene_seed,
                realization_seed=realization_seed,
                activation_gain=float(gain),
                activation_vessel_names=tuple(design_spec.activation_vessel_names),
            )
        )
    return design, hemo, out


def _worker_write_ensemble(
    *,
    out_root: str,
    base_profile: str,
    tier: str,
    base_seed: int,
    ensemble_spec: FunctionalEnsembleSpec,
    threads_per_worker: int | None,
) -> dict[str, Any]:
    if threads_per_worker is not None:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = str(int(threads_per_worker))
    cfg = default_profile_config(profile=base_profile, tier=tier, seed=int(base_seed))  # type: ignore[arg-type]
    cfg = dataclasses.replace(
        cfg,
        seed=int(ensemble_spec.realization_seed),
        scene_seed=int(ensemble_spec.scene_seed),
        realization_seed=int(ensemble_spec.realization_seed),
        activation_vessel_names=tuple(ensemble_spec.activation_vessel_names),
        activation_rc_gain=float(ensemble_spec.activation_gain),
    )
    ensemble_dir = Path(out_root) / f"ensemble_{ensemble_spec.index:03d}"
    outputs = write_simus_run(out_root=ensemble_dir, cfg=cfg, skip_bundle=True)
    dataset_dir = Path(outputs["dataset_dir"])
    mask_activation_roi = dataset_dir / "mask_activation_roi.npy"
    mask_h0_bg = dataset_dir / "mask_h0_bg.npy"
    mask_h0_nuisance_pa = dataset_dir / "mask_h0_nuisance_pa.npy"
    mask_h0_specular_struct = dataset_dir / "mask_h0_specular_struct.npy"
    return {
        "ensemble_index": int(ensemble_spec.index),
        "ensemble_dir": str(ensemble_dir),
        "dataset_dir": str(dataset_dir),
        "scene_seed": int(ensemble_spec.scene_seed),
        "realization_seed": int(ensemble_spec.realization_seed),
        "activation_gain": float(ensemble_spec.activation_gain),
        "mask_activation_roi": str(mask_activation_roi) if mask_activation_roi.is_file() else None,
        "mask_h0_bg": str(mask_h0_bg) if mask_h0_bg.is_file() else None,
        "mask_h0_nuisance_pa": str(mask_h0_nuisance_pa) if mask_h0_nuisance_pa.is_file() else None,
        "mask_h0_specular_struct": str(mask_h0_specular_struct) if mask_h0_specular_struct.is_file() else None,
    }


def write_functional_case(
    *,
    out_root: Path,
    base_profile: str,
    tier: str,
    seed: int,
    null_run: bool,
    design_spec: FunctionalDesignSpec | None = None,
    max_workers: int = 1,
    threads_per_worker: int | None = None,
    reuse_existing: bool = True,
) -> dict[str, Any]:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    spec = design_spec or default_functional_design(base_profile, ensemble_count=16)
    design, hemo, ensemble_specs = build_functional_ensemble_specs(
        base_profile=base_profile,
        seed=int(seed),
        design_spec=spec,
        null_run=bool(null_run),
    )

    design_payload = {
        "schema_version": "simus_functional_case.v1",
        "base_profile": str(base_profile),
        "tier": str(tier),
        "seed": int(seed),
        "null_run": bool(null_run),
        "design_spec": dataclasses.asdict(spec),
        "scene_seed": int(ensemble_specs[0].scene_seed) if ensemble_specs else None,
    }
    _write_json(out_root / "design.json", design_payload)
    np.save(out_root / "task_regressor.npy", design.astype(np.float32, copy=False), allow_pickle=False)
    np.save(out_root / "hemo_regressor.npy", hemo.astype(np.float32, copy=False), allow_pickle=False)

    rows: list[dict[str, Any]] = []
    pending = [spec_i for spec_i in ensemble_specs]
    if bool(reuse_existing):
        reused: list[dict[str, Any]] = []
        missing: list[FunctionalEnsembleSpec] = []
        for spec_i in pending:
            ensemble_dir = out_root / f"ensemble_{spec_i.index:03d}"
            dataset_dir = ensemble_dir / "dataset"
            if (dataset_dir / "meta.json").is_file() and (dataset_dir / "icube.npy").is_file():
                reused.append(
                    {
                        "ensemble_index": int(spec_i.index),
                        "ensemble_dir": str(ensemble_dir),
                        "dataset_dir": str(dataset_dir),
                        "scene_seed": int(spec_i.scene_seed),
                        "realization_seed": int(spec_i.realization_seed),
                        "activation_gain": float(spec_i.activation_gain),
                        "mask_activation_roi": str(dataset_dir / "mask_activation_roi.npy"),
                        "mask_h0_bg": str(dataset_dir / "mask_h0_bg.npy"),
                        "mask_h0_nuisance_pa": str(dataset_dir / "mask_h0_nuisance_pa.npy"),
                        "mask_h0_specular_struct": str(dataset_dir / "mask_h0_specular_struct.npy"),
                    }
                )
            else:
                missing.append(spec_i)
        rows.extend(reused)
        pending = missing

    if pending:
        if int(max_workers) <= 1:
            for spec_i in pending:
                rows.append(
                    _worker_write_ensemble(
                        out_root=str(out_root),
                        base_profile=str(base_profile),
                        tier=str(tier),
                        base_seed=int(seed),
                        ensemble_spec=spec_i,
                        threads_per_worker=threads_per_worker,
                    )
                )
        else:
            with ProcessPoolExecutor(max_workers=int(max_workers)) as pool:
                futures = [
                    pool.submit(
                        _worker_write_ensemble,
                        out_root=str(out_root),
                        base_profile=str(base_profile),
                        tier=str(tier),
                        base_seed=int(seed),
                        ensemble_spec=spec_i,
                        threads_per_worker=threads_per_worker,
                    )
                    for spec_i in pending
                ]
                for fut in as_completed(futures):
                    rows.append(dict(fut.result()))

    rows = sorted(rows, key=lambda r: int(r["ensemble_index"]))
    _write_csv(out_root / "ensemble_table.csv", rows)
    _write_json(
        out_root / "ensemble_table.json",
        {
            "schema_version": "simus_functional_case_table.v1",
            "base_profile": str(base_profile),
            "tier": str(tier),
            "seed": int(seed),
            "null_run": bool(null_run),
            "rows": rows,
        },
    )

    if rows:
        first = rows[0]
        for src_key, dst_name in (
            ("mask_activation_roi", "mask_activation_roi.npy"),
            ("mask_h0_bg", "mask_h0_bg.npy"),
            ("mask_h0_nuisance_pa", "mask_h0_nuisance_pa.npy"),
            ("mask_h0_specular_struct", "mask_h0_specular_struct.npy"),
        ):
            src = first.get(src_key)
            if src and Path(src).is_file():
                dst = out_root / dst_name
                if not dst.exists():
                    arr = np.load(src).astype(bool, copy=False)
                    np.save(dst, arr, allow_pickle=False)

    return {
        "case_root": out_root,
        "design_json": out_root / "design.json",
        "task_regressor_npy": out_root / "task_regressor.npy",
        "hemo_regressor_npy": out_root / "hemo_regressor.npy",
        "ensemble_table_csv": out_root / "ensemble_table.csv",
        "ensemble_table_json": out_root / "ensemble_table.json",
        "rows": rows,
    }
