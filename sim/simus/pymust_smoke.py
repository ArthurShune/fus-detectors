from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


try:  # optional dependency
    import pymust  # type: ignore
except Exception:  # pragma: no cover - optional
    pymust = None  # type: ignore


@dataclass(frozen=True)
class SimusSmokeConfig:
    # Randomness
    seed: int = 0

    # Acquisition / slow-time
    prf_hz: float = 1500.0
    T: int = 8

    # Beamforming grid (output image)
    H: int = 24  # depth
    W: int = 24  # lateral
    x_min_m: float = -0.020
    x_max_m: float = 0.020
    z_min_m: float = 0.010
    z_max_m: float = 0.050

    # PyMUST probe + medium
    probe: str = "P4-2v"  # small element count for smoke
    c_mps: float = 1540.0
    fs_mult: float = 4.0  # must be >= 4 for SIMUS
    tilt_deg: float = 0.0  # plane-wave tilt

    # Scatterers
    tissue_count: int = 400
    blood_count: int = 200
    tissue_rc_scale: float = 1.0
    blood_rc_scale: float = 0.25

    # Simple in-plane vessel: vertical strip (axis ~ +z)
    vessel_center_x_m: float = 0.0
    vessel_radius_m: float = 0.004

    # Plug flow along +z (m/s)
    blood_vz_mps: float = 0.020

    # Deterministic reinjection for blood (new scatterers enter at inflow z)
    reservoir_scale: int = 4
    reinject_depth_span_m: float = 0.003  # depth band near z_min used for reinjection


def _require_pymust() -> Any:
    if pymust is None:  # pragma: no cover - optional
        raise ImportError(
            "PyMUST is not installed. Install into the stap-fus env via:\n"
            "  conda run -n stap-fus pip install PyMUST==0.1.9\n"
        )
    return pymust


def _vessel_mask_strip(*, X: np.ndarray, Z: np.ndarray, cx: float, r: float) -> np.ndarray:
    return (np.abs(X - float(cx)) <= float(r)) & np.isfinite(Z)


def _sample_uniform_excluding_mask(
    rng: np.random.Generator,
    *,
    n: int,
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
    exclude_mask_fn,
    max_tries: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Rejection-sample (x,z) points, excluding a region.

    exclude_mask_fn(x,z)->bool returns True for points to exclude.
    """
    xs = np.empty((n,), dtype=np.float32)
    zs = np.empty((n,), dtype=np.float32)
    filled = 0
    for _ in range(max_tries):
        need = n - filled
        if need <= 0:
            break
        x = rng.uniform(float(x_min), float(x_max), size=need).astype(np.float32)
        z = rng.uniform(float(z_min), float(z_max), size=need).astype(np.float32)
        bad = exclude_mask_fn(x, z)
        keep = ~bad
        k = int(np.sum(keep))
        if k <= 0:
            continue
        xs[filled : filled + k] = x[keep][:k]
        zs[filled : filled + k] = z[keep][:k]
        filled += k
    if filled < n:
        # Fall back: accept remaining samples (smoke only).
        need = n - filled
        xs[filled:] = rng.uniform(float(x_min), float(x_max), size=need).astype(np.float32)
        zs[filled:] = rng.uniform(float(z_min), float(z_max), size=need).astype(np.float32)
    return xs, zs


def generate_icube(cfg: SimusSmokeConfig) -> dict[str, Any]:
    pm = _require_pymust()
    if int(cfg.T) <= 0:
        raise ValueError("T must be positive")
    if float(cfg.prf_hz) <= 0.0:
        raise ValueError("prf_hz must be positive")
    if int(cfg.H) <= 0 or int(cfg.W) <= 0:
        raise ValueError("H and W must be positive")
    if float(cfg.z_min_m) <= 0.0:
        raise ValueError("z_min_m must be > 0")
    if float(cfg.z_max_m) <= float(cfg.z_min_m):
        raise ValueError("z_max_m must be > z_min_m")

    rng = np.random.default_rng(int(cfg.seed))
    dt_prp = 1.0 / float(cfg.prf_hz)

    # ---- PyMUST parameters ----
    param = pm.getparam(str(cfg.probe))
    # Medium / acquisition parameters used by SIMUS/DASMTX.
    param.c = float(cfg.c_mps)
    fs_mult = float(max(4.0, float(cfg.fs_mult)))
    param.fs = fs_mult * float(param.fc)

    tilt_rad = float(np.deg2rad(float(cfg.tilt_deg)))
    txdel = pm.txdelay(param, tilt_rad).astype(np.float32, copy=False)

    # ---- Output grid + masks ----
    xg = np.linspace(float(cfg.x_min_m), float(cfg.x_max_m), int(cfg.W), dtype=np.float32)
    zg = np.linspace(float(cfg.z_min_m), float(cfg.z_max_m), int(cfg.H), dtype=np.float32)
    X, Z = np.meshgrid(xg, zg)  # (H,W)

    mask_flow = _vessel_mask_strip(X=X, Z=Z, cx=float(cfg.vessel_center_x_m), r=float(cfg.vessel_radius_m))
    # Conservative background mask: avoid top rows (near transducer) and flow region.
    mask_bg = (~mask_flow).copy()
    mask_bg[: max(1, int(round(0.1 * int(cfg.H)))), :] = False
    if int(mask_bg.sum()) < 16:
        mask_bg = ~mask_flow

    fd_flow_hz = float(2.0 * float(param.fc) * float(cfg.blood_vz_mps) / max(float(param.c), 1e-9))
    expected_fd_hz = np.zeros((int(cfg.H), int(cfg.W)), dtype=np.float32)
    expected_fd_hz[mask_flow] = float(fd_flow_hz)
    mask_alias_expected = np.abs(expected_fd_hz) > (0.5 * float(cfg.prf_hz))

    # ---- Scatterers ----
    x_min, x_max = float(cfg.x_min_m), float(cfg.x_max_m)
    z_min, z_max = float(cfg.z_min_m), float(cfg.z_max_m)
    cx = float(cfg.vessel_center_x_m)
    r = float(cfg.vessel_radius_m)

    def _in_vessel(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.abs(x - cx) <= r

    # Tissue: uniform over ROI excluding the vessel strip.
    xs_tissue, zs_tissue = _sample_uniform_excluding_mask(
        rng,
        n=int(cfg.tissue_count),
        x_min=x_min,
        x_max=x_max,
        z_min=z_min,
        z_max=z_max,
        exclude_mask_fn=_in_vessel,
    )
    rc_tissue = (float(cfg.tissue_rc_scale) * rng.standard_normal(int(cfg.tissue_count))).astype(np.float32)

    # Blood: uniform in the vessel strip.
    xs_blood = rng.uniform(cx - r, cx + r, size=int(cfg.blood_count)).astype(np.float32)
    zs_blood = rng.uniform(z_min, z_max, size=int(cfg.blood_count)).astype(np.float32)
    rc_blood = (float(cfg.blood_rc_scale) * rng.standard_normal(int(cfg.blood_count))).astype(np.float32)
    xs_blood_init = xs_blood.copy()
    zs_blood_init = zs_blood.copy()

    # Deterministic reinjection reservoir (positions near inflow).
    res_scale = max(1, int(cfg.reservoir_scale))
    res_n = int(res_scale * int(cfg.blood_count))
    rng_res = np.random.default_rng(int(cfg.seed) + 1337)
    res_x = rng_res.uniform(cx - r, cx + r, size=res_n).astype(np.float32)
    span = float(max(1e-6, float(cfg.reinject_depth_span_m)))
    res_z = rng_res.uniform(z_min, min(z_min + span, z_max), size=res_n).astype(np.float32)
    res_ptr = 0

    # ---- Pulse loop: SIMUS -> RF -> IQch -> DAS -> IQimg ----
    Icube = np.empty((int(cfg.T), int(cfg.H), int(cfg.W)), dtype=np.complex64)
    M = None

    for t in range(int(cfg.T)):
        xs = np.concatenate([xs_tissue, xs_blood]).astype(np.float32, copy=False)
        zs = np.concatenate([zs_tissue, zs_blood]).astype(np.float32, copy=False)
        rc = np.concatenate([rc_tissue, rc_blood]).astype(np.float32, copy=False)

        rf, _rf_spec = pm.simus(xs, zs, rc, txdel, param)
        rf = np.asarray(rf, dtype=np.float32)
        iq_ch = pm.rf2iq(rf, param)

        if M is None:
            # Complex DAS matrix for I/Q data; SIG.shape convention is MATLAB-like.
            M = pm.dasmtx(1j * np.array(iq_ch.shape, dtype=np.int64), X, Z, txdel, param)

        img = (M @ iq_ch.flatten(order="F")).reshape(X.shape, order="F")
        Icube[t] = np.asarray(img, dtype=np.complex64)

        # Update blood scatterers (plug flow).
        zs_blood = (zs_blood + np.float32(float(cfg.blood_vz_mps) * dt_prp)).astype(np.float32, copy=False)

        # Reinjection when exiting the ROI depth range.
        out = zs_blood > np.float32(z_max)
        n_out = int(np.sum(out))
        if n_out > 0:
            idx = (np.arange(n_out, dtype=np.int64) + int(res_ptr)) % int(res_n)
            res_ptr = int((res_ptr + n_out) % int(res_n))
            xs_blood[out] = res_x[idx].astype(np.float32, copy=False)
            zs_blood[out] = res_z[idx].astype(np.float32, copy=False)

    # Debug artifacts: initial scatterers + reservoir (small enough for smoke).
    debug = {
        "expected_fd_hz": expected_fd_hz,
        "scatterers_init": {
            "xs_tissue": xs_tissue,
            "zs_tissue": zs_tissue,
            "rc_tissue": rc_tissue,
            "xs_blood0": xs_blood_init,
            "zs_blood0": zs_blood_init,
            "rc_blood": rc_blood,
            "res_x": res_x,
            "res_z": res_z,
        },
        "txdel_s": txdel,
        "grid_x_m": xg,
        "grid_z_m": zg,
        "fd_flow_hz": float(fd_flow_hz),
    }

    return {
        "Icube": Icube,
        "mask_flow": mask_flow.astype(bool, copy=False),
        "mask_bg": mask_bg.astype(bool, copy=False),
        "mask_alias_expected": mask_alias_expected.astype(bool, copy=False),
        "debug": debug,
        "param": {"probe": str(cfg.probe), "fc_hz": float(param.fc), "fs_hz": float(param.fs), "c_mps": float(param.c)},
    }


def dataset_meta(cfg: SimusSmokeConfig) -> dict[str, Any]:
    return dataclasses.asdict(cfg)
