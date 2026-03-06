from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates

from sim.simus.config import MotionSpec, PhaseScreenSpec, SimusConfig


@dataclass(frozen=True)
class MotionArtifacts:
    dx_px: np.ndarray
    dz_px: np.ndarray
    rigid_dx_px: np.ndarray
    rigid_dz_px: np.ndarray
    elastic_base_dx_px: np.ndarray
    elastic_base_dz_px: np.ndarray
    elastic_coef_x: np.ndarray
    elastic_coef_z: np.ndarray
    telemetry: dict[str, Any]


def _normalized_random_walk(T: int, *, rng: np.random.Generator, step_sigma: float) -> np.ndarray:
    if T <= 0:
        return np.zeros((0,), dtype=np.float32)
    steps = rng.normal(scale=float(step_sigma), size=T).astype(np.float32)
    walk = np.cumsum(steps).astype(np.float32)
    walk -= float(np.mean(walk))
    rms = float(np.sqrt(np.mean(walk * walk))) + 1e-12
    return (walk / rms).astype(np.float32, copy=False)


def _ar1_series(T: int, *, rho: float, rng: np.random.Generator) -> np.ndarray:
    rho = float(np.clip(rho, 0.0, 0.9999))
    out = np.zeros((T,), dtype=np.float32)
    if T <= 0:
        return out
    out[0] = np.float32(rng.normal())
    noise_scale = float(np.sqrt(max(1.0 - rho * rho, 1e-6)))
    for t in range(1, T):
        out[t] = np.float32(rho * out[t - 1] + noise_scale * rng.normal())
    out -= np.float32(np.mean(out))
    rms = float(np.sqrt(np.mean(out * out))) + 1e-12
    return (out / rms).astype(np.float32, copy=False)


def _smooth_unit_field(H: int, W: int, *, sigma_px: float, rng: np.random.Generator) -> np.ndarray:
    field = rng.standard_normal((H, W)).astype(np.float32)
    field = gaussian_filter(field, sigma=float(max(sigma_px, 0.5)), mode="nearest")
    field -= float(np.mean(field))
    rms = float(np.sqrt(np.mean(field * field))) + 1e-12
    return (field / rms).astype(np.float32, copy=False)


def build_motion_artifacts(
    *,
    cfg: SimusConfig,
    seed: int,
) -> MotionArtifacts:
    T = int(cfg.T)
    H = int(cfg.H)
    W = int(cfg.W)
    spec = cfg.motion
    zeros_t = np.zeros((T,), dtype=np.float32)
    zeros_hw = np.zeros((H, W), dtype=np.float32)
    zeros_thw = np.zeros((T, H, W), dtype=np.float32)
    if not spec.enabled:
        return MotionArtifacts(
            dx_px=zeros_thw,
            dz_px=zeros_thw,
            rigid_dx_px=zeros_t,
            rigid_dz_px=zeros_t,
            elastic_base_dx_px=zeros_hw,
            elastic_base_dz_px=zeros_hw,
            elastic_coef_x=zeros_t,
            elastic_coef_z=zeros_t,
            telemetry={"enabled": False, "disp_rms_px": 0.0, "disp_p90_px": 0.0},
        )

    rng = np.random.default_rng(int(seed))
    tt = np.arange(T, dtype=np.float32) / max(float(cfg.prf_hz), 1e-9)

    dx_rigid = (
        float(spec.breathing_amp_x_px) * np.sin(2.0 * np.pi * float(spec.breathing_hz) * tt + 0.3)
        + float(spec.cardiac_amp_x_px) * np.sin(2.0 * np.pi * float(spec.cardiac_hz) * tt + 1.1)
        + np.linspace(0.0, float(spec.drift_x_px), T, dtype=np.float32)
    ).astype(np.float32, copy=False)
    dz_rigid = (
        float(spec.breathing_amp_z_px) * np.sin(2.0 * np.pi * float(spec.breathing_hz) * tt + 0.8)
        + float(spec.cardiac_amp_z_px) * np.sin(2.0 * np.pi * float(spec.cardiac_hz) * tt + 0.2)
        + np.linspace(0.0, float(spec.drift_z_px), T, dtype=np.float32)
    ).astype(np.float32, copy=False)

    if float(spec.random_walk_sigma_px) > 0.0:
        dx_rigid += _normalized_random_walk(T, rng=rng, step_sigma=float(spec.random_walk_sigma_px))
        dz_rigid += _normalized_random_walk(T, rng=rng, step_sigma=float(spec.random_walk_sigma_px))

    if float(spec.elastic_amp_px) > 0.0:
        base_dx = _smooth_unit_field(H, W, sigma_px=float(spec.elastic_sigma_px), rng=rng)
        base_dz = _smooth_unit_field(H, W, sigma_px=float(spec.elastic_sigma_px), rng=rng)
        depth_decay = float(np.clip(spec.elastic_depth_decay_frac, 1e-3, 5.0))
        z = (np.arange(H, dtype=np.float32) + 0.5) / max(float(H), 1.0)
        depth_w = np.exp(-z / depth_decay).astype(np.float32)
        depth_w /= float(np.sqrt(np.mean(depth_w * depth_w)) + 1e-12)
        base_dx = (float(spec.elastic_lateral_scale) * base_dx * depth_w[:, None]).astype(np.float32, copy=False)
        base_dz = (float(spec.elastic_axial_scale) * base_dz * depth_w[:, None]).astype(np.float32, copy=False)
        coef_x = (float(spec.elastic_amp_px) * _ar1_series(T, rho=float(spec.elastic_temporal_rho), rng=rng)).astype(
            np.float32, copy=False
        )
        coef_z = (float(spec.elastic_amp_px) * _ar1_series(T, rho=float(spec.elastic_temporal_rho), rng=rng)).astype(
            np.float32, copy=False
        )
        dx_el = (coef_x[:, None, None] * base_dx[None, :, :]).astype(np.float32, copy=False)
        dz_el = (coef_z[:, None, None] * base_dz[None, :, :]).astype(np.float32, copy=False)
    else:
        base_dx = zeros_hw
        base_dz = zeros_hw
        coef_x = zeros_t
        coef_z = zeros_t
        dx_el = zeros_thw
        dz_el = zeros_thw

    dx = (dx_el + dx_rigid[:, None, None]).astype(np.float32, copy=False)
    dz = (dz_el + dz_rigid[:, None, None]).astype(np.float32, copy=False)
    disp = np.sqrt(dx * dx + dz * dz)
    rigid_disp = np.sqrt(dx_rigid * dx_rigid + dz_rigid * dz_rigid)

    tele = {
        "enabled": True,
        "rigid_rms_px": float(np.sqrt(np.mean(rigid_disp * rigid_disp))) if rigid_disp.size else 0.0,
        "rigid_p90_px": float(np.quantile(rigid_disp, 0.90)) if rigid_disp.size else 0.0,
        "elastic_amp_px": float(spec.elastic_amp_px),
        "elastic_sigma_px": float(spec.elastic_sigma_px),
        "elastic_depth_decay_frac": float(spec.elastic_depth_decay_frac),
        "elastic_temporal_rho": float(spec.elastic_temporal_rho),
        "disp_rms_px": float(np.sqrt(np.mean(disp * disp))) if disp.size else 0.0,
        "disp_p90_px": float(np.quantile(disp, 0.90)) if disp.size else 0.0,
    }
    return MotionArtifacts(
        dx_px=dx,
        dz_px=dz,
        rigid_dx_px=dx_rigid.astype(np.float32, copy=False),
        rigid_dz_px=dz_rigid.astype(np.float32, copy=False),
        elastic_base_dx_px=base_dx.astype(np.float32, copy=False),
        elastic_base_dz_px=base_dz.astype(np.float32, copy=False),
        elastic_coef_x=coef_x.astype(np.float32, copy=False),
        elastic_coef_z=coef_z.astype(np.float32, copy=False),
        telemetry=tele,
    )


def sample_motion_displacements_m(
    *,
    field_dx_px: np.ndarray,
    field_dz_px: np.ndarray,
    x_m: np.ndarray,
    z_m: np.ndarray,
    cfg: SimusConfig,
) -> tuple[np.ndarray, np.ndarray]:
    dx_m = float((float(cfg.x_max_m) - float(cfg.x_min_m)) / max(int(cfg.W) - 1, 1))
    dz_m = float((float(cfg.z_max_m) - float(cfg.z_min_m)) / max(int(cfg.H) - 1, 1))
    x_idx = (np.asarray(x_m, dtype=np.float32) - float(cfg.x_min_m)) / max(dx_m, 1e-9)
    z_idx = (np.asarray(z_m, dtype=np.float32) - float(cfg.z_min_m)) / max(dz_m, 1e-9)
    coords = np.vstack([z_idx, x_idx]).astype(np.float32, copy=False)
    dx_px = map_coordinates(
        np.asarray(field_dx_px, dtype=np.float32),
        coords,
        order=1,
        mode="nearest",
        prefilter=False,
    )
    dz_px = map_coordinates(
        np.asarray(field_dz_px, dtype=np.float32),
        coords,
        order=1,
        mode="nearest",
        prefilter=False,
    )
    return dx_px.astype(np.float32, copy=False) * np.float32(dx_m), dz_px.astype(np.float32, copy=False) * np.float32(dz_m)


def build_phase_screen_series(
    *,
    T: int,
    n_elem: int,
    spec: PhaseScreenSpec,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if (not spec.enabled) or float(spec.std_rad) <= 0.0 or n_elem <= 0:
        return np.zeros((T, max(n_elem, 0)), dtype=np.float32), {"enabled": False, "phase_rms_rad": 0.0}

    rng = np.random.default_rng(int(seed))

    def _smooth_phase(scale: float) -> np.ndarray:
        row = rng.standard_normal(n_elem).astype(np.float32)
        row = gaussian_filter1d(row, sigma=float(max(spec.corr_len_elem, 1.0) / 2.355), mode="nearest")
        row -= float(np.mean(row))
        rms = float(np.sqrt(np.mean(row * row))) + 1e-12
        return (float(scale) * row / rms).astype(np.float32, copy=False)

    phase = np.zeros((T, n_elem), dtype=np.float32)
    phase[0] = _smooth_phase(float(spec.std_rad))
    for t in range(1, T):
        innovation = _smooth_phase(float(spec.drift_sigma_rad)) if float(spec.drift_sigma_rad) > 0.0 else 0.0
        phase[t] = (float(spec.drift_rho) * phase[t - 1] + innovation).astype(np.float32, copy=False)

    tele = {
        "enabled": True,
        "std_rad": float(spec.std_rad),
        "corr_len_elem": float(spec.corr_len_elem),
        "drift_rho": float(spec.drift_rho),
        "drift_sigma_rad": float(spec.drift_sigma_rad),
        "phase_rms_rad": float(np.sqrt(np.mean(phase * phase))) if phase.size else 0.0,
    }
    return phase.astype(np.float32, copy=False), tele


def apply_phase_screen(iq_ch: np.ndarray, phase_rad: np.ndarray) -> np.ndarray:
    arr = np.asarray(iq_ch)
    phase = np.asarray(phase_rad, dtype=np.float32)
    if arr.ndim != 2 or phase.size == 0:
        return arr
    weights = np.exp(1j * phase).astype(np.complex64, copy=False)
    if arr.shape[1] == phase.size:
        return (arr.astype(np.complex64, copy=False) * weights[None, :]).astype(np.complex64, copy=False)
    if arr.shape[0] == phase.size:
        return (weights[:, None] * arr.astype(np.complex64, copy=False)).astype(np.complex64, copy=False)
    return arr
