from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates

from sim.simus.config import BackgroundCompartmentSpec, MotionSpec, OrdinaryBackgroundSpec, PhaseScreenSpec, SimusConfig


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


@dataclass(frozen=True)
class CoupledBackgroundArtifacts:
    dx_series_px: np.ndarray
    dz_series_px: np.ndarray
    structured_dx_px: np.ndarray
    structured_dz_px: np.ndarray
    telemetry: dict[str, Any]


def _normalized_random_walk(T: int, *, rng: np.random.Generator, step_sigma: float) -> np.ndarray:
    if T <= 0:
        return np.zeros((0,), dtype=np.float32)
    steps = rng.normal(scale=float(step_sigma), size=T).astype(np.float32)
    walk = np.cumsum(steps).astype(np.float32)
    walk -= float(np.mean(walk))
    # Keep the physical "px per pulse" scale encoded in `step_sigma`.
    return walk.astype(np.float32, copy=False)


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


def _series_from_cutoff(
    T: int,
    *,
    prf_hz: float,
    cutoff_hz: float,
    rms_amp: float,
    rng: np.random.Generator,
) -> np.ndarray:
    cutoff = float(max(cutoff_hz, 1e-3))
    rho = float(np.exp(-2.0 * np.pi * cutoff / max(float(prf_hz), 1e-6)))
    return (float(rms_amp) * _ar1_series(T, rho=rho, rng=rng)).astype(np.float32, copy=False)


def _delay_series(values: np.ndarray, delay: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    d = max(int(delay), 0)
    if d <= 0 or arr.size == 0:
        return arr.astype(np.float32, copy=False)
    out = np.empty_like(arr)
    out[:d] = arr[0]
    out[d:] = arr[:-d]
    return out.astype(np.float32, copy=False)


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
    if float(spec.pulse_jitter_sigma_px) > 0.0:
        dx_jitter = rng.normal(scale=float(spec.pulse_jitter_sigma_px), size=T).astype(np.float32)
        dz_jitter = rng.normal(scale=float(spec.pulse_jitter_sigma_px), size=T).astype(np.float32)
        dx_jitter -= np.float32(np.mean(dx_jitter))
        dz_jitter -= np.float32(np.mean(dz_jitter))
        dx_rigid += dx_jitter
        dz_rigid += dz_jitter

    if float(spec.elastic_amp_px) > 0.0:
        mode_count = max(int(spec.elastic_mode_count), 1)
        depth_decay = float(np.clip(spec.elastic_depth_decay_frac, 1e-3, 5.0))
        z = (np.arange(H, dtype=np.float32) + 0.5) / max(float(H), 1.0)
        depth_w = np.exp(-z / depth_decay).astype(np.float32)
        depth_w /= float(np.sqrt(np.mean(depth_w * depth_w)) + 1e-12)
        amp_scale = float(spec.elastic_amp_px) / float(np.sqrt(mode_count))
        dx_el = np.zeros((T, H, W), dtype=np.float32)
        dz_el = np.zeros((T, H, W), dtype=np.float32)
        base_dx = zeros_hw
        base_dz = zeros_hw
        coef_x = zeros_t
        coef_z = zeros_t
        for midx in range(mode_count):
            mode_dx = _smooth_unit_field(H, W, sigma_px=float(spec.elastic_sigma_px), rng=rng)
            mode_dz = _smooth_unit_field(H, W, sigma_px=float(spec.elastic_sigma_px), rng=rng)
            if mode_count > 1:
                x = np.linspace(-1.0, 1.0, W, dtype=np.float32)
                z_axis = np.linspace(0.0, 1.0, H, dtype=np.float32)
                x_center = np.float32(rng.uniform(-0.7, 0.7))
                z_center = np.float32(rng.uniform(0.15, 0.85))
                x_width = np.float32(rng.uniform(0.20, 0.45))
                z_width = np.float32(rng.uniform(0.18, 0.35))
                x_env = np.exp(-0.5 * ((x - x_center) / max(float(x_width), 1e-3)) ** 2).astype(np.float32)
                z_env = np.exp(-0.5 * ((z_axis - z_center) / max(float(z_width), 1e-3)) ** 2).astype(np.float32)
                region = (z_env[:, None] * x_env[None, :]).astype(np.float32, copy=False)
                region /= float(np.sqrt(np.mean(region * region)) + 1e-12)
                mode_dx = (mode_dx * region).astype(np.float32, copy=False)
                mode_dz = (mode_dz * region).astype(np.float32, copy=False)
            mode_dx = (float(spec.elastic_lateral_scale) * mode_dx * depth_w[:, None]).astype(np.float32, copy=False)
            mode_dz = (float(spec.elastic_axial_scale) * mode_dz * depth_w[:, None]).astype(np.float32, copy=False)
            mode_coef_x = (amp_scale * _ar1_series(T, rho=float(spec.elastic_temporal_rho), rng=rng)).astype(
                np.float32, copy=False
            )
            mode_coef_z = (amp_scale * _ar1_series(T, rho=float(spec.elastic_temporal_rho), rng=rng)).astype(
                np.float32, copy=False
            )
            dx_el += (mode_coef_x[:, None, None] * mode_dx[None, :, :]).astype(np.float32, copy=False)
            dz_el += (mode_coef_z[:, None, None] * mode_dz[None, :, :]).astype(np.float32, copy=False)
            if midx == 0:
                base_dx = mode_dx
                base_dz = mode_dz
                coef_x = mode_coef_x
                coef_z = mode_coef_z
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
        "pulse_jitter_sigma_px": float(spec.pulse_jitter_sigma_px),
        "elastic_amp_px": float(spec.elastic_amp_px),
        "elastic_sigma_px": float(spec.elastic_sigma_px),
        "elastic_depth_decay_frac": float(spec.elastic_depth_decay_frac),
        "elastic_temporal_rho": float(spec.elastic_temporal_rho),
        "elastic_mode_count": int(max(spec.elastic_mode_count, 1)),
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


def build_localized_motion_component(
    *,
    cfg: SimusConfig,
    center_x_m: float,
    center_z_m: float,
    sigma_x_m: float,
    sigma_z_m: float,
    amp_px: float,
    sigma_px: float,
    rho: float,
    jitter_sigma_px: float,
    lateral_scale: float,
    axial_scale: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    T = int(cfg.T)
    H = int(cfg.H)
    W = int(cfg.W)
    zeros = np.zeros((T, H, W), dtype=np.float32)
    if T <= 0 or H <= 0 or W <= 0 or float(amp_px) <= 0.0:
        return zeros, zeros, {"enabled": False, "disp_rms_px": 0.0, "disp_p90_px": 0.0}

    rng = np.random.default_rng(int(seed))
    base_dx = _smooth_unit_field(H, W, sigma_px=float(max(sigma_px, 0.5)), rng=rng)
    base_dz = _smooth_unit_field(H, W, sigma_px=float(max(sigma_px, 0.5)), rng=rng)

    x_axis = np.linspace(float(cfg.x_min_m), float(cfg.x_max_m), W, dtype=np.float32)
    z_axis = np.linspace(float(cfg.z_min_m), float(cfg.z_max_m), H, dtype=np.float32)
    xx, zz = np.meshgrid(x_axis, z_axis)
    env = np.exp(
        -0.5
        * (
            ((xx - np.float32(float(center_x_m))) / np.float32(max(float(sigma_x_m), 1e-6))) ** 2
            + ((zz - np.float32(float(center_z_m))) / np.float32(max(float(sigma_z_m), 1e-6))) ** 2
        )
    ).astype(np.float32, copy=False)
    env_rms = float(np.sqrt(np.mean(env * env))) + 1e-12
    env = (env / env_rms).astype(np.float32, copy=False)

    mode_dx = (float(lateral_scale) * base_dx * env).astype(np.float32, copy=False)
    mode_dz = (float(axial_scale) * base_dz * env).astype(np.float32, copy=False)

    coef_x = (float(amp_px) * _ar1_series(T, rho=float(rho), rng=rng)).astype(np.float32, copy=False)
    coef_z = (float(amp_px) * _ar1_series(T, rho=float(rho), rng=rng)).astype(np.float32, copy=False)
    dx = (coef_x[:, None, None] * mode_dx[None, :, :]).astype(np.float32, copy=False)
    dz = (coef_z[:, None, None] * mode_dz[None, :, :]).astype(np.float32, copy=False)

    if float(jitter_sigma_px) > 0.0:
        jitter_x = rng.normal(scale=float(jitter_sigma_px), size=T).astype(np.float32)
        jitter_z = rng.normal(scale=float(jitter_sigma_px), size=T).astype(np.float32)
        jitter_x -= np.float32(np.mean(jitter_x))
        jitter_z -= np.float32(np.mean(jitter_z))
        dx += (jitter_x[:, None, None] * env[None, :, :]).astype(np.float32, copy=False)
        dz += (jitter_z[:, None, None] * env[None, :, :]).astype(np.float32, copy=False)

    disp = np.sqrt(dx * dx + dz * dz)
    telemetry = {
        "enabled": True,
        "center_x_m": float(center_x_m),
        "center_z_m": float(center_z_m),
        "sigma_x_m": float(sigma_x_m),
        "sigma_z_m": float(sigma_z_m),
        "amp_px": float(amp_px),
        "sigma_px": float(sigma_px),
        "rho": float(rho),
        "jitter_sigma_px": float(jitter_sigma_px),
        "disp_rms_px": float(np.sqrt(np.mean(disp * disp))) if disp.size else 0.0,
        "disp_p90_px": float(np.quantile(disp, 0.90)) if disp.size else 0.0,
    }
    return dx, dz, telemetry


def build_coupled_background_artifacts(
    *,
    cfg: SimusConfig,
    compartments: tuple[BackgroundCompartmentSpec, ...],
    seed: int,
) -> CoupledBackgroundArtifacts:
    spec: OrdinaryBackgroundSpec = cfg.ordinary_background
    T = int(cfg.T)
    n_comp = len(tuple(compartments))
    zeros = np.zeros((n_comp, T), dtype=np.float32)
    zeros_t = np.zeros((T,), dtype=np.float32)
    if str(spec.mode) != "coupled_mass_shear_v2" or T <= 0 or n_comp <= 0:
        return CoupledBackgroundArtifacts(
            dx_series_px=zeros,
            dz_series_px=zeros,
            structured_dx_px=zeros_t,
            structured_dz_px=zeros_t,
            telemetry={"enabled": False, "mode": str(spec.mode)},
        )

    rng = np.random.default_rng(int(seed))
    g0 = _series_from_cutoff(
        T,
        prf_hz=float(cfg.prf_hz),
        cutoff_hz=float(spec.global_cutoff_hz),
        rms_amp=float(spec.global_disp_px),
        rng=rng,
    )
    s1n = _series_from_cutoff(
        T,
        prf_hz=float(cfg.prf_hz),
        cutoff_hz=float(spec.shear1_cutoff_hz),
        rms_amp=float(spec.shear1_disp_px),
        rng=rng,
    )
    s2n = _series_from_cutoff(
        T,
        prf_hz=float(cfg.prf_hz),
        cutoff_hz=float(spec.shear2_cutoff_hz),
        rms_amp=float(spec.shear2_disp_px),
        rng=rng,
    )
    rn = _series_from_cutoff(
        T,
        prf_hz=float(cfg.prf_hz),
        cutoff_hz=float(spec.residual_cutoff_hz),
        rms_amp=float(spec.residual_disp_px),
        rng=rng,
    )
    s1 = (0.70 * _delay_series(g0, 1) + s1n).astype(np.float32, copy=False)
    s2 = (0.60 * _delay_series(g0, 3) + s2n).astype(np.float32, copy=False)

    dx_series = np.zeros((n_comp, T), dtype=np.float32)
    dz_series = np.zeros((n_comp, T), dtype=np.float32)
    for idx, comp in enumerate(tuple(compartments)):
        name = str(comp.name).lower()
        if "left" in name:
            x_sign = -1.0
        elif "right" in name:
            x_sign = 1.0
        else:
            x_sign = 0.0
        if "deep" in name:
            depth_mix = float(spec.deep_dz_scale)
        elif "superficial" in name:
            depth_mix = 0.04
        else:
            depth_mix = 0.08
        sector_mix = float(spec.sector_mix_scale) * x_sign if ("mid" in name and bool(spec.sectorize_mid)) else 0.0
        dz = (
            float(spec.global_dz_scale) * g0
            + (0.35 + sector_mix) * float(spec.shear_dz_scale) * s1
            + depth_mix * s2
            + 0.35 * rn
        ).astype(np.float32, copy=False)
        dx = (
            float(spec.global_dx_scale) * g0
            + x_sign * float(spec.shear_dx_scale) * s1
            + sector_mix * s2
            + 0.20 * rn
        ).astype(np.float32, copy=False)
        dx_series[idx] = dx
        dz_series[idx] = dz

    structured_dx = (
        float(spec.structured_motion_scale) * float(spec.global_dx_scale) * g0
        + 0.25 * float(spec.shear_dx_scale) * s1
    ).astype(np.float32, copy=False)
    structured_dz = (
        float(spec.structured_motion_scale) * float(spec.global_dz_scale) * g0
        + 0.35 * float(spec.shear_dz_scale) * s1
        + 0.10 * s2
    ).astype(np.float32, copy=False)

    telemetry = {
        "enabled": True,
        "mode": str(spec.mode),
        "global_cutoff_hz": float(spec.global_cutoff_hz),
        "shear1_cutoff_hz": float(spec.shear1_cutoff_hz),
        "shear2_cutoff_hz": float(spec.shear2_cutoff_hz),
        "residual_cutoff_hz": float(spec.residual_cutoff_hz),
        "global_disp_rms_px": float(np.sqrt(np.mean(g0 * g0))) if g0.size else 0.0,
        "shear1_disp_rms_px": float(np.sqrt(np.mean(s1 * s1))) if s1.size else 0.0,
        "shear2_disp_rms_px": float(np.sqrt(np.mean(s2 * s2))) if s2.size else 0.0,
        "residual_disp_rms_px": float(np.sqrt(np.mean(rn * rn))) if rn.size else 0.0,
        "sectorize_mid": bool(spec.sectorize_mid),
    }
    return CoupledBackgroundArtifacts(
        dx_series_px=dx_series,
        dz_series_px=dz_series,
        structured_dx_px=structured_dx,
        structured_dz_px=structured_dz,
        telemetry=telemetry,
    )


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
