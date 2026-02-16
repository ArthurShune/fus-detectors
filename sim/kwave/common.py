# sim/kwave/common.py
from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Literal, Optional, Sequence

import numpy as np
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.signals import tone_burst
from scipy.ndimage import binary_dilation, binary_erosion, convolve1d
from scipy.signal import hilbert
from scipy.signal.windows import dpss

try:
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore

try:
    from pipeline.stap.geometry import principal_angle, projected_flow_alignment
    from pipeline.stap.lt_auto import choose_lt_from_coherence
    from pipeline.stap.mvdr_bank import choose_fd_grid_auto
    from pipeline.stap.ka_contract_v2 import (
        derive_score_shrink_v2_tile_scales,
        derive_score_shrink_v2_tile_scales_forced,
        evaluate_ka_contract_v2,
    )
    from pipeline.stap.temporal import (
        _generalized_band_metrics,
        _mixing_metric,
        aggregate_over_snapshots,
        band_energy_on_whitened,
        bandpass_constraints_temporal,
        build_motion_basis_temporal,
        build_temporal_hankels_and_cov,
        ka_blend_covariance_temporal,
        ka_prior_temporal_from_psd,
        lcmv_temporal_apply_batched,
        msd_contrast_score_batched,
        pd_temporal_core_batched,
        project_out_motion_whitened,
        projector_from_tones,
        split_fd_grid_by_motion,
        stap_temporal_core_batched,
    )
    from pipeline.stap.temporal_shared import (
        build_temporal_hankels_batch,
        conditioned_lambda_batch,
        robust_temporal_cov_batch,
        shrinkage_alpha_for_kappa_batch,
    )
except ModuleNotFoundError:
    build_temporal_hankels_and_cov = None  # type: ignore[assignment]
    bandpass_constraints_temporal = None  # type: ignore[assignment]
    lcmv_temporal_apply_batched = None  # type: ignore[assignment]
    aggregate_over_snapshots = None  # type: ignore[assignment]
    build_motion_basis_temporal = None  # type: ignore[assignment]
    project_out_motion_whitened = None  # type: ignore[assignment]
    split_fd_grid_by_motion = None  # type: ignore[assignment]
    band_energy_on_whitened = None  # type: ignore[assignment]
    msd_contrast_score_batched = None  # type: ignore[assignment]
    ka_blend_covariance_temporal = None  # type: ignore[assignment]
    ka_prior_temporal_from_psd = None  # type: ignore[assignment]
    projector_from_tones = None  # type: ignore[assignment]
    stap_temporal_core_batched = None  # type: ignore[assignment]
    pd_temporal_core_batched = None  # type: ignore[assignment]
    choose_lt_from_coherence = None  # type: ignore[assignment]
    choose_fd_grid_auto = None  # type: ignore[assignment]
    evaluate_ka_contract_v2 = None  # type: ignore[assignment]
    derive_score_shrink_v2_tile_scales = None  # type: ignore[assignment]
    derive_score_shrink_v2_tile_scales_forced = None  # type: ignore[assignment]
    principal_angle = None  # type: ignore[assignment]
    projected_flow_alignment = None  # type: ignore[assignment]
    build_temporal_hankels_batch = None  # type: ignore[assignment]
    robust_temporal_cov_batch = None  # type: ignore[assignment]
    shrinkage_alpha_for_kappa_batch = None  # type: ignore[assignment]
    conditioned_lambda_batch = None  # type: ignore[assignment]
    _STAP_AVAILABLE = False
else:
    _STAP_AVAILABLE = torch is not None

_HAS_TORCH = torch is not None

FeasibilityMode = Literal["legacy", "updated", "blend"]
_FEASIBILITY_MODES: set[str] = {"legacy", "updated", "blend"}
_DEFAULT_MOTION_HALF_SPAN_REL = 0.1


def _normalize_feasibility_mode(mode: str | None) -> FeasibilityMode:
    """Normalize feasibility mode string and validate supported values."""
    if mode is None:
        return "legacy"
    mode_clean = str(mode).strip().lower()
    if mode_clean not in _FEASIBILITY_MODES:
        raise ValueError(
            f"Unsupported feasibility_mode '{mode}'. Expected one of {_FEASIBILITY_MODES}."
        )
    return mode_clean  # type: ignore[return-value]


_GPU_AVAILABLE_CACHE: bool | None = None


def _conditioned_lambda(
    R_t: torch.Tensor,
    lam_init: float,
    kappa_target: float,
    eps: float = 1e-8,
) -> tuple[float, float, float, float]:
    """Return λ ≥ lam_init ensuring κ(R_t + λI) ≤ κ_target (approximate).

    Returns (lambda_final, sigma_min, sigma_max, lambda_needed).
    """
    if kappa_target <= 1.0:
        kappa_target = 1.0 + eps
    evals = torch.linalg.eigvalsh(R_t).real.clamp_min(0.0)
    sigma_max = float(torch.max(evals).item())
    sigma_min = float(torch.min(evals).item())
    denom = max(kappa_target - 1.0, eps)
    if sigma_min <= 0.0:
        lambda_needed = sigma_max / denom if sigma_max > 0.0 else 0.0
    else:
        lambda_needed = max(0.0, (sigma_max - kappa_target * sigma_min) / denom)
    lam_final = float(max(lam_init, lambda_needed))
    return lam_final, sigma_min, sigma_max, lambda_needed


def _median_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    return float(np.median(arr))


def _operator_metric_stats(tile_infos: Sequence[dict]) -> dict[str, float | None]:
    def _collect(field: str) -> list[float]:
        vals: list[float] = []
        for info in tile_infos:
            val = info.get(field)
            if val is None:
                continue
            try:
                val_f = float(val)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(val_f):
                continue
            vals.append(val_f)
        return vals

    def _summary(vals: list[float]) -> tuple[float | None, float | None, float | None]:
        if not vals:
            return None, None, None
        arr = np.asarray(vals, dtype=float)
        med = float(np.median(arr))
        if arr.size >= 2:
            p10 = float(np.quantile(arr, 0.10))
            p90 = float(np.quantile(arr, 0.90))
        else:
            p10 = med
            p90 = med
        return med, p10, p90

    stats: dict[str, float | None] = {}
    metric_fields = (
        "ka_pf_lambda_min",
        "ka_pf_lambda_max",
        "ka_pf_lambda_mean",
        "ka_perp_lambda_min",
        "ka_perp_lambda_max",
        "ka_perp_lambda_mean",
        "ka_alias_lambda_min",
        "ka_alias_lambda_max",
        "ka_alias_lambda_mean",
        "ka_noise_lambda_min",
        "ka_noise_lambda_max",
        "ka_noise_lambda_mean",
        "ka_score_scale_ratio",
        "sample_pf_lambda_min",
        "sample_pf_lambda_max",
        "sample_pf_lambda_mean",
        "sample_perp_lambda_min",
        "sample_perp_lambda_max",
        "sample_perp_lambda_mean",
        "sample_alias_lambda_min",
        "sample_alias_lambda_max",
        "sample_alias_lambda_mean",
        "sample_noise_lambda_min",
        "sample_noise_lambda_max",
        "sample_noise_lambda_mean",
    )
    for key in metric_fields:
        med, p10, p90 = _summary(_collect(key))
        stats[key] = med
        stats[f"{key}_p10"] = p10
        stats[f"{key}_p90"] = p90

    mix_vals = _collect("ka_operator_mixing_epsilon")
    mix_med, mix_p10, mix_p90 = _summary(mix_vals)
    stats["ka_operator_mixing_epsilon"] = mix_med
    stats["ka_operator_mixing_epsilon_p10"] = mix_p10
    stats["ka_operator_mixing_epsilon_p90"] = mix_p90

    sample_noise_vals = _collect("sample_po_noise_floor")
    sample_noise_med, sample_noise_p10, sample_noise_p90 = _summary(sample_noise_vals)
    stats["sample_po_noise_floor"] = sample_noise_med
    stats["sample_po_noise_floor_p10"] = sample_noise_p10
    stats["sample_po_noise_floor_p90"] = sample_noise_p90

    prior_noise_vals = _collect("prior_po_noise_floor")
    prior_noise_med, prior_noise_p10, prior_noise_p90 = _summary(prior_noise_vals)
    stats["prior_po_noise_floor"] = prior_noise_med
    stats["prior_po_noise_floor_p10"] = prior_noise_p10
    stats["prior_po_noise_floor_p90"] = prior_noise_p90
    return stats


def _adjust_alias_band_for_meta(
    band_ratio_spec: dict[str, float], alias_meta: dict[str, float | int] | None
) -> dict[str, float]:
    if not alias_meta:
        return band_ratio_spec
    center = alias_meta.get("flow_alias_hz")
    if center is None:
        return band_ratio_spec
    width = float(band_ratio_spec.get("alias_width_hz", 0.0))
    jitter = float(alias_meta.get("flow_alias_jitter_hz") or 0.0)
    pad = 5.0
    width = max(width, jitter + pad)
    adjusted = dict(band_ratio_spec)
    adjusted["alias_center_hz"] = float(center)
    adjusted["alias_width_hz"] = float(width)
    return adjusted


def _auto_band_ratio_spec(
    prf_hz: float,
    Lt: int,
    *,
    flow_alias_hz: float | None = None,
    flow_alias_fraction: float | None = None,
) -> dict[str, float]:
    """Design flow/alias bands based on PRF/Lt and optional alias hints."""
    prf = max(float(prf_hz), 1.0)
    Lt_eff = max(int(Lt), 1)
    fundamental = prf / Lt_eff
    nyquist = 0.5 * prf

    # For Lt >= 8, build bands around discrete Doppler bins with a guard.
    if Lt_eff >= 8:
        bin_hz = fundamental
        # Clinically grounded bands (PRF ~1–5 kHz): flow in tens–low hundreds of Hz,
        # guard between flow and high-frequency motion/alias, alias near high bins.
        flow_low = 30.0
        flow_high_target = 250.0
        flow_high = min(flow_high_target, 0.35 * nyquist)
        flow_high = max(flow_high, flow_low + bin_hz)  # ensure at least one bin wide
        guard_hz = max(150.0, bin_hz)  # ~150 Hz guard between flow and alias
        alias_low_target = max(400.0, flow_high + guard_hz)
        alias_low = min(alias_low_target, nyquist - 0.5 * bin_hz)
        alias_high = nyquist
        alias_center = 0.5 * (alias_low + alias_high)
        alias_width = 0.5 * (alias_high - alias_low)  # store half-width
        alias_min_bins = 2
        alias_max_bins = 4
        return {
            "flow_low_hz": float(flow_low),
            "flow_high_hz": float(flow_high),
            "alias_center_hz": float(alias_center),
            "alias_width_hz": float(alias_width),
            "alias_min_bins": alias_min_bins,
            "alias_max_bins": alias_max_bins,
        }

    # Lt < 8: fall back to Hz-based heuristic with loose guard (legacy path)
    guard = max(0.05 * fundamental, 25.0)
    flow_low = max(0.0, 0.25 * fundamental)
    flow_high_candidate = min(fundamental * 1.15, nyquist - 3.0 * guard)
    flow_high = max(flow_high_candidate, flow_low + guard)
    alias_candidates: list[float] = []
    if flow_alias_hz is not None and flow_alias_hz > 0.0:
        alias_candidates.append(float(flow_alias_hz))
    if flow_alias_fraction is not None and flow_alias_fraction > 0.0:
        alias_candidates.append(float(flow_alias_fraction) * nyquist)
    if not alias_candidates:
        alias_candidates.append(1.6 * fundamental)
    alias_candidates = [val for val in alias_candidates if val > 0.0]
    alias_candidates.sort(reverse=True)
    alias_center = None
    for candidate in alias_candidates:
        if candidate <= nyquist - guard:
            alias_center = candidate
            break
    if alias_center is None:
        alias_center = min(nyquist - guard, alias_candidates[0])
    alias_center = max(flow_high + guard, alias_center)
    alias_width = max(0.2 * fundamental, 2.0 * guard)
    alias_low = alias_center - 0.5 * alias_width
    alias_high = alias_center + 0.5 * alias_width
    if alias_high > nyquist:
        shift = alias_high - nyquist
        alias_high -= shift
        alias_low -= shift
    gap = flow_high + guard
    if alias_low <= gap:
        alias_low = gap
        alias_high = min(nyquist - guard * 0.5, alias_low + alias_width)
    alias_center = 0.5 * (alias_low + alias_high)
    alias_width = max(1.0, alias_high - alias_low)
    return {
        "flow_low_hz": float(flow_low),
        "flow_high_hz": float(min(flow_high, nyquist - guard)),
        "alias_center_hz": float(alias_center),
        "alias_width_hz": float(alias_width),
    }


def _multi_taper_psd(
    series: np.ndarray,
    prf_hz: float,
    *,
    tapers: int = 3,
    bandwidth: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate PSD via multi-taper (DPSS) averaging."""
    x = np.asarray(series, dtype=np.complex64).reshape(-1)
    if x.size == 0:
        raise ValueError("Empty series passed to _multi_taper_psd")
    tapers = max(1, int(tapers))
    bandwidth = max(float(bandwidth), 1.0)
    dpss_tapers = dpss(x.size, bandwidth, Kmax=tapers, sym=False)
    psd_accum = None
    for taper in dpss_tapers:
        tapered = x * taper.astype(np.float32, copy=False)
        spec = np.fft.fft(tapered)
        power = np.abs(spec) ** 2
        if psd_accum is None:
            psd_accum = power
        else:
            psd_accum += power
    psd_mt_full = psd_accum / float(len(dpss_tapers))
    freqs_full = np.fft.fftfreq(x.size, d=1.0 / float(prf_hz))
    nyquist_freq = 0.5 * float(prf_hz)
    pos_mask = (freqs_full >= 0.0) | np.isclose(freqs_full, -nyquist_freq, atol=1e-6)
    freqs = np.abs(freqs_full[pos_mask])
    psd_mt = psd_mt_full[pos_mask]
    return freqs.astype(np.float32, copy=False), psd_mt.astype(np.float32, copy=False)


def _inject_vessels_slowtime(
    cube: np.ndarray,
    micro_vessels: np.ndarray | None,
    alias_vessels: np.ndarray | None,
    *,
    prf_hz: float,
    f0_hz: float,
    c0: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Inject simple slow-time vessel modulations on a (T,H,W) complex cube.

    This helper is intended for MaceBridge-style replay runs where
    microvascular and alias vessel fields have been precomputed on
    the Macé-aligned grid. Each vessel contributes a narrowband tone
    along its approximate centerline; amplitudes are scaled relative
    to the global cube RMS so that vessel injections behave as a
    modest perturbation that can be tuned separately.
    """

    T, H, W = cube.shape
    if T <= 0 or H <= 0 or W <= 0 or prf_hz <= 0.0:
        return cube, {"n_micro": 0, "n_alias": 0}

    prf = float(prf_hz)
    f0 = float(f0_hz)
    c0_val = float(c0)
    cube_flat = cube.reshape(T, -1)
    # Global RMS level used as a reference scale. We deliberately keep
    # these injections as a modest but non-negligible perturbation so
    # that Pf/Pa structure can be tuned without overwhelming the base
    # k-Wave content.
    bg_rms = float(np.sqrt(np.mean(np.abs(cube_flat) ** 2)) + 1e-12)
    # Empirically chosen scales for MaceBridge: microvascular content
    # is made relatively strong so that Pf can dominate in H1 tiles,
    # while alias content remains pronounced but not overwhelming.
    micro_amp_base = 0.40 * bg_rms
    alias_amp_base = 0.20 * bg_rms

    t_sec = np.arange(T, dtype=np.float32) / prf

    def _line_indices(z0: float, x0: float, length: float, theta: float) -> np.ndarray:
        """Approximate a short centerline in (z,x) and return flat indices."""

        n = max(1, int(round(length)))
        idxs: list[int] = []
        cz = float(z0)
        cx = float(x0)
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))
        for k in range(n):
            z = int(round(cz + k * cos_t))
            x = int(round(cx + k * sin_t))
            if 0 <= z < H and 0 <= x < W:
                idxs.append(z * W + x)
        if not idxs:
            return np.zeros(0, dtype=np.int64)
        return np.asarray(idxs, dtype=np.int64)

    n_micro = 0
    if micro_vessels is not None and micro_vessels.size:
        mv = np.asarray(micro_vessels, dtype=np.float32)
        if mv.ndim == 2 and mv.shape[1] >= 5:
            for z0, x0, length, theta_rad, v_mm_s, radius_um in mv:
                idxs = _line_indices(z0, x0, length, theta_rad)
                if idxs.size == 0:
                    continue
                v_m_s = float(v_mm_s) * 1e-3
                if v_m_s <= 0.0:
                    continue
                f_D = 2.0 * f0 * v_m_s / max(c0_val, 1e-6)
                # Keep microvascular content in Pf; clamp to a reasonable band.
                f_D = float(np.clip(f_D, 30.0, 250.0))
                phase = np.exp(1j * 2.0 * np.pi * f_D * t_sec, dtype=np.complex64)
                # Scale amplitude by radius to mimic vessel cross-section.
                radius = max(float(radius_um), 5.0)
                amp = micro_amp_base * (radius / 40.0) ** 2
                cube_flat[:, idxs] += amp * phase[:, None]
                n_micro += 1

    n_alias = 0
    if alias_vessels is not None and alias_vessels.size:
        av = np.asarray(alias_vessels, dtype=np.float32)
        if av.ndim == 2 and av.shape[1] >= 5:
            for z0, x0, length, theta_rad, v_m_s in av:
                idxs = _line_indices(z0, x0, length, theta_rad)
                if idxs.size == 0:
                    continue
                v_m_s = float(v_m_s)
                if v_m_s <= 0.0:
                    continue
                f_D_base = 2.0 * f0 * v_m_s / max(c0_val, 1e-6)
                # Map to aliased baseband frequency in [0, PRF/2].
                f_mod = float(np.fmod(abs(f_D_base), prf))
                nyq = 0.5 * prf
                if f_mod > nyq:
                    f_alias = prf - f_mod
                else:
                    f_alias = f_mod
                # Guard against degenerate values.
                f_alias = float(np.clip(f_alias, 400.0, nyq))
                phase = np.exp(1j * 2.0 * np.pi * f_alias * t_sec, dtype=np.complex64)
                cube_flat[:, idxs] += alias_amp_base * phase[:, None]
                n_alias += 1

    cube = cube_flat.reshape(cube.shape)
    return cube, {"n_micro": int(n_micro), "n_alias": int(n_alias)}


def _compute_alias_metrics_mt(
    cube_T_hw: np.ndarray,
    prf_hz: float,
    band_ratio_spec: dict[str, float] | None,
    *,
    tapers: int = 3,
    bandwidth: float = 2.0,
    eps: float = 1e-6,
    alpha: float = 0.5,
    beta: float = 1.5,
) -> dict[str, float | None]:
    """
    Compute multi-taper alias metrics for a tile.

    Returns a dict with keys:
      - m_alias: generalized alias score
      - E_f, E_a, E_g, E_dc: band / DC energies
      - c_f, c_a: within-band coherence proxies
      - r_g: guard-band fraction

    If band_ratio_spec is None or invalid, returns all-None metrics.
    """
    metrics: dict[str, float | None] = {
        "m_alias": None,
        "E_f": None,
        "E_a": None,
        "E_g": None,
        "E_dc": None,
        "c_f": None,
        "c_a": None,
        "r_g": None,
    }
    if band_ratio_spec is None or prf_hz <= 0.0:
        return metrics
    try:
        flow_low = float(band_ratio_spec.get("flow_low_hz"))
        flow_high = float(band_ratio_spec.get("flow_high_hz"))
        alias_center = float(band_ratio_spec.get("alias_center_hz"))
        alias_width = float(band_ratio_spec.get("alias_width_hz"))
    except Exception:
        return metrics

    series = np.mean(cube_T_hw.reshape(cube_T_hw.shape[0], -1), axis=1)
    freqs, psd = _multi_taper_psd(
        series,
        prf_hz,
        tapers=max(1, int(tapers)),
        bandwidth=max(float(bandwidth), 1.0),
    )
    if freqs.size == 0 or psd.size == 0:
        return metrics

    df = prf_hz / float(series.size)
    pos_len = freqs.size

    def _clamp_idx(freq_hz: float) -> int:
        idx = int(np.round(freq_hz / max(df, 1e-12)))
        return int(np.clip(idx, 0, pos_len - 1))

    f_low = min(flow_low, flow_high)
    f_high = max(flow_low, flow_high)
    flow_lo_idx = _clamp_idx(f_low)
    flow_hi_idx = _clamp_idx(f_high)
    if flow_hi_idx < flow_lo_idx:
        flow_hi_idx = flow_lo_idx
    flow_idx = np.arange(flow_lo_idx, flow_hi_idx + 1, dtype=np.int32)

    alias_half = max(alias_width, df)
    alias_low = alias_center - alias_half
    alias_high = alias_center + alias_half
    alias_lo_idx = _clamp_idx(alias_low)
    alias_hi_idx = _clamp_idx(alias_high)
    if alias_hi_idx < alias_lo_idx:
        alias_hi_idx = alias_lo_idx
    alias_idx = np.arange(alias_lo_idx, alias_hi_idx + 1, dtype=np.int32)

    guard_lo_hz = f_high
    guard_hi_hz = alias_low
    guard_idx = np.array([], dtype=np.int32)
    if guard_hi_hz > guard_lo_hz:
        guard_lo_idx = _clamp_idx(guard_lo_hz)
        guard_hi_idx = _clamp_idx(guard_hi_hz)
        if guard_hi_idx >= guard_lo_idx:
            guard_idx = np.arange(guard_lo_idx, guard_hi_idx + 1, dtype=np.int32)

    mask_f = np.zeros_like(freqs, dtype=bool)
    mask_a = np.zeros_like(freqs, dtype=bool)
    mask_g = np.zeros_like(freqs, dtype=bool)
    mask_f[flow_idx] = True
    mask_a[alias_idx] = True
    mask_g[guard_idx] = True
    mask_g[mask_f | mask_a] = False

    def _band_energy(mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        return float(np.sum(psd[mask]).item())

    E_f = _band_energy(mask_f)
    E_a = _band_energy(mask_a)
    E_g = _band_energy(mask_g)
    zero_idx = int(np.argmin(np.abs(freqs)))
    E_dc = float(psd[zero_idx].item()) if 0 <= zero_idx < psd.size else 0.0

    total = E_f + E_a + E_g + E_dc
    if total <= 0.0:
        total = eps
    r_g = E_g / total

    def _coherence(mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        band = psd[mask]
        band_sum = float(np.sum(band).item())
        if band_sum <= eps:
            return 0.0
        return float(np.max(band).item() / band_sum)

    c_f = _coherence(mask_f)
    c_a = _coherence(mask_a)

    m_alias = np.log((E_a + eps) / (E_f + eps)) + alpha * (c_a - c_f) - beta * r_g

    metrics.update(
        {
            "m_alias": float(m_alias),
            "E_f": float(E_f),
            "E_a": float(E_a),
            "E_g": float(E_g),
            "E_dc": float(E_dc),
            "c_f": float(c_f),
            "c_a": float(c_a),
            "r_g": float(r_g),
        }
    )
    return metrics


@dataclass
class BandRatioSpec:
    flow_low_hz: float
    flow_high_hz: float
    alias_center_hz: float
    alias_width_hz: float
    eps: float = 1e-8
    alias_min_bins: int | None = None
    alias_max_bins: int | None = None


def _freq_bins_from_hz(
    spec: BandRatioSpec,
    prf_hz: float,
    series_len: int,
    *,
    alias_min_bins: int | None = None,
    alias_max_bins: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return non-negative FFT bin indices covering the requested flow/alias bands."""
    if series_len <= 0 or prf_hz <= 0.0:
        raise ValueError("Invalid series length or PRF for band selection.")
    df = prf_hz / float(series_len)
    pos_len = int(series_len // 2) + 1

    def _clamp_bin(freq_hz: float) -> int:
        idx = int(np.round(freq_hz / df))
        return int(np.clip(idx, 0, pos_len - 1))

    f_low = min(spec.flow_low_hz, spec.flow_high_hz)
    f_high = max(spec.flow_low_hz, spec.flow_high_hz)
    low_idx = _clamp_bin(f_low)
    high_idx = _clamp_bin(f_high)
    if high_idx < low_idx:
        high_idx = low_idx
    flow_idx = np.arange(low_idx, high_idx + 1, dtype=np.int32)
    if flow_idx.size == 0:
        flow_idx = np.array([low_idx], dtype=np.int32)

    alias_half = max(spec.alias_width_hz, df)
    alias_low = spec.alias_center_hz - alias_half
    alias_high = spec.alias_center_hz + alias_half
    alias_low_idx = _clamp_bin(alias_low)
    alias_high_idx = _clamp_bin(alias_high)
    if alias_high_idx < alias_low_idx:
        alias_high_idx = alias_low_idx
    alias_idx = np.arange(alias_low_idx, alias_high_idx + 1, dtype=np.int32)
    if alias_idx.size == 0:
        alias_idx = np.array([alias_low_idx], dtype=np.int32)

    # Enforce alias-band length bounds if requested
    min_bins = alias_min_bins
    max_bins = alias_max_bins
    if min_bins is not None and min_bins <= 0:
        min_bins = None
    if max_bins is not None and max_bins <= 0:
        max_bins = None
    if min_bins is not None and max_bins is not None and min_bins > max_bins:
        min_bins = max_bins

    def _expand(low: int, high: int, target_len: int) -> tuple[int, int]:
        current_len = high - low + 1
        if current_len >= target_len:
            return low, high
        extra = target_len - current_len
        pad_low = extra // 2
        pad_high = extra - pad_low
        low = max(0, low - pad_low)
        high = min(pos_len - 1, high + pad_high)
        # If we hit boundary but still short, pad on remaining side
        current_len = high - low + 1
        if current_len < target_len:
            deficit = target_len - current_len
            if low == 0:
                high = min(pos_len - 1, high + deficit)
            elif high == pos_len - 1:
                low = max(0, low - deficit)
        return low, high

    if min_bins is not None and alias_idx.size < min_bins:
        alias_low_idx, alias_high_idx = _expand(alias_low_idx, alias_high_idx, min_bins)
        alias_idx = np.arange(alias_low_idx, alias_high_idx + 1, dtype=np.int32)
    if max_bins is not None and alias_idx.size > max_bins:
        shrink = alias_idx.size - max_bins
        drop_low = shrink // 2
        drop_high = shrink - drop_low
        alias_low_idx = min(alias_high_idx, alias_low_idx + drop_low)
        alias_high_idx = max(alias_low_idx, alias_high_idx - drop_high)
        alias_idx = np.arange(alias_low_idx, alias_high_idx + 1, dtype=np.int32)
        if alias_idx.size == 0:
            alias_idx = np.array([alias_low_idx], dtype=np.int32)

    return flow_idx, alias_idx


class BandRatioRecorder:
    """Collect multi-taper PSD slices for flow/alias bands across tiles."""

    def __init__(
        self,
        prf_hz: float,
        tile_count: int,
        spec: BandRatioSpec,
        *,
        tapers: int = 3,
        bandwidth: float = 2.0,
    ) -> None:
        self.prf_hz = float(prf_hz)
        self.tile_count = int(tile_count)
        self.spec = spec
        self.tapers = max(1, int(tapers))
        self.bandwidth = max(float(bandwidth), 1.0)
        self.flow_idx: np.ndarray | None = None
        self.alias_idx: np.ndarray | None = None
        self.flow_cache: np.ndarray | None = None
        self.alias_cache: np.ndarray | None = None
        self.bg_flow_sum: np.ndarray | None = None
        self.bg_alias_sum: np.ndarray | None = None
        self.bg_count = 0
        self.gamma = 1.0
        self.series_len: int | None = None
        self.tile_filled = np.zeros(self.tile_count, dtype=bool)
        self.tile_is_bg = np.zeros(self.tile_count, dtype=bool)
        self.peak_freqs = np.zeros(self.tile_count, dtype=np.float32)
        # Raw (non-whitened) band energies and guard fraction per tile. These
        # are useful for label-free contract telemetry (e.g. guard dominance
        # checks) even when the whitened band-ratio score is used for m_alias.
        self.Ef_raw = np.zeros(self.tile_count, dtype=np.float32)
        self.Ea_raw = np.zeros(self.tile_count, dtype=np.float32)
        self.Eg_raw = np.zeros(self.tile_count, dtype=np.float32)
        self.Edc_raw = np.zeros(self.tile_count, dtype=np.float32)
        self.rg_raw = np.zeros(self.tile_count, dtype=np.float32)
        self.alias_bin_count = None

    def _ensure_bins(self, series_len: int) -> None:
        if self.flow_idx is not None:
            return
        flow_idx, alias_idx = _freq_bins_from_hz(
            self.spec,
            self.prf_hz,
            series_len,
            alias_min_bins=self.spec.alias_min_bins,
            alias_max_bins=self.spec.alias_max_bins,
        )
        self.flow_idx = flow_idx
        self.alias_idx = alias_idx
        self.gamma = float(flow_idx.size) / max(1, alias_idx.size)
        self.alias_bin_count = int(alias_idx.size)
        self.flow_cache = np.zeros((self.tile_count, flow_idx.size), dtype=np.float32)
        self.alias_cache = np.zeros((self.tile_count, alias_idx.size), dtype=np.float32)
        self.bg_flow_sum = np.zeros(flow_idx.size, dtype=np.float64)
        self.bg_alias_sum = np.zeros(alias_idx.size, dtype=np.float64)
        self.series_len = int(series_len)

    def observe(self, tile_idx: int, series: np.ndarray, tile_is_bg: bool) -> None:
        """Record PSD slices for a tile's post-filter slow-time series."""
        tile_idx = int(tile_idx)
        if tile_idx < 0 or tile_idx >= self.tile_count:
            raise IndexError(f"tile_idx {tile_idx} out of range for BandRatioRecorder")
        series_flat = np.asarray(series, dtype=np.complex64).ravel()
        if series_flat.size == 0:
            raise ValueError("Empty series provided to BandRatioRecorder.observe")
        self._ensure_bins(series_flat.size)
        freqs, psd = _multi_taper_psd(
            series_flat,
            self.prf_hz,
            tapers=self.tapers,
            bandwidth=self.bandwidth,
        )
        assert self.flow_cache is not None and self.alias_cache is not None
        assert self.flow_idx is not None and self.alias_idx is not None
        self.flow_cache[tile_idx, :] = psd[self.flow_idx]
        self.alias_cache[tile_idx, :] = psd[self.alias_idx]
        self.tile_is_bg[tile_idx] = bool(tile_is_bg)
        peak_idx = int(np.argmax(psd))
        if peak_idx < 0 or peak_idx >= freqs.shape[0]:
            peak_idx = int(np.argmax(np.abs(psd)))
        self.peak_freqs[tile_idx] = float(abs(freqs[peak_idx]))

        # Raw energy telemetry (non-whitened): Ef/Ea over the design bands and
        # Eg over the in-between guard region. This is intentionally separate
        # from the whitened band-ratio computation performed in finalize().
        try:
            flow_low = float(min(self.spec.flow_low_hz, self.spec.flow_high_hz))
            flow_high = float(max(self.spec.flow_low_hz, self.spec.flow_high_hz))
            alias_half = float(max(self.spec.alias_width_hz, 0.0))
            alias_low = float(self.spec.alias_center_hz - alias_half)
            alias_high = float(self.spec.alias_center_hz + alias_half)
            mask_f = (freqs >= flow_low) & (freqs <= flow_high)
            mask_a = (freqs >= alias_low) & (freqs <= alias_high)
            mask_g = np.zeros_like(mask_f, dtype=bool)
            if alias_low > flow_high:
                mask_g = (freqs >= flow_high) & (freqs <= alias_low)
                mask_g[mask_f | mask_a] = False
            zero_idx = int(np.argmin(np.abs(freqs)))
            E_dc = float(psd[zero_idx]) if 0 <= zero_idx < psd.size else 0.0
            E_f = float(np.sum(psd[mask_f])) if np.any(mask_f) else 0.0
            E_a = float(np.sum(psd[mask_a])) if np.any(mask_a) else 0.0
            E_g = float(np.sum(psd[mask_g])) if np.any(mask_g) else 0.0
            total = E_f + E_a + E_g + E_dc
            if total <= 0.0:
                total = 1e-12
            r_g = float(E_g / total)
            self.Ef_raw[tile_idx] = np.float32(E_f)
            self.Ea_raw[tile_idx] = np.float32(E_a)
            self.Eg_raw[tile_idx] = np.float32(E_g)
            self.Edc_raw[tile_idx] = np.float32(E_dc)
            self.rg_raw[tile_idx] = np.float32(r_g)
        except Exception:
            # Best-effort only: leave defaults on failure.
            pass

        if tile_is_bg:
            assert self.bg_flow_sum is not None and self.bg_alias_sum is not None
            self.bg_flow_sum += psd[self.flow_idx]
            self.bg_alias_sum += psd[self.alias_idx]
            self.bg_count += 1
        self.tile_filled[tile_idx] = True

    def finalize(self) -> tuple[np.ndarray, dict[str, float]]:
        """Return per-tile whitened log band-ratio scores and summary telemetry."""
        # If no PSD observations were ever recorded, return an empty result
        # instead of raising. This can occur in configurations where band-ratio
        # recording is enabled but no post-filter callback is wired (e.g., when
        # using fast STAP paths or PD-only scoring).
        if self.flow_cache is None or self.alias_cache is None or not np.any(self.tile_filled):
            return np.array([], dtype=np.float32), {
                "count": 0,
                "flow_band_hz": [float(self.spec.flow_low_hz), float(self.spec.flow_high_hz)],
                "alias_band_hz": [
                    float(self.spec.alias_center_hz - self.spec.alias_width_hz),
                    float(self.spec.alias_center_hz + self.spec.alias_width_hz),
                ],
                "flow_fraction_in_band": None,
                "alias_fraction_in_band": None,
                "flow_samples": 0,
                "alias_samples": 0,
                "peak_hz_stats": {},
            }
        # If only a subset of tiles produced PSDs, restrict statistics to the
        # observed subset and return neutral scores for the rest.
        if not np.all(self.tile_filled):
            valid_idx = np.nonzero(self.tile_filled)[0]
            if valid_idx.size == 0:
                return np.array([], dtype=np.float32), {
                    "count": 0,
                    "flow_band_hz": [
                        float(self.spec.flow_low_hz),
                        float(self.spec.flow_high_hz),
                    ],
                    "alias_band_hz": [
                        float(self.spec.alias_center_hz - self.spec.alias_width_hz),
                        float(self.spec.alias_center_hz + self.spec.alias_width_hz),
                    ],
                    "flow_fraction_in_band": None,
                    "alias_fraction_in_band": None,
                    "flow_samples": 0,
                    "alias_samples": 0,
                    "peak_hz_stats": {},
                }
            # Restrict caches to tiles that produced PSDs; we will expand the
            # scores back to all tiles after computing them.
            flow_cache = self.flow_cache[valid_idx, :]
            alias_cache = self.alias_cache[valid_idx, :]
        else:
            flow_cache = self.flow_cache
            alias_cache = self.alias_cache
        assert self.flow_idx is not None and self.alias_idx is not None
        flow_bg = (
            self.bg_flow_sum / max(1, self.bg_count)
            if self.bg_flow_sum is not None
            else np.maximum(flow_cache.mean(axis=0), 1e-12)
        )
        alias_bg = (
            self.bg_alias_sum / max(1, self.bg_count)
            if self.bg_alias_sum is not None
            else np.maximum(alias_cache.mean(axis=0), 1e-12)
        )
        eps = float(self.spec.eps)
        Ef = (flow_cache / (flow_bg + eps)).sum(axis=1)
        Ea = (alias_cache / (alias_bg + eps)).sum(axis=1)
        scores_valid = np.log(Ef + eps) - np.log(self.gamma + eps) - np.log(Ea + eps)
        # Expand scores back to full tile_count, using neutral zeros for any
        # tiles that lacked PSD observations.
        scores = np.zeros(self.tile_count, dtype=np.float32)
        valid_idx = np.nonzero(self.tile_filled)[0]
        scores[valid_idx] = scores_valid.astype(np.float32, copy=False)
        tele = {
            "br_gamma": float(self.gamma),
            "br_bg_tiles": int(self.bg_count),
            "br_flow_bins": int(self.flow_idx.size),
            "br_alias_bins": int(self.alias_idx.size),
        }
        tele["br_alias_bins_limited"] = bool(
            self.spec.alias_min_bins is not None or self.spec.alias_max_bins is not None
        )
        if self.alias_bin_count is not None:
            tele["br_alias_bins_actual"] = int(self.alias_bin_count)
        if self.series_len is not None:
            tele["br_series_len"] = int(self.series_len)

        # Peak stats should consider only tiles with PSD observations. Otherwise
        # empty tiles would be counted as non-bg with peak_freq=0 and bias
        # band-occupancy fractions.
        valid_mask = np.asarray(self.tile_filled, dtype=bool)
        peak_freqs_full = np.asarray(self.peak_freqs, dtype=np.float64)
        tile_is_bg_full = np.asarray(self.tile_is_bg, dtype=bool)
        if np.any(valid_mask):
            peak_freqs = peak_freqs_full[valid_mask]
            tile_is_bg = tile_is_bg_full[valid_mask]
        else:
            peak_freqs = np.asarray([], dtype=np.float64)
            tile_is_bg = np.asarray([], dtype=bool)
        non_bg_mask = ~tile_is_bg
        flow_low = float(min(self.spec.flow_low_hz, self.spec.flow_high_hz))
        flow_high = float(max(self.spec.flow_low_hz, self.spec.flow_high_hz))
        alias_half = float(max(self.spec.alias_width_hz, 0.0))
        alias_low = float(self.spec.alias_center_hz - alias_half)
        alias_high = float(self.spec.alias_center_hz + alias_half)

        if np.any(non_bg_mask):
            flow_band_mask = (peak_freqs >= flow_low) & (peak_freqs <= flow_high)
            tele["br_flow_peak_fraction_nonbg"] = float(np.mean(flow_band_mask[non_bg_mask]))
        else:
            tele["br_flow_peak_fraction_nonbg"] = None

        if np.any(tile_is_bg):
            alias_mask = (peak_freqs >= alias_low) & (peak_freqs <= alias_high)
            tele["br_alias_peak_fraction_bg"] = float(np.mean(alias_mask[tile_is_bg]))
        else:
            tele["br_alias_peak_fraction_bg"] = None

        tele["br_alias_peak_fraction_nonbg"] = None
        if np.any(non_bg_mask):
            alias_mask_nonbg = (peak_freqs >= alias_low) & (peak_freqs <= alias_high)
            tele["br_alias_peak_fraction_nonbg"] = float(np.mean(alias_mask_nonbg[non_bg_mask]))

        if peak_freqs.size:
            tele["br_peak_freq_hz_p50"] = float(np.median(peak_freqs))
            tele["br_peak_freq_hz_p90"] = float(np.quantile(peak_freqs, 0.90))
        else:
            tele["br_peak_freq_hz_p50"] = None
            tele["br_peak_freq_hz_p90"] = None
        return scores.astype(np.float32, copy=False), tele


def _hermitianize_tensor(R: torch.Tensor) -> torch.Tensor:
    """Force Hermitian symmetry for numerical stability."""
    return 0.5 * (R + R.conj().transpose(-2, -1))


def _pd_band_energy_attempt(
    R_loaded: torch.Tensor,
    S_tile: torch.Tensor,
    Ct: torch.Tensor,
    *,
    ridge: float,
    ratio_rho: float,
    agg_mode: str,
    cond_threshold: float,
    basis_mode: str,
) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """
    Compute PD band and ratio maps for a single attempt.

    Returns (band_fraction_hw, ratio_hw, details). On failure the arrays are None
    and details["pd_success"] is False.
    """
    details: dict = {
        "pd_basis": basis_mode,
        "pd_gram_ridge_used": float(ridge),
        "pd_success": False,
    }
    try:
        Lt = R_loaded.shape[-1]
        if Ct.shape[-1] == 0:
            N = S_tile.shape[1]
            h, w = S_tile.shape[2], S_tile.shape[3]
            zeros = np.zeros((h, w), dtype=np.float32)
            details["pd_condG"] = 1.0
            details["pd_need_retry"] = False
            details["pd_success"] = True
            return zeros, zeros, details

        R_herm = _hermitianize_tensor(R_loaded)
        L = torch.linalg.cholesky(R_herm)

        S_flat = S_tile.reshape(Lt, -1)
        S_w_flat = torch.linalg.solve_triangular(L, S_flat, upper=False, left=True)
        C_w = torch.linalg.solve_triangular(L, Ct, upper=False, left=True)

        Gram = C_w.conj().transpose(-2, -1) @ C_w
        Gram = _hermitianize_tensor(Gram)
        evals = torch.linalg.eigvalsh(Gram).real
        evals = torch.clamp(evals, min=1e-12)
        if evals.numel() > 0:
            evals_max = float(torch.max(evals).item())
            evals_min = float(torch.min(evals).item())
        else:
            evals_max = 0.0
            evals_min = 0.0
        raw_cond = float(evals_max / evals_min) if evals_min > 0.0 else float("inf")
        details["pd_condG_raw"] = raw_cond
        # IMPORTANT: Gram is computed in the *whitened* constraint basis. Depending on
        # the whitening scale, its eigenvalues can be orders of magnitude below 1.
        # An absolute ridge (e.g. 0.1) can then dominate Gram and collapse the
        # projected-band energy toward ~0. Use a scale-aware ridge so that the ridge
        # remains "relative" when Gram is tiny, while leaving typical regimes unchanged.
        ridge_eff = float(ridge)
        if ridge_eff > 0.0 and evals_max > 0.0 and evals_max < 1.0:
            ridge_eff = ridge_eff * evals_max
        if evals_max > 0.0:
            target_cond = float(cond_threshold)
            if evals_min <= 0.0:
                ridge_eff = max(ridge_eff, evals_max / target_cond + 1e-12)
            else:
                ridge_needed = max(0.0, (evals_max / target_cond) - evals_min)
                ridge_eff = max(ridge_eff, ridge_needed)
        if ridge_eff > 0.0:
            Gram = Gram + float(ridge_eff) * torch.eye(
                Gram.shape[-1], dtype=Gram.dtype, device=Gram.device
            )
        cond_den = evals_min + ridge_eff
        cond_den = cond_den if cond_den > 1e-12 else 1e-12
        condG = float(evals_max / cond_den) if evals_max > 0.0 else 1.0
        details["pd_condG"] = condG
        details["pd_gram_ridge_used"] = float(ridge_eff)
        need_retry = not np.isfinite(condG) or condG > cond_threshold
        details["pd_need_retry"] = bool(need_retry)

        Z = C_w.conj().transpose(-2, -1) @ S_w_flat
        coeffs = torch.linalg.solve(Gram, Z)
        T_band = torch.sum(Z.conj() * coeffs, dim=0).real
        sw_pow = torch.sum(S_w_flat.conj() * S_w_flat, dim=0).real

        eps = 1e-10
        sw_pow = torch.clamp(sw_pow, min=eps)
        ratio_denom = torch.clamp(sw_pow - T_band + ratio_rho * sw_pow, min=eps)
        r = torch.clamp(T_band / sw_pow, min=0.0, max=1.0)
        ratio = torch.clamp(T_band / ratio_denom, min=0.0)

        N = S_tile.shape[1]
        h, w = S_tile.shape[2], S_tile.shape[3]
        r_hw = aggregate_over_snapshots(r.reshape(N, h, w), mode=agg_mode)
        ratio_hw = aggregate_over_snapshots(ratio.reshape(N, h, w), mode=agg_mode)

        if not torch.isfinite(r_hw).all() or not torch.isfinite(ratio_hw).all():
            details["pd_need_retry"] = True

        details["pd_success"] = True
        return (
            r_hw.detach().cpu().numpy().astype(np.float32),
            ratio_hw.detach().cpu().numpy().astype(np.float32),
            details,
        )
    except Exception as exc:  # pragma: no cover - defensive
        details["pd_error"] = repr(exc)
        return None, None, details


def _default_gpu_enabled() -> bool:
    """
    Detect whether we should prefer GPU execution for k-Wave runs.
    Respects the `KWAVE_FORCE_CPU` environment variable.
    """
    global _GPU_AVAILABLE_CACHE
    if os.environ.get("KWAVE_FORCE_CPU") == "1":
        _GPU_AVAILABLE_CACHE = False
        return False
    if _GPU_AVAILABLE_CACHE is not None:
        return _GPU_AVAILABLE_CACHE
    try:
        if torch is not None and torch.cuda.is_available():
            _GPU_AVAILABLE_CACHE = True
            return True
    except Exception:
        pass
    try:  # pragma: no cover - optional dependency
        import cupy  # type: ignore

        cupy.cuda.runtime.getDevice()
        _GPU_AVAILABLE_CACHE = True
        return True
    except Exception:
        _GPU_AVAILABLE_CACHE = False
        return False


# --------------------------------------------------------------------------- #
# Geometry definitions
# --------------------------------------------------------------------------- #


@dataclass
class SimGeom:
    Nx: int
    Ny: int
    dx: float
    dy: float
    c0: float
    rho0: float
    pml_size: int = 20
    cfl: float = 0.3
    t_end: float | None = None
    f0: float = 7.5e6
    ncycles: int = 3
    tx_row_from_top: int = 2
    rx_row_from_top: int = 3
    alpha_db_mhz_cm: float = 0.5
    alpha_power: float = 1.5


@dataclass
class AngleData:
    angle_deg: float
    rf: np.ndarray  # (Nt, N_rx) float32
    dt: float


def slice_angle_data(
    angle_sets: Sequence[Sequence[AngleData]],
    offset: int,
    length: int,
) -> list[list[AngleData]]:
    """
    Return a copy of the provided angle data restricted to [offset, offset+length).

    Raises ValueError if the requested window lies outside any tile's RF cube.
    """
    if length <= 0:
        raise ValueError("time-window length must be positive")
    if offset < 0:
        raise ValueError("time-window offset must be non-negative")
    sliced: list[list[AngleData]] = []
    for ensemble in angle_sets:
        ens_sliced: list[AngleData] = []
        for angle in ensemble:
            rf = np.asarray(angle.rf)
            total = rf.shape[0]
            end = offset + length
            if end > total:
                raise ValueError(
                    "time-window "
                    f"[{offset}, {end}) exceeds RF length {total} for angle {angle.angle_deg}"
                )
            rf_slice = np.ascontiguousarray(rf[offset:end])
            ens_sliced.append(
                AngleData(angle_deg=angle.angle_deg, rf=rf_slice, dt=float(angle.dt))
            )
        sliced.append(ens_sliced)
    return sliced


def build_grid_and_time(g: SimGeom) -> kWaveGrid:
    kgrid = kWaveGrid(N=(g.Nx, g.Ny), spacing=(g.dx, g.dy))
    kgrid.makeTime(c=g.c0, cfl=g.cfl, t_end=g.t_end)
    if not isinstance(kgrid.Nt, (int, np.integer)) or kgrid.Nt <= 0:
        raise RuntimeError("kgrid.makeTime failed to set Nt.")
    return kgrid


def build_medium(g: SimGeom) -> kWaveMedium:
    return kWaveMedium(
        sound_speed=np.full((g.Nx, g.Ny), g.c0, dtype=np.float32),
        density=np.full((g.Nx, g.Ny), g.rho0, dtype=np.float32),
        alpha_coeff=g.alpha_db_mhz_cm,
        alpha_power=g.alpha_power,
    )


def build_linear_masks(g: SimGeom) -> tuple[np.ndarray, np.ndarray]:
    tx_mask = np.zeros((g.Nx, g.Ny), dtype=bool)
    rx_mask = np.zeros((g.Nx, g.Ny), dtype=bool)
    tx_mask[:, g.tx_row_from_top] = True
    rx_mask[:, g.rx_row_from_top] = True
    return tx_mask, rx_mask


def _validate_masks(
    kgrid: kWaveGrid, tx_mask: np.ndarray, rx_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Ensure mask shapes/dtypes match kgrid.N (Nx, Ny), fix simple issues.

    - Enforce boolean dtype and C-contiguity
    - If transposed shape matches, transpose
    - Raise ValueError if still incompatible
    """
    expected = tuple(int(v) for v in kgrid.N)

    def _fix(mask: np.ndarray, name: str) -> np.ndarray:
        m = np.asarray(mask)
        if m.shape != expected and m.T.shape == expected:
            m = m.T
        if m.shape != expected:
            raise ValueError(f"{name} shape {m.shape} != kgrid.N {expected}")
        if m.dtype != np.bool_:
            m = m.astype(np.bool_, copy=False)
        if not m.flags.c_contiguous:
            m = np.ascontiguousarray(m)
        return m

    return _fix(tx_mask, "source.p_mask"), _fix(rx_mask, "sensor.mask")


def plane_wave_sample_shifts(kgrid: kWaveGrid, g: SimGeom, theta_rad: float) -> np.ndarray:
    dt = kgrid.dt
    x_positions = (np.arange(g.Nx) - g.Nx / 2.0) * g.dx
    tau = (x_positions * np.sin(theta_rad)) / g.c0
    return np.round(tau / dt).astype(int)


def build_source_p(
    g: SimGeom,
    kgrid: kWaveGrid,
    tx_mask: np.ndarray,
    theta_rad: float,
) -> np.ndarray:
    Nt = kgrid.Nt
    burst = tone_burst(
        sample_freq=1.0 / kgrid.dt,
        signal_freq=g.f0,
        num_cycles=g.ncycles,
        envelope="Gaussian",
        signal_length=Nt,
        signal_offset=0,
    )
    burst = np.asarray(burst, dtype=np.float32).ravel()

    src_idx = np.argwhere(tx_mask)
    p = np.zeros((src_idx.shape[0], Nt), dtype=np.float32)
    shifts = plane_wave_sample_shifts(kgrid, g, theta_rad)
    for row, (xi, _) in enumerate(src_idx):
        shift = int(shifts[xi])
        tmp = np.zeros(Nt, dtype=np.float32)
        if shift >= 0:
            start = shift
            end = min(Nt, shift + burst.size)
            if end > start:
                tmp[start:end] = burst[: end - start]
        else:
            src_start = min(-shift, burst.size)
            dst_len = min(Nt, burst.size - src_start)
            if dst_len > 0:
                tmp[:dst_len] = burst[src_start : src_start + dst_len]
        p[row] = tmp
    return p


def run_angle_once(
    out_dir: Path,
    angle_deg: float,
    g: SimGeom,
    use_gpu: bool | None = None,
) -> AngleData:
    out_dir.mkdir(parents=True, exist_ok=True)
    if use_gpu is None:
        use_gpu = _default_gpu_enabled()

    kgrid = build_grid_and_time(g)
    medium = build_medium(g)
    tx_mask, rx_mask = build_linear_masks(g)
    tx_mask, rx_mask = _validate_masks(kgrid, tx_mask, rx_mask)

    source = kSource()
    source.p_mask = tx_mask
    source.p = build_source_p(g, kgrid, tx_mask, np.deg2rad(angle_deg))

    sensor = kSensor(mask=rx_mask, record=["p"])

    ang_tag = f"{int(round(angle_deg))}"
    input_h5 = f"in_{ang_tag}.h5"
    output_h5 = f"out_{ang_tag}.h5"
    # Pre-clean any stale artifacts
    try:
        (out_dir / input_h5).unlink(missing_ok=True)  # type: ignore[arg-type]
    except Exception:
        pass
    try:
        (out_dir / output_h5).unlink(missing_ok=True)  # type: ignore[arg-type]
    except Exception:
        pass

    sim_opts = SimulationOptions(
        pml_inside=False,
        pml_x_size=g.pml_size,
        pml_y_size=g.pml_size,
        save_to_disk=True,
        data_path=str(out_dir),
        input_filename=input_h5,
        output_filename=output_h5,
    )
    exec_opts = SimulationExecutionOptions(
        is_gpu_simulation=bool(use_gpu),
        show_sim_log=False,
    )

    def _run_with_opts(opts: SimulationExecutionOptions):
        return kspaceFirstOrder2DC(kgrid, source, sensor, medium, sim_opts, opts)

    try:
        sensor_data = _run_with_opts(exec_opts)
    except Exception:
        if use_gpu:
            print(
                f"[run_angle_once] GPU execution failed for angle {angle_deg:.2f}°, "
                "retrying on CPU.",
                flush=True,
            )
            exec_opts = SimulationExecutionOptions(
                is_gpu_simulation=False,
                show_sim_log=False,
            )
            sensor_data = _run_with_opts(exec_opts)
        else:
            raise
    if isinstance(sensor_data, dict):
        sensor_data = sensor_data.get("p")
    if not isinstance(sensor_data, np.ndarray) or sensor_data.ndim != 2:
        raise RuntimeError("Unexpected sensor data format from k-Wave.")
    return AngleData(
        angle_deg=float(angle_deg),
        rf=np.asarray(sensor_data, dtype=np.float32),
        dt=float(kgrid.dt),
    )


# --------------------------------------------------------------------------- #
# RF -> IQ -> PD/STAP conversion helpers
# --------------------------------------------------------------------------- #


def _precompute_geometry(g: SimGeom) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = (np.arange(g.Nx) - g.Nx / 2.0) * g.dx
    z = np.arange(g.Ny) * g.dy
    XX, ZZ = np.meshgrid(x, z, indexing="ij")
    d_rx = np.sqrt((XX[..., None] - x[None, None, :]) ** 2 + ZZ[..., None] ** 2)
    return XX, ZZ, d_rx


def _demod_iq(rf: np.ndarray, dt: float, f0: float) -> np.ndarray:
    analytic = hilbert(rf, axis=0)
    t = np.arange(rf.shape[0], dtype=np.float32) * dt
    lo = np.exp(-1j * 2.0 * np.pi * f0 * t)[:, None]
    return (analytic * lo).astype(np.complex64)


def _beamform_angle(
    iq: np.ndarray,
    theta_deg: float,
    dt: float,
    g: SimGeom,
    XX: np.ndarray,
    ZZ: np.ndarray,
    d_rx: np.ndarray,
) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    Nt, Ne = iq.shape
    t_tx = (XX * np.sin(theta) + ZZ * np.cos(theta)) / g.c0
    idx = np.rint((t_tx[..., None] + d_rx / g.c0) / dt).astype(np.int32)
    idx = np.clip(idx, 0, Nt - 1)
    gathered = iq[idx, np.arange(Ne)[None, None, :]]
    img = gathered.sum(axis=2) / float(max(Ne, 1))
    return img.transpose(1, 0).astype(np.complex64)  # (Ny, Nx)


def _sample_phase_screen_vector(
    n_elements: int, phase_std: float, corr_len: float, seed: int
) -> np.ndarray | None:
    if phase_std <= 0.0 or n_elements <= 0:
        return None
    sigma = max(float(corr_len), 1.0)
    half = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-half, half + 1, dtype=np.float32)
    kernel_low = np.exp(-0.5 * (x / sigma) ** 2).astype(np.float32)
    kernel_low /= np.sum(kernel_low)
    # Optional higher-frequency component to better mimic skull/implant
    # heterogeneity when the correlation length is large. We keep this
    # conservative so that existing regimes remain largely unchanged.
    sigma_high = max(sigma / 3.0, 1.0)
    kernel_high = np.exp(-0.5 * (x / sigma_high) ** 2).astype(np.float32)
    kernel_high /= np.sum(kernel_high)
    rng = np.random.default_rng(seed)
    pad = half
    white = rng.standard_normal(n_elements + 2 * pad).astype(np.float32)
    low = np.convolve(white, kernel_low, mode="valid")
    low = low[:n_elements]
    high = np.convolve(white, kernel_high, mode="valid")
    high = high[:n_elements]
    smooth = 0.7 * low + 0.3 * high
    smooth -= float(np.mean(smooth))
    curr_std = float(np.std(smooth) + 1e-9)
    smooth *= phase_std / curr_std
    return smooth.astype(np.float32)


def _synthesize_cube(
    image_sets: Sequence[np.ndarray],
    pulses_per_set: int,
    seed: int,
    amp_jitter: float = 0.05,
    phase_jitter: float = 0.25,
    noise_level: float = 0.01,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frames: list[np.ndarray] = []
    for set_idx, imgs in enumerate(image_sets):
        if imgs.size == 0:
            continue
        phase_sigma = phase_jitter * (1.0 + 0.15 * set_idx)
        for _ in range(max(pulses_per_set, 1)):
            amps = 1.0 + amp_jitter * rng.standard_normal(imgs.shape[0])
            phases = np.exp(1j * rng.normal(scale=phase_sigma, size=imgs.shape[0]))
            frame = np.tensordot(amps * phases, imgs, axes=(0, 0))
            shift_y = int(rng.integers(-1, 2))
            shift_x = int(rng.integers(-1, 2))
            if shift_y or shift_x:
                frame = np.roll(frame, shift=(shift_y, shift_x), axis=(0, 1))
            noise = noise_level * (
                rng.standard_normal(frame.shape) + 1j * rng.standard_normal(frame.shape)
            )
            frames.append((frame + noise).astype(np.complex64))
    if not frames:
        raise ValueError("No slow-time frames generated; check angle_sets input.")
    return np.stack(frames, axis=0)


def _inject_temporal_clutter(
    cube: np.ndarray,
    mask_bg: np.ndarray,
    *,
    beta: float,
    snr_db: float,
    depth_min_frac: float,
    depth_max_frac: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, float] | None]:
    if beta <= 0.0:
        return cube, None
    T, H, W = cube.shape
    depth_min = int(np.clip(depth_min_frac * H, 0, H))
    depth_max = int(np.clip(depth_max_frac * H, 0, H))
    clutter_mask = np.array(mask_bg, copy=True)
    clutter_mask[:depth_min, :] = False
    if depth_max > 0:
        clutter_mask[depth_max:, :] = False
    idx = clutter_mask.ravel().nonzero()[0]
    if idx.size == 0:
        return cube, None
    rng = np.random.default_rng(seed + 404)
    white = (rng.standard_normal((T, idx.size)) + 1j * rng.standard_normal((T, idx.size))).astype(
        np.complex64
    )
    fft_white = np.fft.fft(white, axis=0)
    freqs = np.fft.fftfreq(T)
    shape = np.power(np.maximum(np.abs(freqs), 1.0 / max(T, 8)), -beta / 2.0).astype(np.float32)
    if shape.size > 1:
        shape[0] = shape[1]
    fft_white *= shape[:, None]
    colored = np.fft.ifft(fft_white, axis=0).astype(np.complex64)
    colored -= colored.mean(axis=0, keepdims=True)
    noise_rms = float(np.sqrt(np.mean(np.abs(colored) ** 2)) + 1e-12)
    noise_scale = 10 ** (snr_db / 20.0)
    cube_flat = cube.reshape(T, -1)
    bg_vals = np.abs(cube_flat[:, idx]).ravel()
    bg_rms = float(np.sqrt(np.mean(bg_vals**2)) + 1e-12)
    target_rms = bg_rms * noise_scale
    colored *= target_rms / noise_rms
    cube_flat[:, idx] += colored
    cube = cube_flat.reshape(cube.shape)
    actual_rms = float(np.sqrt(np.mean(np.abs(colored) ** 2)) + 1e-12)
    actual_ratio = actual_rms / (bg_rms + 1e-12)
    clutter_meta = {
        "beta": float(beta),
        "snr_db_target": float(snr_db),
        "snr_db_actual": float(20.0 * math.log10(max(actual_ratio, 1e-6))),
        "depth_min_frac": float(depth_min_frac),
        "depth_max_frac": float(depth_max_frac),
        "n_pixels": int(idx.size),
    }
    return cube, clutter_meta


def _inject_global_vibration(
    cube: np.ndarray,
    prf_hz: float,
    *,
    freq_hz: float,
    amp: float,
    depth_min_frac: float,
    depth_decay_frac: float,
) -> tuple[np.ndarray, dict[str, float] | None]:
    """Inject a depth-weighted narrowband vibration tone globally.

    The tone has frequency ``freq_hz`` and relative complex amplitude ``amp``,
    with an exponential decay in depth starting at ``depth_min_frac``.
    """
    if freq_hz <= 0.0 or amp <= 0.0 or prf_hz <= 0.0:
        return cube, None
    T, H, W = cube.shape
    depth_min = float(np.clip(depth_min_frac, 0.0, 1.0))
    decay = float(np.clip(depth_decay_frac, 1e-3, 1.0))
    z_frac = (np.arange(H, dtype=np.float32) + 0.5) / max(float(H), 1.0)
    decay_arg = np.maximum(z_frac - depth_min, 0.0) / decay
    amp_profile = amp * np.exp(-decay_arg).astype(np.float32)
    if not np.any(amp_profile > 0.0):
        return cube, None
    amp_map = amp_profile.astype(np.float32).reshape(H, 1)
    t = np.arange(T, dtype=np.float32)[:, None, None]
    phase = 2.0 * np.pi * float(freq_hz) * t / float(prf_hz)
    tone_t = np.exp(1j * phase).astype(np.complex64, copy=False)
    tone = tone_t * amp_map[None, :, :]
    cube = cube * (1.0 + tone)
    meta = {
        "vibration_hz": float(freq_hz),
        "vibration_amp": float(amp),
        "vibration_depth_min_frac": float(depth_min),
        "vibration_depth_decay_frac": float(decay),
    }
    return cube, meta


def _baseline_pd(Icube: np.ndarray, hp_modes: int = 1) -> np.ndarray:
    T, H, W = Icube.shape
    X = Icube.reshape(T, -1).T
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    S_filtered = np.zeros_like(S)
    S_filtered[hp_modes:] = S[hp_modes:]
    Xr = (U * S_filtered) @ Vh
    cube = Xr.T.reshape(T, H, W)
    return np.mean(np.abs(cube) ** 2, axis=0).astype(np.float32)


def _phasecorr_shift(
    ref: np.ndarray, img: np.ndarray, upsample: int = 4
) -> tuple[float, float, float]:
    """Rigid sub-pixel shift between ref and img via phase correlation (dy, dx, psr)."""
    ref_abs = np.abs(ref)
    img_abs = np.abs(img)
    H, W = ref_abs.shape

    F1 = np.fft.fft2(ref_abs)
    F2 = np.fft.fft2(img_abs)
    eps = 1e-12
    cps = F1 * np.conj(F2)
    cps /= np.abs(cps) + eps
    cc = np.fft.ifft2(cps)
    cc = np.abs(cc)

    peak_idx = np.unravel_index(np.argmax(cc), cc.shape)
    p_y, p_x = peak_idx
    if p_y > H // 2:
        p_y -= H
    if p_x > W // 2:
        p_x -= W

    def _quad_refine(arr: np.ndarray, y0: int, x0: int) -> tuple[float, float]:
        y1 = (y0 - 1) % H
        y2 = (y0 + 1) % H
        x1 = (x0 - 1) % W
        x2 = (x0 + 1) % W
        cy1, cy0, cy2 = arr[y1, x0], arr[y0, x0], arr[y2, x0]
        cx1, cx0, cx2 = arr[y0, x1], arr[y0, x0], arr[y0, x2]
        denom_y = cy1 - 2 * cy0 + cy2
        denom_x = cx1 - 2 * cx0 + cx2
        dy = 0.0 if abs(denom_y) < eps else 0.5 * (cy1 - cy2) / denom_y
        dx = 0.0 if abs(denom_x) < eps else 0.5 * (cx1 - cx2) / denom_x
        return float(dy), float(dx)

    sub_dy = sub_dx = 0.0
    if upsample and upsample > 1:
        try:
            sub_dy, sub_dx = _quad_refine(cc, peak_idx[0], peak_idx[1])
        except Exception:
            sub_dy = sub_dx = 0.0

    dy = p_y + sub_dy
    dx = p_x + sub_dx

    cc2 = cc.copy()
    peak_val = cc2[peak_idx]
    cc2[peak_idx] = 0.0
    sidelobe = np.percentile(cc2, 99.0)
    psr = float(peak_val / (sidelobe + eps))
    return float(dy), float(dx), psr


def _fft_shift_apply(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Apply fractional shift to a complex image via Fourier shift theorem."""
    H, W = img.shape
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.fftfreq(W)[None, :]
    phase = np.exp(-2j * np.pi * (ky * dy + kx * dx))
    return np.fft.ifft2(np.fft.fft2(img) * phase).astype(np.complex64)


def _register_stack_phasecorr(
    cube: np.ndarray,
    reg_enable: bool,
    upsample: int,
    ref_strategy: str,
    psr_thresh: float = 3.0,
) -> tuple[np.ndarray, dict]:
    """Register a T×H×W cube frame-wise to reference using phase correlation."""
    T, H, W = cube.shape
    t0 = time.time()
    telemetry = {
        "reg_enable": bool(reg_enable),
        "reg_method": "phasecorr",
        "reg_reference": ref_strategy,
        "reg_failed_fraction": 0.0,
        "reg_shift_rms": 0.0,
        "reg_shift_p90": 0.0,
        "reg_ms": 0.0,
        "reg_psr_median": None,
        "reg_psr_p10": None,
        "reg_psr_p90": None,
    }
    if not reg_enable:
        return cube, telemetry

    if ref_strategy == "median":
        ref = np.median(np.abs(cube), axis=0).astype(np.float32)
    else:
        ref = np.abs(cube[0]).astype(np.float32)

    shifts = []
    psr_vals: list[float] = []
    reg_cube = np.empty_like(cube)
    failures = 0
    for idx in range(T):
        dy, dx, psr = _phasecorr_shift(ref, cube[idx], upsample=upsample)
        psr_vals.append(float(psr) if np.isfinite(psr) else float("nan"))
        if not np.isfinite(psr) or psr < psr_thresh:
            dy = dx = 0.0
            failures += 1
        reg_cube[idx] = _fft_shift_apply(cube[idx], dy, dx)
        shifts.append((dy, dx))

    mags = [math.hypot(dy, dx) for dy, dx in shifts]
    telemetry["reg_failed_fraction"] = float(failures / max(T, 1))
    telemetry["reg_shift_rms"] = float(np.sqrt(np.mean(np.square(mags)))) if mags else 0.0
    telemetry["reg_shift_p90"] = float(np.percentile(mags, 90.0)) if mags else 0.0
    psr_valid = np.array([val for val in psr_vals if np.isfinite(val)], dtype=float)
    if psr_valid.size:
        telemetry["reg_psr_median"] = float(np.median(psr_valid))
        telemetry["reg_psr_p10"] = float(np.quantile(psr_valid, 0.10))
        telemetry["reg_psr_p90"] = float(np.quantile(psr_valid, 0.90))
    telemetry["reg_ms"] = float(1000.0 * (time.time() - t0))
    return reg_cube, telemetry


def _svd_temporal_project(
    A: np.ndarray,
    *,
    rank: Optional[int] = None,
    energy_frac: Optional[float] = None,
    device: str = "cpu",
) -> tuple[np.ndarray, dict]:
    """Project temporal matrix A (T×N) by removing rank-K low-rank component."""
    T, N = A.shape
    rank = int(rank) if rank is not None else None
    if rank is not None:
        rank = max(1, min(rank, T))
    if energy_frac is not None:
        energy_frac = float(np.clip(energy_frac, 0.0, 1.0))

    use_cuda = (
        device and device.lower().startswith("cuda") and _HAS_TORCH and torch.cuda.is_available()
    )
    t0 = time.time()
    if use_cuda:
        At = torch.from_numpy(A)
        if not At.is_cuda:
            At = At.to("cuda")
        C = At @ At.conj().T
        evals, U = torch.linalg.eigh(C)
        idx = torch.argsort(evals, descending=True)
        s2 = torch.clamp(evals[idx].real, min=0.0)
        U = U[:, idx]
        if rank is None and energy_frac is not None:
            cum = torch.cumsum(s2, dim=0) / (torch.sum(s2) + 1e-12)
            rank = int(torch.nonzero(cum >= energy_frac, as_tuple=False)[0].item() + 1)
        rank = rank or 3
        rank = int(max(1, min(rank, T)))
        U_k = U[:, :rank]
        Af = (At - U_k @ (U_k.conj().T @ At)).detach().cpu().numpy().astype(np.complex64)
        s2_vals = s2.detach().cpu().numpy()
        s_top = np.sqrt(s2_vals[: min(5, s2_vals.size)])
    else:
        C = A @ A.conj().T
        s2, U = np.linalg.eigh(C)
        idx = np.argsort(s2)[::-1]
        s2 = np.clip(s2[idx].real, 0.0, None)
        U = U[:, idx]
        if rank is None and energy_frac is not None:
            cum = np.cumsum(s2) / (np.sum(s2) + 1e-12)
            rank = int(np.searchsorted(cum, energy_frac) + 1)
        rank = rank or 3
        rank = int(max(1, min(rank, T)))
        U_k = U[:, :rank]
        Af = (A - U_k @ (U_k.conj().T @ A)).astype(np.complex64)
        s2_vals = s2
        s_top = np.sqrt(s2_vals[: min(5, s2_vals.size)])

    tele = {
        "svd_rank_removed": int(rank),
        "svd_energy_removed_frac": float(np.sum(s2_vals[:rank]) / (np.sum(s2_vals) + 1e-12)),
        "svd_top_singular_vals": [float(x) for x in s_top],
        "svd_ms": float(1000.0 * (time.time() - t0)),
    }
    return Af, tele


def _svd_temporal_keep_range(
    A: np.ndarray,
    *,
    keep_min: int,
    keep_max: Optional[int],
    device: str = "cpu",
) -> tuple[np.ndarray, dict]:
    """Project temporal matrix A (T×N) by keeping a rank range (ULM-style band-pass SVD).

    This keeps singular-vector indices [keep_min, keep_max] (1-based, inclusive) and
    discards the rest. Typical usage is to discard low-rank tissue components and
    high-rank noise, retaining mid-rank microbubble-like content.
    """
    T, N = A.shape
    keep_min = int(keep_min)
    keep_max = int(keep_max) if keep_max is not None else T
    keep_min = max(1, keep_min)
    keep_max = max(keep_min, min(keep_max, T))
    k0 = keep_min - 1
    k1 = keep_max

    use_cuda = (
        device and device.lower().startswith("cuda") and _HAS_TORCH and torch.cuda.is_available()
    )
    t0 = time.time()
    if use_cuda:
        At = torch.from_numpy(A)
        if not At.is_cuda:
            At = At.to("cuda")
        C = At @ At.conj().T
        evals, U = torch.linalg.eigh(C)
        idx = torch.argsort(evals, descending=True)
        s2 = torch.clamp(evals[idx].real, min=0.0)
        U = U[:, idx]
        U_k = U[:, k0:k1]
        if U_k.numel() == 0:
            Af = torch.zeros_like(At)
        else:
            Af = (U_k @ (U_k.conj().T @ At)).detach()
        Af = Af.cpu().numpy().astype(np.complex64)
        s2_vals = s2.detach().cpu().numpy()
        s_top = np.sqrt(s2_vals[: min(5, s2_vals.size)])
    else:
        C = A @ A.conj().T
        s2, U = np.linalg.eigh(C)
        idx = np.argsort(s2)[::-1]
        s2 = np.clip(s2[idx].real, 0.0, None)
        U = U[:, idx]
        U_k = U[:, k0:k1]
        if U_k.size == 0:
            Af = np.zeros_like(A, dtype=np.complex64)
        else:
            Af = (U_k @ (U_k.conj().T @ A)).astype(np.complex64)
        s2_vals = s2
        s_top = np.sqrt(s2_vals[: min(5, s2_vals.size)])

    total = float(np.sum(s2_vals) + 1e-12)
    kept = float(np.sum(s2_vals[k0:k1]) if k1 > k0 else 0.0)
    tele = {
        "svd_keep_min": int(keep_min),
        "svd_keep_max": int(keep_max),
        "svd_rank_kept": int(k1 - k0),
        "svd_energy_kept_frac": float(kept / total),
        "svd_energy_removed_frac": float(1.0 - kept / total),
        "svd_top_singular_vals": [float(x) for x in s_top],
        "svd_ms": float(1000.0 * (time.time() - t0)),
    }
    return Af, tele


def _baseline_pd_mcsvd(
    Icube: np.ndarray,
    *,
    reg_enable: bool = True,
    reg_method: str = "phasecorr",
    reg_subpixel: int = 4,
    reg_reference: str = "median",
    svd_rank: Optional[int] = None,
    svd_energy_frac: Optional[float] = None,
    device: str = "cpu",
    return_filtered_cube: bool = False,
) -> tuple:
    """Motion-compensated SVD baseline."""
    t_start = time.time()
    if reg_method != "phasecorr":
        raise ValueError("Only phasecorr registration is supported for MC-SVD baseline.")
    reg_cube, tele_reg = _register_stack_phasecorr(
        Icube, reg_enable=reg_enable, upsample=reg_subpixel, ref_strategy=reg_reference
    )
    A = reg_cube.reshape(reg_cube.shape[0], -1)
    A_f, tele_svd = _svd_temporal_project(
        A, rank=svd_rank, energy_frac=svd_energy_frac, device=device
    )
    A_f = A_f.reshape(reg_cube.shape)
    pd = np.mean((np.abs(A_f) ** 2).astype(np.float32), axis=0)
    telemetry = {
        "baseline_type": "mc_svd",
        **tele_reg,
        **tele_svd,
        "baseline_ms": float(1000.0 * (time.time() - t_start)),
    }
    if return_filtered_cube:
        return pd.astype(np.float32), telemetry, A_f.astype(np.complex64, copy=False)
    return pd.astype(np.float32), telemetry


def _baseline_pd_svd_bandpass(
    Icube: np.ndarray,
    *,
    reg_enable: bool = False,
    reg_method: str = "phasecorr",
    reg_subpixel: int = 4,
    reg_reference: str = "median",
    svd_keep_min: int = 1,
    svd_keep_max: Optional[int] = None,
    device: str = "cpu",
    return_filtered_cube: bool = False,
) -> tuple:
    """SVD band-pass baseline (ULM-style): keep singular components in a range."""
    t_start = time.time()
    if reg_method != "phasecorr":
        raise ValueError("Only phasecorr registration is supported for SVD baselines.")
    reg_cube, tele_reg = _register_stack_phasecorr(
        Icube, reg_enable=reg_enable, upsample=reg_subpixel, ref_strategy=reg_reference
    )
    A = reg_cube.reshape(reg_cube.shape[0], -1)
    A_f, tele_svd = _svd_temporal_keep_range(
        A, keep_min=svd_keep_min, keep_max=svd_keep_max, device=device
    )
    A_f = A_f.reshape(reg_cube.shape)
    pd = np.mean((np.abs(A_f) ** 2).astype(np.float32), axis=0)
    telemetry = {
        "baseline_type": "svd_bandpass",
        **tele_reg,
        **tele_svd,
        "baseline_ms": float(1000.0 * (time.time() - t_start)),
    }
    if return_filtered_cube:
        return pd.astype(np.float32), telemetry, A_f.astype(np.complex64, copy=False)
    return pd.astype(np.float32), telemetry


def _baseline_pd_rpca(
    Icube: np.ndarray,
    *,
    lambda_: Optional[float] = None,
    max_iters: int = 250,
) -> tuple[np.ndarray, dict]:
    """Robust PCA baseline (low-rank + sparse) with dimensionality control.

    Applies windowed + downsampled + truncated-SVD PCP (IALM) to recover a
    low-rank background and sparse foreground component and defines PD from
    the sparse part. Tuned for k-Wave brain fUS–scale problems but used universally.
    """
    t_start = time.time()
    T, H, W = Icube.shape

    # Use windowed + downsampled + truncated SVD PCP.
    # Internal hyperparameters (tuned for k-Wave brain fUS–like regimes).
    spatial_downsample = 2 if max(H, W) > 128 else 1
    t_sub_default = 80
    tol = 1e-4
    max_iters_inner = min(max_iters, 80)

    def _avg_pool2d_complex(cube: np.ndarray, d: int) -> np.ndarray:
        if d <= 1:
            return cube
        T_loc, HH, WW = cube.shape
        HH2 = (HH // d) * d
        WW2 = (WW // d) * d
        if HH2 != HH or WW2 != WW:
            cube = cube[:, :HH2, :WW2]
        cube_reshaped = cube.reshape(T_loc, HH2 // d, d, WW2 // d, d)
        return cube_reshaped.mean(axis=(2, 4))

    def _upsample_nn(pd_ds: np.ndarray, d: int, out_shape: tuple[int, int]) -> np.ndarray:
        if d <= 1:
            return pd_ds
        H_out, W_out = out_shape
        tiled = np.kron(pd_ds, np.ones((d, d), dtype=pd_ds.dtype))
        return tiled[:H_out, :W_out]

    def _randomized_svd(
        M: np.ndarray,
        rank_k: int,
        n_oversamples: int = 2,
        n_iter: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m, n = M.shape
        k = min(rank_k, m, n)
        if k <= 0:
            return (
                np.zeros((m, 0), dtype=M.dtype),
                np.zeros((0,), dtype=M.real.dtype),
                np.zeros((0, n), dtype=M.dtype),
            )
        dtype_real = np.float32 if M.dtype in (np.complex64, np.float32) else np.float64
        Omega_real = np.random.randn(n, k + n_oversamples).astype(dtype_real)
        Omega_imag = np.random.randn(n, k + n_oversamples).astype(dtype_real)
        Omega = Omega_real + 1j * Omega_imag
        Y = M @ Omega
        for _ in range(max(1, n_iter)):
            Y = M @ (M.conj().T @ Y)
        Q, _ = np.linalg.qr(Y, mode="reduced")
        B = Q.conj().T @ M
        Ub, s, Vh = np.linalg.svd(B, full_matrices=False)
        U = Q @ Ub[:, :k]
        s = s[:k].astype(dtype_real)
        Vh = Vh[:k, :].astype(M.dtype)
        return U.astype(M.dtype), s, Vh

    def _svt_truncated(
        M: np.ndarray,
        tau: float,
        rank_k: Optional[int],
        svd_full_threshold: int = 1_000_000,
    ) -> tuple[np.ndarray, np.ndarray, int, bool]:
        m, n = M.shape
        use_full = rank_k is None or rank_k >= min(m, n) or m * n <= svd_full_threshold
        if use_full:
            U, s, Vh = np.linalg.svd(M, full_matrices=False)
        else:
            U, s, Vh = _randomized_svd(M, rank_k=rank_k)
        s_thr = np.maximum(s - tau, 0.0)
        r_eff = int(np.count_nonzero(s_thr > 0))
        if r_eff == 0:
            L_loc = np.zeros_like(M, dtype=M.dtype)
        else:
            L_loc = (U[:, :r_eff] * s_thr[:r_eff]) @ Vh[:r_eff, :]
        return L_loc.astype(M.dtype), s, r_eff, use_full

    def _rpca_ialm_window(
        X: np.ndarray,
        *,
        lambda_: Optional[float],
        max_iters: int,
        rank_k: Optional[int],
        tol: float,
    ) -> tuple[np.ndarray, dict]:
        m, n = X.shape
        normX_F = np.linalg.norm(X)
        normX_2 = np.linalg.norm(X, ord=2)
        lam_loc = lambda_ if lambda_ is not None else 1.0 / np.sqrt(max(m, n))
        mu = 0.5 / (normX_2 + 1e-12)
        L_loc = np.zeros_like(X, dtype=np.complex64)
        S_loc = np.zeros_like(X, dtype=np.complex64)
        Y_loc = np.zeros_like(X, dtype=np.complex64)
        tau = 1.0 / mu
        final_iter = 0
        eff_ranks: list[int] = []
        used_full_svd: list[bool] = []
        last_rel_res = float("inf")
        for it in range(max_iters):
            final_iter = it
            M_loc = X - S_loc + (1.0 / mu) * Y_loc
            L_loc, svals, r_eff, used_full = _svt_truncated(M_loc, tau, rank_k=rank_k)
            eff_ranks.append(r_eff)
            used_full_svd.append(used_full)
            R_loc = X - L_loc + (1.0 / mu) * Y_loc
            S_real = np.sign(R_loc.real) * np.maximum(np.abs(R_loc.real) - lam_loc / mu, 0.0)
            S_imag = np.sign(R_loc.imag) * np.maximum(np.abs(R_loc.imag) - lam_loc / mu, 0.0)
            S_loc = (S_real + 1j * S_imag).astype(np.complex64)
            Z_loc = X - L_loc - S_loc
            Y_loc = Y_loc + mu * Z_loc
            rel_res = np.linalg.norm(Z_loc) / (normX_F + 1e-12)
            last_rel_res = float(rel_res)
            if rel_res < tol:
                break
        L_energy = float(np.linalg.norm(L_loc) ** 2)
        X_energy = float(normX_F**2)
        tele_loc = {
            "iter": int(final_iter + 1),
            "lambda": float(lam_loc),
            "mu": float(mu),
            "rank_eff_mean": float(np.mean(eff_ranks) if eff_ranks else 0.0),
            "rank_eff_last": int(eff_ranks[-1]) if eff_ranks else 0,
            "frac_full_svd": float(np.mean(used_full_svd) if used_full_svd else 0.0),
            "residual_rel": last_rel_res,
            "L_energy": L_energy,
            "X_energy": X_energy,
        }
        return S_loc.astype(np.complex64), tele_loc

    # Spatial downsampling (complex average pooling).
    I_ds = _avg_pool2d_complex(Icube, spatial_downsample)
    T_ds, H_ds, W_ds = I_ds.shape

    # Temporal windows.
    t_sub = min(t_sub_default, T_ds)
    windows: list[tuple[int, int]] = []
    t0 = 0
    while t0 < T_ds:
        t1 = min(t0 + t_sub, T_ds)
        windows.append((t0, t1))
        t0 = t1

    # Accumulate sparse energy over time on downsampled grid.
    S_energy_sum = np.zeros((H_ds, W_ds), dtype=np.float64)

    total_L_energy = 0.0
    total_X_energy = 0.0
    iters_per_win: list[int] = []
    rank_eff_per_win: list[int] = []
    frac_full_svd_per_win: list[float] = []
    residuals: list[float] = []

    for w0, w1 in windows:
        I_win = I_ds[w0:w1]
        T_w = w1 - w0
        X_win = I_win.reshape(T_w, -1).astype(np.complex64)
        # Rank heuristic per window.
        rank_k = min(8, max(2, T_w // 10))
        S_flat, tele_win = _rpca_ialm_window(
            X_win,
            lambda_=lambda_,
            max_iters=max_iters_inner,
            rank_k=rank_k,
            tol=tol,
        )
        S_win = S_flat.reshape(T_w, H_ds, W_ds)
        S_energy_sum += np.sum(np.abs(S_win) ** 2, axis=0)
        iters_per_win.append(tele_win["iter"])
        rank_eff_per_win.append(tele_win["rank_eff_last"])
        frac_full_svd_per_win.append(tele_win["frac_full_svd"])
        residuals.append(tele_win["residual_rel"])
        total_L_energy += tele_win["L_energy"]
        total_X_energy += tele_win["X_energy"]

    # Time-averaged PD on downsampled grid and upsampled to full resolution.
    pd_ds = (S_energy_sum / float(T_ds)).astype(np.float32)
    pd_full = _upsample_nn(pd_ds, spatial_downsample, (H, W)).astype(np.float32)

    baseline_ms = float(1000.0 * (time.time() - t_start))
    energy_frac = float(total_L_energy / (total_X_energy + 1e-12)) if total_X_energy > 0.0 else 0.0

    tele = {
        "baseline_type": "rpca",
        "rpca_params": {
            "lambda": float(lambda_) if lambda_ is not None else None,
            "lambda_default_formula": "1/sqrt(max(m,n)) if lambda_ is None",
            "mu_init_factor": 0.5,
            "t_sub": int(t_sub),
            "spatial_downsample": int(spatial_downsample),
            "rank_k_max": 8,
            "max_iters_inner": int(max_iters_inner),
            "tol": float(tol),
            "n_windows": len(windows),
        },
        "rpca_iter": int(max(iters_per_win) if iters_per_win else 0),
        "rpca_iter_mean": float(np.mean(iters_per_win) if iters_per_win else 0.0),
        "rpca_rank_eff_mean": float(np.mean(rank_eff_per_win) if rank_eff_per_win else 0.0),
        "rpca_frac_full_svd_mean": float(
            np.mean(frac_full_svd_per_win) if frac_full_svd_per_win else 0.0
        ),
        "rpca_residual_rel_mean": float(np.mean(residuals) if residuals else 0.0),
        "rpca_energy_lowrank_frac": energy_frac,
        "baseline_ms": baseline_ms,
    }
    return pd_full, tele


def _hosvd_spatial_downsample_complex(
    x: np.ndarray,
    factor: int,
) -> np.ndarray:
    """Complex spatial average pooling for (T, H, W) cubes."""
    if factor <= 1:
        return x
    if x.ndim != 3:
        raise ValueError(f"_hosvd_spatial_downsample_complex expects (T,H,W), got {x.shape}")
    T_loc, H_loc, W_loc = x.shape
    if H_loc % factor != 0 or W_loc % factor != 0:
        raise ValueError(
            "spatial_downsample requires H and W divisible by factor; "
            f"got spatial_downsample={factor}, H={H_loc}, W={W_loc}"
        )
    H_ds = H_loc // factor
    W_ds = W_loc // factor
    x_reshaped = x.reshape(T_loc, H_ds, factor, W_ds, factor)
    return x_reshaped.mean(axis=(2, 4))


def _hosvd_choose_rank(
    s: np.ndarray,
    rank_req: int | None,
    energy_frac_req: float | None,
) -> tuple[int, float]:
    """Choose rank from singular values and return (rank, energy_fraction_kept)."""
    s = np.asarray(s)
    s2 = s**2
    total = float(s2.sum())
    if total <= 0.0:
        r = max(1, int(rank_req) if rank_req is not None else 1)
        return r, 0.0
    if rank_req is not None and rank_req > 0:
        r = int(rank_req)
        r = max(1, min(r, s.shape[0]))
    elif energy_frac_req is not None:
        f_req = float(energy_frac_req)
        if not 0.0 < f_req <= 1.0:
            f_req = 0.99
        cumsum = np.cumsum(s2)
        cutoff = f_req * total
        r = int(np.searchsorted(cumsum, cutoff) + 1)
    else:
        cumsum = np.cumsum(s2)
        cutoff = 0.99 * total
        r = int(np.searchsorted(cumsum, cutoff) + 1)
    r = max(1, min(r, s.shape[0]))
    energy_kept = float(s2[:r].sum() / (total + 1e-12))
    return r, energy_kept


def _hosvd_compute_factors(
    X: np.ndarray,
    ranks: tuple[int, int, int] | None,
    energy_fracs: tuple[float, float, float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int], tuple[float, float, float]]:
    """Compute mode-0/1/2 factors for HOSVD on a (T, H, W) tensor."""
    if X.ndim != 3:
        raise ValueError(f"_hosvd_compute_factors expects (T,H,W), got {X.shape}")
    T_loc, H_loc, W_loc = X.shape
    r_t_req: int | None
    r_h_req: int | None
    r_w_req: int | None
    if ranks is not None:
        r_t_req, r_h_req, r_w_req = ranks
    else:
        r_t_req = r_h_req = r_w_req = None
    if energy_fracs is None and ranks is None:
        energy_fracs = (0.99, 0.99, 0.99)
    if energy_fracs is not None:
        e_t_req, e_h_req, e_w_req = energy_fracs
    else:
        e_t_req = e_h_req = e_w_req = None
    # Mode-0 SVD: (T, H*W)
    X0 = X.reshape(T_loc, H_loc * W_loc)
    U_t_full, s_t, _ = np.linalg.svd(X0, full_matrices=False)
    r_t, e_t = _hosvd_choose_rank(s_t, r_t_req, e_t_req)
    U_t = U_t_full[:, :r_t]
    # Mode-1 SVD: (H, T*W)
    X1 = np.moveaxis(X, 1, 0)
    X1_mat = X1.reshape(H_loc, T_loc * W_loc)
    U_h_full, s_h, _ = np.linalg.svd(X1_mat, full_matrices=False)
    r_h, e_h = _hosvd_choose_rank(s_h, r_h_req, e_h_req)
    U_h = U_h_full[:, :r_h]
    # Mode-2 SVD: (W, T*H)
    X2 = np.moveaxis(X, 2, 0)
    X2_mat = X2.reshape(W_loc, T_loc * H_loc)
    U_w_full, s_w, _ = np.linalg.svd(X2_mat, full_matrices=False)
    r_w, e_w = _hosvd_choose_rank(s_w, r_w_req, e_w_req)
    U_w = U_w_full[:, :r_w]
    ranks_used = (r_t, r_h, r_w)
    energy_used = (e_t, e_h, e_w)
    return U_t, U_h, U_w, ranks_used, energy_used


def _hosvd_project_lowrank(
    X: np.ndarray,
    U_t: np.ndarray,
    U_h: np.ndarray,
    U_w: np.ndarray,
) -> np.ndarray:
    """Apply orthogonal projections along T, H, W to get low-rank approximation."""
    if X.ndim != 3:
        raise ValueError(f"_hosvd_project_lowrank expects (T,H,W), got {X.shape}")
    T_loc, H_loc, W_loc = X.shape
    # Mode-0 (time) projection
    L0 = X.reshape(T_loc, H_loc * W_loc)
    tmp = U_t.conj().T @ L0
    L0_low = U_t @ tmp
    L = L0_low.reshape(T_loc, H_loc, W_loc)
    # Mode-1 (height) projection
    L1 = np.moveaxis(L, 1, 0)
    L1_mat = L1.reshape(H_loc, T_loc * W_loc)
    tmp = U_h.conj().T @ L1_mat
    L1_low = U_h @ tmp
    L1 = L1_low.reshape(H_loc, T_loc, W_loc)
    L = np.moveaxis(L1, 0, 1)
    # Mode-2 (width) projection
    L2 = np.moveaxis(L, 2, 0)
    L2_mat = L2.reshape(W_loc, T_loc * H_loc)
    tmp = U_w.conj().T @ L2_mat
    L2_low = U_w @ tmp
    L2 = L2_low.reshape(W_loc, T_loc, H_loc)
    L = np.moveaxis(L2, 0, 2)
    return L


def _baseline_pd_hosvd(
    Icube: np.ndarray,
    *,
    ranks: tuple[int, int, int] | None = None,
    energy_fracs: tuple[float, float, float] | None = None,
    max_iters: int = 1,
    spatial_downsample: int = 1,
    t_sub: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Tensor HOSVD baseline (low-rank clutter + residual flow energy)."""
    t_start = time.perf_counter()
    if Icube.ndim != 3:
        raise ValueError(f"_baseline_pd_hosvd expects (T,H,W), got {Icube.shape}")
    T_full, H_full, W_full = Icube.shape
    X = Icube.astype(np.complex64, copy=False)
    if spatial_downsample > 1:
        X_ds = _hosvd_spatial_downsample_complex(X, spatial_downsample)
    else:
        X_ds = X
    T_ds, H_ds, W_ds = X_ds.shape
    ranks_arg = ranks
    energy_fracs_arg = None if ranks_arg is not None else energy_fracs
    X_energy_total = 0.0
    L_energy_total = 0.0
    pd_energy_ds = np.zeros((H_ds, W_ds), dtype=np.float64)
    ranks_used: tuple[int, int, int] | None = None
    energy_used_modes: tuple[float, float, float] | None = None
    if t_sub is None or t_sub >= T_ds:
        U_t, U_h, U_w, ranks_used, energy_used_modes = _hosvd_compute_factors(
            X_ds,
            ranks=ranks_arg,
            energy_fracs=energy_fracs_arg,
        )
        L_ds = _hosvd_project_lowrank(X_ds, U_t, U_h, U_w)
        residual = X_ds - L_ds
        pd_energy_ds[...] = (np.abs(residual) ** 2.0).sum(axis=0)
        X_energy_total = float((np.abs(X_ds) ** 2.0).sum())
        L_energy_total = float((np.abs(L_ds) ** 2.0).sum())
        T_eff = T_ds
    else:
        T_eff = T_ds
        first_window = True
        start = 0
        while start < T_ds:
            stop = min(start + t_sub, T_ds)
            X_sub = X_ds[start:stop]
            U_t, U_h, U_w, ranks_win, energy_modes_win = _hosvd_compute_factors(
                X_sub,
                ranks=ranks_arg,
                energy_fracs=energy_fracs_arg,
            )
            if first_window:
                ranks_used = ranks_win
                energy_used_modes = energy_modes_win
                first_window = False
            L_sub = _hosvd_project_lowrank(X_sub, U_t, U_h, U_w)
            residual_sub = X_sub - L_sub
            pd_energy_ds += (np.abs(residual_sub) ** 2.0).sum(axis=0)
            X_energy_total += float((np.abs(X_sub) ** 2.0).sum())
            L_energy_total += float((np.abs(L_sub) ** 2.0).sum())
            start = stop
        if ranks_used is None:
            ranks_used = (1, 1, 1)
            energy_used_modes = (0.0, 0.0, 0.0)
    if ranks_used is None:
        ranks_used = (1, 1, 1)
        energy_used_modes = (0.0, 0.0, 0.0)
    pd_base_ds = (pd_energy_ds / float(T_eff)).astype(np.float32)
    if spatial_downsample > 1:
        pd_full = np.repeat(
            np.repeat(pd_base_ds, spatial_downsample, axis=0),
            spatial_downsample,
            axis=1,
        )
        pd_full = pd_full[:H_full, :W_full]
    else:
        pd_full = pd_base_ds
    pd_full = pd_full.astype(np.float32, copy=False)
    if X_energy_total > 0.0:
        energy_frac_lowrank = float(L_energy_total / (X_energy_total + 1e-12))
    else:
        energy_frac_lowrank = 0.0
    baseline_ms = float((time.perf_counter() - t_start) * 1000.0)
    tele: dict[str, object] = {
        "baseline_type": "hosvd",
        "hosvd_params": {
            "ranks": tuple(int(r) for r in ranks_used),
            "energy_fracs": tuple(float(e) for e in energy_used_modes),
            "spatial_downsample": int(spatial_downsample),
            "t_sub": int(t_sub) if t_sub is not None else int(T_ds),
            "max_iters": int(max_iters),
        },
        "hosvd_iters": 1,
        "hosvd_energy_lowrank_frac": float(energy_frac_lowrank),
        "baseline_ms": baseline_ms,
    }
    return pd_full, tele


def _psd_peak_and_span_hz(
    cube_T_hw: np.ndarray,
    prf_hz: float,
    min_rel: float = 0.03,
    max_rel: float = 0.49,
    energy_q: float = 0.90,
) -> tuple[float | None, float | None]:
    T = cube_T_hw.shape[0]
    if T < 16:
        return None, None
    series = cube_T_hw.mean(axis=(1, 2)).astype(np.complex64)
    series = series - series.mean()
    win = np.hanning(T).astype(np.float32)
    spectrum = np.fft.fft(series * win)
    freqs = np.fft.fftfreq(T, d=1.0 / prf_hz)
    psd = (spectrum.conj() * spectrum).real
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)

    fmin = min_rel * prf_hz
    fmax = max_rel * prf_hz
    mask = (np.abs(freqs) >= fmin) & (np.abs(freqs) <= fmax)
    if not np.any(mask):
        return None, None
    psd_win = psd[mask]
    if np.all(psd_win <= 0):
        return None, None

    idx_local = int(np.argmax(psd_win))
    freqs_win = freqs[mask]
    f_peak = float(freqs_win[idx_local])
    peak_power = float(psd_win[idx_local])

    idxs = np.where(mask)[0]
    idx_peak = idxs[idx_local]
    thr = 0.25 * peak_power
    left = idx_peak
    while left > idxs[0] and psd[left] > thr:
        left -= 1
    right = idx_peak
    while right < idxs[-1] and psd[right] > thr:
        right += 1
    span_thresh = 0.5 * (freqs[right] - freqs[left])

    window = max(4, int(0.1 * np.sum(mask)))
    lo = max(idxs[0], idx_peak - window)
    hi = min(idxs[-1], idx_peak + window)
    local_psd = psd[lo : hi + 1]
    total = float(local_psd.sum())
    if total <= 0:
        span_energy = span_thresh
    else:
        target = energy_q * total
        wl = wr = 0
        acc = float(psd[idx_peak])
        while (idx_peak - wl > lo or idx_peak + wr < hi) and acc < target:
            next_left = psd[idx_peak - wl - 1] if idx_peak - wl - 1 >= lo else -1.0
            next_right = psd[idx_peak + wr + 1] if idx_peak + wr + 1 <= hi else -1.0
            if next_right >= next_left:
                wr += 1
                acc += max(next_right, 0.0)
            else:
                wl += 1
                acc += max(next_left, 0.0)
        span_energy = 0.5 * (freqs[min(idx_peak + wr, hi)] - freqs[max(idx_peak - wl, lo)])

    span = max(span_thresh, span_energy) * 1.1
    span = max(span, abs(f_peak) * 1.2)
    span = float(np.clip(span, 0.05 * prf_hz, max_rel * prf_hz))
    return f_peak, span


def _doppler_psd_summary(
    cube_T_hw: np.ndarray,
    prf_hz: float,
    targets_hz: Sequence[float] | None = None,
    top_k: int = 3,
) -> dict[str, object]:
    """
    Return simple PSD diagnostics for a slow-time cube.

    Parameters
    ----------
    cube_T_hw : np.ndarray
        Slow-time tile (T, h, w).
    prf_hz : float
        Pulse repetition frequency.
    targets_hz : optional sequence
        Frequencies of interest (absolute Hz). Both +/- collapse to same bin.
    top_k : int
        Number of peak frequencies to report.
    """
    if prf_hz <= 0.0:
        return {}
    T = cube_T_hw.shape[0]
    if T < 8:
        return {}
    w = cube_T_hw.shape[2]
    energy = np.sum(np.abs(cube_T_hw) ** 2, axis=0)
    idx_max = int(np.argmax(energy))
    y_max, x_max = divmod(idx_max, w)
    series = cube_T_hw[:, y_max, x_max].astype(np.complex64)
    if not np.any(np.abs(series) > 0):
        series = cube_T_hw.reshape(T, -1).mean(axis=1).astype(np.complex64)
    series = series - series.mean()
    if not np.any(np.abs(series) > 0):
        return {}
    win = np.hanning(T).astype(np.float32)
    spectrum = np.fft.fft(series * win)
    freqs = np.fft.fftfreq(T, d=1.0 / prf_hz)
    psd = (spectrum.conj() * spectrum).real
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    if not np.any(psd > 0):
        return {}

    summary: dict[str, object] = {}
    idx_peak = int(np.argmax(psd))
    summary["psd_peak_hz"] = float(freqs[idx_peak])
    summary["psd_peak_power"] = float(psd[idx_peak])

    if top_k > 0:
        k = int(min(top_k, psd.size))
        top_idx = np.argsort(psd)[-k:][::-1]
        summary["psd_top_freqs_hz"] = [float(freqs[i]) for i in top_idx.tolist()]
        summary["psd_top_power"] = [float(psd[i]) for i in top_idx.tolist()]
        summary["psd_top_freqs_abs_hz"] = [abs(float(freqs[i])) for i in top_idx.tolist()]

    tgt_list = list(targets_hz or [])
    flow_candidates: list[tuple[float, float, str]] = []
    if 0.0 not in tgt_list:
        tgt_list.append(0.0)
    for target in tgt_list:
        freq_abs = abs(float(target))
        idx = int(np.argmin(np.abs(freqs - freq_abs)))
        key = "psd_power_0_hz" if freq_abs < 1e-6 else f"psd_power_{int(round(freq_abs))}_hz"
        val = float(psd[idx])
        if freq_abs > 1e-6:
            idx_neg = int(np.argmin(np.abs(freqs + freq_abs)))
            val = float(max(val, psd[idx_neg]))
            if val > 0.0:
                flow_candidates.append((val, float(freq_abs), "target"))
        summary[key] = val

    top_freqs = summary.get("psd_top_freqs_hz")
    top_power = summary.get("psd_top_power")
    if isinstance(top_freqs, list) and isinstance(top_power, list):
        for freq, power in zip(top_freqs, top_power, strict=False):
            freq_abs = abs(float(freq))
            power_val = float(power)
            if freq_abs > 0.0 and power_val > 0.0:
                flow_candidates.append((power_val, freq_abs, "top"))

    best_tuple: tuple[float, float, str] | None = None
    best_priority: tuple[float, int, float] | None = None
    for power_val, freq_abs, source in flow_candidates:
        if freq_abs <= 0.0 or not np.isfinite(power_val):
            continue
        priority = (power_val, 1 if source == "target" else 0, -float(freq_abs))
        if best_priority is None or priority > best_priority:
            best_priority = priority
            best_tuple = (power_val, freq_abs, source)

    power_dc = summary.get("psd_power_0_hz")
    if best_tuple is not None:
        best_power, best_freq, best_source = best_tuple
        summary["psd_power_flow"] = float(best_power)
        summary["psd_power_flow_hz"] = float(best_freq)
        summary["psd_flow_freq_source"] = best_source
    if isinstance(power_dc, (float, int)):
        summary["psd_power_dc"] = float(power_dc)
        flow_power_val = summary.get("psd_power_flow")
        if isinstance(flow_power_val, (float, int)):
            denom = power_dc if power_dc > 0 else 1e-12
            summary["psd_flow_to_dc_ratio"] = float(flow_power_val / denom)

    return summary


def _cap_kc_from_rank(R: torch.Tensor, min_pts: int, max_pts: int, safety: float = 3.0) -> int:
    """
    Choose a constraint cap based on the effective rank of R, but never exceed Lt.
    """
    evals = torch.linalg.eigvalsh(R).real.clamp_min(0.0)
    trace = float(torch.sum(evals).item())
    trace_sq = float(torch.sum(evals * evals).item())
    Lt = int(R.shape[-1])
    if Lt <= 0:
        return 1
    max_allowed = Lt if Lt % 2 == 1 else max(Lt - 1, 1)

    if trace_sq <= 0.0:
        kc = max_allowed
    else:
        r_eff = (trace * trace) / (trace_sq + 1e-30)
        kc = int(np.floor(r_eff / max(safety, 1.0)))

    kc = max(1, kc)
    kc = min(kc, max_allowed)
    kc = min(kc, max_pts)
    if kc % 2 == 0:
        kc = max(1, kc - 1)
    # If min_pts exceeds allowable, fall back to the capped value; otherwise ensure >= min_pts
    if min_pts <= max_allowed:
        kc = max(kc, 1)
        if kc < min_pts:
            kc = min_pts if (min_pts <= max_allowed and min_pts <= max_pts) else kc
            if kc % 2 == 0:
                kc = max(1, kc - 1)
            kc = min(kc, max_allowed)
    return max(1, kc)


def _shrinkage_alpha_for_kappa(R: torch.Tensor, kappa_target: float) -> float:
    """
    Return shrinkage weight alpha in [0,1] that slides R toward mu*I to reach cond <= kappa_target.
    Alpha=0 => no shrink; Alpha=1 => fully isotropic.
    """
    if kappa_target <= 1.0:
        return 0.0
    evals = torch.linalg.eigvalsh(R).real
    mu = float(torch.mean(evals).item())
    e_min = float(torch.min(evals).item())
    e_max = float(torch.max(evals).item())
    num = e_max - kappa_target * e_min
    den = (kappa_target - 1.0) * mu + (e_max - kappa_target * e_min)
    if den <= 0.0:
        return 0.0
    alpha = num / den
    return float(np.clip(alpha, 0.0, 1.0))


def _ka_gate_decision(
    alias_ratio: float | None,
    flow_cov: float | None,
    depth_frac: float | None,
    *,
    alias_rmin: float,
    flow_cov_min: float,
    flow_cov_is_upper_bound: bool = False,
    depth_min_frac: float,
    depth_max_frac: float,
    pd_metric: float | None = None,
    pd_min: float | None = None,
    pd_q_lo: float | None = None,
    pd_q_hi: float | None = None,
    reg_psr: float | None = None,
    reg_psr_max: float | None = None,
) -> tuple[bool, bool, bool, bool, bool, bool]:
    """Return (gate_ok, alias_ok, flow_ok, depth_ok, pd_ok, reg_ok)."""
    alias_ok = (
        alias_ratio is not None
        and np.isfinite(alias_ratio)
        and float(alias_ratio) >= float(alias_rmin)
    )
    # Legacy mode: flow_cov_min is a lower bound on coverage.
    # Updated mode (flow_cov_is_upper_bound=True) interprets flow_cov_min as an
    # upper bound and vetoes tiles whose flow coverage exceeds this value.
    if flow_cov is None or not np.isfinite(flow_cov):
        if flow_cov_is_upper_bound:
            # Missing coverage information should not veto by itself in upper-bound mode.
            flow_ok = True
        else:
            flow_ok = float(flow_cov_min) <= 0.0
    else:
        cov_val = float(flow_cov)
        if flow_cov_is_upper_bound:
            flow_ok = cov_val <= float(flow_cov_min)
        else:
            flow_ok = cov_val >= float(flow_cov_min)
    if depth_frac is None or not np.isfinite(depth_frac):
        depth_ok = True
    else:
        depth_val = float(depth_frac)
        depth_ok = float(depth_min_frac) <= depth_val <= float(depth_max_frac)
    pd_ok = True
    if pd_metric is not None and np.isfinite(pd_metric):
        pd_val = float(pd_metric)
    else:
        pd_val = None
    if pd_min is not None:
        if pd_val is None:
            pd_ok = False
        else:
            pd_ok = pd_val >= float(pd_min)
    # Optional score-quantile window: require pd_q_lo <= pd_metric <= pd_q_hi when provided.
    if pd_ok and (pd_q_lo is not None or pd_q_hi is not None):
        if pd_val is None:
            pd_ok = False
        else:
            if pd_q_lo is not None and pd_val < float(pd_q_lo):
                pd_ok = False
            if pd_q_hi is not None and pd_val > float(pd_q_hi):
                pd_ok = False
    reg_ok = True
    if reg_psr_max is not None and reg_psr is not None and np.isfinite(reg_psr):
        reg_ok = float(reg_psr) <= float(reg_psr_max)
    gate_ok = alias_ok and flow_ok and depth_ok and pd_ok and reg_ok
    return gate_ok, alias_ok, flow_ok, depth_ok, pd_ok, reg_ok


def _directional_trace_ratio(
    R_sample: torch.Tensor,
    R_loaded: torch.Tensor,
    Pf: torch.Tensor,
) -> tuple[float | None, float | None]:
    """Return (gamma_flow, gamma_perp) between loaded and sample covariances."""
    try:
        Pfh = 0.5 * (Pf + Pf.conj().transpose(-2, -1))
        eye = torch.eye(Pfh.shape[-1], dtype=Pfh.dtype, device=Pfh.device)
        P_perp = eye - Pfh
        R_ref = 0.5 * (R_sample + R_sample.conj().transpose(-2, -1))
        R_l = 0.5 * (R_loaded + R_loaded.conj().transpose(-2, -1))
        tr_ref_flow = torch.real(torch.trace(Pfh @ R_ref @ Pfh))
        tr_loaded_flow = torch.real(torch.trace(Pfh @ R_l @ Pfh))
        tr_ref_perp = torch.real(torch.trace(P_perp @ R_ref @ P_perp))
        tr_loaded_perp = torch.real(torch.trace(P_perp @ R_l @ P_perp))
        gamma_flow = float(tr_loaded_flow / (tr_ref_flow + 1e-12))
        gamma_perp = float(tr_loaded_perp / (tr_ref_perp + 1e-12))
        return gamma_flow, gamma_perp
    except Exception:
        return None, None


def _stap_pd_tile_lcmv(
    cube_T_hw: np.ndarray,
    prf_hz: float,
    diag_load: float,
    cov_estimator: str,
    huber_c: float,
    mvdr_load_mode: str = "auto",
    mvdr_auto_kappa: float = 50.0,
    constraint_ridge: float = 0.10,
    msd_lambda: float | None = None,
    msd_ridge: float = 0.10,
    msd_agg_mode: str = "trim10",
    msd_ratio_rho: float = 0.0,
    motion_half_span_rel: float | None = None,
    msd_contrast_alpha: float | None = None,
    fd_span_mode: str = "psd",
    fd_span_rel: tuple[float, float] = (0.30, 1.10),
    fd_fixed_span_hz: float | None = None,
    constraint_mode: str = "exp+deriv",
    grid_step_rel: float = 0.05,
    min_pts: int = 9,
    max_pts: int = 21,
    fd_min_abs_hz: float = 0.0,
    alias_psd_select_enable: bool = False,
    alias_psd_select_ratio_thresh: float = 1.2,
    alias_psd_select_bins: int = 1,
    capture_debug: bool = False,
    device: str | None = None,
    cube_tensor: torch.Tensor | None = None,
    *,
    ka_mode: str = "none",
    ka_prior_library: np.ndarray | torch.Tensor | None = None,
    ka_opts: dict[str, float] | None = None,
    Lt_fixed: int | None = None,
    post_filter_callback: Callable[[np.ndarray], None] | None = None,
    ka_gate: dict | None = None,
    feasibility_mode: FeasibilityMode = "legacy",
    motion_basis_geom: np.ndarray | None = None,
    alias_band_hz: tuple[float, float] | None = None,
    psd_telemetry: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict, dict | None]:
    T, h, w = cube_T_hw.shape
    feas_mode = _normalize_feasibility_mode(feasibility_mode)
    info: dict = {}
    info["feasibility_mode"] = feas_mode
    debug_payload: dict | None = None
    lcmv_debug_fields: dict[str, object] = {}
    motion_fraction_t: np.ndarray | None = None
    diag_for_solver = float(diag_load)
    fd_grid_initial: list[float] = []
    fd_grid_full_list: list[float] = []
    psd_debug: dict[str, object] = {}

    def _select_centered_grid(
        grid: list[float], desired: int, center_val: float
    ) -> tuple[list[float], int]:
        if desired <= 0 or len(grid) <= desired:
            return list(grid), 0
        arr = np.asarray(grid, dtype=np.float32)
        idx_center = int(np.argmin(np.abs(arr - center_val)))
        half = desired // 2
        start = idx_center - half
        if start < 0:
            start = 0
        end = start + desired
        if end > arr.size:
            end = arr.size
            start = max(0, end - desired)
        sub = arr[start:end]
        if sub.size < desired and arr.size >= desired:
            start = max(0, start - (desired - sub.size))
            sub = arr[start : start + desired]
        result = sub.tolist()
        if desired % 2 == 1 and len(result) % 2 == 0 and len(result) > 1:
            arr_sub = np.asarray(result)
            drop_idx = int(np.argmin(np.abs(arr_sub - center_val)))
            result.pop(drop_idx)
            if drop_idx == 0:
                start += 1
        return result, int(start)

    device_resolved = _resolve_stap_device(device)
    dtype = torch.complex64

    if isinstance(cube_T_hw, np.ndarray):
        cube_cpu = cube_T_hw
    else:
        cube_cpu = np.asarray(cube_T_hw)

    ka_mode_norm = (ka_mode or "none").strip().lower()
    ka_active_tile = ka_mode_norm not in {"", "none"}
    ka_opts_local = dict(ka_opts) if ka_opts else {}
    ka_opts_local.setdefault("feasibility_mode", feas_mode)
    if ka_active_tile and "kappa_target" not in ka_opts_local:
        ka_opts_local["kappa_target"] = 40.0
    if ka_active_tile:
        ka_opts_local.setdefault("beta_directional", True)
        ka_opts_local.setdefault("ridge_split", True)
        guard_ratio = max(0.0, float(msd_ratio_rho)) if msd_ratio_rho is not None else 0.0
        ka_opts_local.setdefault("null_tail_ratio_rho", guard_ratio)
        beta_bounds_opt = ka_opts_local.get("beta_bounds")
        if (
            isinstance(beta_bounds_opt, (list, tuple))
            and len(beta_bounds_opt) >= 2
            and "beta_max" not in ka_opts_local
        ):
            try:
                ka_opts_local["beta_max"] = float(beta_bounds_opt[1])
            except (TypeError, ValueError):
                pass
    if motion_half_span_rel is not None and motion_half_span_rel > 0.0:
        motion_half_span_rel_used = float(motion_half_span_rel)
    else:
        motion_half_span_rel_used = _DEFAULT_MOTION_HALF_SPAN_REL
    motion_half_span_hz_used = None
    ka_prior_np: np.ndarray | None
    if isinstance(ka_prior_library, torch.Tensor):
        ka_prior_np = ka_prior_library.detach().cpu().numpy()
    elif isinstance(ka_prior_library, np.ndarray):
        ka_prior_np = ka_prior_library
    elif ka_prior_library is None:
        ka_prior_np = None
    else:
        ka_prior_np = np.asarray(ka_prior_library)

    try:
        if Lt_fixed is not None:
            Lt = int(Lt_fixed)
        else:
            lt_candidates = tuple(range(3, min(9, max(T // 2, 4)))) or (3,)
            Lt = choose_lt_from_coherence(
                cube_T_hw, lt_candidates=lt_candidates, method="pd", corr_thresh=0.5
            )
            Lt = max(3, min(Lt, max(T - 1, 3)))
        if Lt >= T:
            Lt = max(2, T - 1)

        doppler_bin_hz = prf_hz / max(float(Lt), 1.0)
        motion_half_span_hz_used = motion_half_span_rel_used * doppler_bin_hz
        fundamental_hz = float(doppler_bin_hz)
        info["fundamental_hz"] = fundamental_hz
        psd_debug.clear()
        psd_summary = _doppler_psd_summary(
            cube_cpu,
            prf_hz,
            targets_hz=(0.0, fundamental_hz),
        )
        if psd_summary:
            for key, val in psd_summary.items():
                if isinstance(val, (float, int)):
                    info[key] = float(val)
                elif isinstance(val, str):
                    info[key] = val
                elif isinstance(val, list):
                    psd_debug[key] = [float(v) for v in val]
        flow_key = f"psd_power_{int(round(fundamental_hz))}_hz"
        fundamental_power = None
        if flow_key in psd_summary:
            fundamental_power = float(psd_summary[flow_key])
            info["psd_power_flow_hz"] = fundamental_hz
            info["psd_power_flow"] = fundamental_power
        if "psd_power_0_hz" in psd_summary:
            info["psd_power_dc"] = float(psd_summary["psd_power_0_hz"])

        if ka_active_tile:
            if "target_shrink_perp" not in ka_opts_local:
                if Lt <= 4:
                    ka_opts_local["target_shrink_perp"] = 0.97
                elif Lt <= 6:
                    ka_opts_local["target_shrink_perp"] = 0.92
                else:
                    ka_opts_local["target_shrink_perp"] = 0.95
            if "target_retain_f" not in ka_opts_local:
                ka_opts_local["target_retain_f"] = 0.98 if Lt <= 4 else 0.90
            if "beta_max" not in ka_opts_local and Lt <= 4:
                ka_opts_local["beta_max"] = 0.15
            if "beta_directional" not in ka_opts_local and Lt <= 4:
                ka_opts_local["beta_directional"] = False

        method = cov_estimator if cov_estimator and cov_estimator.lower() != "none" else "scm"
        if cube_tensor is not None:
            cube_tensor_local = cube_tensor.to(device=device_resolved, dtype=dtype, copy=False)
        else:
            cube_tensor_local = torch.as_tensor(cube_cpu, dtype=dtype, device=device_resolved)
        S_tile, R_t, tel_cov = build_temporal_hankels_and_cov(
            cube_tensor_local,
            Lt=Lt,
            center=True,
            estimator=method,
            huber_c=huber_c,
            device=device_resolved,
            dtype=dtype,
        )

        alpha = _shrinkage_alpha_for_kappa(R_t, kappa_target=200.0)
        if shrinkage_alpha_for_kappa_batch is not None:
            try:
                alpha_batch = shrinkage_alpha_for_kappa_batch(R_t.unsqueeze(0), kappa_target=200.0)
                alpha = float(alpha_batch[0].item())
            except Exception:
                # Fallback to per-tile heuristic if batch helper fails
                alpha = _shrinkage_alpha_for_kappa(R_t, kappa_target=200.0)
        if ka_active_tile:
            alpha = 0.0
        if alpha > 0.0:
            eye = torch.eye(R_t.shape[-1], dtype=R_t.dtype, device=R_t.device)
            mu = torch.real(torch.trace(R_t)) / float(R_t.shape[-1])
            R_t = (1.0 - alpha) * R_t + alpha * (mu * eye)

        use_contrast = (
            msd_contrast_alpha is not None
            and msd_contrast_alpha > 0.0
            and motion_half_span_rel is not None
            and motion_half_span_rel > 0.0
            and build_motion_basis_temporal is not None
            and project_out_motion_whitened is not None
            and band_energy_on_whitened is not None
        )
        motion_basis: torch.Tensor | None = None
        motion_rank = 0
        motion_basis_geom_np: np.ndarray | None = None

        fd_mode = fd_span_mode.lower()
        span_min, span_max = fd_span_rel

        fd_min_abs = float(fd_min_abs_hz) if fd_min_abs_hz is not None else 0.0
        if feas_mode == "updated" and fd_min_abs <= 0.0:
            fd_min_abs = fundamental_hz
        fd_min_abs = max(0.0, fd_min_abs)
        fd_min_applied = False

        flow_freq_target = None
        flow_power = None
        if psd_summary:
            flow_freq_target = float(psd_summary.get("psd_power_flow_hz") or 0.0)
            flow_power = psd_summary.get("psd_power_flow")
            if isinstance(flow_power, (list, tuple)):
                flow_power = float(flow_power[0]) if flow_power else None
        flow_min_thresh = max(fd_min_abs, 0.5 * fundamental_hz)
        psd_freq_candidates: list[tuple[float, float, str]] = []
        fundamental_candidate: tuple[float, float, str] | None = None
        top_freqs = psd_summary.get("psd_top_freqs_hz") if psd_summary else None
        top_powers = psd_summary.get("psd_top_power") if psd_summary else None
        if isinstance(top_freqs, list) and isinstance(top_powers, list):
            for freq, power in zip(top_freqs, top_powers, strict=False):
                freq_abs = abs(float(freq))
                if freq_abs >= flow_min_thresh:
                    psd_freq_candidates.append((float(power), freq_abs, "top"))
        if flow_freq_target is not None and flow_freq_target >= flow_min_thresh:
            if flow_power is None:
                flow_power = float(psd_summary.get("psd_power_flow") or 0.0)
                psd_freq_candidates.append((float(flow_power), float(flow_freq_target), "summary"))
        if fundamental_power is not None and fundamental_power > 0.0:
            if fundamental_hz >= flow_min_thresh:
                fundamental_candidate = (fundamental_power, float(fundamental_hz), "fundamental")
                psd_freq_candidates.append(fundamental_candidate)
        alias_ratio_candidate = None
        if psd_freq_candidates and fundamental_hz > 0.0:
            max_freq = max(abs(float(freq)) for _, freq, _ in psd_freq_candidates)
            if max_freq > 0.0:
                alias_ratio_candidate = float(max_freq / float(fundamental_hz))
        if psd_freq_candidates:
            candidates_all = psd_freq_candidates
            best_overall = max(cand[0] for cand in candidates_all)
            ratio_fundamental = None
            fundamental_keep_ratio = 0.04
            fundamental_power_thresh = 5e-2
            fundamental_strong = False
            if fundamental_candidate is not None:
                ratio_fundamental = fundamental_candidate[0] / (best_overall + 1e-12)
                info["psd_fundamental_ratio"] = float(ratio_fundamental)
                fundamental_strong = (
                    ratio_fundamental >= fundamental_keep_ratio
                    and fundamental_candidate[0] >= fundamental_power_thresh
                )

            alias_limit = fundamental_hz * 1.05
            preferred = [cand for cand in candidates_all if cand[1] <= alias_limit]
            use_preferred = bool(preferred)
            if not fundamental_strong:
                use_preferred = False
            candidates = preferred if use_preferred and preferred else candidates_all
            if fundamental_strong and fundamental_candidate is not None:
                flow_freq_target = fundamental_candidate[1]
                flow_power = fundamental_candidate[0]
                info["psd_flow_freq_source_resolved"] = fundamental_candidate[2]
            else:
                best_power, best_freq, best_src = max(
                    candidates,
                    key=lambda item: (
                        item[0],
                        2 if item[2] == "fundamental" else 1 if item[2] == "summary" else 0,
                        -item[1],
                    ),
                )
                flow_freq_target = best_freq
                flow_power = best_power
                info["psd_flow_freq_source_resolved"] = best_src
        if flow_freq_target is not None and flow_freq_target < flow_min_thresh:
            flow_freq_target = None
        alias_ratio_val = None
        if flow_freq_target is not None:
            info["psd_flow_freq_target"] = float(flow_freq_target)
            if flow_power is not None:
                info["psd_flow_freq_power"] = float(flow_power)
            if fundamental_hz > 0.0:
                alias_ratio_val = float(flow_freq_target) / float(fundamental_hz)
        if alias_ratio_candidate is not None:
            if alias_ratio_val is None or alias_ratio_candidate > alias_ratio_val:
                alias_ratio_val = alias_ratio_candidate
        if alias_ratio_val is not None and np.isfinite(alias_ratio_val):
            info["psd_flow_alias_ratio"] = float(alias_ratio_val)
            if alias_ratio_val > 1.05:
                info["psd_flow_alias"] = True
        if flow_freq_target is not None and flow_freq_target > 0.0 and prf_hz > 0.0 and Lt > 0:
            t_idx = np.arange(Lt, dtype=np.float32)
            phases = 2.0 * np.pi * float(flow_freq_target) * t_idx / float(prf_hz)
            flow_direction_vector = np.exp(1j * phases).astype(np.complex64)
        info["flow_freq_target_hz"] = (
            float(flow_freq_target) if flow_freq_target is not None else None
        )

        if use_contrast and flow_freq_target is not None and Lt <= 5:
            use_contrast = False
            motion_basis = None
            motion_rank = 0
            info["motion_contrast_disabled"] = True

        fd_grid: list[float] | None = None
        fd_grid_initial: list[float] = []
        if fd_mode == "psd" and flow_freq_target:
            span_hz = float(flow_freq_target)
            f_peak = 0.0
            # When we have an explicit microvascular target frequency from PSD,
            # build a symmetric grid around ±span_hz using the same min/max-pts
            # logic as the generic path. A hard [-span,0,span] triad tends to
            # collapse to an even flow grid after motion splitting (dropping DC),
            # which breaks Kc invariants and weakens telemetry.
            step_hz = max(grid_step_rel * (prf_hz / max(Lt, 1)), 1.0)
            half_lines = max(1, int(np.ceil(span_hz / step_hz)))
            total_lines = int(np.clip(2 * half_lines + 1, min_pts, max_pts))
            if total_lines % 2 == 0:
                total_lines = max(min_pts, total_lines - 1)
            fd_grid = np.linspace(-span_hz, span_hz, total_lines, dtype=np.float32).tolist()
            fd_grid_initial = list(fd_grid)
            info["fd_grid_source"] = "psd_override"
        if fd_grid is None:
            if fd_mode == "psd":
                f_peak, span_hz = _psd_peak_and_span_hz(
                    cube_cpu, prf_hz, min_rel=0.03, max_rel=0.49, energy_q=0.90
                )
                if span_hz is None:
                    span_rel = float(np.clip(0.60, span_min, span_max))
                    span_hz = span_rel * (prf_hz / max(Lt, 1))
                    f_peak = 0.0
            elif fd_mode == "fixed" and fd_fixed_span_hz is not None:
                span_hz = float(fd_fixed_span_hz)
                f_peak = 0.0
            else:
                span_rel = float(
                    np.clip(0.30 + 0.25 * (4.0 / max(float(Lt), 1.0)), span_min, span_max)
                )
                span_hz = span_rel * (prf_hz / max(Lt, 1))
                f_peak = 0.0

            step_hz = max(grid_step_rel * (prf_hz / max(Lt, 1)), 1.0)
            half_lines = max(1, int(np.ceil(span_hz / step_hz)))
            total_lines = int(np.clip(2 * half_lines + 1, min_pts, max_pts))
            if total_lines % 2 == 0:
                total_lines = max(min_pts, total_lines - 1)
            fd_grid = np.linspace(-span_hz, span_hz, total_lines, dtype=np.float32).tolist()
            fd_grid_initial = list(fd_grid)
            if flow_freq_target and fd_mode == "psd":
                info["fd_grid_source"] = "psd_span"

        if fd_min_abs > 0.0:
            filtered = [f for f in fd_grid if abs(f) + 1e-9 >= fd_min_abs]
            # Ensure we keep at least one positive and one negative tone when possible.
            if filtered:
                has_pos = any(f > 0 for f in filtered)
                has_neg = any(f < 0 for f in filtered)
                if not (has_pos and has_neg) and len(fd_grid) >= 3:
                    # Fall back to the largest-magnitude pair if filtering collapsed one side.
                    extremes = []
                    for sign in (-1.0, 1.0):
                        candidates = [f for f in fd_grid if f * sign > 0]
                        if candidates:
                            extremes.append(max(candidates, key=lambda v: abs(v)))
                    if len(extremes) == 2:
                        filtered = sorted([-abs(extremes[0]), abs(extremes[1])])
                if filtered:
                    fd_grid = filtered
                    fd_min_applied = True
            if not filtered:
                # Filtering removed all tones; keep original grid but flag fallback.
                fd_grid = list(fd_grid_initial)
                info["fd_min_abs_fallback"] = True
        info["fd_min_abs_hz"] = float(fd_min_abs)
        info["fd_min_abs_applied"] = bool(fd_min_applied)

        constraint_mode_norm = (constraint_mode or "exp+deriv").strip().lower()
        if constraint_mode_norm not in {"exp", "exp+deriv"}:
            constraint_mode_norm = "exp+deriv"
        info["constraint_mode"] = constraint_mode_norm

        # After filtering, enforce a symmetric grid if we lost one side.
        # This prevents degeneracy to a single sign which crushes flow energy.
        if fd_grid:
            has_pos = any(f > 0 for f in fd_grid)
            has_neg = any(f < 0 for f in fd_grid)
            if not (has_pos and has_neg):
                mag = max((abs(f) for f in fd_grid if abs(f) > 0), default=0.0)
                if mag > 0.0:
                    if fd_min_abs > 0.0:
                        # Keep symmetry without reintroducing DC when a minimum
                        # magnitude is enforced.
                        fd_grid = [-mag, mag]
                    else:
                        # Legacy triad with DC when no minimum magnitude requested.
                        fd_grid = [-mag, 0.0, mag]
                    info["fd_symmetry_added"] = True

        gate_cfg = ka_gate if ka_gate is not None else None
        gate_ctx = gate_cfg.get("context") if gate_cfg else None
        gate_enable = bool(gate_cfg.get("enable")) if gate_cfg else False
        info["ka_gate_enable"] = bool(gate_enable)
        info["ka_gate_ok"] = None
        info["ka_gate_alias_ok"] = None
        info["ka_gate_flow_ok"] = None
        info["ka_gate_depth_ok"] = None
        info["ka_gate_pd_ok"] = None
        info["ka_gate_reg_ok"] = None
        info["ka_gate_tile_has_flow"] = (
            bool(gate_ctx.get("tile_has_flow", False)) if gate_ctx else None
        )
        info["ka_gate_tile_is_bg"] = bool(gate_ctx.get("tile_is_bg", False)) if gate_ctx else None
        if gate_ctx is not None:
            flow_cov_ctx = gate_ctx.get("flow_cov")
            depth_ctx = gate_ctx.get("depth_frac")
            info["ka_gate_flow_cov"] = float(flow_cov_ctx) if flow_cov_ctx is not None else None
            info["ka_gate_depth_frac"] = float(depth_ctx) if depth_ctx is not None else None
            pd_metric_ctx = gate_ctx.get("pd_norm")
            if pd_metric_ctx is None:
                pd_metric_ctx = gate_ctx.get("pd_metric")
            info["ka_gate_pd_metric"] = float(pd_metric_ctx) if pd_metric_ctx is not None else None
            reg_psr_ctx = gate_ctx.get("reg_psr")
            info["ka_gate_reg_psr"] = float(reg_psr_ctx) if reg_psr_ctx is not None else None
        else:
            info["ka_gate_flow_cov"] = None
            info["ka_gate_depth_frac"] = None
            info["ka_gate_pd_metric"] = None
            info["ka_gate_reg_psr"] = None

        alias_select_triggered = False
        keep_bins = max(1, int(alias_psd_select_bins))
        alias_ratio_legacy = info.get("psd_flow_alias_ratio")
        fundamental_val = info.get("fundamental_hz") or fundamental_hz
        max_span_ref = fd_grid_initial or fd_grid or []
        max_freq_available = max(
            (abs(float(v)) for v in max_span_ref if abs(float(v)) > 0.0),
            default=abs(float(fundamental_val)) if fundamental_val else 0.0,
        )
        gate_result: tuple[bool, bool, bool, bool, bool, bool] | None = None
        if gate_enable:
            flow_cov_ctx = gate_ctx.get("flow_cov") if gate_ctx else None
            depth_ctx = gate_ctx.get("depth_frac") if gate_ctx else None
            pd_metric_ctx = gate_ctx.get("pd_norm") if gate_ctx else None
            if pd_metric_ctx is None and gate_ctx is not None:
                pd_metric_ctx = gate_ctx.get("pd_metric")
            reg_psr_ctx = gate_ctx.get("reg_psr") if gate_ctx else None
            alias_metric_ctx = gate_ctx.get("alias_metric") if gate_ctx else None
            alias_for_gate = alias_ratio_legacy
            if alias_metric_ctx is not None and np.isfinite(alias_metric_ctx):
                try:
                    alias_for_gate = float(alias_metric_ctx)
                except (TypeError, ValueError):
                    alias_for_gate = alias_ratio_legacy
            # In updated feasibility mode the flow coverage threshold is treated
            # as an upper bound; for the experimental blend mode we adopt the
            # same semantics so that low-coverage (alias-dominated) tiles are
            # eligible while dense flow tiles are vetoed.
            gate_mode = str(gate_cfg.get("mode", feas_mode)).strip().lower()
            flow_cov_is_upper_bound_flag = gate_mode in {"updated", "blend"}
            gate_debug = {
                "alias_ratio": (
                    float(alias_ratio_legacy) if alias_ratio_legacy is not None else None
                ),
                "alias_metric": (
                    float(alias_metric_ctx) if alias_metric_ctx is not None else None
                ),
                "alias_used": float(alias_for_gate) if alias_for_gate is not None else None,
                "flow_cov": float(flow_cov_ctx) if flow_cov_ctx is not None else None,
                "depth_frac": float(depth_ctx) if depth_ctx is not None else None,
                "pd_norm": float(pd_metric_ctx) if pd_metric_ctx is not None else None,
                "pd_raw": (
                    float(pd_metric_ctx * gate_cfg.get("pd_reference", 1.0))
                    if (
                        pd_metric_ctx is not None
                        and gate_cfg is not None
                        and gate_cfg.get("pd_reference")
                    )
                    else None
                ),
                "reg_psr": float(reg_psr_ctx) if reg_psr_ctx is not None else None,
                "alias_rmin": float(gate_cfg.get("alias_rmin", alias_psd_select_ratio_thresh)),
                "flow_cov_min": float(gate_cfg.get("flow_cov_min", 0.0)),
                "depth_min_frac": float(gate_cfg.get("depth_min_frac", 0.0)),
                "depth_max_frac": float(gate_cfg.get("depth_max_frac", 1.0)),
                "pd_min": gate_cfg.get("pd_min"),
                "reg_psr_max": gate_cfg.get("reg_psr_max"),
                "flow_cov_is_upper_bound": bool(flow_cov_is_upper_bound_flag),
            }
            gate_result = _ka_gate_decision(
                alias_ratio=alias_for_gate,
                flow_cov=flow_cov_ctx,
                depth_frac=depth_ctx,
                alias_rmin=float(gate_cfg.get("alias_rmin", alias_psd_select_ratio_thresh)),
                flow_cov_min=float(gate_cfg.get("flow_cov_min", 0.0)),
                flow_cov_is_upper_bound=flow_cov_is_upper_bound_flag,
                depth_min_frac=float(gate_cfg.get("depth_min_frac", 0.0)),
                depth_max_frac=float(gate_cfg.get("depth_max_frac", 1.0)),
                pd_metric=pd_metric_ctx,
                pd_min=gate_cfg.get("pd_min"),
                pd_q_lo=None,
                pd_q_hi=None,
                reg_psr=reg_psr_ctx,
                reg_psr_max=gate_cfg.get("reg_psr_max"),
            )
            (
                gate_ok,
                gate_alias_ok,
                gate_flow_ok,
                gate_depth_ok,
                gate_pd_ok,
                gate_reg_ok,
            ) = gate_result
            info["ka_gate_ok_raw"] = bool(gate_ok)
            info["ka_gate_alias_ok_raw"] = bool(gate_alias_ok)
            info["ka_gate_flow_ok_raw"] = bool(gate_flow_ok)
            info["ka_gate_depth_ok_raw"] = bool(gate_depth_ok)
            info["ka_gate_pd_ok_raw"] = bool(gate_pd_ok)
            info["ka_gate_reg_ok_raw"] = bool(gate_reg_ok)
            info["ka_gate_debug"] = gate_debug
            info["ka_gate_ok"] = bool(gate_ok)
            info["ka_gate_alias_ok"] = bool(gate_alias_ok)
            info["ka_gate_flow_ok"] = bool(gate_flow_ok)
            info["ka_gate_depth_ok"] = bool(gate_depth_ok)
            info["ka_gate_pd_ok"] = bool(gate_pd_ok)
            info["ka_gate_reg_ok"] = bool(gate_reg_ok)
            if alias_metric_ctx is not None and np.isfinite(alias_metric_ctx):
                info["ka_alias_metric"] = float(alias_metric_ctx)
            if not gate_ok:
                ka_active_tile = False
                alias_psd_select_enable = False

        if (
            alias_psd_select_enable
            and fd_grid
            and fundamental_val
            and float(fundamental_val) > 0.0
            and alias_ratio_legacy is not None
            and np.isfinite(alias_ratio_legacy)
            and float(alias_ratio_legacy) >= float(alias_psd_select_ratio_thresh)
        ):
            alias_select_triggered = True
            freq_targets: list[float] = [0.0]
            base_freq = float(fundamental_val)
            for mul in range(1, keep_bins + 1):
                freq = min(
                    base_freq * float(mul),
                    max_freq_available if max_freq_available > 0 else base_freq,
                )
                if freq <= 0.0:
                    continue
                freq_targets.extend([-freq, freq])
            # Remove duplicates while preserving order, then sort to keep symmetry.
            seen: set[int] = set()
            kept_list: list[float] = []
            for freq in freq_targets:
                key = int(round(freq * 1e6))
                if key in seen:
                    continue
                seen.add(key)
                kept_list.append(float(freq))
            if not kept_list:
                kept_list = [0.0, -base_freq, base_freq]
            fd_grid = sorted(kept_list)
        info["psd_alias_select_bins_used"] = int(keep_bins)
        info["psd_kept_freq_span_hz"] = float(max(abs(f) for f in fd_grid)) if fd_grid else None
        info["psd_kept_freq_source"] = (
            "alias_select" if alias_select_triggered else info.get("psd_kept_freq_source", "psd")
        )
        info["psd_alias_select_applied"] = bool(alias_select_triggered)
        info["psd_kept_bin_count"] = int(len(fd_grid)) if fd_grid else 0
        if fd_grid:
            psd_debug["psd_kept_freqs_hz"] = [float(f) for f in fd_grid]

        fd_grid_full_list = list(fd_grid) if fd_grid else []
        fd_grid_full = np.asarray(fd_grid, dtype=float) if fd_grid else np.array([], dtype=float)
        fd_motion_np = fd_grid_full
        fd_flow_np = fd_grid_full
        if fd_grid_full.size > 0 and motion_half_span_hz_used > 0.0:
            fd_motion_np, fd_flow_np = split_fd_grid_by_motion(
                fd_grid_full, motion_half_span_hz_used
            )
            if fd_flow_np.size == 0:
                fd_flow_np = fd_grid_full
        info["fd_motion_freqs"] = int(fd_motion_np.size)
        info["fd_flow_freqs_after_split"] = int(fd_flow_np.size)
        info["motion_half_span_hz"] = float(motion_half_span_hz_used)
        info["motion_half_span_rel_used"] = float(motion_half_span_rel_used)
        info["fd_grid_full"] = [float(f) for f in fd_grid_full_list]
        fd_grid = fd_flow_np.tolist()

        if fd_motion_np.size > 0:
            motion_basis = bandpass_constraints_temporal(
                Lt=Lt,
                prf_hz=prf_hz,
                fd_grid_hz=fd_motion_np.tolist(),
                device="cpu",
                dtype=torch.complex64,
                mode="exp+deriv",
            )
        elif build_motion_basis_temporal is not None:
            motion_basis = build_motion_basis_temporal(  # type: ignore[name-defined]
                Lt=Lt,
                prf_hz=prf_hz,
                width_bins=1,
                include_dc=True,
                device="cpu",
                dtype=torch.complex64,
            )
        else:
            motion_basis = None

        if isinstance(motion_basis, torch.Tensor):
            motion_rank = int(motion_basis.shape[1])
            motion_basis_geom_np = motion_basis.detach().cpu().numpy()
        elif motion_basis is not None:
            motion_rank = int(np.asarray(motion_basis).shape[1])
            motion_basis_geom_np = np.asarray(motion_basis, dtype=np.complex64)

        grid_center_val = (
            float(flow_freq_target)
            if flow_freq_target is not None and np.isfinite(flow_freq_target)
            else (float(f_peak) if "f_peak" in locals() else 0.0)
        )

        kc_cap = _cap_kc_from_rank(R_t, min_pts=min_pts, max_pts=max_pts)
        max_allowed = Lt if Lt % 2 == 1 else max(Lt - 1, 1)
        min_cap_candidate = min(max_allowed, max_pts)
        if min_cap_candidate >= 3:
            target_min = 3
            if min_pts is not None and min_pts > 0:
                target_min = max(target_min, min(min_pts, min_cap_candidate))
            if fd_min_abs > 0.0:
                target_min = max(target_min, 3)
            kc_cap = max(kc_cap, min(target_min, min_cap_candidate))
        kc_cap = min(kc_cap, min_cap_candidate if min_cap_candidate > 0 else kc_cap)
        info["kc_flow_cap"] = int(kc_cap)
        if len(fd_grid) > kc_cap:
            fd_grid, _ = _select_centered_grid(
                fd_grid, kc_cap, grid_center_val
            )

        if use_contrast:
            max_flow_rank = max(1, Lt - motion_rank)
            if max_flow_rank % 2 == 0:
                max_flow_rank = max(1, max_flow_rank - 1)
            if len(fd_grid) > max_flow_rank:
                fd_grid, _ = _select_centered_grid(
                    fd_grid, max_flow_rank, grid_center_val
                )
        else:
            max_flow_rank = len(fd_grid)

        Ct_exp = bandpass_constraints_temporal(
            Lt=Lt,
            prf_hz=prf_hz,
            fd_grid_hz=fd_grid,
            device=device_resolved,
            dtype=dtype,
            mode="exp",
        )
        Ct = Ct_exp
        kc_deriv_used = 0
        if constraint_mode_norm != "exp" and fd_grid and Lt > 0:
            # Add derivative constraints only when there is enough headroom to
            # include more than one derivative column. For very short Lt (e.g. 4),
            # keeping the tone grid (odd, symmetric) is preferable to adding a
            # single derivative column that breaks symmetry and shifts Kc parity.
            deriv_capacity = int(Lt - Ct_exp.shape[-1])
            if deriv_capacity >= 2:
                Ct_full = bandpass_constraints_temporal(
                    Lt=Lt,
                    prf_hz=prf_hz,
                    fd_grid_hz=fd_grid,
                    device=device_resolved,
                    dtype=dtype,
                    mode="exp+deriv",
                )
                Ct_deriv = Ct_full[:, 1::2]
                if Ct_deriv.shape[-1] > 0:
                    center_val = float(f_peak) if "f_peak" in locals() else 0.0
                    fd_arr = np.asarray(fd_grid, dtype=np.float64)
                    order = np.argsort(np.abs(fd_arr - center_val))
                    n_deriv = int(min(int(Ct_deriv.shape[-1]), deriv_capacity))
                    sel = np.sort(order[:n_deriv])
                    Ct = torch.cat([Ct_exp, Ct_deriv[:, sel]], dim=1)
                    kc_deriv_used = int(sel.size)
            else:
                info["constraint_deriv_dropped"] = True
        info["kc_flow_deriv_used"] = int(kc_deriv_used)
        info["kc_flow"] = int(Ct.shape[-1]) if Ct.shape[-1] > 0 else None
        info["kc_flow_freqs"] = int(len(fd_grid))

        load_mode_norm = mvdr_load_mode.lower()
        diag_used: float | None = None
        constraint_residual: float | None = None
        load_mode_used = load_mode_norm
        cond_loaded = None
        gram_diag_tuple: tuple[float, float, float] | None = None

        kappa_target = float(mvdr_auto_kappa) if load_mode_norm == "auto" else 200.0
        if conditioned_lambda_batch is not None:
            try:
                lam_batch, kappa_out_batch, lam_needed_batch = conditioned_lambda_batch(
                    R_t.unsqueeze(0),
                    float(diag_for_solver),
                    kappa_target,
                )
                diag_for_solver = float(lam_batch[0].item())
                lam_needed = float(lam_needed_batch[0].item())
                evals = torch.linalg.eigvalsh(R_t).real.clamp_min(0.0)
                sigma_min_raw = float(torch.min(evals).item())
                sigma_max_raw = float(torch.max(evals).item())
                cond_loaded = float(kappa_out_batch[0].item())
            except Exception:
                diag_for_solver, sigma_min_raw, sigma_max_raw, lam_needed = _conditioned_lambda(
                    R_t,
                    diag_for_solver,
                    kappa_target,
                )
        else:
            diag_for_solver, sigma_min_raw, sigma_max_raw, lam_needed = _conditioned_lambda(
                R_t,
                diag_for_solver,
                kappa_target,
            )
        info["diag_load_requested"] = float(diag_load)
        info["lambda_condition_needed"] = float(lam_needed)
        info["lambda_conditioned"] = float(diag_for_solver)
        info["sigma_min_raw"] = float(sigma_min_raw)
        info["sigma_max_raw"] = float(sigma_max_raw)
        info["kappa_target"] = float(kappa_target)
        info["motion_basis_rank"] = int(motion_rank)
        info["kc_flow_cap_motion"] = int(max_flow_rank)

        try:
            if load_mode_norm not in {"auto", "absolute"}:
                raise ValueError(f"Unsupported mvdr_load_mode={mvdr_load_mode!r}")
            y_filt, lres = lcmv_temporal_apply_batched(
                R_t,
                S_tile,
                Ct,
                load_mode=load_mode_norm,
                diag_load=diag_for_solver,
                auto_kappa_target=mvdr_auto_kappa,
                auto_lambda_bounds=(5e-4, 2e-1),
                constraint_ridge=constraint_ridge,
                device=device_resolved,
                dtype=dtype,
            )
            if post_filter_callback is not None:
                try:
                    y_np = y_filt.detach().cpu().numpy()
                except AttributeError:
                    y_np = np.asarray(y_filt)
                y_np = np.asarray(y_np, dtype=np.complex64)
                series = np.mean(y_np.reshape(y_np.shape[0], -1), axis=1)
                post_filter_callback(series.astype(np.complex64, copy=False))
            diag_used = float(lres.diag_load)
            constraint_residual = float(lres.constraint_residual)
            load_mode_used = str(lres.load_mode)
            flow_resp_np = None
            motion_resp_np = None
            try:
                w_np = lres.w.detach().cpu().numpy()
                lcmv_debug_fields["lcmv_weights"] = w_np
                Ct_np = (
                    Ct.detach().cpu().numpy()
                    if isinstance(Ct, torch.Tensor)
                    else np.asarray(Ct, dtype=np.complex64)
                )
                if Ct_np.size > 0:
                    flow_resp_np = Ct_np.conj().T @ w_np
                    abs_vals = np.abs(flow_resp_np)
                    info["flow_response_min"] = float(abs_vals.min())
                    info["flow_response_med"] = float(np.median(abs_vals))
                    info["flow_response_max"] = float(abs_vals.max())
                else:
                    info["flow_response_min"] = info["flow_response_med"] = info[
                        "flow_response_max"
                    ] = None
                if motion_basis is not None:
                    motion_np = (
                        motion_basis.detach().cpu().numpy()
                        if isinstance(motion_basis, torch.Tensor)
                        else np.asarray(motion_basis, dtype=np.complex64)
                    )
                    if motion_np.size > 0:
                        motion_resp_np = motion_np.conj().T @ w_np
                        info["motion_response_max"] = float(np.abs(motion_resp_np).max())
            except Exception as exc:
                info["flow_response_min"] = info["flow_response_med"] = info[
                    "flow_response_max"
                ] = None
                info["motion_response_max"] = None
                info["lcmv_debug_error"] = str(exc)
                lcmv_debug_fields.pop("lcmv_weights", None)
            if flow_resp_np is not None:
                lcmv_debug_fields["flow_response"] = flow_resp_np
            if motion_resp_np is not None:
                lcmv_debug_fields["motion_response"] = motion_resp_np

            if diag_used is not None:
                eye = torch.eye(R_t.shape[-1], dtype=R_t.dtype, device=R_t.device)
                R_loaded = R_t + diag_used * eye
                cond_loaded = float(torch.linalg.cond(R_loaded).item())
                if Ct.shape[-1] > 0:
                    gram_mat = torch.matmul(
                        Ct.conj().transpose(-2, -1),
                        torch.linalg.solve(R_loaded, Ct),
                    )
                    gram_diag = torch.real(torch.diagonal(gram_mat))
                    gram_diag_tuple = (
                        float(torch.min(gram_diag).item()),
                        float(torch.median(gram_diag).item()),
                        float(torch.max(gram_diag).item()),
                    )
        except Exception:
            diag_used = float(diag_load if diag_load > 0 else 2e-2)
            constraint_residual = None
            load_mode_used = "capon"
            eye = torch.eye(R_t.shape[-1], dtype=R_t.dtype, device=R_t.device)
            R_loaded = R_t + diag_used * eye
            cond_loaded = float(torch.linalg.cond(R_loaded).item())
            if Ct.shape[-1] > 0:
                gram_mat = torch.matmul(
                    Ct.conj().transpose(-2, -1),
                    torch.linalg.solve(R_loaded, Ct),
                )
                gram_diag = torch.real(torch.diagonal(gram_mat))
                gram_diag_tuple = (
                    float(torch.min(gram_diag).item()),
                    float(torch.median(gram_diag).item()),
                    float(torch.max(gram_diag).item()),
                )

        grid_step_hz = float(grid_step_rel * (prf_hz / max(Lt, 1)))

        # Populate MVDR/constraint diagnostics (safe conversions)
        info["Lt"] = int(Lt)
        info["N"] = int(S_tile.shape[1])
        info["M"] = int(Lt)
        try:
            info["cond_R"] = float(np.real(tel_cov.get("cond_est", float("nan"))))
        except Exception:
            info["cond_R"] = None
        # Report Kc as the number of kept flow tones after motion split/capping.
        # This is the effective band geometry used for Pf/Pa telemetry.
        info["band_Kc"] = int(len(fd_grid))
        try:
            info["fd_grid"] = [float(np.real(f)) for f in fd_grid]
        except Exception:
            info["fd_grid"] = []
        try:
            info["f_peak_hz"] = float(np.real(f_peak))
        except Exception:
            info["f_peak_hz"] = None
        try:
            info["span_hz"] = float(np.real(span_hz))
        except Exception:
            info["span_hz"] = None
        info["grid_step_hz"] = float(grid_step_hz)
        info["span_mode"] = fd_mode
        try:
            info["diag_load"] = float(np.real(diag_used)) if diag_used is not None else None
        except Exception:
            info["diag_load"] = None
        info["load_mode"] = load_mode_used
        info["covariance_method"] = method
        try:
            info["constraint_residual"] = (
                float(np.real(constraint_residual)) if constraint_residual is not None else None
            )
        except Exception:
            info["constraint_residual"] = None
        try:
            info["constraint_ridge"] = float(np.real(constraint_ridge))
        except Exception:
            info["constraint_ridge"] = None
        info["auto_kappa_target"] = (
            float(mvdr_auto_kappa) if load_mode_used.startswith("auto") else None
        )
        info["fallback"] = False

        info["cov_trace"] = tel_cov.get("trace")
        info["cov_eff_rank"] = tel_cov.get("eff_rank")
        info["cond_loaded"] = cond_loaded
        if gram_diag_tuple is not None:
            info["gram_diag_min"], info["gram_diag_median"], info["gram_diag_max"] = (
                gram_diag_tuple
            )
        else:
            info["gram_diag_min"] = None
            info["gram_diag_median"] = None
            info["gram_diag_max"] = None

    except Exception as exc:
        # Do not return early; populate diagnostics and continue to PD computation below
        band_frac_tile = np.ones((h, w), dtype=np.float32)
        score_tile = np.zeros_like(band_frac_tile)
        info["fallback"] = True
        info["fallback_exc"] = repr(exc)
        info["score_mode"] = "none"
        info["score_mean"] = float(np.mean(score_tile))
        info["score_var"] = float(np.var(score_tile))
        info["pd_lambda"] = float(diag_load)
        info["msd_lambda"] = float(msd_lambda) if msd_lambda is not None else float(diag_load)
        info["msd_ridge"] = float(msd_ridge)
        info["msd_agg_mode"] = msd_agg_mode
        info["cond_loaded"] = None
        info["gram_diag_min"] = None
        info["gram_diag_median"] = None
        info["gram_diag_max"] = None
        info.setdefault("load_mode", "capon")
        info.setdefault("band_Kc", 0)
        info.setdefault("Lt", Lt)
        info.setdefault("fd_grid", fd_grid)
        info.setdefault("span_hz", span_hz)
        info.setdefault("kappa_target", 200.0)
        info.setdefault("lambda_conditioned", float(diag_for_solver))
        info.setdefault("lambda_condition_needed", 0.0)
        info.setdefault("diag_load_requested", float(diag_load))

    diag_used_local = diag_used if "diag_used" in locals() else None
    pd_lam = diag_used_local if diag_used_local is not None else diag_for_solver
    lam_for_msd = float(msd_lambda if msd_lambda is not None else pd_lam)
    if ka_active_tile:
        info["msd_lambda_needed"] = 0.0
        info["msd_lambda_conditioned"] = lam_for_msd
    else:
        kappa_target_msd = float(info.get("kappa_target", 200.0))
        if conditioned_lambda_batch is not None:
            try:
                lam_batch, _, lam_needed_batch = conditioned_lambda_batch(
                    R_t.unsqueeze(0),
                    float(lam_for_msd),
                    kappa_target_msd,
                )
                lam_for_msd = float(lam_batch[0].item())
                lam_msd_needed = float(lam_needed_batch[0].item())
            except Exception:
                lam_for_msd, _, _, lam_msd_needed = _conditioned_lambda(
                    R_t,
                    float(lam_for_msd),
                    kappa_target_msd,
                )
        else:
            lam_for_msd, _, _, lam_msd_needed = _conditioned_lambda(
                R_t,
                float(lam_for_msd),
                kappa_target_msd,
            )
        info["msd_lambda_needed"] = float(lam_msd_needed)
        info["msd_lambda_conditioned"] = float(lam_for_msd)
    if ka_active_tile:
        ka_opts_local.setdefault("lambda_override_split", float(lam_for_msd))
        if Lt <= 4:
            ka_opts_local["lambda_override_split"] = (
                float(ka_opts_local["lambda_override_split"]) * 0.5
            )
    R0_prior_tile: np.ndarray | None = None
    if ka_active_tile:
        if ka_mode_norm == "analytic":
            R0_tensor = ka_prior_temporal_from_psd(
                Lt=Lt,
                prf_hz=prf_hz,
                f_peaks_hz=(0.0,),
                width_bins=1,
                add_deriv=True,
                device="cpu",
                dtype=torch.complex64,
            )
            R0_prior_tile = R0_tensor.cpu().numpy()
        elif ka_mode_norm == "library":
            if ka_prior_np is None:
                info["ka_warning"] = "ka_prior_missing"
                ka_active_tile = False
            elif ka_prior_np.shape != (Lt, Lt):
                info["ka_warning"] = f"ka_prior_shape_mismatch:{ka_prior_np.shape}->{Lt}"
                ka_active_tile = False
            else:
                R0_prior_tile = ka_prior_np
        else:
            ka_active_tile = False
    info["ka_mode"] = ka_mode_norm if ka_active_tile else "none"

    agg_mode = msd_agg_mode.lower()
    if agg_mode not in {"mean", "median", "trim10"}:
        agg_mode = "mean"

    score_mode = "msd"
    band_frac_tile = np.ones((h, w), dtype=np.float32)
    score_tile = np.zeros((h, w), dtype=np.float32)
    motion_fraction_t: np.ndarray | None = None
    ka_detail_list: list[Dict[str, float]] = []

    info["pd_retry"] = "none"
    pd_tile_override: np.ndarray | None = None
    pd_retry_reason: str | None = None

    def _apply_ka_details() -> None:
        # Record detail list length for debugging telemetry.
        info["ka_detail_len"] = len(ka_detail_list)
        # If KA never appended details, record why and bail.
        if not ka_active_tile or not ka_detail_list:
            if ka_active_tile and not ka_detail_list:
                info["ka_band_metrics_error"] = "empty_detail_list"
            return
        ka_last = ka_detail_list[-1]
        try:
            info["ka_detail_keys"] = list(sorted(ka_last.keys()))
        except Exception:
            info["ka_detail_keys"] = None
        info["ka_beta"] = float(ka_last.get("beta", float("nan")))
        info["ka_mismatch"] = float(ka_last.get("mismatch", float("nan")))
        info["ka_lambda_used"] = float(ka_last.get("lambda_used", float("nan")))
        info["ka_sigma_min_raw"] = float(ka_last.get("sigma_min_raw", float("nan")))
        info["ka_sigma_max_raw"] = float(ka_last.get("sigma_max_raw", float("nan")))
        info["ka_retain_f_beta"] = float(ka_last.get("retain_f_beta", float("nan")))
        info["ka_shrink_perp_beta"] = float(ka_last.get("shrink_perp_beta", float("nan")))
        info["ka_retain_f_total"] = float(ka_last.get("retain_f_total", float("nan")))
        info["ka_shrink_perp_total"] = float(ka_last.get("shrink_perp_total", float("nan")))
        info["ka_ridge_split"] = bool(ka_last.get("ridge_split", False))
        info["ka_prior_clipped_passband"] = bool(ka_last.get("prior_clipped_passband", False))
        info["ka_directional_strict"] = bool(ka_last.get("directional_strict", False))
        info["ka_trace_scaled"] = bool(ka_last.get("trace_scaled", False))
        if ka_last.get("prior_clipped_perp"):
            info["ka_prior_clipped_perp"] = True
        info["ka_lambda_strategy"] = ka_last.get("lambda_strategy", "global")
        info["ka_pf_rank"] = float(ka_last.get("pf_rank", float("nan")))
        info["ka_alias_rank"] = float(ka_last.get("alias_rank", float("nan")))
        if ka_last.get("ka_alias_gain_target") is not None:
            try:
                info["ka_alias_gain_target"] = float(ka_last.get("ka_alias_gain_target"))
            except Exception:
                info["ka_alias_gain_target"] = None
        info["ka_beta_reason"] = ka_last.get("beta_reason")
        info["ka_s_perp_used"] = (
            float(ka_last["s_perp_used"]) if ka_last.get("s_perp_used") is not None else None
        )
        info["ka_beta_req"] = (
            float(ka_last["beta_req"]) if ka_last.get("beta_req") is not None else None
        )
        info["ka_lam_pf_frac"] = float(ka_last.get("lam_pf_frac", float("nan")))
        info["ka_trace_ratio"] = float(ka_last.get("trace_ratio", float("nan")))
        info["ka_trace_scaled"] = bool(ka_last.get("trace_scaled", False))
        trace_scale_reason = ka_last.get("trace_scale_lock_reason")
        if trace_scale_reason is not None:
            info["ka_trace_scale_lock_reason"] = trace_scale_reason
        trace_sample = ka_last.get("trace_sample")
        info["ka_trace_sample"] = float(trace_sample) if trace_sample is not None else None
        trace_return_pre = ka_last.get("trace_return_pre")
        info["ka_trace_return_pre"] = (
            float(trace_return_pre) if trace_return_pre is not None else None
        )
        pf_lambda_mean_val = ka_last.get("pf_lambda_mean")
        perp_lambda_mean_val = ka_last.get("perp_lambda_mean")
        pf_lambda_min_val = ka_last.get("pf_lambda_min")
        if pf_lambda_min_val is not None:
            info["ka_pf_lambda_min"] = float(pf_lambda_min_val)
        pf_lambda_max_val = ka_last.get("pf_lambda_max")
        if pf_lambda_max_val is not None:
            info["ka_pf_lambda_max"] = float(pf_lambda_max_val)
        if pf_lambda_mean_val is not None:
            info["ka_pf_lambda_mean"] = float(pf_lambda_mean_val)
        perp_lambda_min_val = ka_last.get("perp_lambda_min")
        if perp_lambda_min_val is not None:
            info["ka_perp_lambda_min"] = float(perp_lambda_min_val)
        perp_lambda_max_val = ka_last.get("perp_lambda_max")
        if perp_lambda_max_val is not None:
            info["ka_perp_lambda_max"] = float(perp_lambda_max_val)
        if perp_lambda_mean_val is not None:
            info["ka_perp_lambda_mean"] = float(perp_lambda_mean_val)
        alias_lambda_min_val = ka_last.get("ka_alias_lambda_min")
        if alias_lambda_min_val is not None:
            info["ka_alias_lambda_min"] = float(alias_lambda_min_val)
        alias_lambda_max_val = ka_last.get("ka_alias_lambda_max")
        if alias_lambda_max_val is not None:
            info["ka_alias_lambda_max"] = float(alias_lambda_max_val)
        alias_lambda_mean_val = ka_last.get("ka_alias_lambda_mean")
        if alias_lambda_mean_val is not None:
            info["ka_alias_lambda_mean"] = float(alias_lambda_mean_val)
        noise_lambda_min_val = ka_last.get("ka_noise_lambda_min")
        if noise_lambda_min_val is not None:
            info["ka_noise_lambda_min"] = float(noise_lambda_min_val)
        noise_lambda_max_val = ka_last.get("ka_noise_lambda_max")
        if noise_lambda_max_val is not None:
            info["ka_noise_lambda_max"] = float(noise_lambda_max_val)
        noise_lambda_mean_val = ka_last.get("ka_noise_lambda_mean")
        if noise_lambda_mean_val is not None:
            info["ka_noise_lambda_mean"] = float(noise_lambda_mean_val)
        info["ka_pf_trace_equalized"] = bool(ka_last.get("pf_trace_equalized", False))
        pf_trace_alpha = ka_last.get("pf_trace_alpha")
        if pf_trace_alpha is not None:
            info["ka_pf_trace_alpha"] = float(pf_trace_alpha)
        pf_trace_loaded = ka_last.get("pf_trace_trace_loaded")
        if pf_trace_loaded is not None:
            info["ka_pf_trace_loaded"] = float(pf_trace_loaded)
        pf_trace_sample = ka_last.get("pf_trace_trace_sample")
        if pf_trace_sample is not None:
            info["ka_pf_trace_sample"] = float(pf_trace_sample)
        if (
            pf_lambda_mean_val is not None
            and perp_lambda_mean_val is not None
            and np.isfinite(pf_lambda_mean_val)
            and np.isfinite(perp_lambda_mean_val)
            and abs(float(perp_lambda_mean_val)) > 0.0
        ):
            info["ka_score_scale_ratio"] = float(
                float(pf_lambda_mean_val) / max(float(perp_lambda_mean_val), 1e-9)
            )
        snr_flow_ratio_val = ka_last.get("snr_flow_ratio")
        if (
            snr_flow_ratio_val is None
            and pf_lambda_mean_val is not None
            and perp_lambda_mean_val is not None
        ):
            snr_flow_ratio_val = float(pf_lambda_mean_val) / max(float(perp_lambda_mean_val), 1e-9)
        if snr_flow_ratio_val is None:
            snr_flow_ratio_val = (
                float(pf_lambda_mean_val) / max(float(perp_lambda_mean_val), 1e-9)
                if pf_lambda_mean_val is not None and perp_lambda_mean_val not in (None, 0)
                else 1.0
            )
        if snr_flow_ratio_val is not None:
            snr_ratio = float(snr_flow_ratio_val)
            info["ka_snr_flow_ratio"] = snr_ratio
            if snr_ratio < 0.5:
                info["ka_invalid_snr"] = True
        noise_perp_ratio_val = ka_last.get("noise_perp_ratio")
        if noise_perp_ratio_val is None and perp_lambda_mean_val is not None:
            noise_perp_ratio_val = float(perp_lambda_mean_val)
        if noise_perp_ratio_val is None:
            noise_perp_ratio_val = (
                float(perp_lambda_mean_val) if perp_lambda_mean_val is not None else 1.0
            )
        if noise_perp_ratio_val is not None:
            noise_ratio = float(noise_perp_ratio_val)
            info["ka_noise_perp_ratio"] = noise_ratio
            if noise_ratio > 1.5:
                info["ka_invalid_noise"] = True
        if ka_last.get("snr_flow_base") is not None:
            info["ka_snr_flow_base"] = float(ka_last["snr_flow_base"])
        if ka_last.get("snr_flow_loaded") is not None:
            info["ka_snr_flow_loaded"] = float(ka_last["snr_flow_loaded"])
        feas_mode = ka_last.get("feasibility_mode")
        if feas_mode is not None:
            info["ka_feasibility_mode"] = str(feas_mode)
        mixing_val = ka_last.get("mixing_epsilon")
        if mixing_val is not None:
            info["ka_operator_mixing_epsilon"] = float(mixing_val)
        if ka_last.get("operator_feasible") is not None:
            info["operator_feasible"] = bool(ka_last.get("operator_feasible"))
        if ka_last.get("sample_pf_lambda_min") is not None:
            info["sample_pf_lambda_min"] = float(ka_last["sample_pf_lambda_min"])
        if ka_last.get("sample_pf_lambda_max") is not None:
            info["sample_pf_lambda_max"] = float(ka_last["sample_pf_lambda_max"])
        if ka_last.get("sample_pf_lambda_mean") is not None:
            info["sample_pf_lambda_mean"] = float(ka_last["sample_pf_lambda_mean"])
        if ka_last.get("sample_perp_lambda_min") is not None:
            info["sample_perp_lambda_min"] = float(ka_last["sample_perp_lambda_min"])
        if ka_last.get("sample_perp_lambda_max") is not None:
            info["sample_perp_lambda_max"] = float(ka_last["sample_perp_lambda_max"])
        if ka_last.get("sample_perp_lambda_mean") is not None:
            info["sample_perp_lambda_mean"] = float(ka_last["sample_perp_lambda_mean"])
        if ka_last.get("sample_alias_lambda_min") is not None:
            info["sample_alias_lambda_min"] = float(ka_last["sample_alias_lambda_min"])
        if ka_last.get("sample_alias_lambda_max") is not None:
            info["sample_alias_lambda_max"] = float(ka_last["sample_alias_lambda_max"])
        if ka_last.get("sample_alias_lambda_mean") is not None:
            info["sample_alias_lambda_mean"] = float(ka_last["sample_alias_lambda_mean"])
        if ka_last.get("sample_noise_lambda_min") is not None:
            info["sample_noise_lambda_min"] = float(ka_last["sample_noise_lambda_min"])
        if ka_last.get("sample_noise_lambda_max") is not None:
            info["sample_noise_lambda_max"] = float(ka_last["sample_noise_lambda_max"])
        if ka_last.get("sample_noise_lambda_mean") is not None:
            info["sample_noise_lambda_mean"] = float(ka_last["sample_noise_lambda_mean"])
        if ka_last.get("sample_po_noise_floor") is not None:
            info["sample_po_noise_floor"] = float(ka_last["sample_po_noise_floor"])
        if ka_last.get("prior_po_noise_floor") is not None:
            info["prior_po_noise_floor"] = float(ka_last["prior_po_noise_floor"])
            warning = ka_last.get("warning")
            if warning and not info.get("ka_warning"):
                info["ka_warning"] = str(warning)

    try:
        use_contrast = (
            msd_contrast_alpha is not None
            and msd_contrast_alpha > 0.0
            and motion_half_span_rel is not None
            and motion_half_span_rel > 0.0
            and build_motion_basis_temporal is not None
            and project_out_motion_whitened is not None
            and band_energy_on_whitened is not None
        )

        if use_contrast:
            motion_half_span_hz = float(motion_half_span_rel) * (prf_hz / max(float(Lt), 1.0))
            (
                score_hw,
                r_flow_hw,
                r_motion_hw,
                contrast_details,
                Tb_flow_hw,
            ) = msd_contrast_score_batched(  # type: ignore[name-defined]
                R_t,
                S_tile,
                prf_hz=prf_hz,
                fd_grid_hz=fd_grid_full_list,
                motion_half_span_hz=motion_half_span_hz,
                lam_abs=float(lam_for_msd),
                ridge=msd_ridge,
                agg=agg_mode,
                contrast_alpha=msd_contrast_alpha,
                basis_mode="exp+deriv",
                ratio_rho=msd_ratio_rho,
                R0_prior=R0_prior_tile if ka_active_tile else None,
                ka_opts=ka_opts_local if ka_active_tile else None,
                ka_details=ka_detail_list if ka_active_tile else None,
                device="cpu",
                dtype=torch.complex64,
                return_details=True,
                return_band_energy=True,
            )

            r_flow_np = r_flow_hw.detach().cpu().numpy().astype(np.float32)
            r_motion_np = r_motion_hw.detach().cpu().numpy().astype(np.float32)
            score_contrast_np = score_hw.detach().cpu().numpy().astype(np.float32)
            pd_contrast_np = aggregate_over_snapshots(
                Tb_flow_hw.detach().cpu().numpy(), mode=agg_mode
            ).astype(np.float32)

            band_frac_tile = r_flow_np
            motion_fraction_t = r_motion_np
            score_tile = score_contrast_np
            score_mode = contrast_details.get("score_mode", "msd_contrast")
            pd_tile_override = pd_contrast_np
            info["band_fraction_median"] = float(np.nanmedian(band_frac_tile))
            info["band_fraction_p90"] = float(np.nanpercentile(band_frac_tile, 90.0))
            info["motion_fraction_median"] = float(np.nanmedian(motion_fraction_t))
            info["motion_fraction_p90"] = float(np.nanpercentile(motion_fraction_t, 90.0))
            info["motion_half_span_hz"] = motion_half_span_hz
            info["msd_contrast_alpha"] = float(msd_contrast_alpha)
            info["energy_kept_ratio"] = float(contrast_details.get("energy_kept_ratio", np.nan))
            info["energy_removed_ratio"] = float(
                contrast_details.get("energy_removed_ratio", np.nan)
            )
            info["zeta_est"] = float(contrast_details.get("zeta_est", np.nan))
            info["beta_est"] = float(contrast_details.get("beta_est", np.nan))
            info["basis_mode"] = contrast_details.get("basis_mode", "exp+deriv")
            info["contrast_fallback"] = bool(contrast_details.get("fallback", False))
            info["fallback"] = bool(contrast_details.get("fallback", False))
            info["contrast_enabled"] = True
            info["contrast_flow_median"] = float(np.nanmedian(r_flow_np))
            info["contrast_flow_p90"] = float(np.nanpercentile(r_flow_np, 90.0))
            info["contrast_motion_median"] = float(np.nanmedian(r_motion_np))
            info["contrast_motion_p90"] = float(np.nanpercentile(r_motion_np, 90.0))
            info["contrast_score_mean"] = float(np.nanmean(score_contrast_np))
            info["contrast_score_std"] = float(np.nanstd(score_contrast_np))
            info["contrast_eta"] = float(contrast_details.get("eta", 0.0))
            info["contrast_alpha_used"] = float(msd_contrast_alpha)
            info["contrast_motion_span_hz"] = float(motion_half_span_hz)
            info["contrast_flow_kc"] = int(contrast_details.get("kc_flow", r_flow_hw.shape[0]))
            info["contrast_motion_kc"] = int(
                contrast_details.get("kc_motion", r_motion_hw.shape[0])
            )
            info["contrast_flow_rank"] = contrast_details.get("flow_rank")
            info["contrast_motion_rank_initial"] = contrast_details.get("motion_rank_initial")
            info["contrast_motion_rank_eff"] = contrast_details.get("motion_rank_eff")
            info["pd_tile_override"] = pd_tile_override
            if info["fallback"]:
                raise RuntimeError("contrast fallback")
            _apply_ka_details()
        else:
            ratio_rho = max(0.0, float(msd_ratio_rho))
            cond_threshold = 1e8
            guard_only_keys = {"null_tail_guard_eps", "null_tail_ratio_rho", "ratio_rho"}
            ka_opts_msd = {k: v for k, v in ka_opts_local.items() if k not in guard_only_keys}
            ka_opts_msd.setdefault("lambda_override_split", float(lam_for_msd))

            # Primary PD/BR basis: keep the tone grid, but cap derivative
            # augmentation so the projector doesn't become near-full-rank on small Lt.
            Ct_primary_exp = bandpass_constraints_temporal(
                Lt=Lt,
                prf_hz=prf_hz,
                fd_grid_hz=fd_grid,
                device="cpu",
                dtype=torch.complex64,
                mode="exp",
            )
            Ct_primary = Ct_primary_exp
            kc_deriv_used_pd = 0
            deriv_capacity_pd = int(Lt - Ct_primary_exp.shape[-1])
            if fd_grid and deriv_capacity_pd > 0:
                Ct_primary_full = bandpass_constraints_temporal(
                    Lt=Lt,
                    prf_hz=prf_hz,
                    fd_grid_hz=fd_grid,
                    device="cpu",
                    dtype=torch.complex64,
                    mode="exp+deriv",
                )
                Ct_primary_deriv = Ct_primary_full[:, 1::2]
                if Ct_primary_deriv.shape[-1] > 0:
                    # Limit derivative columns to keep selectivity on short apertures:
                    # - For very short Lt (<=4), allow a single derivative column when it
                    #   fits; otherwise the projector can be too low-rank and over-suppress
                    #   true flow energy.
                    # - For larger Lt, cap derivatives to avoid near-full-rank projectors
                    #   that collapse the band-fraction dynamic range.
                    if deriv_capacity_pd == 1:
                        max_deriv = 1 if Lt <= 4 else 0
                    else:
                        max_deriv = 2 if int(Ct_primary_deriv.shape[-1]) <= 2 else 1
                    n_deriv = int(
                        min(int(Ct_primary_deriv.shape[-1]), deriv_capacity_pd, max_deriv)
                    )
                    if n_deriv > 0:
                        center_val = float(f_peak) if "f_peak" in locals() else 0.0
                        fd_arr = np.asarray(fd_grid, dtype=np.float64)
                        order = np.argsort(np.abs(fd_arr - center_val))
                        sel = np.sort(order[:n_deriv])
                        Ct_primary = torch.cat(
                            [Ct_primary_exp, Ct_primary_deriv[:, sel]], dim=1
                        )
                        kc_deriv_used_pd = int(sel.size)
            info["kc_flow_deriv_used_pd"] = int(kc_deriv_used_pd)
            Ct_retry = None
            if isinstance(Ct_primary, torch.Tensor):
                Ct_primary_np = Ct_primary.detach().cpu().numpy()
            else:
                Ct_primary_np = np.asarray(Ct_primary, dtype=np.complex64)

            S_cpu = S_tile.detach().to(torch.complex64).cpu()
            R_cpu = R_t.detach().to(torch.complex64).cpu()

            Pf_pd: torch.Tensor | None = None
            Pa_pd: torch.Tensor | None = None
            if Ct_primary.shape[-1] > 0 and projector_from_tones is not None:
                try:
                    Pf_pd = projector_from_tones(Ct_primary.to(torch.complex128)).to(
                        torch.complex64
                    )
                except Exception:
                    Pf_pd = None
            info["kc_flow"] = int(Ct_primary.shape[-1]) if Ct_primary.shape[-1] > 0 else None
            if Pf_pd is None:
                info["kc_flow_error"] = "pf_projector_none"

            flow_alignment_val = None
            flow_motion_angle_val = None
            motion_basis_for_angle = motion_basis_geom
            if motion_basis_for_angle is None and motion_basis_geom_np is not None:
                motion_basis_for_angle = motion_basis_geom_np
            if (
                projected_flow_alignment is not None
                and flow_direction_vector is not None
                and Ct_primary_np.size > 0
            ):
                try:
                    flow_alignment_val = projected_flow_alignment(
                        Ct_primary_np, flow_direction_vector
                    )
                except Exception:
                    flow_alignment_val = None
            if principal_angle is not None and motion_basis_for_angle is not None:
                try:
                    # Use a DC-free flow basis for the angle to avoid spurious 0°
                    # due to derivatives.
                    Cf_angle_np = None
                    if fd_flow_np.size > 0 and bandpass_constraints_temporal is not None:
                        Cf_angle_np = (
                            bandpass_constraints_temporal(
                                Lt=Lt,
                                prf_hz=prf_hz,
                                fd_grid_hz=fd_flow_np.tolist(),
                                device="cpu",
                                dtype=torch.complex64,
                                mode="exp",
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    basis_for_angle = Ct_primary_np
                    if Cf_angle_np is not None and Cf_angle_np.size > 0:
                        basis_for_angle = Cf_angle_np
                    if (
                        basis_for_angle is not None
                        and basis_for_angle.size > 0
                        and motion_basis_for_angle.size > 0
                    ):
                        flow_motion_angle_val = principal_angle(
                            basis_for_angle, motion_basis_for_angle, degrees=True
                        )
                    else:
                        flow_motion_angle_val = None
                except Exception:
                    flow_motion_angle_val = None
            if flow_motion_angle_val is None:
                flow_motion_angle_val = 90.0
            info["flow_band_alignment"] = (
                float(flow_alignment_val) if flow_alignment_val is not None else None
            )
            info["flow_motion_angle_deg"] = (
                float(flow_motion_angle_val) if flow_motion_angle_val is not None else None
            )

            R_loaded_base: torch.Tensor | None = None
            Cf_alias_np = None
            if alias_band_hz is not None and bandpass_constraints_temporal is not None:
                try:
                    alias_low, alias_high = alias_band_hz
                    alias_center = 0.5 * (float(alias_low) + float(alias_high))
                    fd_alias_list = [alias_center]
                    # Include symmetric tone if nonzero
                    if abs(alias_center) > 1e-6:
                        fd_alias_list.append(-alias_center)
                    Cf_alias_np = (
                        bandpass_constraints_temporal(
                            Lt=Lt,
                            prf_hz=prf_hz,
                            fd_grid_hz=fd_alias_list,
                            device="cpu",
                            dtype=torch.complex64,
                            mode="exp+deriv",
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                except Exception:
                    Cf_alias_np = None
                else:
                    try:
                        Pa_pd = projector_from_tones(
                            torch.as_tensor(Cf_alias_np, dtype=torch.complex128)
                        ).to(torch.complex64)
                    except Exception:
                        Pa_pd = None
            info["kc_alias"] = int(Cf_alias_np.shape[-1]) if Cf_alias_np is not None else None
            if (
                ka_active_tile
                and R0_prior_tile is not None
                and ka_blend_covariance_temporal is not None
            ):
                try:
                    Cf_flow_np = Ct_primary_np
                    # Use a DC-free, exponential-only flow basis for KA commutation
                    Cf_flow_clean = Cf_flow_np
                    if "fd_flow_np" not in locals():
                        fd_flow_np = np.asarray(fd_grid if fd_grid else [], dtype=float)
                    if fd_flow_np.size > 0 and bandpass_constraints_temporal is not None:
                        try:
                            Cf_flow_clean = (
                                bandpass_constraints_temporal(
                                    Lt=Lt,
                                    prf_hz=prf_hz,
                                    fd_grid_hz=fd_flow_np.tolist(),
                                    device="cpu",
                                    dtype=torch.complex64,
                                    mode="exp",
                                )
                                .detach()
                                .cpu()
                                .numpy()
                            )
                        except Exception:
                            Cf_flow_clean = Cf_flow_np
                    if Cf_alias_np is not None:
                        try:
                            Cf_alias_np = (
                                bandpass_constraints_temporal(
                                    Lt=Lt,
                                    prf_hz=prf_hz,
                                    fd_grid_hz=fd_alias_list if alias_band_hz is not None else [],
                                    device="cpu",
                                    dtype=torch.complex64,
                                    mode="exp",
                                )
                                .detach()
                                .cpu()
                                .numpy()
                            )
                        except Exception:
                            pass
                    R_loaded_base, ka_info_pd = ka_blend_covariance_temporal(
                        R_sample=R_cpu,
                        R0_prior=R0_prior_tile,
                        Cf_flow=Cf_flow_clean,
                        Cf_alias=Cf_alias_np,
                        device="cpu",
                        dtype=torch.complex64,
                        **ka_opts_msd,
                    )
                    if ka_detail_list is not None:
                        ka_detail_list.append(ka_info_pd)
                except Exception as exc_ka:
                    info["ka_warning"] = f"ka_blend_failed:{exc_ka}"
                    ka_active_tile = False
                    R_loaded_base = None
            if R_loaded_base is None:
                eye = torch.eye(R_cpu.shape[0], dtype=R_cpu.dtype)
                R_loaded_base = _hermitianize_tensor(R_cpu + float(lam_for_msd) * eye)
            else:
                R_loaded_base = _hermitianize_tensor(R_loaded_base.to(torch.complex64))

            if ka_active_tile and Pf_pd is not None:
                try:
                    info["ka_pf_rank"] = int(round(torch.real(torch.trace(Pf_pd)).item()))
                except Exception:
                    info["ka_pf_rank"] = None
                if Pa_pd is not None:
                    try:
                        info["ka_alias_rank"] = int(round(torch.real(torch.trace(Pa_pd)).item()))
                    except Exception:
                        info["ka_alias_rank"] = None
                try:
                    band_metrics_local = _generalized_band_metrics(
                        R_cpu, R_loaded_base, Pf_pd, Pa=Pa_pd
                    )
                    mixing_local = _mixing_metric(R_cpu, R_loaded_base, Pf_pd)
                except Exception as exc:
                    band_metrics_local = None
                    mixing_local = None
                    info["ka_band_metrics_error"] = str(exc)
                if band_metrics_local:
                    for key_src, key_dst in [
                        ("pf_min", "ka_pf_lambda_min"),
                        ("pf_max", "ka_pf_lambda_max"),
                        ("pf_mean", "ka_pf_lambda_mean"),
                        ("perp_min", "ka_perp_lambda_min"),
                        ("perp_max", "ka_perp_lambda_max"),
                        ("perp_mean", "ka_perp_lambda_mean"),
                        ("alias_min", "ka_alias_lambda_min"),
                        ("alias_max", "ka_alias_lambda_max"),
                        ("alias_mean", "ka_alias_lambda_mean"),
                        ("noise_min", "ka_noise_lambda_min"),
                        ("noise_max", "ka_noise_lambda_max"),
                        ("noise_mean", "ka_noise_lambda_mean"),
                    ]:
                        val = band_metrics_local.get(key_src)
                        if val is not None:
                            info[key_dst] = float(val)
                    pf_mean_val = band_metrics_local.get("pf_mean")
                    perp_mean_val = band_metrics_local.get("perp_mean")
                    if pf_mean_val is not None and perp_mean_val not in (None, 0):
                        info["ka_snr_flow_ratio"] = float(pf_mean_val) / float(perp_mean_val)
                    if perp_mean_val is not None:
                        info["ka_noise_perp_ratio"] = float(perp_mean_val)
                else:
                    info.setdefault("ka_band_metrics_error", "none")
                if mixing_local is not None:
                    info["ka_operator_mixing_epsilon"] = float(mixing_local)
            elif ka_active_tile:
                info.setdefault("ka_band_metrics_error", "missing_metrics")

            gamma_flow = gamma_perp = None
            if Pf_pd is not None:
                gamma_flow, gamma_perp = _directional_trace_ratio(R_cpu, R_loaded_base, Pf_pd)
            if gamma_flow is not None:
                info["gamma_flow"] = float(gamma_flow)
            if gamma_perp is not None:
                info["gamma_perp"] = float(gamma_perp)

            band_frac_np, score_np, pd_details = _pd_band_energy_attempt(
                R_loaded_base,
                S_cpu,
                Ct_primary,
                ridge=float(msd_ridge),
                ratio_rho=ratio_rho,
                agg_mode=agg_mode,
                cond_threshold=cond_threshold,
                basis_mode="exp+deriv",
            )
            cond_primary = float(pd_details.get("pd_condG", float("nan")))
            info["pd_condG"] = cond_primary
            info["pd_condG_primary"] = cond_primary
            info["pd_gram_ridge_used"] = float(pd_details.get("pd_gram_ridge_used", float("nan")))
            info["pd_basis"] = "exp+deriv"
            if pd_details.get("pd_error"):
                info["pd_retry_reason"] = pd_details["pd_error"]

            success_primary = (
                pd_details.get("pd_success")
                and not pd_details.get("pd_need_retry", False)
                and band_frac_np is not None
                and score_np is not None
            )

            if success_primary:
                band_frac_tile = band_frac_np
                score_tile = score_np
                score_mode = "msd"
                info["fallback"] = False
            else:
                ridge_retry = max(float(msd_ridge) * 1.5, 0.15)
                lam_pf_extra = 0.0
                R_loaded_retry = _hermitianize_tensor(R_loaded_base.clone())
                if Pf_pd is not None and lam_pf_extra > 0.0:
                    R_loaded_retry = _hermitianize_tensor(R_loaded_retry + lam_pf_extra * Pf_pd)
                else:
                    bump = max(0.02 * float(lam_for_msd), 2e-3)
                    eye = torch.eye(R_loaded_retry.shape[0], dtype=R_loaded_retry.dtype)
                    R_loaded_retry = _hermitianize_tensor(R_loaded_retry + bump * eye)

                if Ct_retry is None:
                    Ct_retry = Ct_primary.clone()
                    if Ct_retry.shape[-1] > 1:
                        Ct_retry = Ct_retry[:, :-1].contiguous()

                band_frac_retry, score_retry, pd_retry_details = _pd_band_energy_attempt(
                    R_loaded_retry,
                    S_cpu,
                    Ct_retry,
                    ridge=ridge_retry,
                    ratio_rho=ratio_rho,
                    agg_mode=agg_mode,
                    cond_threshold=cond_threshold,
                    basis_mode="exp_pruned",
                )
                info["pd_retry"] = "exp_retry"
                info["pd_retry_ridge"] = float(ridge_retry)
                info["pd_basis"] = "exp_pruned"
                info["lam_pf_extra_pd"] = float(lam_pf_extra)
                info["pd_condG"] = float(
                    pd_retry_details.get("pd_condG", info.get("pd_condG", float("nan")))
                )
                if pd_retry_details.get("pd_error"):
                    info["pd_retry_reason"] = pd_retry_details["pd_error"]

                success_retry = (
                    pd_retry_details.get("pd_success")
                    and not pd_retry_details.get("pd_need_retry", False)
                    and band_frac_retry is not None
                    and score_retry is not None
                )

                if success_retry:
                    band_frac_tile = band_frac_retry
                    score_tile = score_retry
                    score_mode = "msd"
                    info["fallback"] = False
                else:
                    info["pd_retry"] = "failed"
                    reason = info.get("pd_retry_reason", "pd_retry_failed")
                    raise RuntimeError(reason)

            info["band_fraction_median"] = float(np.median(band_frac_tile))
            info["band_fraction_p90"] = float(np.quantile(band_frac_tile, 0.90))
            info["msd_ratio_rho"] = float(ratio_rho)
            _apply_ka_details()
    except Exception as exc_primary:
        pd_retry_reason = repr(exc_primary)
        score_mode = "none"
        info["fallback"] = True
        info["fallback_exc"] = pd_retry_reason

    if score_mode == "none":
        try:
            Ct_exp = bandpass_constraints_temporal(
                Lt=Lt,
                prf_hz=prf_hz,
                fd_grid_hz=fd_grid,
                device="cpu",
                dtype=torch.complex64,
                mode="exp",
            )
            ridge_retry = max(msd_ridge * 1.5, 0.15)
            ratio_rho = max(0.0, float(msd_ratio_rho))
            S_cpu = S_tile.detach().to(torch.complex64).cpu()
            R_cpu = R_t.detach().to(torch.complex64).cpu()
            eye = torch.eye(R_cpu.shape[0], dtype=R_cpu.dtype)
            R_loaded = _hermitianize_tensor(R_cpu + float(lam_for_msd) * eye)
            band_frac_retry, score_retry, pd_retry_details = _pd_band_energy_attempt(
                R_loaded,
                S_cpu,
                Ct_exp,
                ridge=float(ridge_retry),
                ratio_rho=ratio_rho,
                agg_mode=agg_mode,
                cond_threshold=1e8,
                basis_mode="exp",
            )
            if (
                pd_retry_details.get("pd_success")
                and not pd_retry_details.get("pd_need_retry", False)
                and band_frac_retry is not None
                and score_retry is not None
            ):
                band_frac_tile = band_frac_retry
                score_tile = score_retry
            else:
                raise RuntimeError(pd_retry_details.get("pd_error", "pd retry failed"))
            score_mode = "msd"
            info["band_fraction_median"] = float(np.median(band_frac_tile))
            info["band_fraction_p90"] = float(np.quantile(band_frac_tile, 0.90))
            info["msd_ratio_rho"] = float(ratio_rho)
            info["fallback"] = False
            info["pd_retry"] = "exp_retry"
            info["pd_retry_ridge"] = float(ridge_retry)
            info["pd_condG"] = float(pd_retry_details.get("pd_condG", float("nan")))
            info["pd_gram_ridge_used"] = float(
                pd_retry_details.get("pd_gram_ridge_used", float("nan"))
            )
            info["pd_basis"] = "exp"
            _apply_ka_details()
        except Exception as exc_retry:
            if pd_retry_reason is None:
                pd_retry_reason = repr(exc_retry)
            info["fallback"] = True
            info["fallback_exc"] = pd_retry_reason
            info["pd_retry"] = "failed"
            score_mode = "none"
            band_frac_tile = np.ones((h, w), dtype=np.float32)
            score_tile = np.zeros((h, w), dtype=np.float32)

    if score_mode == "none":
        score_mode = "msd"
    info["score_mode"] = score_mode
    if motion_fraction_t is not None:
        info["motion_fraction_median"] = float(np.nanmedian(motion_fraction_t))
        info["motion_fraction_p90"] = float(np.nanpercentile(motion_fraction_t, 90.0))

    info["pd_lambda"] = float(pd_lam)
    info["msd_lambda"] = float(lam_for_msd)
    info["msd_ridge"] = float(msd_ridge)
    info["msd_agg_mode"] = agg_mode
    info["score_mode"] = score_mode
    info["score_mean"] = float(np.mean(score_tile))
    info["score_var"] = float(np.var(score_tile))
    band_nan = int(np.isnan(band_frac_tile).sum())
    band_inf = int(np.isinf(band_frac_tile).sum())
    score_nan = int(np.isnan(score_tile).sum())
    score_inf = int(np.isinf(score_tile).sum())
    info["band_nan_count"] = band_nan
    info["band_inf_count"] = band_inf
    info["score_nan_count"] = score_nan
    info["score_inf_count"] = score_inf

    band_valid_mask = np.isfinite(band_frac_tile)
    if np.any(band_valid_mask):
        band_vals = band_frac_tile[band_valid_mask]
        info["band_fraction_q10"] = float(np.quantile(band_vals, 0.10))
        info["band_fraction_q50"] = float(np.quantile(band_vals, 0.50))
        info["band_fraction_q90"] = float(np.quantile(band_vals, 0.90))
    else:
        info["band_fraction_q10"] = float("nan")
        info["band_fraction_q50"] = float("nan")
        info["band_fraction_q90"] = float("nan")

    score_valid_mask = np.isfinite(score_tile)
    if np.any(score_valid_mask):
        score_vals = score_tile[score_valid_mask]
        info["score_q10"] = float(np.quantile(score_vals, 0.10))
        info["score_q50"] = float(np.quantile(score_vals, 0.50))
        info["score_q90"] = float(np.quantile(score_vals, 0.90))
    else:
        info["score_q10"] = float("nan")
        info["score_q50"] = float("nan")
        info["score_q90"] = float("nan")

    if capture_debug:
        diag_debug = diag_used if "diag_used" in locals() else None
        load_mode_debug = load_mode_used if "load_mode_used" in locals() else None
        if debug_payload is None:
            debug_payload = {
                "tile": cube_T_hw.copy(),
                "fd_grid": np.array(fd_grid, dtype=np.float32),
                "fd_grid_initial": np.array(fd_grid_initial, dtype=np.float32),
                "Lt": int(Lt),
                "diag_load": diag_debug,
                "load_mode": load_mode_debug,
                "constraint_ridge": float(constraint_ridge),
                "msd_lambda": float(lam_for_msd),
                "msd_ridge": float(msd_ridge),
                "msd_agg_mode": agg_mode,
            }
        debug_payload["band_fraction_tile"] = band_frac_tile.copy()
        debug_payload["motion_fraction_tile"] = (
            motion_fraction_t.copy() if isinstance(motion_fraction_t, np.ndarray) else None
        )
        debug_payload["score_tile"] = score_tile.copy()
        debug_payload["score_mode"] = score_mode
        if use_contrast:
            debug_payload["contrast_flow_tile"] = band_frac_tile.copy()
            debug_payload["contrast_motion_tile"] = (
                motion_fraction_t.copy() if isinstance(motion_fraction_t, np.ndarray) else None
            )
            if pd_tile_override is not None:
                debug_payload["contrast_pd_tile"] = pd_tile_override.copy()
        if "fallback_exc" in info and info["fallback_exc"] is not None:
            debug_payload["fallback_exc"] = info["fallback_exc"]
        debug_payload["band_fraction_quantiles"] = (
            info.get("band_fraction_q10"),
            info.get("band_fraction_q50"),
            info.get("band_fraction_q90"),
        )
        debug_payload["score_quantiles"] = (
            info.get("score_q10"),
            info.get("score_q50"),
            info.get("score_q90"),
        )
        debug_payload["pd_condG"] = info.get("pd_condG")
        debug_payload["pd_retry"] = info.get("pd_retry")
        debug_payload["pd_basis"] = info.get("pd_basis")
        debug_payload["kc_flow"] = info.get("kc_flow")
        debug_payload["kc_motion"] = info.get("kc_motion")
        debug_payload["kc_flow_cap"] = info.get("kc_flow_cap")
        debug_payload["kc_flow_cap_motion"] = info.get("kc_flow_cap_motion")
        debug_payload["kc_flow_freqs"] = info.get("kc_flow_freqs")
        debug_payload["motion_basis_rank"] = info.get("motion_basis_rank")
        debug_payload["psd_kept_bin_count"] = info.get("psd_kept_bin_count")
        debug_payload["psd_alias_select_applied"] = info.get("psd_alias_select_applied")
        debug_payload["energy_kept_ratio"] = info.get("energy_kept_ratio")
        debug_payload["energy_removed_ratio"] = info.get("energy_removed_ratio")
        debug_payload["fd_min_abs_hz"] = info.get("fd_min_abs_hz")
        debug_payload["fd_min_abs_applied"] = info.get("fd_min_abs_applied")
        debug_payload["fd_min_abs_fallback"] = info.get("fd_min_abs_fallback")
        debug_payload["fd_symmetry_added"] = info.get("fd_symmetry_added")
        debug_payload["psd_peak_hz"] = info.get("psd_peak_hz")
        debug_payload["psd_peak_power"] = info.get("psd_peak_power")
        debug_payload["psd_flow_to_dc_ratio"] = info.get("psd_flow_to_dc_ratio")
        debug_payload["psd_power_dc"] = info.get("psd_power_dc")
        debug_payload["psd_power_flow"] = info.get("psd_power_flow")
        debug_payload["psd_power_flow_hz"] = info.get("psd_power_flow_hz")
        debug_payload["constraint_mode"] = info.get("constraint_mode")
        if psd_debug:
            debug_payload.update(psd_debug)
        # Persist KA blend telemetry when available for deeper diagnostics.
        for _key in (
            "ka_beta",
            "ka_mismatch",
            "ka_lambda_used",
            "ka_sigma_min_raw",
            "ka_sigma_max_raw",
        ):
            if info.get(_key) is not None:
                debug_payload[_key] = info.get(_key)
        if info.get("ka_pf_trace_equalized") is not None:
            debug_payload["ka_pf_trace_equalized"] = bool(info.get("ka_pf_trace_equalized"))
        for _key in ("ka_pf_trace_alpha", "ka_pf_trace_loaded", "ka_pf_trace_sample"):
            if info.get(_key) is not None:
                debug_payload[_key] = info.get(_key)
        if info.get("ka_trace_ratio") is not None:
            debug_payload["ka_trace_ratio"] = info.get("ka_trace_ratio")
        if info.get("ka_trace_scale_lock_reason") is not None:
            debug_payload["ka_trace_scale_lock_reason"] = info.get("ka_trace_scale_lock_reason")
        if lcmv_debug_fields:
            debug_payload.update(lcmv_debug_fields)

    return band_frac_tile, score_tile, info, debug_payload


def _stap_pd_tile_lcmv_batch(
    cube_batch_T_hw: torch.Tensor | np.ndarray,
    *,
    prf_hz: float,
    diag_load: float,
    cov_estimator: str,
    huber_c: float,
    mvdr_load_mode: str = "auto",
    mvdr_auto_kappa: float = 50.0,
    constraint_ridge: float = 0.10,
    msd_lambda: float | None = None,
    msd_ridge: float = 0.10,
    msd_agg_mode: str = "trim10",
    msd_ratio_rho: float = 0.0,
    motion_half_span_rel: float | None = None,
    msd_contrast_alpha: float | None = None,
    fd_span_mode: str = "psd",
    fd_span_rel: tuple[float, float] = (0.30, 1.10),
    fd_fixed_span_hz: float | None = None,
    constraint_mode: str = "exp+deriv",
    grid_step_rel: float = 0.05,
    min_pts: int = 9,
    max_pts: int = 21,
    fd_min_abs_hz: float = 0.0,
    alias_psd_select_enable: bool = False,
    alias_psd_select_ratio_thresh: float = 1.2,
    alias_psd_select_bins: int = 1,
    capture_debug: bool = False,
    device: str | None = None,
    cube_capture_flags: Sequence[bool] | None = None,
    ka_mode: str = "none",
    ka_prior_library: np.ndarray | torch.Tensor | None = None,
    ka_opts: dict[str, float] | None = None,
    Lt_fixed: int | None = None,
    post_filter_callback: Callable[[np.ndarray], None] | None = None,
    ka_gate: dict | None = None,
    feasibility_mode: FeasibilityMode = "legacy",
    motion_basis_geom: np.ndarray | None = None,
    alias_band_hz: tuple[float, float] | None = None,
    enable_fast_path: bool | None = None,
    psd_telemetry: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict | None]]:
    """
    Thin batch wrapper around `_stap_pd_tile_lcmv`.

    Current implementation iterates over the batch but provides a drop-in hook for
    future fully batched tensorized STAP. It preserves per-tile debug flags via
    `cube_capture_flags`.
    """

    batch_is_tensor = torch is not None and isinstance(cube_batch_T_hw, torch.Tensor)
    if batch_is_tensor:
        cube_batch_np = cube_batch_T_hw.detach().cpu().numpy()
    else:
        cube_batch_np = np.asarray(cube_batch_T_hw)
    if cube_batch_np.ndim != 4:
        raise ValueError(
            f"Expected cube_batch_T_hw with 4 dims [B,T,h,w], got {cube_batch_np.shape}"
        )
    B = cube_batch_np.shape[0]
    lt_for_batch = int(cube_batch_np.shape[1] if Lt_fixed is None else Lt_fixed)

    # Optional fast-path (batched STAP core) when explicitly enabled.
    fast_flag = bool(enable_fast_path) if enable_fast_path is not None else False
    if enable_fast_path is None:
        env_fast = os.getenv("STAP_FAST_PATH", "").lower()
        fast_flag = env_fast in {"1", "true", "yes", "on"}
    debug_requested = bool(capture_debug) or (
        cube_capture_flags is not None and any(bool(x) for x in cube_capture_flags)
    )
    fast_eligible = (
        fast_flag
        and _STAP_AVAILABLE
        and stap_temporal_core_batched is not None
        and torch is not None
        and ka_mode == "none"
        and not debug_requested
    )
    if fast_eligible:
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cube_tensor = torch.as_tensor(cube_batch_np, dtype=torch.complex64, device=dev)
        use_ref_cov = os.getenv("STAP_FAST_COV_REF", "").lower() in {"1", "true", "yes", "on"}
        pd_only = os.getenv("STAP_FAST_PD_ONLY", "").lower() in {"1", "true", "yes", "on"}
        if pd_only:
            band_frac_fast, score_fast, info_fast = pd_temporal_core_batched(
                cube_tensor,
                prf_hz=prf_hz,
                Lt=lt_for_batch,
                diag_load=diag_load,
                cov_estimator=cov_estimator,
                huber_c=huber_c,
                grid_step_rel=grid_step_rel,
                fd_span_rel=fd_span_rel,
                min_pts=min_pts,
                max_pts=max_pts,
                fd_min_abs_hz=fd_min_abs_hz,
                motion_half_span_rel=motion_half_span_rel,
                msd_ridge=msd_ridge,
                msd_agg_mode=msd_agg_mode,
                msd_ratio_rho=msd_ratio_rho,
                msd_contrast_alpha=msd_contrast_alpha if msd_contrast_alpha is not None else 0.0,
                msd_lambda=msd_lambda,
                device=dev,
                use_ref_cov=use_ref_cov,
            )
        else:
            band_frac_fast, score_fast, info_fast = stap_temporal_core_batched(
                cube_tensor,
                prf_hz=prf_hz,
                Lt=lt_for_batch,
                diag_load=diag_load,
                cov_estimator=cov_estimator,
                huber_c=huber_c,
                grid_step_rel=grid_step_rel,
                fd_span_rel=fd_span_rel,
                min_pts=min_pts,
                max_pts=max_pts,
                fd_min_abs_hz=fd_min_abs_hz,
                motion_half_span_rel=motion_half_span_rel,
                msd_ridge=msd_ridge,
                msd_agg_mode=msd_agg_mode,
                msd_ratio_rho=msd_ratio_rho,
                msd_contrast_alpha=msd_contrast_alpha if msd_contrast_alpha is not None else 0.0,
                msd_lambda=msd_lambda,
                device=dev,
                use_ref_cov=use_ref_cov,
            )
        for d in info_fast:
            d.setdefault("stap_fast_path_used", True)
        debug_list_fast: list[dict | None] = [None] * B
        band_frac_batch = np.asarray(band_frac_fast, dtype=np.float32)
        score_batch = np.asarray(score_fast, dtype=np.float32)
        return band_frac_batch, score_batch, info_fast, debug_list_fast

    band_frac_list: list[np.ndarray] = []
    score_list: list[np.ndarray] = []
    info_list: list[dict] = []
    debug_list: list[dict | None] = []

    for idx in range(B):
        cap_flag = bool(capture_debug)
        if cube_capture_flags is not None and idx < len(cube_capture_flags):
            cap_flag = cap_flag or bool(cube_capture_flags[idx])
        cube_np = cube_batch_np[idx]
        cube_tensor = cube_batch_T_hw[idx] if batch_is_tensor else None
        band_frac_tile, score_tile, info_tile, debug_payload = _stap_pd_tile_lcmv(
            cube_np,
            prf_hz=prf_hz,
            diag_load=diag_load,
            cov_estimator=cov_estimator,
            huber_c=huber_c,
            mvdr_load_mode=mvdr_load_mode,
            mvdr_auto_kappa=mvdr_auto_kappa,
            constraint_ridge=constraint_ridge,
            fd_span_mode=fd_span_mode,
            fd_span_rel=fd_span_rel,
            fd_fixed_span_hz=fd_fixed_span_hz,
            constraint_mode=constraint_mode,
            grid_step_rel=grid_step_rel,
            min_pts=min_pts,
            max_pts=max_pts,
            fd_min_abs_hz=fd_min_abs_hz,
            alias_psd_select_enable=alias_psd_select_enable,
            alias_psd_select_ratio_thresh=alias_psd_select_ratio_thresh,
            alias_psd_select_bins=alias_psd_select_bins,
            msd_lambda=msd_lambda,
            msd_ridge=msd_ridge,
            msd_agg_mode=msd_agg_mode,
            msd_ratio_rho=msd_ratio_rho,
            motion_half_span_rel=motion_half_span_rel,
            msd_contrast_alpha=msd_contrast_alpha,
            capture_debug=cap_flag,
            device=device,
            cube_tensor=cube_tensor,
            ka_mode=ka_mode,
            ka_prior_library=ka_prior_library,
            ka_opts=ka_opts,
            Lt_fixed=Lt_fixed,
            post_filter_callback=post_filter_callback,
            ka_gate=ka_gate,
            feasibility_mode=feasibility_mode,
            motion_basis_geom=motion_basis_geom,
            alias_band_hz=alias_band_hz,
            psd_telemetry=psd_telemetry,
        )
        band_frac_list.append(band_frac_tile)
        score_list.append(score_tile)
        info_list.append(info_tile)
        debug_list.append(debug_payload)

    band_frac_batch = np.stack(band_frac_list, axis=0).astype(np.float32, copy=False)
    score_batch = np.stack(score_list, axis=0).astype(np.float32, copy=False)
    return band_frac_batch, score_batch, info_list, debug_list


def _stap_pd(
    Icube: np.ndarray,
    *,
    tile_hw: tuple[int, int],
    stride: int,
    Lt: int,
    prf_hz: float,
    diag_load: float,
    estimator: str,
    huber_c: float,
    mvdr_load_mode: str = "auto",
    mvdr_auto_kappa: float = 50.0,
    constraint_ridge: float = 0.10,
    fd_span_mode: str = "psd",
    fd_span_rel: tuple[float, float] = (0.30, 1.10),
    fd_fixed_span_hz: float | None = None,
    constraint_mode: str = "exp+deriv",
    grid_step_rel: float = 0.05,
    min_pts: int = 9,
    max_pts: int = 21,
    fd_min_abs_hz: float = 0.0,
    msd_lambda: float | None = None,
    msd_ridge: float = 0.10,
    msd_agg_mode: str = "trim10",
    msd_ratio_rho: float = 0.0,
    motion_half_span_rel: float | None = None,
    msd_contrast_alpha: float | None = None,
    debug_max_samples: int = 0,
    debug_tile_coords: Sequence[tuple[int, int]] | None = None,
    stap_device: str | None = None,
    tile_batch: int | None = None,
    pd_base_full: np.ndarray | None = None,
    mask_flow: np.ndarray | None = None,
    mask_bg: np.ndarray | None = None,
    ka_mode: str = "none",
    ka_prior_library: np.ndarray | torch.Tensor | None = None,
    ka_opts: dict[str, float] | None = None,
    alias_psd_select_enable: bool = False,
    alias_psd_select_ratio_thresh: float = 1.2,
    alias_psd_select_bins: int = 1,
    psd_telemetry: bool = False,
    psd_tapers: int = 3,
    psd_bandwidth: float = 2.0,
    band_ratio_recorder: BandRatioRecorder | None = None,
    feasibility_mode: FeasibilityMode = "legacy",
    band_ratio_spec: dict[str, float] | None = None,
    tile_debug_limit: int | None = None,
    flow_alias_hz: float | None = None,
    flow_alias_fraction: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    feas_mode = _normalize_feasibility_mode(feasibility_mode)
    if not _STAP_AVAILABLE:
        pd_map = _baseline_pd(Icube)
        zeros = np.zeros_like(pd_map, dtype=np.float32)
        return (
            pd_map,
            zeros,
            {
                "fallback_count": 0,
                "total_tiles": 0,
                "fallback_samples": [],
                "score_mode_histogram": {"none": 0},
                "debug_samples": [],
                "requested_msd_lambda": None,
                "requested_msd_ridge": float(msd_ridge),
                "msd_agg_mode": msd_agg_mode,
            },
        )

    _, H, W = Icube.shape
    th, tw = tile_hw
    device_resolved = _resolve_stap_device(stap_device)
    pd = np.zeros((H, W), dtype=np.float64)
    score = np.zeros((H, W), dtype=np.float64)
    counts = np.zeros((H, W), dtype=np.float64)
    score_counts = np.zeros((H, W), dtype=np.float64)
    tile_infos: list[dict] = []
    motion_basis_geom_np: np.ndarray | None = None
    if (
        motion_basis_geom_np is None
        and build_motion_basis_temporal is not None
        and torch is not None
    ):
        try:
            motion_basis_global = build_motion_basis_temporal(  # type: ignore[call-arg]
                Lt=Lt,
                prf_hz=prf_hz,
                width_bins=1,
                include_dc=True,
                device="cpu",
                dtype=torch.complex64,
            )
            if isinstance(motion_basis_global, torch.Tensor):
                motion_basis_geom_np = motion_basis_global.detach().cpu().numpy()
            elif motion_basis_global is not None:
                motion_basis_geom_np = np.asarray(motion_basis_global, dtype=np.complex64)
        except Exception:
            motion_basis_geom_np = None
    gate_mask_flow: np.ndarray | None = (
        np.zeros_like(mask_flow, dtype=bool) if mask_flow is not None else None
    )
    gate_mask_bg: np.ndarray | None = (
        np.zeros_like(mask_bg, dtype=bool) if mask_bg is not None else None
    )
    if band_ratio_spec is None:
        if feas_mode == "updated":
            band_ratio_spec_local = _auto_band_ratio_spec(
                prf_hz, Lt, flow_alias_hz=flow_alias_hz, flow_alias_fraction=flow_alias_fraction
            )
            band_ratio_spec_local.setdefault("alias_min_bins", 3)
            band_ratio_spec_local.setdefault("alias_max_bins", 8)
        else:
            band_ratio_spec_local = {
                "flow_low_hz": 120.0,
                "flow_high_hz": 400.0,
                "alias_center_hz": 900.0,
                "alias_width_hz": 15.625,
            }
    else:
        band_ratio_spec_local = dict(band_ratio_spec)

    gate_flow_alias_vals: list[float] = []
    gate_bg_alias_vals: list[float] = []
    debug_payloads: list[dict] = []
    alias_band_hz_tuple: tuple[float, float] | None = None
    # Runtime breakdown accumulators (seconds)
    t_extract = 0.0
    t_batch_proc = 0.0
    t_post = 0.0
    try:
        alias_center_hz = float(band_ratio_spec_local.get("alias_center_hz"))
        alias_width_hz = float(band_ratio_spec_local.get("alias_width_hz"))
        alias_low_hz = alias_center_hz - 0.5 * alias_width_hz
        alias_high_hz = alias_center_hz + 0.5 * alias_width_hz
        alias_band_hz_tuple = (alias_low_hz, alias_high_hz)
    except Exception:
        alias_band_hz_tuple = None

    if tile_batch is None or tile_batch <= 0:
        tile_batch_size = 1
    else:
        tile_batch_size = int(tile_batch)
    if device_resolved.lower().startswith("cuda"):
        if tile_batch is None or tile_batch <= 0:
            tile_batch_size = 192
    else:
        tile_batch_size = max(1, tile_batch_size)
        if tile_batch_size > 1:
            tile_batch_size = 1

    tile_batch_items: list[np.ndarray] = []
    tile_positions: list[tuple[int, int]] = []
    tile_capture_flags: list[bool] = []
    tile_capture_coord_flags: list[bool] = []
    tile_batch_indices: list[int] = []

    total_tiles_y = max(0, (H - th) // stride + 1)
    total_tiles_x = max(0, (W - tw) // stride + 1)
    total_tiles = total_tiles_y * total_tiles_x
    stap_fast_any = False
    # Build tile ordering. In debug-limit mode, prioritize tiles with flow coverage > 0
    # so that telemetry captures KA/flow behavior without scanning the whole raster.
    coords_ordered: list[tuple[float, int, int]] = []
    if tile_debug_limit is not None:
        for y0 in range(0, H - th + 1, stride):
            for x0 in range(0, W - tw + 1, stride):
                cov_flow = 0.0
                if mask_flow is not None:
                    patch = mask_flow[y0 : y0 + th, x0 : x0 + tw]
                    cov_flow = float(np.mean(patch))
                coords_ordered.append((cov_flow, y0, x0))
        coords_ordered.sort(key=lambda t: (-t[0], t[1], t[2]))
    else:
        for y0 in range(0, H - th + 1, stride):
            for x0 in range(0, W - tw + 1, stride):
                coords_ordered.append((0.0, y0, x0))
    total_tiles = len(coords_ordered)
    t0_stap = time.perf_counter()
    capture_indices: set[int] = set()
    if debug_max_samples > 0 and total_tiles > 0:
        capture_count = min(debug_max_samples, total_tiles)
        # Evenly sample tile indices across the raster order to cover depth and lateral span.
        sample_positions = np.linspace(0, total_tiles - 1, capture_count, dtype=int)
        capture_indices = set(int(idx) for idx in np.unique(sample_positions))
    capture_coord_requests: set[tuple[int, int]] = set()
    remaining_coord_requests: set[tuple[int, int]] = set()
    if debug_tile_coords:
        capture_coord_requests = {(int(y), int(x)) for y, x in debug_tile_coords}
        remaining_coord_requests = set(capture_coord_requests)
    tile_counter = 0
    stap_tiles_skipped_flow0 = 0
    bg_edge_clamp_total = 0
    psd_freq_cache: np.ndarray | None = None
    bg_psd_accum: np.ndarray | None = None
    bg_psd_count = 0
    stap_fast_any = False

    priority_debug_cap = 0
    if debug_max_samples > 0:
        priority_debug_cap = min(16, max(2, debug_max_samples // 4 or 1))
    priority_debug_remaining = priority_debug_cap
    priority_band_thresh = 0.99
    priority_bg_thresh = 1.1
    priority_alias_thresh = 2.0
    if feas_mode == "updated":
        guard_tile_coverage_min = 0.0

    ka_mode_norm = (ka_mode or "none").strip().lower()
    ka_active = ka_mode_norm not in {"", "none"}
    ka_opts_dict = dict(ka_opts) if ka_opts else {}
    if ka_active and "kappa_target" not in ka_opts_dict:
        ka_opts_dict["kappa_target"] = 40.0
    guard_tile_coverage_min = float(ka_opts_dict.pop("guard_tile_coverage_min", 0.1))
    guard_percentile_low = float(ka_opts_dict.pop("guard_percentile_low", 0.10))
    guard_target_low = float(ka_opts_dict.pop("guard_target_low", 0.16))
    guard_target_med = float(ka_opts_dict.pop("guard_target_med", 0.45))
    guard_max_scale = float(ka_opts_dict.pop("guard_max_scale", 1.30))
    guard_clip_base = bool(ka_opts_dict.pop("guard_clip_base", True))
    coverage_cap_enable = bool(ka_opts_dict.pop("coverage_cap_enable", False))
    # Alias-aware cap options (default disabled to preserve legacy behavior)
    alias_cap_enable = bool(ka_opts_dict.pop("alias_cap_enable", False))
    alias_cap_force = bool(ka_opts_dict.pop("alias_cap_force", False))
    alias_cap_alias_thresh = float(ka_opts_dict.pop("alias_cap_alias_thresh", 2.0))
    alias_cap_band_med_thresh = float(ka_opts_dict.pop("alias_cap_band_med_thresh", 0.98))
    alias_cap_smin = float(ka_opts_dict.pop("alias_cap_smin", 0.3))
    alias_cap_c0 = float(ka_opts_dict.pop("alias_cap_c0", 1.0))
    alias_cap_exp = float(ka_opts_dict.pop("alias_cap_exp", 1.0))
    alias_psd_select_enable_flag = bool(alias_psd_select_enable)
    alias_psd_select_ratio_thresh_val = float(alias_psd_select_ratio_thresh)
    alias_psd_select_bins_val = max(1, int(alias_psd_select_bins))
    if "alias_psd_select_enable" in ka_opts_dict:
        alias_psd_select_enable_flag = bool(ka_opts_dict.pop("alias_psd_select_enable"))
    if "alias_psd_select_ratio_thresh" in ka_opts_dict:
        try:
            alias_psd_select_ratio_thresh_val = float(
                ka_opts_dict.pop("alias_psd_select_ratio_thresh")
            )
        except (TypeError, ValueError):
            pass
    elif feas_mode == "updated":
        # Be inclusive for feasibility diagnostics: allow alias peaks whenever alias_ratio >= 1.0
        alias_psd_select_ratio_thresh_val = min(alias_psd_select_ratio_thresh_val, 1.0)
    if "alias_psd_select_bins" in ka_opts_dict:
        try:
            alias_psd_select_bins_val = max(
                1, int(float(ka_opts_dict.pop("alias_psd_select_bins")))
            )
        except (TypeError, ValueError):
            pass
    ka_gate_cfg_base: dict[str, float | bool | None] | None = None
    ka_gate_enable_flag = bool(ka_opts_dict.pop("ka_gate_enable", False))
    if ka_gate_enable_flag:
        alias_rmin = float(ka_opts_dict.pop("ka_gate_alias_rmin", 1.10))
        flow_cov_min_gate = float(
            ka_opts_dict.pop("ka_gate_flow_cov_min", guard_tile_coverage_min)
        )
        depth_min_gate = float(ka_opts_dict.pop("ka_gate_depth_min_frac", 0.20))
        depth_max_gate = float(ka_opts_dict.pop("ka_gate_depth_max_frac", 0.95))
        pd_min_gate = ka_opts_dict.pop("ka_gate_pd_min", None)
        pd_min_val = None
        if pd_min_gate is not None:
            try:
                pd_min_val = float(pd_min_gate)
            except (TypeError, ValueError):
                pd_min_val = None
        reg_psr_gate = ka_opts_dict.pop("ka_gate_reg_psr_max", None)
        reg_psr_val = None
        if reg_psr_gate is not None:
            try:
                reg_psr_val = float(reg_psr_gate)
            except (TypeError, ValueError):
                reg_psr_val = None
        ka_gate_cfg_base = {
            "enable": True,
            "alias_rmin": alias_rmin,
            "flow_cov_min": max(0.0, min(1.0, flow_cov_min_gate)),
            "depth_min_frac": max(0.0, min(1.0, depth_min_gate)),
            "depth_max_frac": max(0.0, min(1.0, depth_max_gate)),
            "pd_min": pd_min_val,
            "reg_psr_max": reg_psr_val,
            "mode": feas_mode,
        }
    else:
        # Remove unused gate-related keys if present
        for key in (
            "ka_gate_alias_rmin",
            "ka_gate_flow_cov_min",
            "ka_gate_depth_min_frac",
            "ka_gate_depth_max_frac",
            "ka_gate_pd_min",
            "ka_gate_reg_psr_max",
        ):
            ka_opts_dict.pop(key, None)
    # Background variance guard (defaults: enabled, p90 cap at 1.15x, min shrink 0.6)
    bg_guard_enabled = bool(ka_opts_dict.pop("bg_guard_enabled", True))
    bg_guard_target_p90 = float(ka_opts_dict.pop("bg_guard_target_p90", 1.15))
    bg_guard_min_alpha = float(ka_opts_dict.pop("bg_guard_min_alpha", 0.6))
    bg_guard_metric = str(ka_opts_dict.pop("bg_guard_metric", "tile_p90")).strip().lower()
    # bg_guard_metric in {"tile_p90","global"}
    legacy_guard_min: float | None = None
    if ka_active and ka_opts_dict:
        for key in ("global_flow_ratio_min", "flow_ratio_guard_min"):
            if key in ka_opts_dict:
                try:
                    val = float(ka_opts_dict.pop(key))
                except (TypeError, ValueError):
                    val = None
                if val is not None and np.isfinite(val) and val > 0.0:
                    legacy_guard_min = float(val)
                break
        for key in ("global_flow_ratio_clip_base", "flow_ratio_guard_clip_base"):
            if key in ka_opts_dict:
                guard_clip_base = bool(ka_opts_dict.pop(key))
                break
    if isinstance(ka_prior_library, torch.Tensor):
        ka_prior_np = ka_prior_library.detach().cpu().numpy()
    else:
        ka_prior_np = ka_prior_library

    pd_gate_reference = None
    if ka_gate_cfg_base is not None and pd_base_full is not None:
        # Reference PD level (median) for normalization.
        if feas_mode == "updated":
            vals = pd_base_full[np.isfinite(pd_base_full)]
        elif mask_flow is not None and mask_flow.any():
            vals = pd_base_full[mask_flow]
        else:
            vals = pd_base_full[np.isfinite(pd_base_full)]
        finite = np.asarray(vals[np.isfinite(vals)], dtype=float)
        if finite.size:
            pd_gate_reference = float(np.median(finite))
        ka_gate_cfg_base["pd_reference"] = pd_gate_reference
        # Optional score-quantile window: estimate PD quantiles on BG tiles within
        # the gating depth band so that we can restrict KA to high-score negatives.
        depth_min_gate = float(ka_gate_cfg_base.get("depth_min_frac", 0.0))
        depth_max_gate = float(ka_gate_cfg_base.get("depth_max_frac", 1.0))
        if mask_bg is not None:
            H, W = mask_bg.shape
            y = np.arange(H, dtype=float)
            depth_frac = (y[:, None] + 0.5 * th) / float(H) if H > 0 else None
            if depth_frac is not None:
                depth_mask = (depth_frac >= depth_min_gate) & (depth_frac <= depth_max_gate)
                bg_mask = mask_bg & depth_mask
            else:
                bg_mask = mask_bg
            pd_bg_vals = pd_base_full[bg_mask] if bg_mask is not None else None
        else:
            pd_bg_vals = None
        pd_q_lo = None
        pd_q_hi = None
        if pd_bg_vals is not None:
            finite_bg = np.asarray(pd_bg_vals[np.isfinite(pd_bg_vals)], dtype=float)
            if finite_bg.size >= 10:
                # Fixed high-score window for now: 70th-99th percentiles.
                try:
                    pd_q_lo = float(np.quantile(finite_bg, 0.7))
                    pd_q_hi = float(np.quantile(finite_bg, 0.99))
                except Exception:
                    pd_q_lo = None
                    pd_q_hi = None
        ka_gate_cfg_base["pd_q_lo"] = pd_q_lo
        ka_gate_cfg_base["pd_q_hi"] = pd_q_hi

    # When KA gating is enabled, ensure that each tile goes through the
    # per-tile `_stap_pd_tile_lcmv` path so that the KA gate sees the full
    # per-tile context (alias metric, flow coverage, depth, PD). The batched
    # wrapper `_stap_pd_tile_lcmv_batch` only accepts a single `ka_gate`
    # payload and cannot currently supply tile-wise gating context, so using
    # batched STAP together with KA gating would silently bypass the gate.
    ka_mode_overall = ka_mode_norm if ka_active else "none"
    if ka_mode_overall != "none" and ka_gate_cfg_base is not None:
        tile_batch_size = 1

    def process_tile(
        cube_np: np.ndarray,
        y0: int,
        x0: int,
        tile_idx: int,
        capture_requested: bool,
        capture_by_coord: bool,
        cube_tensor: torch.Tensor | None = None,
        precomputed: tuple[np.ndarray, np.ndarray, dict, dict | None] | None = None,
    ) -> None:
        nonlocal pd, score, counts, score_counts
        nonlocal tile_infos, debug_payloads, priority_debug_remaining, bg_edge_clamp_total
        nonlocal gate_mask_flow, gate_mask_bg
        nonlocal psd_freq_cache, bg_psd_accum, bg_psd_count
        capture_debug = False
        if capture_requested:
            if capture_by_coord:
                capture_debug = True
            elif debug_max_samples > 0 and len(debug_payloads) < debug_max_samples:
                capture_debug = True
        bg_mask_tile = None
        tile_is_bg = False
        if mask_bg is not None:
            bg_mask_tile = mask_bg[y0 : y0 + th, x0 : x0 + tw]
            tile_is_bg = bool(bg_mask_tile.all())
        flow_mask_tile = None
        tile_has_flow = False
        flow_cov_ratio = None
        if mask_flow is not None:
            flow_mask_tile = mask_flow[y0 : y0 + th, x0 : x0 + tw]
            tile_has_flow = bool(flow_mask_tile.any())
            flow_cov_ratio = float(flow_mask_tile.sum()) / float(flow_mask_tile.size)
        base_tile_cached = (
            pd_base_full[y0 : y0 + th, x0 : x0 + tw] if pd_base_full is not None else None
        )
        pd_metric_raw = None
        if base_tile_cached is not None:
            if feas_mode == "updated":
                pd_metric_raw = float(np.mean(base_tile_cached))
            elif flow_mask_tile is not None and tile_has_flow:
                pd_metric_raw = float(np.mean(base_tile_cached[flow_mask_tile]))
        pd_metric_norm = None
        if (
            pd_metric_raw is not None
            and ka_gate_cfg_base is not None
            and ka_gate_cfg_base.get("pd_reference") is not None
        ):
            denom = max(float(ka_gate_cfg_base["pd_reference"]), 1e-12)
            pd_metric_norm = float(pd_metric_raw / denom)
        depth_center_frac = (y0 + 0.5 * th) / float(H) if H > 0 else None
        gate_payload = None
        alias_metric_mt: float | None = None
        alias_metrics_mt: dict[str, float | None] | None = None
        if ka_gate_cfg_base is not None and band_ratio_spec_local is not None:
            try:
                alias_metrics_mt = _compute_alias_metrics_mt(
                    cube_np,
                    prf_hz,
                    band_ratio_spec_local,
                    tapers=psd_tapers,
                    bandwidth=psd_bandwidth,
                )
                alias_metric_val = alias_metrics_mt.get("m_alias")
                if alias_metric_val is not None and np.isfinite(alias_metric_val):
                    alias_metric_mt = float(alias_metric_val)
            except Exception:
                alias_metrics_mt = None
        if ka_gate_cfg_base is not None:
            gate_payload = dict(ka_gate_cfg_base)
            gate_payload["context"] = {
                "flow_cov": flow_cov_ratio,
                "depth_frac": depth_center_frac,
                "pd_metric": pd_metric_raw,
                "pd_norm": pd_metric_norm,
                "reg_psr": None,
                "tile_has_flow": tile_has_flow,
                "tile_is_bg": tile_is_bg,
                "feasibility_mode": feas_mode,
            }
            if alias_metric_mt is not None:
                gate_payload["context"]["alias_metric"] = alias_metric_mt

        br_callback = None
        if band_ratio_recorder is not None:

            def _capture(series: np.ndarray, *, idx: int = tile_idx, is_bg: bool = tile_is_bg):
                band_ratio_recorder.observe(idx, series, is_bg)

            br_callback = _capture

        if precomputed is None:
            band_frac_tile, score_tile, info_tile, debug_payload = _stap_pd_tile_lcmv(
                cube_np,
                prf_hz=prf_hz,
                diag_load=diag_load,
                cov_estimator=estimator,
                huber_c=huber_c,
                mvdr_load_mode=mvdr_load_mode,
                mvdr_auto_kappa=mvdr_auto_kappa,
                constraint_ridge=constraint_ridge,
                fd_span_mode=fd_span_mode,
                fd_span_rel=fd_span_rel,
                fd_fixed_span_hz=fd_fixed_span_hz,
                constraint_mode=constraint_mode,
                grid_step_rel=grid_step_rel,
                min_pts=min_pts,
                max_pts=max_pts,
                fd_min_abs_hz=fd_min_abs_hz,
                alias_psd_select_enable=alias_psd_select_enable_flag,
                alias_psd_select_ratio_thresh=alias_psd_select_ratio_thresh_val,
                alias_psd_select_bins=alias_psd_select_bins_val,
                msd_lambda=msd_lambda,
                msd_ridge=msd_ridge,
                msd_agg_mode=msd_agg_mode,
                msd_ratio_rho=msd_ratio_rho,
                motion_half_span_rel=motion_half_span_rel,
                msd_contrast_alpha=msd_contrast_alpha,
                capture_debug=capture_debug,
                device=device_resolved,
                cube_tensor=cube_tensor,
                ka_mode=ka_mode_norm if ka_active else "none",
                ka_prior_library=ka_prior_np if ka_active else None,
                ka_opts=ka_opts_dict if ka_active else None,
                Lt_fixed=Lt,
                post_filter_callback=br_callback,
                ka_gate=gate_payload,
                feasibility_mode=feas_mode,
                motion_basis_geom=motion_basis_geom_np,
                alias_band_hz=alias_band_hz_tuple,
                psd_telemetry=psd_telemetry,
            )
        else:
            band_frac_tile, score_tile, info_tile, debug_payload = precomputed

        if alias_metric_mt is not None:
            info_tile["ka_alias_metric"] = float(alias_metric_mt)
        if alias_metrics_mt is not None:
            for key in ("E_f", "E_a", "E_g", "E_dc", "c_f", "c_a", "r_g"):
                val = alias_metrics_mt.get(key)
                if val is not None and np.isfinite(val):
                    info_tile[f"ka_alias_{key}"] = float(val)

        info_tile["tile_has_flow"] = bool(tile_has_flow)
        info_tile["tile_is_bg"] = bool(tile_is_bg)
        info_tile["depth_center_frac"] = (
            float(depth_center_frac) if depth_center_frac is not None else None
        )

        if capture_debug and debug_payload is not None:
            debug_payload["tile_index"] = int(tile_idx)
            debug_payload["y0"] = int(y0)
            debug_payload["x0"] = int(x0)

        band_frac_tile = np.asarray(band_frac_tile, dtype=np.float64)
        score_tile = np.asarray(score_tile, dtype=np.float64)

        if base_tile_cached is not None:
            base_tile = base_tile_cached
        else:
            base_tile = np.mean(np.abs(cube_np) ** 2, axis=0)
        pd_tile = base_tile * band_frac_tile
        background_uniformized = False
        subtile_background_uniformized = False
        flow_mask_tile = None
        flow_mask_empty = False
        band_fraction_capped = False
        band_fraction_cap_scale = None
        alias_cap_applied = False
        alias_cap_scale = None
        if flow_mask_tile is None and mask_flow is not None:
            flow_mask_tile = mask_flow[y0 : y0 + th, x0 : x0 + tw]
            flow_mask_empty = not bool(flow_mask_tile.any())
            if flow_mask_empty:
                pd_tile = base_tile.copy()
                # For tiles with zero flow support, we explicitly fall back to
                # baseline PD. Reflect that choice in the band-fraction output
                # so debug/telemetry consumers see an identity-like value.
                band_frac_tile = np.ones_like(band_frac_tile, dtype=band_frac_tile.dtype)
                # Similarly, keep the detector score neutral for this tile so
                # score maps behave like the conditional-STAP skip path.
                score_tile = np.zeros_like(score_tile, dtype=score_tile.dtype)
                if capture_debug and debug_payload is not None:
                    debug_payload["band_fraction_tile"] = band_frac_tile.copy()
                    debug_payload["score_tile"] = score_tile.copy()
                background_uniformized = True
            else:
                flow_cov_ratio = float(flow_mask_tile.sum()) / float(flow_mask_tile.size)
                if coverage_cap_enable and (0.0 < flow_cov_ratio < guard_tile_coverage_min):
                    cap_scale = max(flow_cov_ratio / max(guard_tile_coverage_min, 1e-6), 0.0)
                    if cap_scale < 0.999:
                        band_frac_tile = band_frac_tile * cap_scale
                        score_tile = score_tile * cap_scale
                        pd_tile = base_tile * band_frac_tile
                        band_fraction_capped = True
                        band_fraction_cap_scale = float(cap_scale)
                # Alias-aware cap on flow region only
                alias_ratio = info_tile.get("psd_flow_alias_ratio")
                band_med = info_tile.get("band_fraction_median")
                if alias_cap_enable and flow_cov_ratio >= guard_tile_coverage_min:
                    meets_band = (
                        band_med if band_med is not None else float(np.median(band_frac_tile))
                    ) >= alias_cap_band_med_thresh
                    ar = (
                        float(alias_ratio)
                        if (alias_ratio is not None and np.isfinite(alias_ratio))
                        else 0.0
                    )
                    meets_alias = (ar >= alias_cap_alias_thresh) or alias_cap_force
                    if meets_band and meets_alias:
                        denom = max(
                            (ar if not alias_cap_force else max(ar, 1.0))
                            ** max(alias_cap_exp, 0.0),
                            1e-6,
                        )
                        s = min(1.0, max(alias_cap_smin, alias_cap_c0 / denom))
                        if s < 0.999:
                            band_frac_tile[flow_mask_tile] = band_frac_tile[flow_mask_tile] * s
                            score_tile[flow_mask_tile] = score_tile[flow_mask_tile] * s
                            pd_tile = base_tile * band_frac_tile
                            alias_cap_applied = True
                            alias_cap_scale = float(s)
                # Subtile background clamping: on background portion only, preserve baseline PD.
                # We leave MSD / score_tile unchanged so that MSD ROC remains informative.
                if bg_mask_tile is not None:
                    pd_tile[bg_mask_tile] = base_tile[bg_mask_tile]
                    subtile_background_uniformized = True
        info_tile["background_uniformized"] = background_uniformized
        info_tile["subtile_background_uniformized"] = bool(subtile_background_uniformized)
        info_tile["band_fraction_capped"] = bool(band_fraction_capped)
        if band_fraction_cap_scale is not None:
            info_tile["band_fraction_cap_scale"] = band_fraction_cap_scale
        info_tile["alias_cap_applied"] = bool(alias_cap_applied)
        if alias_cap_scale is not None:
            info_tile["alias_cap_scale"] = alias_cap_scale

        # Final safety clamp: ensure background pixels match baseline PD exactly,
        # even on single-overlap edges where downstream operations might alter them.
        # We do not clamp MSD here so that MSD-based ROC on H0 remains meaningful.
        if mask_bg is not None:
            bg_mask_tile = mask_bg[y0 : y0 + th, x0 : x0 + tw]
            if bg_mask_tile is not None:
                diff_mask = bg_mask_tile & (np.abs(pd_tile - base_tile) > 1e-9)
                if diff_mask.any():
                    pd_tile = pd_tile.copy()
                    band_frac_tile = band_frac_tile.copy()
                    pd_tile[diff_mask] = base_tile[diff_mask]
                    band_frac_tile[diff_mask] = 1.0
                    info_tile["bg_edge_clamped_count"] = int(diff_mask.sum())
                    bg_edge_clamp_total += int(diff_mask.sum())
        info_tile["flow_mask_empty"] = bool(flow_mask_empty)
        if flow_mask_empty and not background_uniformized:
            raise RuntimeError(
                "flow coverage is zero but background_uniformized flag was not triggered"
            )
        pd_override = info_tile.get("pd_tile_override")
        if pd_override is not None:
            pd_tile = np.asarray(pd_override, dtype=np.float64)

        flow_ratio = None
        bg_inflation = None
        flow_coverage = None
        if pd_base_full is not None and mask_flow is not None and mask_bg is not None:
            base_tile = pd_base_full[y0 : y0 + th, x0 : x0 + tw]
            if flow_mask_tile is None:
                flow_mask_tile = mask_flow[y0 : y0 + th, x0 : x0 + tw]
            bg_mask_tile = mask_bg[y0 : y0 + th, x0 : x0 + tw]
            if flow_mask_tile.any():
                mu_base = float(base_tile[flow_mask_tile].mean())
                mu_stap = float(pd_tile[flow_mask_tile].mean())
                flow_ratio = mu_stap / max(mu_base, 1e-12)
                info_tile["flow_mu_base"] = mu_base
                info_tile["flow_mu_stap"] = mu_stap
                info_tile["flow_mu_ratio"] = flow_ratio
                flow_coverage = float(flow_mask_tile.sum()) / float(flow_mask_tile.size)
                info_tile["flow_coverage"] = flow_coverage
            if bg_mask_tile.any():
                var_base = float(base_tile[bg_mask_tile].var())
                var_stap = float(pd_tile[bg_mask_tile].var())
                bg_inflation = var_stap / max(var_base, 1e-12)
                info_tile["bg_var_base"] = var_base
                info_tile["bg_var_stap"] = var_stap
                info_tile["bg_var_inflation"] = bg_inflation
        if "flow_mu_ratio" not in info_tile:
            info_tile["flow_mu_ratio"] = flow_ratio
        if "bg_var_inflation" not in info_tile:
            info_tile["bg_var_inflation"] = bg_inflation
        if flow_coverage is None and mask_flow is not None:
            flow_mask_tile = mask_flow[y0 : y0 + th, x0 : x0 + tw]
            flow_coverage = float(flow_mask_tile.sum()) / float(flow_mask_tile.size)
            info_tile["flow_coverage"] = flow_coverage
        if mask_flow is not None and flow_coverage == 0.0:
            info_tile["flow_mask_empty"] = True

        if psd_telemetry:
            tile_series = np.mean(cube_np.reshape(cube_np.shape[0], -1), axis=1)
            psd_freqs, psd_power = _multi_taper_psd(
                tile_series,
                prf_hz,
                tapers=psd_tapers,
                bandwidth=psd_bandwidth,
            )
            if psd_freq_cache is None:
                psd_freq_cache = psd_freqs
            peak_idx = int(np.argmax(psd_power))
            info_tile["psd_mt_peak_hz"] = float(psd_freqs[peak_idx])
            info_tile["psd_mt_peak_power"] = float(psd_power[peak_idx])
            info_tile["psd_mt_total_power"] = float(np.sum(psd_power))
            info_tile["psd_mt_bg_tile"] = 1.0 if tile_is_bg else 0.0
            if capture_debug and debug_payload is not None:
                debug_payload["psd_mt_freqs_hz"] = psd_freqs.astype(np.float32, copy=False)
                debug_payload["psd_mt_power"] = psd_power.astype(np.float32, copy=False)
            if tile_is_bg:
                if bg_psd_accum is None:
                    bg_psd_accum = np.zeros_like(psd_power, dtype=np.float64)
                bg_psd_accum += psd_power.astype(np.float64, copy=False)
                bg_psd_count += 1

        # If we captured a debug payload for this tile, attach a few useful
        # scalar telemetry fields that are computed at the _stap_pd aggregation
        # level (e.g. flow/background PD ratios and subtile clamping flags).
        if capture_debug and debug_payload is not None:
            for _k in (
                "flow_mu_ratio",
                "bg_var_inflation",
                "flow_coverage",
                "background_uniformized",
                "subtile_background_uniformized",
                "band_fraction_capped",
                "band_fraction_cap_scale",
                "alias_cap_applied",
                "alias_cap_scale",
                "bg_edge_clamped_count",
            ):
                if _k in info_tile:
                    debug_payload[_k] = info_tile.get(_k)

        pd[y0 : y0 + th, x0 : x0 + tw] += pd_tile
        score[y0 : y0 + th, x0 : x0 + tw] += score_tile
        counts[y0 : y0 + th, x0 : x0 + tw] += 1.0
        score_counts[y0 : y0 + th, x0 : x0 + tw] += 1.0

        def _record_debug_payload(payload: dict | None) -> bool:
            if payload is None:
                return False
            debug_payloads.append(payload)
            return True

        tile_infos.append(info_tile)
        if info_tile.get("ka_gate_ok"):
            alias_value = info_tile.get("ka_alias_metric")
            if alias_value is None:
                alias_value = info_tile.get("psd_flow_alias_ratio")
            if alias_value is not None and np.isfinite(alias_value):
                if info_tile.get("ka_gate_tile_has_flow"):
                    gate_flow_alias_vals.append(float(alias_value))
                if info_tile.get("ka_gate_tile_is_bg"):
                    gate_bg_alias_vals.append(float(alias_value))
            if gate_mask_flow is not None and info_tile.get("ka_gate_tile_has_flow"):
                tile_flow_mask = mask_flow[y0 : y0 + th, x0 : x0 + tw]
                if tile_flow_mask is not None and tile_flow_mask.any():
                    gate_mask_flow[y0 : y0 + th, x0 : x0 + tw] |= tile_flow_mask
            if gate_mask_bg is not None and info_tile.get("ka_gate_tile_is_bg"):
                tile_bg_mask = mask_bg[y0 : y0 + th, x0 : x0 + tw]
                if tile_bg_mask is not None and tile_bg_mask.any():
                    gate_mask_bg[y0 : y0 + th, x0 : x0 + tw] |= tile_bg_mask
        recorded_debug = False
        if capture_debug:
            recorded_debug = _record_debug_payload(debug_payload)

        band_med = info_tile.get("band_fraction_median")
        bg_inf_val = info_tile.get("bg_var_inflation") or 0.0
        alias_ratio_val = info_tile.get("psd_flow_alias_ratio") or 0.0
        flow_cov_val = flow_coverage if flow_coverage is not None else 0.0
        priority_condition = False
        if (
            flow_cov_val is not None
            and flow_cov_val >= guard_tile_coverage_min
            and band_med is not None
            and band_med >= priority_band_thresh
        ):
            priority_condition = True
        if bg_inf_val is not None and np.isfinite(bg_inf_val) and bg_inf_val >= priority_bg_thresh:
            priority_condition = True
        if (
            flow_cov_val is not None
            and flow_cov_val > 0.0
            and np.isfinite(alias_ratio_val)
            and alias_ratio_val >= priority_alias_thresh
        ):
            priority_condition = True

        if (
            (not recorded_debug)
            and priority_condition
            and priority_debug_remaining > 0
            and debug_max_samples > 0
        ):
            _, _, _, dbg_extra = _stap_pd_tile_lcmv(
                cube_np,
                prf_hz=prf_hz,
                diag_load=diag_load,
                cov_estimator=estimator,
                huber_c=huber_c,
                mvdr_load_mode=mvdr_load_mode,
                mvdr_auto_kappa=mvdr_auto_kappa,
                constraint_ridge=constraint_ridge,
                fd_span_mode=fd_span_mode,
                fd_span_rel=fd_span_rel,
                fd_fixed_span_hz=fd_fixed_span_hz,
                constraint_mode=constraint_mode,
                grid_step_rel=grid_step_rel,
                min_pts=min_pts,
                max_pts=max_pts,
                fd_min_abs_hz=fd_min_abs_hz,
                alias_psd_select_enable=alias_psd_select_enable_flag,
                alias_psd_select_ratio_thresh=alias_psd_select_ratio_thresh_val,
                alias_psd_select_bins=alias_psd_select_bins_val,
                msd_lambda=msd_lambda,
                msd_ridge=msd_ridge,
                msd_agg_mode=msd_agg_mode,
                msd_ratio_rho=msd_ratio_rho,
                motion_half_span_rel=motion_half_span_rel,
                msd_contrast_alpha=msd_contrast_alpha,
                capture_debug=True,
                device=device_resolved,
                cube_tensor=cube_tensor,
                ka_mode=ka_mode_norm if ka_active else "none",
                ka_prior_library=ka_prior_np if ka_active else None,
                ka_opts=ka_opts_dict if ka_active else None,
                Lt_fixed=Lt,
                post_filter_callback=br_callback,
                ka_gate=gate_payload,
                feasibility_mode=feas_mode,
                motion_basis_geom=motion_basis_geom_np,
                alias_band_hz=alias_band_hz_tuple,
                psd_telemetry=psd_telemetry,
            )
            if dbg_extra is not None:
                dbg_extra["tile_index"] = int(tile_idx)
                dbg_extra["y0"] = int(y0)
                dbg_extra["x0"] = int(x0)
                if precomputed is not None and precomputed[2] is not None:
                    dbg_extra.setdefault(
                        "stap_fast_path_used",
                        bool(
                            precomputed[2].get(
                                "stap_fast_path_used",
                                False,
                            )
                        ),
                    )
            if _record_debug_payload(dbg_extra):
                priority_debug_remaining -= 1

    def flush_batch() -> None:
        nonlocal tile_batch_items, tile_positions, tile_capture_flags
        nonlocal tile_capture_coord_flags, tile_batch_indices, t_batch_proc, t_post
        nonlocal stap_fast_any
        if not tile_batch_items:
            return
        # When KA gating is enabled, we must run the per-tile STAP kernel with a
        # tile-specific gate payload so that KA is actually localized in score
        # space. In that regime we bypass the batched core and call `process_tile`
        # directly so that `_stap_pd_tile_lcmv` sees `ka_gate` and executes its
        # gate branch. This keeps the fast/batched path available for pure STAP
        # (ka_mode == "none") while ensuring correct KA behavior.
        if ka_gate_cfg_base is not None:
            t0_batch = time.perf_counter()
            # Move the batch to the chosen device once so that each tile can
            # still reuse the GPU tensor inside `_stap_pd_tile_lcmv`.
            stack_np = np.stack(tile_batch_items, axis=0).astype(np.complex64, copy=False)
            batch_tensor = torch.as_tensor(stack_np, dtype=torch.complex64, device=device_resolved)
            t_batch_proc += time.perf_counter() - t0_batch
            t0_post = time.perf_counter()
            for idx, (cube_np, (y0, x0), capture_flag, capture_coord_flag, tile_idx) in enumerate(
                zip(
                    tile_batch_items,
                    tile_positions,
                    tile_capture_flags,
                    tile_capture_coord_flags,
                    tile_batch_indices,
                    strict=False,
                )
            ):
                cube_tensor = batch_tensor[idx]
                process_tile(
                    cube_np,
                    y0,
                    x0,
                    tile_idx,
                    capture_flag,
                    capture_coord_flag,
                    cube_tensor=cube_tensor,
                    precomputed=None,
                )
            t_post += time.perf_counter() - t0_post
        else:
            t0_batch = time.perf_counter()
            stack_np = np.stack(tile_batch_items, axis=0).astype(np.complex64, copy=False)
            batch_tensor = torch.as_tensor(stack_np, dtype=torch.complex64, device=device_resolved)
            capture_flags = [
                bool(cf) or bool(cc)
                for cf, cc in zip(tile_capture_flags, tile_capture_coord_flags, strict=False)
            ]
            # If no tile in the batch requests capture, allow fast path by passing None.
            cube_capture_flags = None if not any(capture_flags) else capture_flags
            fast_path_flag = device_resolved.lower().startswith("cuda") and not ka_active
            band_frac_batch, score_batch, info_batch, debug_batch = _stap_pd_tile_lcmv_batch(
                batch_tensor,
                prf_hz=prf_hz,
                diag_load=diag_load,
                cov_estimator=estimator,
                huber_c=huber_c,
                mvdr_load_mode=mvdr_load_mode,
                mvdr_auto_kappa=mvdr_auto_kappa,
                constraint_ridge=constraint_ridge,
                fd_span_mode=fd_span_mode,
                fd_span_rel=fd_span_rel,
                fd_fixed_span_hz=fd_fixed_span_hz,
                constraint_mode=constraint_mode,
                grid_step_rel=grid_step_rel,
                min_pts=min_pts,
                max_pts=max_pts,
                fd_min_abs_hz=fd_min_abs_hz,
                alias_psd_select_enable=alias_psd_select_enable_flag,
                alias_psd_select_ratio_thresh=alias_psd_select_ratio_thresh_val,
                alias_psd_select_bins=alias_psd_select_bins_val,
                msd_lambda=msd_lambda,
                msd_ridge=msd_ridge,
                msd_agg_mode=msd_agg_mode,
                msd_ratio_rho=msd_ratio_rho,
                motion_half_span_rel=motion_half_span_rel,
                msd_contrast_alpha=msd_contrast_alpha,
                capture_debug=False,
                device=device_resolved,
                cube_capture_flags=cube_capture_flags,
                ka_mode=ka_mode_norm if ka_active else "none",
                ka_prior_library=ka_prior_np if ka_active else None,
                ka_opts=ka_opts_dict if ka_active else None,
                Lt_fixed=Lt,
                post_filter_callback=None,
                ka_gate=None,
                feasibility_mode=feas_mode,
                motion_basis_geom=motion_basis_geom_np,
                alias_band_hz=alias_band_hz_tuple,
                psd_telemetry=psd_telemetry,
                enable_fast_path=fast_path_flag,
            )
            t_batch_proc += time.perf_counter() - t0_batch
            t0_post = time.perf_counter()
            for idx, (cube_np, (y0, x0), capture_flag, capture_coord_flag, tile_idx) in enumerate(
                zip(
                    tile_batch_items,
                    tile_positions,
                    tile_capture_flags,
                    tile_capture_coord_flags,
                    tile_batch_indices,
                    strict=False,
                )
            ):
                cube_tensor = batch_tensor[idx]
                precomputed = (
                    band_frac_batch[idx],
                    score_batch[idx],
                    info_batch[idx],
                    debug_batch[idx],
                )
                if info_batch[idx] is not None:
                    stap_fast_any = stap_fast_any or bool(
                        info_batch[idx].get("stap_fast_path_used", False)
                    )
                process_tile(
                    cube_np,
                    y0,
                    x0,
                    tile_idx,
                    capture_flag,
                    capture_coord_flag,
                    cube_tensor=cube_tensor,
                    precomputed=precomputed,
                )
            t_post += time.perf_counter() - t0_post
        tile_batch_items = []
        tile_positions = []
        tile_capture_flags = []
        tile_capture_coord_flags = []
        tile_batch_indices = []

    for _cov_flow, y0, x0 in coords_ordered:
        coord = (y0, x0)
        capture_by_index = tile_counter in capture_indices
        capture_by_coord = coord in remaining_coord_requests
        capture_requested = capture_by_index or capture_by_coord

        # Optional conditional STAP: if the tile has no flow coverage at all,
        # we can safely fall back to the baseline PD map for this tile and
        # skip STAP entirely. When debug capture is requested for a tile,
        # do not skip: run the normal STAP path so debug_samples are populated.
        if mask_flow is not None:
            flow_mask_tile = mask_flow[y0 : y0 + th, x0 : x0 + tw]
            if not bool(flow_mask_tile.any()) and not capture_requested:
                base_tile = None
                if pd_base_full is not None:
                    base_tile = pd_base_full[y0 : y0 + th, x0 : x0 + tw]
                else:
                    t0_ext = time.perf_counter()
                    cube_tmp = Icube[:, y0 : y0 + th, x0 : x0 + tw]
                    t_extract += time.perf_counter() - t0_ext
                    base_tile = np.mean(np.abs(cube_tmp) ** 2, axis=0)
                pd[y0 : y0 + th, x0 : x0 + tw] += base_tile
                counts[y0 : y0 + th, x0 : x0 + tw] += 1.0
                stap_tiles_skipped_flow0 += 1
                tile_counter += 1
                if tile_debug_limit is not None and tile_counter >= int(tile_debug_limit):
                    break
                continue

        t0_ext = time.perf_counter()
        cube_np = np.ascontiguousarray(Icube[:, y0 : y0 + th, x0 : x0 + tw], dtype=np.complex64)
        t_extract += time.perf_counter() - t0_ext
        if capture_by_coord:
            remaining_coord_requests.discard(coord)
        tile_batch_items.append(cube_np)
        tile_positions.append((y0, x0))
        tile_capture_flags.append(capture_requested)
        tile_capture_coord_flags.append(capture_by_coord)
        tile_batch_indices.append(tile_counter)
        tile_counter += 1
        if tile_debug_limit is not None and tile_counter >= int(tile_debug_limit):
            flush_batch()
            break
        if len(tile_batch_items) >= tile_batch_size:
            flush_batch()

    flush_batch()

    # Pixels with zero tile support are outside the tiling grid (typically a
    # small border when (H - th) or (W - tw) is not divisible by stride). For
    # PD-mode scoring these uncovered pixels must not default to 0, since 0 can
    # become an extreme value under the lower-tail PD convention. When a
    # baseline PD map is available, fall back to it on uncovered pixels.
    #
    # This is especially important for full-STAP runs where mask_flow/mask_bg
    # are not provided (conditional execution disabled): downstream ROC code may
    # still include these border pixels via evaluation masks.
    uncovered = counts == 0.0
    if (
        pd_base_full is not None
        and np.any(uncovered)
        and pd_base_full.shape == pd.shape
        and np.isfinite(pd_base_full).any()
    ):
        pd[uncovered] = pd_base_full.astype(pd.dtype, copy=False)[uncovered]

    counts[counts == 0.0] = 1.0
    score_counts[score_counts == 0.0] = 1.0

    Lt_vals = [info["Lt"] for info in tile_infos if not info.get("fallback") and "Lt" in info]
    cond_vals = [
        info["cond_R"]
        for info in tile_infos
        if not info.get("fallback") and np.isfinite(info.get("cond_R", float("nan")))
    ]
    band_vals = [
        info["band_Kc"]
        for info in tile_infos
        if not info.get("fallback") and info.get("band_Kc") is not None
    ]
    span_vals = [
        info["span_hz"]
        for info in tile_infos
        if not info.get("fallback") and info.get("span_hz") is not None
    ]
    peak_vals = [
        info["f_peak_hz"]
        for info in tile_infos
        if not info.get("fallback") and info.get("f_peak_hz") is not None
    ]
    diag_vals = [
        info["diag_load"]
        for info in tile_infos
        if not info.get("fallback") and info.get("diag_load") is not None
    ]
    ridge_vals = [
        info["constraint_ridge"]
        for info in tile_infos
        if not info.get("fallback") and info.get("constraint_ridge") is not None
    ]
    resid_vals = [
        info["constraint_residual"]
        for info in tile_infos
        if not info.get("fallback") and info.get("constraint_residual") is not None
    ]
    load_mode = next(
        (
            info.get("load_mode")
            for info in tile_infos
            if not info.get("fallback") and info.get("load_mode")
        ),
        "absolute",
    )
    fd_examples = next(
        (
            info["fd_grid"]
            for info in tile_infos
            if not info.get("fallback") and info.get("fd_grid")
        ),
        [],
    )
    span_mode = next(
        (
            info.get("span_mode")
            for info in tile_infos
            if not info.get("fallback") and info.get("span_mode")
        ),
        fd_span_mode,
    )
    auto_vals = [
        info["auto_kappa_target"]
        for info in tile_infos
        if not info.get("fallback") and info.get("auto_kappa_target") is not None
    ]
    trace_vals = [
        info.get("cov_trace")
        for info in tile_infos
        if not info.get("fallback") and info.get("cov_trace") is not None
    ]
    eff_rank_vals = [
        info.get("cov_eff_rank")
        for info in tile_infos
        if not info.get("fallback") and info.get("cov_eff_rank") is not None
    ]
    sigma_min_vals = [
        info.get("sigma_min_raw") for info in tile_infos if info.get("sigma_min_raw") is not None
    ]
    sigma_max_vals = [
        info.get("sigma_max_raw") for info in tile_infos if info.get("sigma_max_raw") is not None
    ]
    lambda_needed_vals = [
        info.get("lambda_condition_needed")
        for info in tile_infos
        if info.get("lambda_condition_needed") is not None
    ]
    lambda_conditioned_vals = [
        info.get("lambda_conditioned")
        for info in tile_infos
        if info.get("lambda_conditioned") is not None
    ]
    diag_requested_vals = [
        info.get("diag_load_requested")
        for info in tile_infos
        if info.get("diag_load_requested") is not None
    ]
    cond_loaded_vals = [
        info["cond_loaded"]
        for info in tile_infos
        if not info.get("fallback") and info.get("cond_loaded") is not None
    ]
    gram_med_vals = [
        info["gram_diag_median"]
        for info in tile_infos
        if not info.get("fallback") and info.get("gram_diag_median") is not None
    ]
    gram_max_vals = [
        info["gram_diag_max"]
        for info in tile_infos
        if not info.get("fallback") and info.get("gram_diag_max") is not None
    ]
    bg_inflation = [
        info["bg_var_inflation"]
        for info in tile_infos
        if not info.get("fallback") and info.get("bg_var_inflation") is not None
    ]
    flow_ratio_vals = [
        info["flow_mu_ratio"]
        for info in tile_infos
        if not info.get("fallback") and info.get("flow_mu_ratio") is not None
    ]
    score_mean_vals = [
        info["score_mean"] for info in tile_infos if info.get("score_mean") is not None
    ]
    score_var_vals = [
        info["score_var"] for info in tile_infos if info.get("score_var") is not None
    ]
    msd_lambda_vals = [
        info["msd_lambda"] for info in tile_infos if info.get("msd_lambda") is not None
    ]
    msd_lambda_conditioned_vals = [
        info.get("msd_lambda_conditioned")
        for info in tile_infos
        if info.get("msd_lambda_conditioned") is not None
    ]
    msd_lambda_needed_vals = [
        info.get("msd_lambda_needed")
        for info in tile_infos
        if info.get("msd_lambda_needed") is not None
    ]
    band_frac_vals = [
        info.get("band_fraction_median")
        for info in tile_infos
        if info.get("band_fraction_median") is not None
        and np.isfinite(info.get("band_fraction_median"))
    ]
    band_frac_p90_vals = [
        info.get("band_fraction_p90")
        for info in tile_infos
        if info.get("band_fraction_p90") is not None and np.isfinite(info.get("band_fraction_p90"))
    ]
    motion_frac_vals = [
        info.get("motion_fraction_median")
        for info in tile_infos
        if info.get("motion_fraction_median") is not None
        and np.isfinite(info.get("motion_fraction_median"))
    ]
    motion_frac_p90_vals = [
        info.get("motion_fraction_p90")
        for info in tile_infos
        if info.get("motion_fraction_p90") is not None
        and np.isfinite(info.get("motion_fraction_p90"))
    ]
    motion_span_vals = [
        info.get("motion_half_span_hz")
        for info in tile_infos
        if info.get("motion_half_span_hz") is not None
        and np.isfinite(info.get("motion_half_span_hz"))
    ]
    motion_span_rel_vals = [
        info.get("motion_half_span_rel_used")
        for info in tile_infos
        if info.get("motion_half_span_rel_used") is not None
        and np.isfinite(info.get("motion_half_span_rel_used"))
    ]
    fd_motion_freq_vals = [
        info.get("fd_motion_freqs")
        for info in tile_infos
        if info.get("fd_motion_freqs") is not None and np.isfinite(info.get("fd_motion_freqs"))
    ]
    fd_flow_freq_split_vals = [
        info.get("fd_flow_freqs_after_split")
        for info in tile_infos
        if info.get("fd_flow_freqs_after_split") is not None
        and np.isfinite(info.get("fd_flow_freqs_after_split"))
    ]
    contrast_alpha_vals = [
        info.get("msd_contrast_alpha")
        for info in tile_infos
        if info.get("msd_contrast_alpha") is not None
        and np.isfinite(info.get("msd_contrast_alpha"))
    ]
    motion_rank_vals = [
        info.get("motion_basis_rank")
        for info in tile_infos
        if info.get("motion_basis_rank") is not None
    ]
    energy_kept_vals = [
        info.get("energy_kept_ratio")
        for info in tile_infos
        if info.get("energy_kept_ratio") is not None and np.isfinite(info.get("energy_kept_ratio"))
    ]
    energy_removed_vals = [
        info.get("energy_removed_ratio")
        for info in tile_infos
        if info.get("energy_removed_ratio") is not None
        and np.isfinite(info.get("energy_removed_ratio"))
    ]
    kc_flow_vals = [info.get("kc_flow") for info in tile_infos if info.get("kc_flow") is not None]
    kc_motion_vals = [
        info.get("kc_motion") for info in tile_infos if info.get("kc_motion") is not None
    ]
    kc_flow_cap_vals = [
        info.get("kc_flow_cap") for info in tile_infos if info.get("kc_flow_cap") is not None
    ]
    kc_flow_cap_motion_vals = [
        info.get("kc_flow_cap_motion")
        for info in tile_infos
        if info.get("kc_flow_cap_motion") is not None
    ]
    kc_flow_freq_vals = [
        info.get("kc_flow_freqs") for info in tile_infos if info.get("kc_flow_freqs") is not None
    ]
    psd_peak_vals = [
        info.get("psd_peak_hz")
        for info in tile_infos
        if info.get("psd_peak_hz") is not None and np.isfinite(info.get("psd_peak_hz"))
    ]
    psd_ratio_vals = [
        info.get("psd_flow_to_dc_ratio")
        for info in tile_infos
        if info.get("psd_flow_to_dc_ratio") is not None
        and np.isfinite(info.get("psd_flow_to_dc_ratio"))
    ]
    psd_flow_power_vals = [
        info.get("psd_power_flow")
        for info in tile_infos
        if info.get("psd_power_flow") is not None and np.isfinite(info.get("psd_power_flow"))
    ]
    psd_dc_power_vals = [
        info.get("psd_power_dc")
        for info in tile_infos
        if info.get("psd_power_dc") is not None and np.isfinite(info.get("psd_power_dc"))
    ]
    psd_flow_target_vals = [
        info.get("psd_flow_freq_target")
        for info in tile_infos
        if info.get("psd_flow_freq_target") is not None
        and np.isfinite(info.get("psd_flow_freq_target"))
    ]
    psd_flow_freq_power_vals = [
        info.get("psd_flow_freq_power")
        for info in tile_infos
        if info.get("psd_flow_freq_power") is not None
        and np.isfinite(info.get("psd_flow_freq_power"))
    ]
    psd_fundamental_ratio_vals = [
        info.get("psd_fundamental_ratio")
        for info in tile_infos
        if info.get("psd_fundamental_ratio") is not None
        and np.isfinite(info.get("psd_fundamental_ratio"))
    ]
    psd_alias_ratio_vals = [
        info.get("psd_flow_alias_ratio")
        for info in tile_infos
        if info.get("psd_flow_alias_ratio") is not None
        and np.isfinite(info.get("psd_flow_alias_ratio"))
    ]
    alias_ratio_array = (
        np.array(psd_alias_ratio_vals, dtype=float) if psd_alias_ratio_vals else np.array([])
    )
    alias_ratio_stats = {
        "median": float(np.median(alias_ratio_array)) if alias_ratio_array.size else None,
        "p10": (
            float(np.quantile(alias_ratio_array, 0.10))
            if alias_ratio_array.size >= 2
            else (float(alias_ratio_array[0]) if alias_ratio_array.size else None)
        ),
        "p90": (
            float(np.quantile(alias_ratio_array, 0.90))
            if alias_ratio_array.size >= 2
            else (float(alias_ratio_array[0]) if alias_ratio_array.size else None)
        ),
    }
    psd_alias_flags = [
        bool(info.get("psd_flow_alias"))
        for info in tile_infos
        if info.get("psd_flow_alias") is not None
    ]
    alias_psd_select_flags = [
        bool(info.get("psd_alias_select_applied"))
        for info in tile_infos
        if info.get("psd_alias_select_applied") is not None
    ]
    alias_psd_select_count = int(sum(alias_psd_select_flags))
    alias_psd_kept_bins = [
        info.get("psd_kept_bin_count")
        for info in tile_infos
        if info.get("psd_kept_bin_count") is not None
    ]
    psd_alias_denominator = len(
        [
            1
            for info in tile_infos
            if info.get("psd_flow_freq_target") is not None and not info.get("fallback")
        ]
    )
    fundamental_vals = [
        info.get("fundamental_hz")
        for info in tile_infos
        if info.get("fundamental_hz") is not None and np.isfinite(info.get("fundamental_hz"))
    ]
    fd_min_applied_flags = [
        bool(info.get("fd_min_abs_applied"))
        for info in tile_infos
        if info.get("fd_min_abs_applied") is not None
    ]
    fd_symmetry_flags = [
        bool(info.get("fd_symmetry_added"))
        for info in tile_infos
        if info.get("fd_symmetry_added") is not None
    ]
    grid_step_vals = [
        info.get("grid_step_hz") for info in tile_infos if info.get("grid_step_hz") is not None
    ]
    contrast_flow_med_vals = [
        info.get("contrast_flow_median")
        for info in tile_infos
        if info.get("contrast_flow_median") is not None
        and np.isfinite(info.get("contrast_flow_median"))
    ]
    contrast_flow_p90_vals = [
        info.get("contrast_flow_p90")
        for info in tile_infos
        if info.get("contrast_flow_p90") is not None and np.isfinite(info.get("contrast_flow_p90"))
    ]
    contrast_motion_med_vals = [
        info.get("contrast_motion_median")
        for info in tile_infos
        if info.get("contrast_motion_median") is not None
        and np.isfinite(info.get("contrast_motion_median"))
    ]
    contrast_motion_p90_vals = [
        info.get("contrast_motion_p90")
        for info in tile_infos
        if info.get("contrast_motion_p90") is not None
        and np.isfinite(info.get("contrast_motion_p90"))
    ]
    contrast_score_mean_vals = [
        info.get("contrast_score_mean")
        for info in tile_infos
        if info.get("contrast_score_mean") is not None
        and np.isfinite(info.get("contrast_score_mean"))
    ]
    contrast_score_std_vals = [
        info.get("contrast_score_std")
        for info in tile_infos
        if info.get("contrast_score_std") is not None
        and np.isfinite(info.get("contrast_score_std"))
    ]
    contrast_flow_kc_vals = [
        info.get("contrast_flow_kc")
        for info in tile_infos
        if info.get("contrast_flow_kc") is not None and np.isfinite(info.get("contrast_flow_kc"))
    ]
    contrast_motion_kc_vals = [
        info.get("contrast_motion_kc")
        for info in tile_infos
        if info.get("contrast_motion_kc") is not None
        and np.isfinite(info.get("contrast_motion_kc"))
    ]
    contrast_flow_rank_vals = [
        info.get("contrast_flow_rank")
        for info in tile_infos
        if info.get("contrast_flow_rank") is not None
        and np.isfinite(info.get("contrast_flow_rank"))
    ]
    contrast_motion_rank_eff_vals = [
        info.get("contrast_motion_rank_eff")
        for info in tile_infos
        if info.get("contrast_motion_rank_eff") is not None
        and np.isfinite(info.get("contrast_motion_rank_eff"))
    ]
    contrast_motion_rank_init_vals = [
        info.get("contrast_motion_rank_initial")
        for info in tile_infos
        if info.get("contrast_motion_rank_initial") is not None
        and np.isfinite(info.get("contrast_motion_rank_initial"))
    ]
    ka_beta_vals = [
        info.get("ka_beta")
        for info in tile_infos
        if info.get("ka_beta") is not None and np.isfinite(info.get("ka_beta"))
    ]
    ka_retain_vals = [
        info.get("ka_retain_f_beta")
        for info in tile_infos
        if info.get("ka_retain_f_beta") is not None and np.isfinite(info.get("ka_retain_f_beta"))
    ]
    ka_shrink_vals = [
        info.get("ka_shrink_perp_beta")
        for info in tile_infos
        if info.get("ka_shrink_perp_beta") is not None
        and np.isfinite(info.get("ka_shrink_perp_beta"))
    ]
    ka_retain_total_vals = [
        info.get("ka_retain_f_total")
        for info in tile_infos
        if info.get("ka_retain_f_total") is not None and np.isfinite(info.get("ka_retain_f_total"))
    ]
    ka_shrink_total_vals = [
        info.get("ka_shrink_perp_total")
        for info in tile_infos
        if info.get("ka_shrink_perp_total") is not None
        and np.isfinite(info.get("ka_shrink_perp_total"))
    ]
    pd_cond_vals = [
        info.get("pd_condG")
        for info in tile_infos
        if info.get("pd_condG") is not None and np.isfinite(info.get("pd_condG"))
    ]
    pd_ridge_vals = [
        info.get("pd_gram_ridge_used")
        for info in tile_infos
        if info.get("pd_gram_ridge_used") is not None
        and np.isfinite(info.get("pd_gram_ridge_used"))
    ]
    lam_pf_extra_vals = [
        info.get("lam_pf_extra_pd")
        for info in tile_infos
        if info.get("lam_pf_extra_pd") is not None and np.isfinite(info.get("lam_pf_extra_pd"))
    ]
    ka_ridge_vals = [
        1.0 if info.get("ka_ridge_split") else 0.0
        for info in tile_infos
        if info.get("ka_ridge_split") is not None
    ]
    ka_prior_clip_vals = [
        1.0 if info.get("ka_prior_clipped_passband") else 0.0
        for info in tile_infos
        if info.get("ka_prior_clipped_passband") is not None
    ]
    ka_gate_ok_vals = [
        info.get("ka_gate_ok") for info in tile_infos if info.get("ka_gate_ok") is not None
    ]
    ka_gate_flow_hits = [
        info
        for info in tile_infos
        if info.get("ka_gate_ok") is not None and info.get("ka_gate_tile_has_flow")
    ]
    ka_gate_bg_hits = [
        info
        for info in tile_infos
        if info.get("ka_gate_ok") is not None and info.get("ka_gate_tile_is_bg")
    ]
    ka_lambda_strategy_vals = [
        (
            "split"
            if info.get("ka_lambda_strategy") in {"split", "split_override"}
            else info.get("ka_lambda_strategy")
        )
        for info in tile_infos
        if info.get("ka_lambda_strategy") is not None
    ]
    gamma_flow_vals = [
        float(info.get("gamma_flow"))
        for info in tile_infos
        if info.get("gamma_flow") is not None and np.isfinite(info.get("gamma_flow"))
    ]
    gamma_perp_vals = [
        float(info.get("gamma_perp"))
        for info in tile_infos
        if info.get("gamma_perp") is not None and np.isfinite(info.get("gamma_perp"))
    ]
    ka_pf_rank_vals = [
        info.get("ka_pf_rank")
        for info in tile_infos
        if info.get("ka_pf_rank") is not None and np.isfinite(info.get("ka_pf_rank"))
    ]
    ka_alias_rank_vals = [
        info.get("ka_alias_rank")
        for info in tile_infos
        if info.get("ka_alias_rank") is not None and np.isfinite(info.get("ka_alias_rank"))
    ]
    ka_alias_gain_vals = [
        info.get("ka_alias_gain_target")
        for info in tile_infos
        if info.get("ka_alias_gain_target") is not None
        and np.isfinite(info.get("ka_alias_gain_target"))
    ]

    def _collect_pf_trace_values(field: str) -> tuple[list[float], int, int]:
        vals: list[float] = []
        invalid = 0
        total = 0
        for info in tile_infos:
            val = info.get(field)
            if val is None:
                continue
            total += 1
            try:
                val_float = float(val)
            except (TypeError, ValueError):
                invalid += 1
                continue
            if np.isfinite(val_float):
                vals.append(val_float)
            else:
                invalid += 1
        return vals, invalid, total

    (
        ka_pf_trace_alpha_vals,
        ka_pf_trace_alpha_invalid,
        ka_pf_trace_alpha_total,
    ) = _collect_pf_trace_values("ka_pf_trace_alpha")
    (
        ka_pf_trace_loaded_vals,
        ka_pf_trace_loaded_invalid,
        ka_pf_trace_loaded_total,
    ) = _collect_pf_trace_values("ka_pf_trace_loaded")
    (
        ka_pf_trace_sample_vals,
        ka_pf_trace_sample_invalid,
        ka_pf_trace_sample_total,
    ) = _collect_pf_trace_values("ka_pf_trace_sample")
    ka_trace_ratio_vals = [
        info.get("ka_trace_ratio")
        for info in tile_infos
        if info.get("ka_trace_ratio") is not None and np.isfinite(info.get("ka_trace_ratio"))
    ]
    ka_trace_scaled_vals = [
        1.0 if info.get("ka_trace_scaled") else 0.0
        for info in tile_infos
        if info.get("ka_trace_scaled") is not None
    ]
    ka_trace_scale_lock_vals = [
        info.get("ka_trace_scale_lock_reason")
        for info in tile_infos
        if info.get("ka_trace_scale_lock_reason")
    ]
    ka_mismatch_vals = [
        info.get("ka_mismatch")
        for info in tile_infos
        if info.get("ka_mismatch") is not None and np.isfinite(info.get("ka_mismatch"))
    ]
    ka_lambda_vals = [
        info.get("ka_lambda_used")
        for info in tile_infos
        if info.get("ka_lambda_used") is not None and np.isfinite(info.get("ka_lambda_used"))
    ]
    ka_sigma_min_vals = [
        info.get("ka_sigma_min_raw")
        for info in tile_infos
        if info.get("ka_sigma_min_raw") is not None and np.isfinite(info.get("ka_sigma_min_raw"))
    ]
    ka_sigma_max_vals = [
        info.get("ka_sigma_max_raw")
        for info in tile_infos
        if info.get("ka_sigma_max_raw") is not None and np.isfinite(info.get("ka_sigma_max_raw"))
    ]
    ka_snr_flow_ratio_vals = [
        info.get("ka_snr_flow_ratio")
        for info in tile_infos
        if info.get("ka_snr_flow_ratio") is not None and np.isfinite(info.get("ka_snr_flow_ratio"))
    ]
    ka_snr_flow_base_vals = [
        info.get("ka_snr_flow_base")
        for info in tile_infos
        if info.get("ka_snr_flow_base") is not None and np.isfinite(info.get("ka_snr_flow_base"))
    ]
    ka_snr_flow_loaded_vals = [
        info.get("ka_snr_flow_loaded")
        for info in tile_infos
        if info.get("ka_snr_flow_loaded") is not None
        and np.isfinite(info.get("ka_snr_flow_loaded"))
    ]
    ka_noise_perp_ratio_vals = [
        info.get("ka_noise_perp_ratio")
        for info in tile_infos
        if info.get("ka_noise_perp_ratio") is not None
        and np.isfinite(info.get("ka_noise_perp_ratio"))
    ]

    # If ratios are missing but band means are present, derive them to populate C1/C2.
    if (not ka_snr_flow_ratio_vals) or (not ka_noise_perp_ratio_vals):
        for info in tile_infos:
            if info.get("ka_mode") in (None, "none"):
                continue
            pf_mean = info.get("ka_pf_lambda_mean")
            perp_mean = info.get("ka_perp_lambda_mean")
            if pf_mean is not None and perp_mean not in (None, 0):
                ka_snr_flow_ratio_vals.append(float(pf_mean) / float(perp_mean))
            if perp_mean is not None:
                ka_noise_perp_ratio_vals.append(float(perp_mean))
    band_q10_vals = [
        info.get("band_fraction_q10")
        for info in tile_infos
        if info.get("band_fraction_q10") is not None and np.isfinite(info.get("band_fraction_q10"))
    ]
    band_q50_vals = [
        info.get("band_fraction_q50")
        for info in tile_infos
        if info.get("band_fraction_q50") is not None and np.isfinite(info.get("band_fraction_q50"))
    ]
    band_q90_vals = [
        info.get("band_fraction_q90")
        for info in tile_infos
        if info.get("band_fraction_q90") is not None and np.isfinite(info.get("band_fraction_q90"))
    ]
    score_q10_vals = [
        info.get("score_q10")
        for info in tile_infos
        if info.get("score_q10") is not None and np.isfinite(info.get("score_q10"))
    ]
    score_q50_vals = [
        info.get("score_q50")
        for info in tile_infos
        if info.get("score_q50") is not None and np.isfinite(info.get("score_q50"))
    ]
    score_q90_vals = [
        info.get("score_q90")
        for info in tile_infos
        if info.get("score_q90") is not None and np.isfinite(info.get("score_q90"))
    ]
    band_nan_counts = [
        info.get("band_nan_count") for info in tile_infos if info.get("band_nan_count") is not None
    ]
    band_inf_counts = [
        info.get("band_inf_count") for info in tile_infos if info.get("band_inf_count") is not None
    ]
    score_nan_counts = [
        info.get("score_nan_count")
        for info in tile_infos
        if info.get("score_nan_count") is not None
    ]
    score_inf_counts = [
        info.get("score_inf_count")
        for info in tile_infos
        if info.get("score_inf_count") is not None
    ]
    score_mode_hist = Counter(info.get("score_mode", "unknown") for info in tile_infos)
    ka_warning_hist = Counter(
        info.get("ka_warning") for info in tile_infos if info.get("ka_warning")
    )
    ka_failure_count = int(sum(1 for info in tile_infos if info.get("ka_warning")))
    contrast_tile_count = sum(1 for info in tile_infos if info.get("contrast_enabled"))
    pd_retry_modes = [info.get("pd_retry", "none") for info in tile_infos]
    capon_count = sum(1 for info in tile_infos if info.get("load_mode") == "capon")
    flow_source_resolved_hist = Counter(
        info.get("psd_flow_freq_source_resolved", "unknown")
        for info in tile_infos
        if info.get("psd_flow_freq_source_resolved") is not None
    )

    ka_tile_flags = [
        info.get("ka_mode")
        for info in tile_infos
        if info.get("ka_mode") and info.get("ka_mode") != "none"
    ]
    ka_tile_count = len(ka_tile_flags) if ka_tile_flags else len(ka_beta_vals)
    ka_mode_overall = ka_mode_norm if (ka_active and ka_tile_count > 0) else "none"
    ka_pf_equalized_count = sum(1 for info in tile_infos if info.get("ka_pf_trace_equalized"))
    ka_operator_flags = [
        bool(info.get("operator_feasible"))
        for info in tile_infos
        if info.get("operator_feasible") is not None
    ]
    ka_invalid_snr_count = sum(1 for info in tile_infos if info.get("ka_invalid_snr"))
    ka_invalid_noise_count = sum(1 for info in tile_infos if info.get("ka_invalid_noise"))
    ka_invalid_tile_count = sum(
        1 for info in tile_infos if info.get("ka_invalid_snr") or info.get("ka_invalid_noise")
    )
    ka_invalid_fraction = (
        float(ka_invalid_tile_count) / float(ka_tile_count) if ka_tile_count else 0.0
    )
    ka_disable_reasons: list[str] = []
    if ka_tile_count and ka_invalid_fraction >= 0.5:
        ka_disable_reasons.append("snr_noise")

    background_uniformized_count = sum(
        1 for info in tile_infos if info.get("background_uniformized")
    )
    subtile_background_uniformized_count = sum(
        1 for info in tile_infos if info.get("subtile_background_uniformized")
    )
    flow_cov_zero = [
        info
        for info in tile_infos
        if info.get("flow_coverage") is not None and info.get("flow_coverage") <= 1e-9
    ]
    flow_cov_zero_count = len(flow_cov_zero)
    flow_cov_zero_bg_clamped = sum(
        1 for info in flow_cov_zero if info.get("background_uniformized")
    )
    flow_cov_zero_subtile = sum(
        1 for info in flow_cov_zero if info.get("subtile_background_uniformized")
    )
    band_fraction_capped_count = sum(1 for info in tile_infos if info.get("band_fraction_capped"))
    band_fraction_cap_scales = [
        float(info.get("band_fraction_cap_scale"))
        for info in tile_infos
        if info.get("band_fraction_cap_scale") is not None
    ]
    alias_cap_applied_count = sum(1 for info in tile_infos if info.get("alias_cap_applied"))
    alias_cap_scales = [
        float(info.get("alias_cap_scale"))
        for info in tile_infos
        if info.get("alias_cap_scale") is not None
    ]
    alias_cap_alias_ratios = [
        float(info.get("psd_flow_alias_ratio"))
        for info in tile_infos
        if info.get("alias_cap_applied") and info.get("psd_flow_alias_ratio") is not None
    ]
    alias_cap_band_medians = [
        float(info.get("band_fraction_median"))
        for info in tile_infos
        if info.get("alias_cap_applied") and info.get("band_fraction_median") is not None
    ]

    energy_kept_vals = [
        float(info.get("energy_kept_ratio"))
        for info in tile_infos
        if info.get("energy_kept_ratio") is not None
    ]
    energy_removed_vals = [
        float(info.get("energy_removed_ratio"))
        for info in tile_infos
        if info.get("energy_removed_ratio") is not None
    ]

    flow_alignment_vals = [
        float(info.get("flow_band_alignment"))
        for info in tile_infos
        if info.get("flow_band_alignment") is not None
    ]
    flow_motion_angle_vals = [
        float(info.get("flow_motion_angle_deg"))
        for info in tile_infos
        if info.get("flow_motion_angle_deg") is not None
        and np.isfinite(info.get("flow_motion_angle_deg"))
    ]

    def _scalar_stats(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "median": None, "p10": None, "p90": None}
        arr = np.asarray(values, dtype=float)
        stats: dict[str, float | int | None] = {
            "count": int(arr.size),
            "median": float(np.median(arr)),
        }
        if arr.size >= 2:
            stats["p10"] = float(np.quantile(arr, 0.10))
            stats["p90"] = float(np.quantile(arr, 0.90))
        else:
            val = float(arr[0])
            stats["p10"] = val
            stats["p90"] = val
        return stats

    flow_band_alignment_stats = _scalar_stats(flow_alignment_vals)
    flow_motion_angle_stats = _scalar_stats(flow_motion_angle_vals)
    if flow_motion_angle_stats.get("count", 0) == 0:
        fallback_count = sum(1 for info in tile_infos if info.get("kc_flow"))
        if fallback_count:
            flow_motion_angle_stats = {
                "count": fallback_count,
                "median": 90.0,
                "p10": 90.0,
                "p90": 90.0,
            }

    def _stats_by_class(values: list[float]) -> dict[str, float | int | None]:
        return _scalar_stats(values)

    flow_alignment_vals_flow = [
        float(info.get("flow_band_alignment"))
        for info in tile_infos
        if info.get("tile_has_flow")
        and info.get("flow_band_alignment") is not None
        and np.isfinite(info.get("flow_band_alignment"))
    ]
    flow_alignment_vals_bg = [
        float(info.get("flow_band_alignment"))
        for info in tile_infos
        if info.get("tile_is_bg")
        and info.get("flow_band_alignment") is not None
        and np.isfinite(info.get("flow_band_alignment"))
    ]
    flow_motion_vals_flow = [
        float(info.get("flow_motion_angle_deg"))
        for info in tile_infos
        if info.get("tile_has_flow")
        and info.get("flow_motion_angle_deg") is not None
        and np.isfinite(info.get("flow_motion_angle_deg"))
    ]
    flow_motion_vals_bg = [
        float(info.get("flow_motion_angle_deg"))
        for info in tile_infos
        if info.get("tile_is_bg")
        and info.get("flow_motion_angle_deg") is not None
        and np.isfinite(info.get("flow_motion_angle_deg"))
    ]

    def _stats_by_depth(
        tile_infos: list[dict],
        value_key: str,
        buckets: list[tuple[float, float]],
    ) -> dict[str, dict[str, float | int | None]]:
        depth_stats: dict[str, dict[str, float | int | None]] = {}
        for low, high in buckets:
            vals: list[float] = []
            for info in tile_infos:
                val = info.get(value_key)
                depth = info.get("depth_center_frac")
                if val is None or depth is None or not np.isfinite(val) or not np.isfinite(depth):
                    continue
                depth_f = float(depth)
                high_cmp = high if high < 1.0 else 1.00001
                if float(low) <= depth_f < float(high_cmp):
                    vals.append(float(val))
                elif high >= 1.0 and abs(depth_f - high) < 1e-6:
                    vals.append(float(val))
            stats = _scalar_stats(vals)
            stats["range"] = [float(low), float(high)]
            depth_stats[f"{low:.2f}-{high:.2f}"] = stats
        return depth_stats

    depth_buckets = [(0.0, 0.33), (0.33, 0.66), (0.66, 1.01)]

    flow_band_low = float(band_ratio_spec_local["flow_low_hz"])
    flow_band_high = float(band_ratio_spec_local["flow_high_hz"])
    alias_center = float(band_ratio_spec_local["alias_center_hz"])
    alias_width = float(band_ratio_spec_local["alias_width_hz"])
    alias_low = alias_center - alias_width
    alias_high = alias_center + alias_width
    flow_peak_vals: list[float] = []
    alias_peak_vals: list[float] = []
    all_peak_vals: list[float] = []
    alias_candidate_flags: list[bool] = []
    for info in tile_infos:
        peak = info.get("psd_peak_hz")
        if peak is None or not np.isfinite(peak):
            continue
        all_peak_vals.append(float(peak))
        flow_ratio_val = info.get("flow_coverage")
        if flow_ratio_val is not None and flow_ratio_val >= guard_tile_coverage_min:
            flow_peak_vals.append(abs(float(peak)))
        alias_ratio_val = info.get("psd_flow_alias_ratio")
        is_alias_cand = False
        if alias_ratio_val is not None and np.isfinite(alias_ratio_val):
            if float(alias_ratio_val) >= float(alias_psd_select_ratio_thresh_val):
                alias_peak_vals.append(abs(float(peak)))
                is_alias_cand = True
        alias_candidate_flags.append(is_alias_cand)

    def _fraction_in_band(peaks: list[float], low: float, high: float) -> tuple[int, float | None]:
        if not peaks:
            return 0, None
        arr = np.asarray(peaks, dtype=float)
        if arr.size == 0:
            return 0, None
        in_band = np.logical_and(arr >= low, arr <= high).sum()
        fraction = float(in_band) / float(arr.size) if arr.size else None
        return int(arr.size), fraction

    flow_peak_count, flow_peak_fraction = _fraction_in_band(
        flow_peak_vals, flow_band_low, flow_band_high
    )
    alias_low_band = alias_low
    alias_high_band = alias_high
    alias_peak_count, alias_peak_fraction = _fraction_in_band(
        [abs(p) for p in alias_peak_vals], alias_low_band, alias_high_band
    )

    # Conditional alias alignment on alias-candidate tiles (psd_flow_alias_ratio above threshold)
    alias_cand_count = int(sum(1 for flag in alias_candidate_flags if flag))
    alias_cand_hits = 0
    if alias_cand_count > 0:
        for idx, flag in enumerate(alias_candidate_flags):
            if not flag:
                continue
            if idx >= len(all_peak_vals):
                continue
            peak_abs = abs(float(all_peak_vals[idx]))
            if alias_low_band <= peak_abs <= alias_high_band:
                alias_cand_hits += 1
    alias_cand_fraction = (
        float(alias_cand_hits) / float(alias_cand_count) if alias_cand_count > 0 else None
    )

    # Build approximate global Pf/Pa projectors for telemetry based on the
    # flow/alias band design. These are small (Lt x Lt) and useful for
    # offline prior construction and analysis. Store real/imag parts separately
    # to remain JSON-serializable.
    Pf_real = Pf_imag = Pa_real = Pa_imag = None
    try:
        if bandpass_constraints_temporal is not None and Lt > 0:
            fd_flow = [0.5 * (flow_band_low + flow_band_high)]
            if fd_flow[0] > 1e-6:
                fd_flow.append(-fd_flow[0])
            fd_alias = [alias_center]
            if alias_center > 1e-6:
                fd_alias.append(-alias_center)
            Cf_global = bandpass_constraints_temporal(
                Lt=Lt,
                prf_hz=prf_hz,
                fd_grid_hz=fd_flow,
                device="cpu",
                dtype=torch.complex64,
                mode="exp",
            )
            Ca_global = bandpass_constraints_temporal(
                Lt=Lt,
                prf_hz=prf_hz,
                fd_grid_hz=fd_alias,
                device="cpu",
                dtype=torch.complex64,
                mode="exp",
            )
            Pf_global_t = projector_from_tones(Cf_global.to(torch.complex128))
            Pa_global_t = projector_from_tones(Ca_global.to(torch.complex128))
            Pf_np = Pf_global_t.detach().cpu().numpy()
            Pa_np = Pa_global_t.detach().cpu().numpy()
            Pf_real = Pf_np.real.tolist()
            Pf_imag = Pf_np.imag.tolist()
            Pa_real = Pa_np.real.tolist()
            Pa_imag = Pa_np.imag.tolist()
    except Exception:
        Pf_real = Pf_imag = Pa_real = Pa_imag = None

    info_out = {
        "fallback_count": int(sum(1 for info in tile_infos if info.get("fallback"))),
        "total_tiles": len(tile_infos),
        "feasibility_mode": feas_mode,
        "alias_psd_select_enable": bool(alias_psd_select_enable_flag),
        "psd_alias_select_enable": bool(alias_psd_select_enable_flag),
        "alias_psd_select_ratio_thresh": float(alias_psd_select_ratio_thresh_val),
        "alias_psd_select_bins": int(alias_psd_select_bins_val),
        "psd_mt_enabled": bool(psd_telemetry),
        "psd_mt_bg_tile_count": int(bg_psd_count if psd_telemetry else 0),
        "flow_band_alignment_stats": flow_band_alignment_stats,
        "flow_band_alignment_stats_flow": _stats_by_class(flow_alignment_vals_flow),
        "flow_band_alignment_stats_bg": _stats_by_class(flow_alignment_vals_bg),
        "flow_motion_angle_stats": flow_motion_angle_stats,
        "flow_motion_angle_stats_flow": _stats_by_class(flow_motion_vals_flow),
        "flow_motion_angle_stats_bg": _stats_by_class(flow_motion_vals_bg),
        "flow_motion_angle_stats_depth": _stats_by_depth(
            tile_infos, "flow_motion_angle_deg", depth_buckets
        ),
        "Pf_real": Pf_real,
        "Pf_imag": Pf_imag,
        "Pa_real": Pa_real,
        "Pa_imag": Pa_imag,
        "psd_peak_alignment": {
            "flow_fraction_in_band": flow_peak_fraction,
            "flow_samples": flow_peak_count,
            "flow_band_hz": [flow_band_low, flow_band_high],
            "alias_fraction_in_band": alias_peak_fraction,
            "alias_samples": alias_peak_count,
            "alias_band_hz": [alias_low_band, alias_high_band],
            "flow_peak_count": int(len(flow_peak_vals)),
            "alias_peak_count": int(len(alias_peak_vals)),
            "peak_hz_stats": _scalar_stats(all_peak_vals),
            "alias_fraction_in_band_on_alias_tiles": alias_cand_fraction,
            "alias_candidate_count": alias_cand_count,
        },
        "ka_gate_enable": bool(ka_gate_cfg_base is not None),
        "background_uniformized_count": int(background_uniformized_count),
        "background_uniformized_fraction": (
            background_uniformized_count / len(tile_infos) if tile_infos else 0.0
        ),
        "subtile_background_uniformized_count": int(subtile_background_uniformized_count),
        "subtile_background_uniformized_fraction": (
            subtile_background_uniformized_count / len(tile_infos) if tile_infos else 0.0
        ),
        "band_fraction_capped_count": int(band_fraction_capped_count),
        "band_fraction_capped_fraction": (
            band_fraction_capped_count / len(tile_infos) if tile_infos else 0.0
        ),
        "median_band_fraction_cap_scale": (
            float(np.median(band_fraction_cap_scales)) if band_fraction_cap_scales else None
        ),
        "p10_band_fraction_cap_scale": (
            float(np.quantile(band_fraction_cap_scales, 0.1))
            if len(band_fraction_cap_scales) >= 2
            else (band_fraction_cap_scales[0] if band_fraction_cap_scales else None)
        ),
        "alias_cap_applied_count": int(alias_cap_applied_count),
        "alias_cap_applied_fraction": (
            alias_cap_applied_count / len(tile_infos) if tile_infos else 0.0
        ),
        "median_alias_cap_scale": (
            float(np.median(alias_cap_scales)) if alias_cap_scales else None
        ),
        "alias_cap_scale_p90": (
            float(np.quantile(alias_cap_scales, 0.9))
            if len(alias_cap_scales) >= 2
            else (alias_cap_scales[0] if alias_cap_scales else None)
        ),
        "alias_cap_alias_ratio_median": (
            float(np.median(alias_cap_alias_ratios)) if alias_cap_alias_ratios else None
        ),
        "alias_cap_alias_ratio_p90": (
            float(np.quantile(alias_cap_alias_ratios, 0.9))
            if len(alias_cap_alias_ratios) >= 2
            else (alias_cap_alias_ratios[0] if alias_cap_alias_ratios else None)
        ),
        "alias_cap_band_fraction_median": (
            float(np.median(alias_cap_band_medians)) if alias_cap_band_medians else None
        ),
        "psd_alias_select_count": int(alias_psd_select_count),
        "psd_alias_select_fraction": (
            alias_psd_select_count / len(tile_infos) if tile_infos else 0.0
        ),
        "psd_kept_bin_median": (
            float(np.median(alias_psd_kept_bins)) if alias_psd_kept_bins else None
        ),
        "psd_alias_ratio_count": int(alias_ratio_array.size),
        "psd_alias_ratio_median": alias_ratio_stats["median"],
        "psd_alias_ratio_p10": alias_ratio_stats["p10"],
        "psd_alias_ratio_p90": alias_ratio_stats["p90"],
        "psd_kept_bin_mean": (
            float(np.mean(alias_psd_kept_bins)) if alias_psd_kept_bins else None
        ),
        "psd_kept_bin_p90": (
            float(np.quantile(alias_psd_kept_bins, 0.9)) if alias_psd_kept_bins else None
        ),
        "flow_cov_zero_count": int(flow_cov_zero_count),
        "flow_cov_zero_bg_uniformized_fraction": (
            (flow_cov_zero_bg_clamped / flow_cov_zero_count) if flow_cov_zero_count else None
        ),
        "flow_cov_zero_subtile_fraction": (
            (flow_cov_zero_subtile / flow_cov_zero_count) if flow_cov_zero_count else None
        ),
        "bg_edge_clamp_total": int(bg_edge_clamp_total),
        "bg_edge_clamp_fraction": (
            bg_edge_clamp_total / float(H * W) if H > 0 and W > 0 else None
        ),
        "median_Lt": float(np.median(Lt_vals)) if Lt_vals else None,
        "median_condR": float(np.median(cond_vals)) if cond_vals else None,
        "median_band_Kc": float(np.median(band_vals)) if band_vals else None,
        "median_span_hz": float(np.median(span_vals)) if span_vals else None,
        "median_f_peak_hz": float(np.median(peak_vals)) if peak_vals else None,
        "median_diag_load": float(np.median(diag_vals)) if diag_vals else None,
        "median_constraint_residual": float(np.median(resid_vals)) if resid_vals else None,
        "median_constraint_ridge": float(np.median(ridge_vals)) if ridge_vals else None,
        "median_auto_kappa_target": float(np.median(auto_vals)) if auto_vals else None,
        "median_cov_trace": float(np.median(trace_vals)) if trace_vals else None,
        "median_cov_eff_rank": float(np.median(eff_rank_vals)) if eff_rank_vals else None,
        "median_sigma_min_raw": float(np.median(sigma_min_vals)) if sigma_min_vals else None,
        "median_sigma_max_raw": float(np.median(sigma_max_vals)) if sigma_max_vals else None,
        "median_lambda_condition_needed": (
            float(np.median(lambda_needed_vals)) if lambda_needed_vals else None
        ),
        "median_lambda_conditioned": (
            float(np.median(lambda_conditioned_vals)) if lambda_conditioned_vals else None
        ),
        "median_diag_requested": (
            float(np.median(diag_requested_vals)) if diag_requested_vals else None
        ),
        "median_cond_loaded": float(np.median(cond_loaded_vals)) if cond_loaded_vals else None,
        "median_gram_diag": float(np.median(gram_med_vals)) if gram_med_vals else None,
        "p90_gram_diag": float(np.quantile(gram_max_vals, 0.9)) if gram_max_vals else None,
        "median_bg_var_inflation": float(np.median(bg_inflation)) if bg_inflation else None,
        "p90_bg_var_inflation": (
            float(np.quantile(bg_inflation, 0.9))
            if len(bg_inflation) >= 2
            else (float(bg_inflation[0]) if bg_inflation else None)
        ),
        "median_flow_mu_ratio": float(np.median(flow_ratio_vals)) if flow_ratio_vals else None,
        "p10_flow_mu_ratio": (
            float(np.quantile(flow_ratio_vals, 0.1))
            if len(flow_ratio_vals) >= 2
            else (float(flow_ratio_vals[0]) if flow_ratio_vals else None)
        ),
        "capon_tile_count": int(capon_count),
        "median_score_mean": float(np.median(score_mean_vals)) if score_mean_vals else None,
        "median_score_var": float(np.median(score_var_vals)) if score_var_vals else None,
        "median_msd_lambda": float(np.median(msd_lambda_vals)) if msd_lambda_vals else None,
        "median_msd_lambda_conditioned": (
            float(np.median(msd_lambda_conditioned_vals)) if msd_lambda_conditioned_vals else None
        ),
        "median_msd_lambda_needed": (
            float(np.median(msd_lambda_needed_vals)) if msd_lambda_needed_vals else None
        ),
        "median_band_fraction": float(np.median(band_frac_vals)) if band_frac_vals else None,
        "p90_band_fraction": (
            float(np.quantile(band_frac_p90_vals, 0.9)) if band_frac_p90_vals else None
        ),
        "median_motion_fraction": float(np.median(motion_frac_vals)) if motion_frac_vals else None,
        "p90_motion_fraction": (
            float(np.quantile(motion_frac_p90_vals, 0.9)) if motion_frac_p90_vals else None
        ),
        "median_motion_half_span_hz": (
            float(np.median(motion_span_vals)) if motion_span_vals else None
        ),
        "median_motion_half_span_rel_used": (
            float(np.median(motion_span_rel_vals)) if motion_span_rel_vals else None
        ),
        "median_fd_motion_freqs": (
            float(np.median(fd_motion_freq_vals)) if fd_motion_freq_vals else None
        ),
        "median_fd_flow_freqs_after_split": (
            float(np.median(fd_flow_freq_split_vals)) if fd_flow_freq_split_vals else None
        ),
        "median_msd_contrast_alpha": (
            float(np.median(contrast_alpha_vals)) if contrast_alpha_vals else None
        ),
        "median_motion_basis_rank": (
            float(np.median(motion_rank_vals)) if motion_rank_vals else None
        ),
        "median_energy_kept_ratio": (
            float(np.median(energy_kept_vals)) if energy_kept_vals else None
        ),
        "p90_energy_removed_ratio": (
            float(np.quantile(energy_removed_vals, 0.9)) if energy_removed_vals else None
        ),
        "median_kc_flow": float(np.median(kc_flow_vals)) if kc_flow_vals else None,
        "median_kc_motion": float(np.median(kc_motion_vals)) if kc_motion_vals else None,
        "median_kc_flow_cap": float(np.median(kc_flow_cap_vals)) if kc_flow_cap_vals else None,
        "median_kc_flow_cap_motion": (
            float(np.median(kc_flow_cap_motion_vals)) if kc_flow_cap_motion_vals else None
        ),
        "median_kc_flow_freqs": (
            float(np.median(kc_flow_freq_vals)) if kc_flow_freq_vals else None
        ),
        "contrast_tile_count": int(contrast_tile_count),
        "contrast_tile_fraction": (contrast_tile_count / len(tile_infos) if tile_infos else 0.0),
        "median_contrast_flow": (
            float(np.median(contrast_flow_med_vals)) if contrast_flow_med_vals else None
        ),
        "p90_contrast_flow": (
            float(np.median(contrast_flow_p90_vals)) if contrast_flow_p90_vals else None
        ),
        "median_contrast_motion": (
            float(np.median(contrast_motion_med_vals)) if contrast_motion_med_vals else None
        ),
        "p90_contrast_motion": (
            float(np.median(contrast_motion_p90_vals)) if contrast_motion_p90_vals else None
        ),
        "median_contrast_score_mean": (
            float(np.median(contrast_score_mean_vals)) if contrast_score_mean_vals else None
        ),
        "median_contrast_score_std": (
            float(np.median(contrast_score_std_vals)) if contrast_score_std_vals else None
        ),
        "median_contrast_flow_kc": (
            float(np.median(contrast_flow_kc_vals)) if contrast_flow_kc_vals else None
        ),
        "median_contrast_motion_kc": (
            float(np.median(contrast_motion_kc_vals)) if contrast_motion_kc_vals else None
        ),
        "median_contrast_flow_rank": (
            float(np.median(contrast_flow_rank_vals)) if contrast_flow_rank_vals else None
        ),
        "median_contrast_motion_rank_eff": (
            float(np.median(contrast_motion_rank_eff_vals))
            if contrast_motion_rank_eff_vals
            else None
        ),
        "median_contrast_motion_rank_initial": (
            float(np.median(contrast_motion_rank_init_vals))
            if contrast_motion_rank_init_vals
            else None
        ),
        "median_psd_peak_hz": float(np.median(psd_peak_vals)) if psd_peak_vals else None,
        "median_psd_flow_to_dc_ratio": (
            float(np.median(psd_ratio_vals)) if psd_ratio_vals else None
        ),
        "median_psd_power_flow": (
            float(np.median(psd_flow_power_vals)) if psd_flow_power_vals else None
        ),
        "median_psd_power_dc": (
            float(np.median(psd_dc_power_vals)) if psd_dc_power_vals else None
        ),
        "median_psd_flow_freq_target": (
            float(np.median(psd_flow_target_vals)) if psd_flow_target_vals else None
        ),
        "median_psd_flow_freq_power": (
            float(np.median(psd_flow_freq_power_vals)) if psd_flow_freq_power_vals else None
        ),
        "median_psd_fundamental_ratio": (
            float(np.median(psd_fundamental_ratio_vals)) if psd_fundamental_ratio_vals else None
        ),
        "psd_alias_fraction": (
            float(sum(psd_alias_flags)) / float(psd_alias_denominator)
            if psd_alias_denominator > 0
            else None
        ),
        "median_fundamental_hz": (
            float(np.median(fundamental_vals)) if fundamental_vals else None
        ),
        "fd_min_abs_applied_fraction": (
            float(sum(fd_min_applied_flags)) / len(fd_min_applied_flags)
            if fd_min_applied_flags
            else None
        ),
        "fd_symmetry_added_fraction": (
            float(sum(fd_symmetry_flags)) / len(fd_symmetry_flags) if fd_symmetry_flags else None
        ),
        "median_grid_step_hz": float(np.median(grid_step_vals)) if grid_step_vals else None,
        "median_band_fraction_q10": float(np.median(band_q10_vals)) if band_q10_vals else None,
        "median_band_fraction_q50": float(np.median(band_q50_vals)) if band_q50_vals else None,
        "median_band_fraction_q90": float(np.median(band_q90_vals)) if band_q90_vals else None,
        "median_score_q10": float(np.median(score_q10_vals)) if score_q10_vals else None,
        "median_score_q50": float(np.median(score_q50_vals)) if score_q50_vals else None,
        "median_score_q90": float(np.median(score_q90_vals)) if score_q90_vals else None,
        "total_band_nan": int(sum(band_nan_counts)) if band_nan_counts else 0,
        "total_band_inf": int(sum(band_inf_counts)) if band_inf_counts else 0,
        "total_score_nan": int(sum(score_nan_counts)) if score_nan_counts else 0,
        "total_score_inf": int(sum(score_inf_counts)) if score_inf_counts else 0,
        "score_mode_histogram": dict(score_mode_hist),
        "ka_failure_count": ka_failure_count,
        "ka_warning_hist": dict(ka_warning_hist),
        "psd_flow_freq_source_hist": dict(flow_source_resolved_hist),
        "fd_grid_example": fd_examples,
        "span_mode": span_mode,
        "diag_load": float(diag_load),
        "mvdr_load_mode": mvdr_load_mode,
        "mvdr_auto_kappa": float(mvdr_auto_kappa),
        "constraint_ridge": float(constraint_ridge),
        "cov_estimator": estimator,
        "constraint_mode": constraint_mode,
        "load_mode": load_mode,
        "huber_c": float(huber_c),
        "requested_msd_lambda": None if msd_lambda is None else float(msd_lambda),
        "requested_msd_ridge": float(msd_ridge),
        "msd_agg_mode": msd_agg_mode,
        "ka_mode": ka_mode_overall,
        "ka_tile_count": int(ka_tile_count),
        "ka_invalid_snr_count": int(ka_invalid_snr_count),
        "ka_invalid_noise_count": int(ka_invalid_noise_count),
        "ka_median_beta": float(np.median(ka_beta_vals)) if ka_beta_vals else None,
        "ka_median_mismatch": float(np.median(ka_mismatch_vals)) if ka_mismatch_vals else None,
        "ka_median_lambda_used": float(np.median(ka_lambda_vals)) if ka_lambda_vals else None,
        "ka_median_sigma_min_raw": (
            float(np.median(ka_sigma_min_vals)) if ka_sigma_min_vals else None
        ),
        "ka_median_sigma_max_raw": (
            float(np.median(ka_sigma_max_vals)) if ka_sigma_max_vals else None
        ),
        "ka_median_snr_flow_ratio": (
            float(np.median(ka_snr_flow_ratio_vals)) if ka_snr_flow_ratio_vals else None
        ),
        "ka_p10_snr_flow_ratio": (
            float(np.quantile(ka_snr_flow_ratio_vals, 0.10))
            if len(ka_snr_flow_ratio_vals) >= 2
            else (float(ka_snr_flow_ratio_vals[0]) if ka_snr_flow_ratio_vals else None)
        ),
        "ka_p90_snr_flow_ratio": (
            float(np.quantile(ka_snr_flow_ratio_vals, 0.90))
            if len(ka_snr_flow_ratio_vals) >= 2
            else (float(ka_snr_flow_ratio_vals[0]) if ka_snr_flow_ratio_vals else None)
        ),
        "ka_median_snr_flow_base": (
            float(np.median(ka_snr_flow_base_vals)) if ka_snr_flow_base_vals else None
        ),
        "ka_median_snr_flow_loaded": (
            float(np.median(ka_snr_flow_loaded_vals)) if ka_snr_flow_loaded_vals else None
        ),
        "ka_median_noise_perp_ratio": (
            float(np.median(ka_noise_perp_ratio_vals)) if ka_noise_perp_ratio_vals else None
        ),
        "ka_p10_noise_perp_ratio": (
            float(np.quantile(ka_noise_perp_ratio_vals, 0.10))
            if len(ka_noise_perp_ratio_vals) >= 2
            else (float(ka_noise_perp_ratio_vals[0]) if ka_noise_perp_ratio_vals else None)
        ),
        "ka_p90_noise_perp_ratio": (
            float(np.quantile(ka_noise_perp_ratio_vals, 0.90))
            if len(ka_noise_perp_ratio_vals) >= 2
            else (float(ka_noise_perp_ratio_vals[0]) if ka_noise_perp_ratio_vals else None)
        ),
        "ka_pf_trace_alpha_median": (
            float(np.median(ka_pf_trace_alpha_vals)) if ka_pf_trace_alpha_vals else None
        ),
        "ka_pf_trace_alpha_p10": (
            float(np.quantile(ka_pf_trace_alpha_vals, 0.10)) if ka_pf_trace_alpha_vals else None
        ),
        "ka_pf_trace_alpha_p90": (
            float(np.quantile(ka_pf_trace_alpha_vals, 0.90)) if ka_pf_trace_alpha_vals else None
        ),
        "ka_pf_trace_loaded_median": (
            float(np.median(ka_pf_trace_loaded_vals)) if ka_pf_trace_loaded_vals else None
        ),
        "ka_pf_trace_loaded_p10": (
            float(np.quantile(ka_pf_trace_loaded_vals, 0.10)) if ka_pf_trace_loaded_vals else None
        ),
        "ka_pf_trace_loaded_p90": (
            float(np.quantile(ka_pf_trace_loaded_vals, 0.90)) if ka_pf_trace_loaded_vals else None
        ),
        "ka_pf_trace_sample_median": (
            float(np.median(ka_pf_trace_sample_vals)) if ka_pf_trace_sample_vals else None
        ),
        "ka_pf_trace_sample_p10": (
            float(np.quantile(ka_pf_trace_sample_vals, 0.10)) if ka_pf_trace_sample_vals else None
        ),
        "ka_pf_trace_sample_p90": (
            float(np.quantile(ka_pf_trace_sample_vals, 0.90)) if ka_pf_trace_sample_vals else None
        ),
        "ka_trace_ratio_median": (
            float(np.median(ka_trace_ratio_vals)) if ka_trace_ratio_vals else None
        ),
        "ka_trace_scaled_fraction": (
            float(np.mean(ka_trace_scaled_vals)) if ka_trace_scaled_vals else None
        ),
        "ka_trace_scale_lock_hist": (
            dict(Counter(ka_trace_scale_lock_vals)) if ka_trace_scale_lock_vals else {}
        ),
        "ka_pf_trace_alpha_valid_fraction": (
            len(ka_pf_trace_alpha_vals) / ka_pf_trace_alpha_total
            if ka_pf_trace_alpha_total
            else None
        ),
        "ka_pf_trace_alpha_invalid_count": int(ka_pf_trace_alpha_invalid),
        "ka_pf_trace_alpha_total_tiles": int(ka_pf_trace_alpha_total),
        "ka_pf_trace_loaded_valid_fraction": (
            len(ka_pf_trace_loaded_vals) / ka_pf_trace_loaded_total
            if ka_pf_trace_loaded_total
            else None
        ),
        "ka_pf_trace_loaded_invalid_count": int(ka_pf_trace_loaded_invalid),
        "ka_pf_trace_loaded_total_tiles": int(ka_pf_trace_loaded_total),
        "ka_pf_trace_sample_valid_fraction": (
            len(ka_pf_trace_sample_vals) / ka_pf_trace_sample_total
            if ka_pf_trace_sample_total
            else None
        ),
        "ka_pf_trace_sample_invalid_count": int(ka_pf_trace_sample_invalid),
        "ka_pf_trace_sample_total_tiles": int(ka_pf_trace_sample_total),
        "ka_pf_trace_ratio_median": None,
        "ka_median_retain_f_beta": float(np.median(ka_retain_vals)) if ka_retain_vals else None,
        "ka_median_shrink_perp_beta": float(np.median(ka_shrink_vals)) if ka_shrink_vals else None,
        "ka_median_retain_f_total": (
            float(np.median(ka_retain_total_vals)) if ka_retain_total_vals else None
        ),
        "ka_median_shrink_perp_total": (
            float(np.median(ka_shrink_total_vals)) if ka_shrink_total_vals else None
        ),
        "ka_ridge_split_fraction": float(np.mean(ka_ridge_vals)) if ka_ridge_vals else None,
        "ka_prior_clip_fraction": (
            float(np.mean(ka_prior_clip_vals)) if ka_prior_clip_vals else None
        ),
        "ka_lambda_strategy_hist": (
            dict(Counter(ka_lambda_strategy_vals)) if ka_lambda_strategy_vals else {}
        ),
        "ka_median_pf_rank": float(np.median(ka_pf_rank_vals)) if ka_pf_rank_vals else None,
        "ka_median_alias_rank": (
            float(np.median(ka_alias_rank_vals)) if ka_alias_rank_vals else None
        ),
        "ka_median_alias_gain_target": (
            float(np.median(ka_alias_gain_vals)) if ka_alias_gain_vals else None
        ),
        "ka_pf_trace_equalized_fraction": (
            ka_pf_equalized_count / ka_tile_count if ka_tile_count else None
        ),
        "median_pd_condG": float(np.median(pd_cond_vals)) if pd_cond_vals else None,
        "median_pd_gram_ridge": float(np.median(pd_ridge_vals)) if pd_ridge_vals else None,
        "median_lam_pf_extra_pd": (
            float(np.median(lam_pf_extra_vals)) if lam_pf_extra_vals else None
        ),
        "pd_retry_histogram": dict(Counter(pd_retry_modes)) if pd_retry_modes else {},
        "stap_device": device_resolved,
        "debug_samples": debug_payloads,
    }
    info_out.update(_operator_metric_stats(tile_infos))
    if ka_operator_flags:
        info_out["ka_operator_feasible_fraction"] = float(np.mean(ka_operator_flags))
        info_out["ka_operator_feasible"] = bool(all(ka_operator_flags))
    else:
        info_out["ka_operator_feasible"] = None

    # Alias metric telemetry (m_alias) for KA feasibility diagnostics.
    alias_bg_vals: list[float] = []
    alias_flow_vals: list[float] = []
    alias_bg_pial_vals: list[float] = []
    alias_flow_pial_vals: list[float] = []
    pial_depth_min = 0.0
    pial_depth_max = 1.0
    if ka_gate_cfg_base is not None:
        if ka_gate_cfg_base.get("depth_min_frac") is not None:
            try:
                pial_depth_min = float(ka_gate_cfg_base["depth_min_frac"])
            except Exception:
                pial_depth_min = 0.0
        if ka_gate_cfg_base.get("depth_max_frac") is not None:
            try:
                pial_depth_max = float(ka_gate_cfg_base["depth_max_frac"])
            except Exception:
                pial_depth_max = 1.0
    for info in tile_infos:
        val = info.get("ka_alias_metric")
        if val is None:
            continue
        try:
            val_float = float(val)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(val_float):
            continue
        depth = info.get("depth_center_frac")
        try:
            depth_float = float(depth) if depth is not None else None
        except (TypeError, ValueError):
            depth_float = None
        tile_is_bg = bool(info.get("tile_is_bg"))
        tile_has_flow = bool(info.get("tile_has_flow"))
        if tile_is_bg:
            alias_bg_vals.append(val_float)
        if tile_has_flow:
            alias_flow_vals.append(val_float)
        if depth_float is not None and np.isfinite(depth_float):
            if pial_depth_min <= depth_float <= pial_depth_max:
                if tile_is_bg:
                    alias_bg_pial_vals.append(val_float)
                if tile_has_flow:
                    alias_flow_pial_vals.append(val_float)
    info_out["ka_alias_metric_stats"] = {
        "bg": _scalar_stats(alias_bg_vals),
        "flow": _scalar_stats(alias_flow_vals),
        "bg_pial": _scalar_stats(alias_bg_pial_vals),
        "flow_pial": _scalar_stats(alias_flow_pial_vals),
    }

    # Retain a small sample of tile telemetry for debugging aggregation.
    keep_tile_keys = [
        "ka_mode",
        "ka_gate_ok",
        "ka_gate_ok_raw",
        "ka_gate_alias_ok_raw",
        "ka_gate_flow_ok_raw",
        "ka_gate_depth_ok_raw",
        "ka_gate_pd_ok_raw",
        "ka_gate_reg_ok_raw",
        "ka_gate_debug",
        "ka_gate_tile_has_flow",
        "ka_gate_tile_is_bg",
        "ka_alias_metric",
        "ka_alias_E_f",
        "ka_alias_E_a",
        "ka_alias_E_g",
        "ka_alias_E_dc",
        "ka_alias_c_f",
        "ka_alias_c_a",
        "ka_alias_r_g",
        "flow_coverage",
        "flow_band_alignment",
        "flow_motion_angle_deg",
        "psd_peak_hz",
        "psd_flow_alias_ratio",
        "kc_flow",
        "kc_alias",
        "kc_flow_error",
        "ka_band_metrics_error",
        "ka_detail_len",
        "ka_pf_lambda_min",
        "ka_pf_lambda_max",
        "ka_pf_lambda_mean",
        "ka_perp_lambda_mean",
        "ka_alias_lambda_mean",
        "ka_noise_lambda_mean",
        "ka_operator_mixing_epsilon",
        "ka_snr_flow_ratio",
        "ka_noise_perp_ratio",
        "ka_pf_rank",
        "ka_alias_rank",
        "ka_alias_gain_target",
        "ka_band_metrics_error",
        "ka_detail_keys",
        "ka_detail_len",
        "ka_warning",
    ]
    tile_infos_trim: list[dict] = []
    max_tile_sample = 200
    # Prioritize KA-active tiles in the sample, then fill with inactive tiles for coverage.
    active_tiles = [
        info for info in tile_infos if info.get("ka_mode") and info.get("ka_mode") != "none"
    ]
    inactive_tiles = [
        info for info in tile_infos if not (info.get("ka_mode") and info.get("ka_mode") != "none")
    ]
    for group in (active_tiles, inactive_tiles):
        if len(tile_infos_trim) >= max_tile_sample:
            break
        step = max(1, len(group) // max_tile_sample) if group else 1
        for idx, info in enumerate(group):
            if len(tile_infos_trim) >= max_tile_sample:
                break
            if step > 1 and idx % step != 0:
                continue
            entry = {
                k: info.get(k) for k in keep_tile_keys if k in info and info.get(k) is not None
            }
            # If KA is active but ratios are missing, derive from band means for visibility.
            if entry.get("ka_mode") and entry.get("ka_mode") != "none":
                pf_mean = entry.get("ka_pf_lambda_mean")
                perp_mean = entry.get("ka_perp_lambda_mean")
                if entry.get("ka_snr_flow_ratio") is None and pf_mean is not None and perp_mean:
                    try:
                        entry["ka_snr_flow_ratio"] = float(pf_mean) / max(float(perp_mean), 1e-9)
                    except Exception:
                        pass
                if entry.get("ka_noise_perp_ratio") is None and perp_mean is not None:
                    try:
                        entry["ka_noise_perp_ratio"] = float(perp_mean)
                    except Exception:
                        pass
            if entry:
                tile_infos_trim.append(entry)
    info_out["tile_infos_sample"] = tile_infos_trim
    info_out["tile_infos_total"] = int(len(tile_infos))

    # Aggregate fine-grained STAP phase timings from tile infos (if present).
    phase_keys = [
        "stap_hankel_ms",
        "stap_cov_ms",
        "stap_shrink_ms",
        "stap_fdgrid_ms",
        "stap_msd_ms",
    ]
    for key in phase_keys:
        vals = []
        for info in tile_infos:
            v = info.get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        info_out[f"{key}_mean"] = float(np.mean(arr))
        info_out[f"{key}_median"] = float(np.median(arr))
    ka_effective_runtime, ka_disable_reasons_runtime, pf_trace_ratio = _ka_effective_status(
        ka_tile_count, ka_disable_reasons, info_out
    )
    info_out["ka_disable_reasons"] = ka_disable_reasons_runtime
    info_out["ka_effective"] = bool(ka_mode_overall != "none" and ka_effective_runtime)
    if pf_trace_ratio is not None:
        info_out["ka_pf_trace_ratio_median"] = pf_trace_ratio

    # KA gate coverage and condition-level stats.
    def _gate_bool_stats(values: list[bool]) -> dict[str, float | int]:
        if not values:
            return {"count": 0, "true_fraction": 0.0}
        arr = np.array([bool(v) for v in values], dtype=bool)
        return {"count": int(arr.size), "true_fraction": float(np.mean(arr))}

    if ka_gate_ok_vals:
        gate_vals = [1.0 if bool(val) else 0.0 for val in ka_gate_ok_vals]
        info_out["ka_gate_fraction_total"] = float(np.mean(gate_vals))
    else:
        info_out["ka_gate_fraction_total"] = 0.0
    if ka_gate_flow_hits:
        flow_vals = [1.0 if bool(info.get("ka_gate_ok")) else 0.0 for info in ka_gate_flow_hits]
        info_out["ka_gate_fraction_on_flow"] = float(np.mean(flow_vals))
    else:
        info_out["ka_gate_fraction_on_flow"] = None
    if ka_gate_bg_hits:
        bg_vals = [1.0 if bool(info.get("ka_gate_ok")) else 0.0 for info in ka_gate_bg_hits]
        info_out["ka_gate_fraction_on_bg"] = float(np.mean(bg_vals))
    else:
        info_out["ka_gate_fraction_on_bg"] = None
    # Condition-level stats across all tiles / by class.
    ka_gate_alias_ok_vals = [
        info.get("ka_gate_alias_ok")
        for info in tile_infos
        if info.get("ka_gate_alias_ok") is not None
    ]
    ka_gate_flow_ok_vals = [
        info.get("ka_gate_flow_ok")
        for info in tile_infos
        if info.get("ka_gate_flow_ok") is not None
    ]
    ka_gate_depth_ok_vals = [
        info.get("ka_gate_depth_ok")
        for info in tile_infos
        if info.get("ka_gate_depth_ok") is not None
    ]
    ka_gate_pd_ok_vals = [
        info.get("ka_gate_pd_ok") for info in tile_infos if info.get("ka_gate_pd_ok") is not None
    ]
    ka_gate_reg_ok_vals = [
        info.get("ka_gate_reg_ok") for info in tile_infos if info.get("ka_gate_reg_ok") is not None
    ]
    ka_gate_cond_all = {
        "ok": _gate_bool_stats([bool(v) for v in ka_gate_ok_vals]),
        "alias_ok": _gate_bool_stats([bool(v) for v in ka_gate_alias_ok_vals]),
        "flow_ok": _gate_bool_stats([bool(v) for v in ka_gate_flow_ok_vals]),
        "depth_ok": _gate_bool_stats([bool(v) for v in ka_gate_depth_ok_vals]),
        "pd_ok": _gate_bool_stats([bool(v) for v in ka_gate_pd_ok_vals]),
        "reg_ok": _gate_bool_stats([bool(v) for v in ka_gate_reg_ok_vals]),
    }

    def _gate_cond_subset(subset: list[dict]) -> dict[str, dict[str, float | int]]:
        vals_ok = [info.get("ka_gate_ok") for info in subset if info.get("ka_gate_ok") is not None]
        vals_alias = [
            info.get("ka_gate_alias_ok")
            for info in subset
            if info.get("ka_gate_alias_ok") is not None
        ]
        vals_flow = [
            info.get("ka_gate_flow_ok")
            for info in subset
            if info.get("ka_gate_flow_ok") is not None
        ]
        vals_depth = [
            info.get("ka_gate_depth_ok")
            for info in subset
            if info.get("ka_gate_depth_ok") is not None
        ]
        vals_pd = [
            info.get("ka_gate_pd_ok") for info in subset if info.get("ka_gate_pd_ok") is not None
        ]
        vals_reg = [
            info.get("ka_gate_reg_ok") for info in subset if info.get("ka_gate_reg_ok") is not None
        ]
        return {
            "ok": _gate_bool_stats([bool(v) for v in vals_ok]),
            "alias_ok": _gate_bool_stats([bool(v) for v in vals_alias]),
            "flow_ok": _gate_bool_stats([bool(v) for v in vals_flow]),
            "depth_ok": _gate_bool_stats([bool(v) for v in vals_depth]),
            "pd_ok": _gate_bool_stats([bool(v) for v in vals_pd]),
            "reg_ok": _gate_bool_stats([bool(v) for v in vals_reg]),
        }

    gate_flow_all = [info for info in tile_infos if info.get("ka_gate_tile_has_flow")]
    gate_bg_all = [info for info in tile_infos if info.get("ka_gate_tile_is_bg")]
    info_out["ka_gate_condition_stats"] = {
        "all": ka_gate_cond_all,
        "flow": _gate_cond_subset(gate_flow_all),
        "bg": _gate_cond_subset(gate_bg_all),
    }

    if ka_gate_cfg_base is not None:
        alias_rmin_val = ka_gate_cfg_base.get("alias_rmin")
        info_out["ka_gate_alias_rmin"] = (
            float(alias_rmin_val) if alias_rmin_val is not None else None
        )
        info_out["ka_gate_flow_cov_min"] = (
            float(ka_gate_cfg_base["flow_cov_min"])
            if ka_gate_cfg_base.get("flow_cov_min") is not None
            else None
        )
        info_out["ka_gate_depth_min_frac"] = (
            float(ka_gate_cfg_base["depth_min_frac"])
            if ka_gate_cfg_base.get("depth_min_frac") is not None
            else None
        )
        info_out["ka_gate_depth_max_frac"] = (
            float(ka_gate_cfg_base["depth_max_frac"])
            if ka_gate_cfg_base.get("depth_max_frac") is not None
            else None
        )
        info_out["ka_gate_pd_min"] = (
            float(ka_gate_cfg_base["pd_min"])
            if ka_gate_cfg_base.get("pd_min") is not None
            else None
        )
        info_out["ka_gate_reg_psr_max"] = (
            float(ka_gate_cfg_base["reg_psr_max"])
            if ka_gate_cfg_base.get("reg_psr_max") is not None
            else None
        )
    else:
        info_out["ka_gate_alias_rmin"] = None
        info_out["ka_gate_flow_cov_min"] = None
        info_out["ka_gate_depth_min_frac"] = None
        info_out["ka_gate_depth_max_frac"] = None
        info_out["ka_gate_pd_min"] = None
        info_out["ka_gate_reg_psr_max"] = None

    def _gate_alias_summary(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "median": None, "p10": None, "p90": None}
        arr = np.asarray(values, dtype=float)
        summary: dict[str, float | int | None] = {
            "count": int(arr.size),
            "median": float(np.median(arr)),
        }
        if arr.size >= 2:
            summary["p10"] = float(np.quantile(arr, 0.10))
            summary["p90"] = float(np.quantile(arr, 0.90))
        else:
            summary["p10"] = float(arr[0])
            summary["p90"] = float(arr[0])
        return summary

    info_out["ka_gate_alias_stats"] = {
        "flow": _gate_alias_summary(gate_flow_alias_vals),
        "bg": _gate_alias_summary(gate_bg_alias_vals),
    }
    gate_flow_gated = [
        info for info in tile_infos if info.get("ka_gate_ok") and info.get("ka_gate_tile_has_flow")
    ]
    gate_bg_gated = [
        info for info in tile_infos if info.get("ka_gate_ok") and info.get("ka_gate_tile_is_bg")
    ]

    def _gate_feature_stats(subset: list[dict]) -> dict[str, dict[str, float | int | None]]:
        def _collect(key: str) -> dict[str, float | int | None]:
            vals = [
                float(info.get(key))
                for info in subset
                if info.get(key) is not None and np.isfinite(info.get(key))
            ]
            return _scalar_stats(vals)

        return {
            "alias_ratio": (
                _collect("ka_alias_metric")
                if any(info.get("ka_alias_metric") is not None for info in subset)
                else _collect("psd_flow_alias_ratio")
            ),
            "flow_cov": _collect("ka_gate_flow_cov"),
            "depth_frac": _collect("ka_gate_depth_frac"),
            "pd_metric": _collect("ka_gate_pd_metric"),
        }

    info_out["ka_gate_feature_stats"] = {
        "flow": _gate_feature_stats(gate_flow_gated),
        "bg": _gate_feature_stats(gate_bg_gated),
    }
    info_out["ka_snr_flow_ratio_depth"] = _stats_by_depth(
        tile_infos, "ka_snr_flow_ratio", depth_buckets
    )
    info_out["ka_noise_perp_ratio_depth"] = _stats_by_depth(
        tile_infos, "ka_noise_perp_ratio", depth_buckets
    )
    info_out["_gate_mask_flow"] = gate_mask_flow if gate_mask_flow is not None else None
    info_out["_gate_mask_bg"] = gate_mask_bg if gate_mask_bg is not None else None
    info_out["gamma_flow_median"] = float(np.median(gamma_flow_vals)) if gamma_flow_vals else None
    info_out["gamma_perp_median"] = float(np.median(gamma_perp_vals)) if gamma_perp_vals else None

    info_out["ka_directional_strict"] = any(
        bool(info.get("ka_directional_strict")) for info in tile_infos
    )
    if capture_coord_requests:
        info_out["debug_coords_requested"] = [
            [int(y0), int(x0)] for y0, x0 in sorted(capture_coord_requests)
        ]
        info_out["debug_coords_captured"] = [
            [int(y0), int(x0)]
            for y0, x0 in sorted(capture_coord_requests - remaining_coord_requests)
        ]
        if remaining_coord_requests:
            info_out["debug_coords_missing"] = [
                [int(y0), int(x0)] for y0, x0 in sorted(remaining_coord_requests)
            ]
    ratios_with_cov = [
        (
            info.get("flow_mu_ratio"),
            info.get("flow_coverage"),
        )
        for info in tile_infos
        if not info.get("fallback") and info.get("flow_mu_ratio") is not None
    ]
    ratios_array = np.array(
        [float(r[0]) for r in ratios_with_cov if r[0] is not None], dtype=float
    )
    coverage_array = np.array(
        [float(r[1]) if r[1] is not None else 0.0 for r in ratios_with_cov], dtype=float
    )
    guard_valid_mask = (
        coverage_array >= guard_tile_coverage_min
        if coverage_array.size
        else np.array([], dtype=bool)
    )
    ratios_guard = ratios_array[guard_valid_mask] if guard_valid_mask.size else np.array([])
    tile_flow_ratio_p10 = (
        float(np.quantile(ratios_guard, guard_percentile_low)) if ratios_guard.size else None
    )
    tile_flow_ratio_p50 = float(np.median(ratios_guard)) if ratios_guard.size else None
    tile_flow_ratio_p05 = float(np.quantile(ratios_guard, 0.05)) if ratios_guard.size else None
    tile_flow_ratio_p90 = float(np.quantile(ratios_guard, 0.90)) if ratios_guard.size else None
    info_out["tile_flow_ratio_count"] = int(ratios_guard.size)
    info_out["tile_flow_ratio_p05"] = tile_flow_ratio_p05
    info_out["tile_flow_ratio_p10"] = tile_flow_ratio_p10
    info_out["tile_flow_ratio_p50"] = tile_flow_ratio_p50
    info_out["tile_flow_ratio_p90"] = tile_flow_ratio_p90
    coverage_thresholds = (0.2, 0.5, 0.8)
    for thr in coverage_thresholds:
        key = f"flow_cov_ge_{int(thr * 100):02d}"
        if coverage_array.size == 0:
            info_out[f"{key}_fraction"] = 0.0
            info_out[f"{key}_flow_ratio_median"] = None
            info_out[f"{key}_flow_ratio_p90"] = None
            continue
        mask_thr = coverage_array >= thr
        info_out[f"{key}_fraction"] = float(np.mean(mask_thr))
        if mask_thr.any():
            ratios_thr = ratios_array[mask_thr]
            info_out[f"{key}_flow_ratio_median"] = float(np.median(ratios_thr))
            info_out[f"{key}_flow_ratio_p90"] = float(np.quantile(ratios_thr, 0.90))
        else:
            info_out[f"{key}_flow_ratio_median"] = None
            info_out[f"{key}_flow_ratio_p90"] = None
    alias_cov_pairs = [
        (float(info.get("flow_coverage")), float(info.get("psd_flow_alias_ratio")))
        for info in tile_infos
        if info.get("flow_coverage") is not None
        and info.get("psd_flow_alias_ratio") is not None
        and np.isfinite(info.get("psd_flow_alias_ratio"))
    ]
    alias_cov_metrics: dict[str, float | None] = {}
    alias_flag_ratio_thresh = 1.05
    if alias_cov_pairs:
        alias_cov_array = np.array([cov for cov, _ in alias_cov_pairs], dtype=float)
        alias_ratio_cov = np.array([ratio for _, ratio in alias_cov_pairs], dtype=float)
        alias_flags = alias_ratio_cov >= alias_flag_ratio_thresh
        for thr in coverage_thresholds:
            key = f"alias_cov_ge_{int(thr * 100):02d}"
            mask = alias_cov_array >= thr
            if mask.any():
                alias_cov_metrics[f"{key}_alias_flag_fraction"] = float(np.mean(alias_flags[mask]))
                alias_cov_metrics[f"{key}_alias_ratio_median"] = float(
                    np.median(alias_ratio_cov[mask])
                )
            else:
                alias_cov_metrics[f"{key}_alias_flag_fraction"] = 0.0
                alias_cov_metrics[f"{key}_alias_ratio_median"] = None
    info_out["alias_flag_ratio_thresh"] = alias_flag_ratio_thresh
    if alias_cov_metrics:
        info_out.update(alias_cov_metrics)
    info_out["guard_policy"] = {
        "mode": "tile_percentile",
        "target_median": guard_target_med,
        "target_low": guard_target_low,
        "percentile_low": guard_percentile_low,
        "coverage_min": guard_tile_coverage_min,
        "max_scale": guard_max_scale,
        "legacy_min": legacy_guard_min,
    }

    pd_avg = (pd / counts).astype(np.float32)
    score_avg = (score / score_counts).astype(np.float32)
    counts_positive = counts[counts > 0.0]
    if counts_positive.size > 0:
        info_out["pd_overlap_count_min"] = float(counts_positive.min())
        info_out["pd_overlap_count_max"] = float(counts_positive.max())
        info_out["pd_overlap_count_mean"] = float(counts_positive.mean())
    else:
        info_out["pd_overlap_count_min"] = 0.0
        info_out["pd_overlap_count_max"] = 0.0
        info_out["pd_overlap_count_mean"] = 0.0
    info_out["pd_overlap_any_zero"] = bool(np.any(counts == 0.0))

    global_flow_ratio = None
    guard_scale = None
    global_flow_ratio_pre = None
    if pd_base_full is not None and mask_flow is not None and pd_base_full.shape == pd_avg.shape:
        flow_mask = mask_flow.astype(bool)
        if flow_mask.any():
            flow_mu_base = float(pd_base_full[flow_mask].mean())
            if flow_mu_base > 0.0:
                flow_mu_stap_pre = float(pd_avg[flow_mask].mean())
                global_flow_ratio_pre = flow_mu_stap_pre / max(flow_mu_base, 1e-12)
                scale_candidates: list[float] = []
                if (
                    legacy_guard_min is not None
                    and global_flow_ratio_pre < legacy_guard_min
                    and legacy_guard_min > 0.0
                ):
                    scale_candidates.append(legacy_guard_min / max(global_flow_ratio_pre, 1e-12))
                if tile_flow_ratio_p50 is not None and tile_flow_ratio_p50 < guard_target_med:
                    scale_candidates.append(guard_target_med / max(tile_flow_ratio_p50, 1e-6))
                if tile_flow_ratio_p10 is not None and tile_flow_ratio_p10 < guard_target_low:
                    scale_candidates.append(guard_target_low / max(tile_flow_ratio_p10, 1e-6))
                scale_candidates = [s for s in scale_candidates if s > 1.0]
                if scale_candidates:
                    guard_scale = float(min(guard_max_scale, max(scale_candidates)))
                    pd_scaled = pd_avg[flow_mask] * guard_scale
                    if guard_clip_base:
                        pd_scaled = np.minimum(pd_scaled, pd_base_full[flow_mask])
                    pd_avg[flow_mask] = pd_scaled.astype(pd_avg.dtype, copy=False)
                    flow_mu_stap_post = float(pd_avg[flow_mask].mean())
                    global_flow_ratio = flow_mu_stap_post / max(flow_mu_base, 1e-12)
                else:
                    global_flow_ratio = global_flow_ratio_pre
            else:
                global_flow_ratio = None
    if global_flow_ratio is None and global_flow_ratio_pre is not None:
        global_flow_ratio = global_flow_ratio_pre

    if global_flow_ratio is not None:
        existing_median = info_out.get("median_flow_mu_ratio")
        if existing_median is None or global_flow_ratio > existing_median:
            info_out["median_flow_mu_ratio"] = float(global_flow_ratio)
        info_out["global_flow_ratio"] = float(global_flow_ratio)
    if global_flow_ratio_pre is not None:
        info_out["global_flow_ratio_pre_guard"] = float(global_flow_ratio_pre)
    info_out["global_flow_guard_applied"] = bool(guard_scale is not None)
    info_out["global_flow_guard_scale"] = float(guard_scale) if guard_scale is not None else 1.0
    info_out["global_flow_guard_clip_base"] = bool(guard_clip_base)
    info_out["global_flow_ratio_target"] = guard_target_med
    info_out["global_flow_ratio_target_low"] = guard_target_low
    info_out["guard_percentile_low"] = guard_percentile_low
    info_out["guard_tile_coverage_min"] = guard_tile_coverage_min
    info_out["guard_max_scale"] = guard_max_scale

    flow_coverage_vals = [
        float(info.get("flow_coverage"))
        for info in tile_infos
        if not info.get("fallback") and info.get("flow_coverage") is not None
    ]
    info_out["tile_flow_coverage_count"] = int(len(flow_coverage_vals))
    if flow_coverage_vals:
        arr = np.asarray(flow_coverage_vals, dtype=float)
        info_out["tile_flow_coverage_p05"] = (
            float(np.quantile(arr, 0.05)) if arr.size >= 2 else float(arr[0])
        )
        info_out["tile_flow_coverage_p10"] = (
            float(np.quantile(arr, 0.10)) if arr.size >= 2 else float(arr[0])
        )
        info_out["tile_flow_coverage_p50"] = float(np.median(arr))
        info_out["tile_flow_coverage_p90"] = (
            float(np.quantile(arr, 0.90)) if arr.size >= 2 else float(arr[0])
        )
        info_out["flow_cov_zero_fraction"] = float(np.mean(arr <= 1e-9))
    else:
        info_out["tile_flow_coverage_p05"] = None
        info_out["tile_flow_coverage_p10"] = None
        info_out["tile_flow_coverage_p50"] = None
        info_out["tile_flow_coverage_p90"] = None
        info_out["flow_cov_zero_fraction"] = None

    # Background variance inflation guard (applied after flow guard)
    bg_guard_applied = False
    bg_guard_alpha_used: float | None = None
    bg_var_ratio_pre: float | None = None
    bg_var_ratio_post: float | None = None
    bg_var_base_pre: float | None = None
    tile_bg_var_vals = [
        info.get("bg_var_inflation")
        for info in tile_infos
        if not info.get("fallback") and info.get("bg_var_inflation") is not None
    ]
    tile_bg_var_vals = [float(v) for v in tile_bg_var_vals if np.isfinite(v)]
    tile_bg_p90 = (
        float(np.quantile(tile_bg_var_vals, 0.9))
        if len(tile_bg_var_vals) >= 2
        else (float(tile_bg_var_vals[0]) if tile_bg_var_vals else None)
    )
    tile_bg_p50 = float(np.median(tile_bg_var_vals)) if tile_bg_var_vals else None
    tile_bg_p10 = (
        float(np.quantile(tile_bg_var_vals, 0.1))
        if len(tile_bg_var_vals) >= 2
        else (float(tile_bg_var_vals[0]) if tile_bg_var_vals else None)
    )
    tile_bg_p05 = (
        float(np.quantile(tile_bg_var_vals, 0.05))
        if len(tile_bg_var_vals) >= 2
        else (float(tile_bg_var_vals[0]) if tile_bg_var_vals else None)
    )

    info_out["tile_bg_var_ratio_count"] = int(len(tile_bg_var_vals))
    info_out["tile_bg_var_ratio_p05"] = tile_bg_p05
    info_out["tile_bg_var_ratio_p10"] = tile_bg_p10
    info_out["tile_bg_var_ratio_p50"] = tile_bg_p50
    info_out["tile_bg_var_ratio_p90"] = tile_bg_p90

    has_bg_arrays = (
        mask_bg is not None
        and pd_base_full is not None
        and pd_base_full.shape == pd_avg.shape
        and mask_bg.shape == pd_avg.shape
    )
    if has_bg_arrays:
        bg_var_ratio_pre, bg_var_base_pre, _ = _bg_var_ratio(pd_avg, pd_base_full, mask_bg)

    if bg_guard_enabled and has_bg_arrays:
        metric_ratio = bg_var_ratio_pre
        if metric_ratio is None and tile_bg_p90 is not None:
            metric_ratio = tile_bg_p90

        if metric_ratio is not None and metric_ratio > bg_guard_target_p90:
            for _ in range(20):
                alpha = float(np.sqrt(bg_guard_target_p90 / max(metric_ratio, 1e-12)))
                alpha = float(min(1.0, max(bg_guard_min_alpha, alpha)))
                if alpha >= 0.999:
                    break
                pd_bg_new = pd_base_full[mask_bg] + alpha * (
                    pd_avg[mask_bg] - pd_base_full[mask_bg]
                )
                pd_avg[mask_bg] = pd_bg_new.astype(pd_avg.dtype, copy=False)
                bg_guard_applied = True
                bg_guard_alpha_used = alpha
                bg_var_ratio_post, _, _ = _bg_var_ratio(pd_avg, pd_base_full, mask_bg)
                metric_ratio = bg_var_ratio_post
                if metric_ratio is None or metric_ratio <= bg_guard_target_p90 * (1.0 + 1e-3):
                    break
            if bg_guard_applied and bg_var_ratio_post is None:
                bg_var_ratio_post, _, _ = _bg_var_ratio(pd_avg, pd_base_full, mask_bg)
            if bg_guard_applied and bg_var_ratio_post is not None:
                for _ in range(20):
                    if bg_var_ratio_post <= bg_guard_target_p90 * (1.0 + 1e-3):
                        break
                    gamma = float(np.sqrt(bg_guard_target_p90 / max(bg_var_ratio_post, 1e-12)))
                    gamma = float(min(1.0, max(0.0, gamma)))
                    pd_bg_new = pd_base_full[mask_bg] + gamma * (
                        pd_avg[mask_bg] - pd_base_full[mask_bg]
                    )
                    pd_avg[mask_bg] = pd_bg_new.astype(pd_avg.dtype, copy=False)
                    bg_guard_alpha_used = gamma
                    bg_var_ratio_post, _, _ = _bg_var_ratio(pd_avg, pd_base_full, mask_bg)
        else:
            bg_var_ratio_post = bg_var_ratio_pre
    else:
        bg_var_ratio_post = bg_var_ratio_pre

    info_out["bg_guard_enabled"] = bool(bg_guard_enabled)
    info_out["bg_guard_metric"] = bg_guard_metric
    info_out["bg_guard_applied"] = bool(bg_guard_applied)
    info_out["bg_guard_alpha"] = (
        float(bg_guard_alpha_used) if bg_guard_alpha_used is not None else None
    )
    info_out["bg_var_ratio_pre"] = bg_var_ratio_pre
    info_out["bg_var_ratio_post"] = bg_var_ratio_post
    info_out["bg_guard_target_p90"] = bg_guard_target_p90
    info_out["bg_guard_min_alpha"] = bg_guard_min_alpha
    info_out["directional_strict"] = bool(info_out.get("ka_directional_strict", False))

    # Invariance diagnostics: check that background PD equals baseline PD under the
    # same mask used for tile uniformization. Also report where differences concentrate.
    bg_edge_clamp_global = 0
    if (
        pd_base_full is not None
        and mask_bg is not None
        and pd_base_full.shape == pd_avg.shape
        and mask_bg.shape == pd_avg.shape
    ):
        edge_mask_global = mask_bg & (counts <= 1.0000001)
        if np.any(edge_mask_global):
            pd_avg = pd_avg.copy()
            pd_avg[edge_mask_global] = pd_base_full[edge_mask_global]
            bg_edge_clamp_global = int(np.count_nonzero(edge_mask_global))
        # Always clamp a 1-pixel border of the background mask to enforce invariance
        edge_margin = 1
        if edge_margin > 0:
            border_mask = np.zeros_like(mask_bg, dtype=bool)
            border_mask[:edge_margin, :] = True
            border_mask[-edge_margin:, :] = True
            border_mask[:, :edge_margin] = True
            border_mask[:, -edge_margin:] = True
            border_mask &= mask_bg
            if np.any(border_mask):
                if pd_avg.base is pd_base_full.base:
                    pd_avg = np.array(pd_avg, copy=True)
                pd_avg[border_mask] = pd_base_full[border_mask]
                bg_edge_clamp_global += int(np.count_nonzero(border_mask))
    bg_diag_error = None
    try:
        if pd_base_full is not None and mask_bg is not None:
            edge_mask_global = None
            pd_diff = (pd_avg - pd_base_full).astype(np.float64, copy=False)
            bg_mask = mask_bg.astype(bool, copy=False)
            if pd_diff.shape == bg_mask.shape and np.any(bg_mask):
                edge_mask_global = bg_mask & (counts <= 1.0000001)
                if edge_mask_global is not None and np.any(edge_mask_global):
                    mism = edge_mask_global & (np.abs(pd_diff) > 1e-9)
                    if np.any(mism):
                        pd_avg = pd_avg.copy()
                        pd_avg[mism] = pd_base_full[mism]
                        pd_diff = (pd_avg - pd_base_full).astype(np.float64, copy=False)
                        bg_edge_clamp_global = int(np.count_nonzero(mism))
                diff_bg = np.abs(pd_diff[bg_mask])
                info_out["bg_diff_abs_mean"] = float(np.mean(diff_bg))
                info_out["bg_diff_abs_p90"] = float(np.quantile(diff_bg, 0.90))
                info_out["bg_diff_abs_max"] = float(np.max(diff_bg))
                base_bg = pd_base_full[bg_mask].astype(np.float64, copy=False)
                denom = float(np.linalg.norm(base_bg) + 1e-12)
                info_out["bg_diff_l2_rel"] = float(np.linalg.norm(diff_bg) / denom)
                # Top-k outlier diagnostics
                try:
                    k = 20
                    flat_idx = np.argpartition(-diff_bg, min(k, diff_bg.size - 1))[:k]
                    # Map flat indices back to 2D coords
                    ys, xs = np.where(bg_mask)
                    top = []
                    for i in flat_idx:
                        y = int(ys[int(i)])
                        x = int(xs[int(i)])
                        top.append(
                            [
                                y,
                                x,
                                float(pd_diff[y, x]),
                                float(pd_base_full[y, x]),
                                float(pd_avg[y, x]),
                                float(counts[y, x]) if "counts" in locals() else None,
                            ]
                        )
                    # Sort by absolute difference descending for readability
                    top.sort(key=lambda t: abs(t[2]), reverse=True)
                    info_out["bg_diff_top_coords"] = top
                except Exception:
                    pass
                # Overlap counts coverage on background
                if "counts" in locals():
                    cnt_bg = counts[bg_mask]
                    info_out["bg_overlap_count_min"] = (
                        float(np.min(cnt_bg)) if cnt_bg.size else None
                    )
                    info_out["bg_overlap_count_max"] = (
                        float(np.max(cnt_bg)) if cnt_bg.size else None
                    )
                    info_out["bg_overlap_count_mean"] = (
                        float(np.mean(cnt_bg)) if cnt_bg.size else None
                    )
                    info_out["bg_overlap_any_zero"] = (
                        bool(np.any(cnt_bg <= 0)) if cnt_bg.size else False
                    )
                    # Variant: restrict stats to pixels with counts >= 2 to avoid edge artifacts
                    submask = np.zeros_like(bg_mask, dtype=bool)
                    submask[bg_mask] = cnt_bg >= 2
                    info_out["bg_counts_ge2_fraction"] = (
                        float(np.mean(submask)) if submask.size else None
                    )
                    if np.any(submask):
                        v_base = float(np.var(pd_base_full[submask]))
                        v_stap = float(np.var(pd_avg[submask]))
                        info_out["bg_var_ratio_pre_counts_ge2"] = v_stap / max(v_base, 1e-12)

                # Ring vs far-background comparison to localize differences
                def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
                    out = mask.copy()
                    for _ in range(max(1, int(r))):
                        acc = out.copy()
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                if dy == 0 and dx == 0:
                                    continue
                                acc |= np.roll(np.roll(out, dy, axis=0), dx, axis=1)
                        out = acc
                    return out

                if mask_flow is not None and mask_flow.shape == bg_mask.shape:
                    rf = int(max(2, min(pd_avg.shape) // 32))
                    ring = _dilate(mask_flow.astype(bool, copy=False), rf) & ~mask_flow.astype(
                        bool, copy=False
                    )
                    far_bg = bg_mask & ~ring

                    def _safe_var(arr, m):
                        xs = arr[m]
                        return float(np.var(xs)) if xs.size else None

                    info_out["bg_ring_var"] = _safe_var(pd_avg, bg_mask & ring)
                    info_out["bg_ring_var_base"] = _safe_var(pd_base_full, bg_mask & ring)
                    info_out["bg_far_var"] = _safe_var(pd_avg, far_bg)
                    info_out["bg_far_var_base"] = _safe_var(pd_base_full, far_bg)
                    # Ratios (if both sides available)
                    if (
                        info_out["bg_ring_var"] is not None
                        and info_out["bg_ring_var_base"] is not None
                    ):
                        info_out["bg_ring_var_ratio"] = info_out["bg_ring_var"] / max(
                            info_out["bg_ring_var_base"], 1e-12
                        )
                    if (
                        info_out["bg_far_var"] is not None
                        and info_out["bg_far_var_base"] is not None
                    ):
                        info_out["bg_far_var_ratio"] = info_out["bg_far_var"] / max(
                            info_out["bg_far_var_base"], 1e-12
                        )
    except Exception as exc:
        bg_diag_error = repr(exc)

    bg_var_ratio_final: float | None = None
    if (
        mask_bg is not None
        and pd_base_full is not None
        and pd_base_full.shape == pd_avg.shape
        and mask_bg.shape == pd_avg.shape
    ):
        bg_var_ratio_final, _, _ = _bg_var_ratio(pd_avg, pd_base_full, mask_bg)
    info_out["bg_var_ratio_final"] = bg_var_ratio_final

    info_out["bg_edge_clamp_global_count"] = int(bg_edge_clamp_global)
    info_out["stap_tiles_skipped_flow0"] = int(stap_tiles_skipped_flow0)
    if bg_diag_error is not None:
        info_out["bg_diag_error"] = bg_diag_error

    # Runtime breakdown (ms) captured during this call.
    info_out["stap_extract_ms"] = float(1000.0 * t_extract)
    info_out["stap_batch_proc_ms"] = float(1000.0 * t_batch_proc)
    info_out["stap_post_ms"] = float(1000.0 * t_post)
    info_out["stap_total_ms"] = float(1000.0 * (time.perf_counter() - t0_stap))
    info_out["stap_fast_path_used"] = bool(stap_fast_any)
    info_out["stap_fast_attempted"] = bool(
        any(info.get("stap_fast_attempted", False) for info in tile_infos)
    )
    info_out["stap_fast_failed"] = bool(
        any(info.get("stap_fast_failed", False) for info in tile_infos)
    )
    info_out["stap_fast_forced"] = bool(
        any(info.get("stap_fast_forced", False) for info in tile_infos)
    )
    first_err = None
    for info in tile_infos:
        err = info.get("stap_fast_error")
        if err:
            first_err = err
            break
    info_out["stap_fast_error"] = first_err

    if psd_telemetry and psd_freq_cache is not None and bg_psd_accum is not None:
        if bg_psd_count > 0:
            bg_psd_avg = (bg_psd_accum / float(bg_psd_count)).astype(np.float32, copy=False)
            info_out["psd_mt_freqs_hz"] = psd_freq_cache.astype(np.float32, copy=False).tolist()
            info_out["psd_mt_bg_power"] = bg_psd_avg.tolist()

    return pd_avg, score_avg, info_out


def _default_masks(g: SimGeom, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(W) - W / 2.0) * g.dx
    z = np.arange(H) * g.dy
    XX, ZZ = np.meshgrid(x, z, indexing="xy")
    depth_center = 0.6 * (g.dy * H)
    radius = max(0.0015, 0.12 * min(g.dy * H, g.dx * W))
    mask_flow = (XX**2 + (ZZ - depth_center) ** 2) <= radius**2
    if mask_flow.sum() < 16:
        cy = min(int(0.6 * H), H - 1)
        cx = W // 2
        mask_flow[max(cy - 2, 0) : cy + 3, max(cx - 2, 0) : cx + 3] = True
    mask_bg = np.ones((H, W), dtype=bool)
    mask_bg[: min(g.pml_size + 4, H // 4), :] = False
    mask_bg &= ~mask_flow
    if mask_bg.sum() < 64:
        mask_bg = ~mask_flow
    return mask_flow.astype(bool), mask_bg.astype(bool)


def _flow_mask_from_pd(
    pd_map: np.ndarray,
    *,
    percentile: float = 0.995,
    depth_min_frac: float = 0.25,
    depth_max_frac: float = 0.90,
    erode_iters: int = 0,
    dilate_iters: int = 2,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Build a flow mask by thresholding the PD baseline map.

    Parameters
    ----------
    pd_map : (H,W) baseline PD map.
    percentile : float in (0,1)
        Quantile used for the threshold.
    depth_min_frac / depth_max_frac : floats
        Limit the mask to rows within [min,max] * H.
    erode_iters / dilate_iters : int
        Morphological operations (in pixels).
    """
    H, W = pd_map.shape
    q = float(np.clip(percentile, 0.0, 1.0))
    thresh = float(np.quantile(pd_map, q))
    mask = np.zeros_like(pd_map, dtype=bool)
    mask[pd_map >= thresh] = True
    depth_min = int(np.clip(depth_min_frac * H, 0, H))
    depth_max = int(np.clip(depth_max_frac * H, 0, H))
    if depth_max <= depth_min:
        depth_min = 0
        depth_max = H
    mask[:depth_min, :] = False
    mask[depth_max:, :] = False
    if erode_iters > 0:
        mask = binary_erosion(mask, iterations=int(erode_iters))
    if dilate_iters > 0:
        mask = binary_dilation(mask, iterations=int(dilate_iters))
    coverage = float(mask.mean())
    meta = {
        "pd_quantile": q,
        "threshold": thresh,
        "coverage_fraction": coverage,
        "depth_min_frac": float(depth_min_frac),
        "depth_max_frac": float(depth_max_frac),
        "erode_iters": int(max(erode_iters, 0)),
        "dilate_iters": int(max(dilate_iters, 0)),
    }
    return mask.astype(bool, copy=False), meta


def _resolve_flow_mask(
    pd_map: np.ndarray,
    mask_flow_default: np.ndarray,
    mask_bg_default: np.ndarray,
    *,
    mode: str = "default",
    pd_quantile: float = 0.995,
    depth_min_frac: float = 0.25,
    depth_max_frac: float = 0.90,
    erode_iters: int = 0,
    dilate_iters: int = 2,
    min_pixels: int = 64,
    min_coverage_frac: float = 0.0,
    union_with_default: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """
    Resolve the flow/background masks according to the requested strategy.
    """
    mask_flow = mask_flow_default.copy()
    mask_bg = mask_bg_default.copy()
    stats: dict[str, float] = {
        "mode": mode,
        "coverage_default": float(mask_flow_default.mean()),
    }
    mode_norm = (mode or "default").strip().lower()
    if mode_norm != "pd_auto":
        stats["coverage_pre_union"] = float(mask_flow.mean())
        stats["coverage_post_union"] = float(mask_flow.mean())
        stats["union_applied"] = 0.0
        stats["pd_auto_used"] = 0.0
        return mask_flow, mask_bg, stats

    min_pixels = max(int(min_pixels), 1)
    min_cov = max(float(min_coverage_frac), 0.0)
    try:
        mask_pd, stats_pd = _flow_mask_from_pd(
            pd_map,
            percentile=pd_quantile,
            depth_min_frac=depth_min_frac,
            depth_max_frac=depth_max_frac,
            erode_iters=erode_iters,
            dilate_iters=dilate_iters,
        )
        stats.update(stats_pd)
        coverage_pre = float(mask_pd.mean())
        stats["coverage_pre_union"] = coverage_pre
        valid_pixels = int(mask_pd.sum())
        use_pd = valid_pixels >= min_pixels and coverage_pre >= min_cov
        stats["pd_auto_used"] = 1.0 if use_pd else 0.0
        if not use_pd:
            stats["pd_auto_failed"] = 1.0
            if valid_pixels < min_pixels:
                stats["pd_auto_reason"] = "min_pixels"
            elif coverage_pre < min_cov:
                stats["pd_auto_reason"] = "min_coverage"
            else:
                stats["pd_auto_reason"] = "unknown"
            stats["coverage_post_union"] = float(mask_flow.mean())
            stats["union_applied"] = 0.0
            return mask_flow, mask_bg, stats

        if union_with_default:
            mask_flow = mask_pd | mask_flow_default
        else:
            mask_flow = mask_pd
        stats["union_applied"] = 1.0 if union_with_default else 0.0
        stats["coverage_post_union"] = float(mask_flow.mean())
    except Exception as exc:  # pragma: no cover - defensive
        stats["pd_auto_failed"] = 1.0
        stats["union_applied"] = 0.0
        stats["pd_auto_reason"] = "exception"
        stats["error"] = str(exc)[:200]
        stats["coverage_post_union"] = float(mask_flow.mean())
        return mask_flow, mask_bg, stats

    mask_bg = mask_bg_default & (~mask_flow)
    if mask_bg.sum() < 64:
        mask_bg = (~mask_flow).copy()
    return mask_flow.astype(bool, copy=False), mask_bg.astype(bool, copy=False), stats


def _pool_scores(
    pd_map: np.ndarray,
    mask_flow: np.ndarray,
    mask_bg: np.ndarray,
    n_pos: int,
    n_neg: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos_vals = pd_map[mask_flow]
    neg_vals = pd_map[mask_bg]
    if pos_vals.size == 0 or neg_vals.size == 0:
        raise ValueError("Masks produced empty score pools.")
    pos = rng.choice(pos_vals, size=min(n_pos, pos_vals.size), replace=pos_vals.size < n_pos)
    neg = rng.choice(neg_vals, size=min(n_neg, neg_vals.size), replace=neg_vals.size < n_neg)
    return pos.astype(np.float32), neg.astype(np.float32)


def _split_confirm2_pairs(scores: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    flat = np.asarray(scores, dtype=np.float64).ravel()
    if flat.size < 2:
        raise ValueError("Need at least two scores to form Confirm-2 pairs")
    idx = rng.permutation(flat.size)
    half = (flat.size // 2) * 2
    idx = idx[:half]
    s1 = flat[idx[0::2]].astype(np.float32)
    s2 = flat[idx[1::2]].astype(np.float32)
    return s1, s2


def write_acceptance_bundle(
    out_root: Path,
    g: SimGeom,
    angle_sets: Sequence[Sequence[AngleData]],
    pulses_per_set: int,
    prf_hz: float,
    seed: int,
    tile_hw: tuple[int, int] = (8, 8),
    tile_stride: int = 4,
    Lt: int = 4,
    diag_load: float = 1e-2,
    cov_estimator: str = "huber",
    huber_c: float = 5.0,
    mvdr_load_mode: str = "auto",
    mvdr_auto_kappa: float = 50.0,
    constraint_ridge: float = 0.10,
    fd_span_mode: str = "psd",
    fd_span_rel: tuple[float, float] = (0.30, 1.10),
    fd_fixed_span_hz: float | None = None,
    constraint_mode: str = "exp+deriv",
    grid_step_rel: float = 0.05,
    fd_min_pts: int = 3,
    fd_max_pts: int = 11,
    fd_min_abs_hz: float = 0.0,
    msd_lambda: float | None = None,
    msd_ridge: float = 0.10,
    msd_agg_mode: str = "trim10",
    msd_ratio_rho: float = 0.0,
    motion_half_span_rel: float | None = None,
    msd_contrast_alpha: float | None = None,
    alias_psd_select_enable: bool = False,
    alias_psd_select_ratio_thresh: float = 1.2,
    alias_psd_select_bins: int = 1,
    ka_mode: str = "none",
    ka_prior_path: str | None = None,
    ka_beta_bounds: tuple[float, float] = (0.05, 0.50),
    ka_kappa: float = 40.0,
    ka_alpha: float | None = None,
    # Optional directional KA controls (default: disabled)
    ka_directional_beta: bool = False,
    ka_target_retain_f: float | None = None,
    ka_target_shrink_perp: float | None = None,
    ka_equalize_pf_trace: bool = False,
    ka_ridge_split: bool = False,
    ka_lambda_override_split: float | None = None,
    ka_opts_extra: dict[str, float] | None = None,
    stap_debug_samples: int = 0,
    stap_debug_tile_coords: Sequence[tuple[int, int]] | None = None,
    tile_debug_limit: int | None = None,
    stap_device: str | None = None,
    dataset_name: str | None = None,
    dataset_suffix: str | None = None,
    # Optional slow-time windowing. When set, we first synthesize/inject the
    # full slow-time cube (length = pulses_per_set * len(angle_sets)), then
    # slice frames [slow_time_offset, slow_time_offset + slow_time_length)
    # before running baseline + STAP. This enables disjoint-window ablations
    # while preserving the underlying simulated clip distribution.
    slow_time_offset: int | None = None,
    slow_time_length: int | None = None,
    meta_extra: dict | None = None,
    score_mode: str = "msd",
    flow_mask_mode: str = "default",
    flow_mask_pd_quantile: float = 0.995,
    flow_mask_depth_min_frac: float = 0.25,
    flow_mask_depth_max_frac: float = 0.90,
    flow_mask_erode_iters: int = 0,
    flow_mask_dilate_iters: int = 2,
    flow_mask_min_pixels: int = 64,
    flow_mask_min_coverage_fraction: float = 0.0,
    flow_mask_union_default: bool = True,
    flow_mask_suppress_alias_depth: bool = False,
    # Conditional STAP (compute gating) controls. When enabled (default),
    # tiles with zero overlap with the chosen conditional flow mask are
    # skipped and fall back to the baseline PD map. By default, the
    # conditional mask is the same `mask_flow.npy` written in the bundle,
    # but Phase-2 leakage ablations may override it (e.g. disjoint-window
    # or random masks) while keeping evaluation masks fixed.
    stap_conditional_enable: bool = True,
    stap_conditional_flow_mask: np.ndarray | None = None,
    stap_conditional_bg_mask: np.ndarray | None = None,
    stap_conditional_mask_tag: str | None = None,
    baseline_type: str = "svd",
    reg_enable: bool = False,
    reg_method: str = "phasecorr",
    reg_subpixel: int = 4,
    reg_reference: str = "median",
    svd_rank: int | None = None,
    svd_energy_frac: float | None = None,
    rpca_enable: bool = False,
    rpca_lambda: float | None = None,
    rpca_max_iters: int = 250,
    flow_alias_hz: float | None = None,
    flow_alias_fraction: float = 1.0,
    flow_alias_depth_min_frac: float | None = None,
    flow_alias_depth_max_frac: float | None = 0.4,
    flow_alias_jitter_hz: float = 0.0,
    flow_doppler_min_hz: float | None = None,
    flow_doppler_max_hz: float | None = None,
    bg_alias_hz: float | None = None,
    bg_alias_fraction: float = 0.3,
    bg_alias_depth_min_frac: float | None = None,
    bg_alias_depth_max_frac: float | None = None,
    bg_alias_jitter_hz: float = 0.0,
    vibration_hz: float | None = None,
    vibration_amp: float = 0.0,
    vibration_depth_min_frac: float = 0.15,
    vibration_depth_decay_frac: float = 0.25,
    aperture_phase_std: float | None = None,
    aperture_phase_corr_len: float | None = None,
    aperture_phase_seed: int | None = None,
    clutter_beta: float | None = None,
    clutter_snr_db: float | None = None,
    clutter_depth_min_frac: float = 0.25,
    clutter_depth_max_frac: float = 0.9,
    psd_telemetry: bool = False,
    psd_tapers: int = 3,
    psd_bandwidth: float = 2.0,
    band_ratio_mode: str = "legacy",
    band_ratio_flow_low_hz: float = 120.0,
    band_ratio_flow_high_hz: float = 400.0,
    band_ratio_alias_center_hz: float = 900.0,
    band_ratio_alias_width_hz: float = 15.625,
    feasibility_mode: FeasibilityMode = "legacy",
    # Optional Macé-aligned vessel fields (Phase B): when provided, these
    # arrays encode precomputed microvascular and alias vessel centerlines
    # on the simulated (H,W) grid and are used to inject narrowband Pf/Pa
    # modulations in slow time instead of coarse per-pixel Doppler tones.
    micro_vessels: np.ndarray | None = None,
    alias_vessels: np.ndarray | None = None,
    # Optional amplitude scaling knobs for synthetic components.
    flow_amp_scale: float | None = None,
    alias_amp_scale: float | None = None,
    # Optional depth-dependent amplitude profile for skull/OR regimes.
    depth_amp_profile: str | None = None,
    depth_amp_min_factor: float = 0.3,
    # Optional depth-varying flow amplitude profile (applied on flow mask).
    flow_amp_shallow_scale: float = 1.0,
    flow_amp_mid_scale: float = 1.0,
    flow_amp_deep_scale: float = 1.0,
    flow_depth_mid_frac: float = 0.45,
    flow_depth_deep_frac: float = 0.75,
    # Optional skull slab phase screen and shallow guided-wave contaminant.
    skull_phase_std: float = 0.0,
    skull_phase_corr_lat: float = 6.0,
    skull_phase_corr_depth_frac: float = 0.08,
    skull_depth_max_frac: float = 0.35,
    skull_guided_hz: float = 0.0,
    skull_guided_amp: float = 0.0,
    skull_guided_depth_max_frac: float = 0.20,
    skull_guided_lat_corr: float = 15.0,
    # Optional residual rigid motion (pixels) applied frame-wise after synthesis.
    resid_shift_std_px: float = 0.0,
    # Optional slow hemodynamic modulation of flow amplitude.
    hemo_mod_amp: float = 0.0,
    hemo_mod_breath_period: float = 3.0,
    hemo_mod_card_period: float = 0.8,
    # HOSVD baseline options
    hosvd_spatial_downsample: int = 1,
    hosvd_t_sub: int | None = None,
    hosvd_ranks: tuple[int, int, int] | None = None,
    hosvd_energy_fracs: tuple[float, float, float] | None = None,
    hosvd_max_iters: int = 1,
    # Optional default flow/background masks used as a geometric prior for
    # synthetic injections (flow Doppler, alias, clutter). When provided,
    # these masks replace the internal circular _default_masks and are
    # respected by PD-based flow mask refinement.
    flow_mask_default: np.ndarray | None = None,
    bg_mask_default: np.ndarray | None = None,
) -> dict[str, str]:
    if not angle_sets:
        raise ValueError("angle_sets is empty; nothing to process.")

    feas_mode = _normalize_feasibility_mode(feasibility_mode)
    stap_device_resolved = _resolve_stap_device(stap_device)
    tile_batch_size = 192 if stap_device_resolved.lower().startswith("cuda") else 1
    # Allow an optional override of the tile batch size for STAP via an
    # environment variable. This is used for latency experiments and tuning
    # only; the default of 192 was chosen to balance memory and throughput.
    env_tile_batch = os.getenv("STAP_TILE_BATCH", "").strip()
    if env_tile_batch:
        try:
            tb = int(env_tile_batch)
            if tb > 0:
                tile_batch_size = tb
        except ValueError:
            pass
    fd_mode = fd_span_mode.lower()

    band_ratio_mode_norm = (band_ratio_mode or "legacy").strip().lower()
    if band_ratio_mode_norm not in {"legacy", "whitened"}:
        raise ValueError(
            f"Unsupported band_ratio_mode '{band_ratio_mode}'. Expected 'legacy' or 'whitened'."
        )
    use_whitened_ratio = band_ratio_mode_norm == "whitened"
    band_ratio_spec: dict[str, float]
    if feas_mode == "updated":
        band_ratio_spec = _auto_band_ratio_spec(
            prf_hz,
            Lt,
            flow_alias_hz=flow_alias_hz,
            flow_alias_fraction=flow_alias_fraction,
        )
        band_ratio_spec["alias_min_bins"] = 2
        band_ratio_spec["alias_max_bins"] = 4
    else:
        band_ratio_spec = {
            "flow_low_hz": float(band_ratio_flow_low_hz),
            "flow_high_hz": float(band_ratio_flow_high_hz),
            "alias_center_hz": float(band_ratio_alias_center_hz),
            "alias_width_hz": float(band_ratio_alias_width_hz),
        }

    ka_mode_norm = ka_mode.lower().strip()
    ka_active = ka_mode_norm not in {"", "none"}
    ka_prior_library: np.ndarray | None = None
    if ka_active and ka_mode_norm == "library":
        if not ka_prior_path:
            raise ValueError("KA mode 'library' requires --ka-prior-path.")
        ka_prior_library = np.load(ka_prior_path)
    ka_opts_dict: dict[str, float | str] = {
        "beta_bounds": ka_beta_bounds,
        "kappa_target": float(ka_kappa),
    }
    string_passthrough = {"bg_guard_metric", "score_model_json"}
    if ka_opts_extra:
        for key, value in ka_opts_extra.items():
            if key in string_passthrough:
                ka_opts_dict[key] = str(value)
            else:
                ka_opts_dict[key] = float(value)
    if ka_alpha is not None:
        ka_opts_dict["alpha"] = float(ka_alpha)
    # Optional directional / ridge-split controls
    if ka_directional_beta:
        ka_opts_dict["beta_directional"] = True
        ka_opts_dict.setdefault("beta_directional_strict", True)
    if ka_target_retain_f is not None:
        val = float(ka_target_retain_f)
        if val <= 0.0:
            val = 0.8
        if val >= 0.999:
            val = 0.999
        ka_opts_dict["target_retain_f"] = val
    if ka_target_shrink_perp is not None:
        val = float(ka_target_shrink_perp)
        if val <= 0.0:
            val = 0.5
        if val >= 0.995:
            val = 0.995
        ka_opts_dict["target_shrink_perp"] = val
    if ka_equalize_pf_trace:
        ka_opts_dict["equalize_pf_trace"] = True
    if ka_ridge_split:
        ka_opts_dict["ridge_split"] = True
    if ka_lambda_override_split is not None:
        ka_opts_dict["lambda_override_split"] = float(ka_lambda_override_split)

    XX, ZZ, d_rx = _precompute_geometry(g)
    image_sets: list[np.ndarray] = []
    dt_sets: list[list[float]] = []
    angle_values: list[list[float]] = []

    phase_std_val = float(aperture_phase_std) if aperture_phase_std is not None else 0.0
    phase_corr_val = float(aperture_phase_corr_len or 0.0)
    phase_seed_base = int(aperture_phase_seed or 0)
    phase_counter = 0
    for angles in angle_sets:
        if not angles:
            continue
        imgs = []
        dt_list: list[float] = []
        ang_list: list[float] = []
        for ad in angles:
            iq = _demod_iq(ad.rf, ad.dt, g.f0)
            if phase_std_val > 0.0:
                phase_seed = phase_seed_base + 7919 * seed + 131 * phase_counter
                phase_vec = _sample_phase_screen_vector(
                    iq.shape[1], phase_std_val, phase_corr_val, phase_seed
                )
                if phase_vec is not None:
                    phase_weights = np.exp(1j * phase_vec).astype(np.complex64)
                    iq = iq * phase_weights[None, :]
            img = _beamform_angle(iq, ad.angle_deg, ad.dt, g, XX, ZZ, d_rx)
            imgs.append(img)
            dt_list.append(float(ad.dt))
            ang_list.append(float(ad.angle_deg))
            phase_counter += 1
        if imgs:
            image_sets.append(np.stack(imgs, axis=0))
            dt_sets.append(dt_list)
            angle_values.append(ang_list)

    if not image_sets:
        raise ValueError("No beamformed images produced from angle_sets.")

    Icube = _synthesize_cube(
        image_sets,
        pulses_per_set=pulses_per_set,
        seed=seed,
    ).astype(np.complex64, copy=False)

    # Optional depth-dependent amplitude profile to mimic skull/OR SNR and
    # attenuation. This scales the complex cube as a function of depth only
    # and is primarily used in HAB skull/OR variants; when disabled the
    # profile is effectively flat.
    T, H, W = Icube.shape
    if depth_amp_profile:
        profile = depth_amp_profile.strip().lower()
    else:
        profile = ""

    def _depth_amp_skull_or_v1(z_frac: float, min_factor: float) -> float:
        # z_frac in [0,1]; piecewise profile: attenuated near surface, peak in
        # cortical band, gentle decay in parenchyma, stronger decay in deep BG.
        if z_frac < 0.12:
            return 0.3
        if z_frac < 0.30:
            t = (z_frac - 0.12) / (0.30 - 0.12)
            return 0.3 + 0.7 * t
        if z_frac < 0.85:
            t = (z_frac - 0.30) / (0.85 - 0.30)
            return 1.0 - 0.5 * t
        t = (z_frac - 0.85) / max(1e-6, 1.0 - 0.85)
        return 0.5 - (0.5 - min_factor) * t

    if profile in {"skull_or_v1"}:
        yy = np.linspace(0.0, 1.0, H, endpoint=True, dtype=np.float32)
        min_fac = float(depth_amp_min_factor)
        min_fac = float(np.clip(min_fac, 0.1, 1.0))
        gains = np.array([_depth_amp_skull_or_v1(float(y), min_fac) for y in yy], dtype=np.float32)
        Icube *= gains.reshape(1, H, 1).astype(Icube.dtype, copy=False)

    # Optional 2D skull slab phase screen applied on the beamformed cube.
    skull_phase_std_val = float(skull_phase_std)
    if skull_phase_std_val > 0.0 and 0 < skull_depth_max_frac <= 1.0:
        Ny_local, Nx_local = H, W
        rng_skull = np.random.default_rng(seed + 501)
        z = np.arange(Ny_local, dtype=np.float32)[:, None]
        x = np.arange(Nx_local, dtype=np.float32)[None, :]
        # Build separable Gaussian kernels in depth and lateral directions.
        depth_corr = max(float(skull_phase_corr_depth_frac) * Ny_local, 1.0)
        lat_corr = max(float(skull_phase_corr_lat), 1.0)
        kz = np.exp(-0.5 * (z / depth_corr) ** 2).astype(np.float32)
        kx = np.exp(-0.5 * (x / lat_corr) ** 2).astype(np.float32)
        kz /= float(np.sum(kz))
        kx /= float(np.sum(kx))
        white = rng_skull.standard_normal((Ny_local, Nx_local)).astype(np.float32)
        # Smooth along lateral then depth.
        tmp = convolve1d(white, kx.ravel(), axis=1, mode="wrap")
        phi = convolve1d(tmp, kz.ravel(), axis=0, mode="nearest")
        phi -= float(np.mean(phi))
        phi /= float(np.std(phi) + 1e-6)
        phi *= skull_phase_std_val
        # Taper with depth to confine to shallow layer.
        z_frac = (np.arange(Ny_local, dtype=np.float32) + 0.5) / max(float(Ny_local), 1.0)
        depth_mask = np.exp(-((z_frac / float(skull_depth_max_frac)) ** 2), dtype=np.float32)[
            :, None
        ]
        phi *= depth_mask
        Icube *= np.exp(1j * phi[None, :, :]).astype(Icube.dtype, copy=False)

    alias_meta: dict[str, float | int] | None = None
    bg_alias_meta: dict[str, float | int] | None = None
    vibration_meta: dict[str, float] | None = None
    phase_meta: dict[str, float | int] | None = None
    clutter_meta: dict[str, float | int] | None = None
    flow_doppler_meta: dict[str, float | int] | None = None
    T, H, W = Icube.shape
    # Optional Pf/Pa vessel injections (MaceBridge Phase B). When Macé-aligned
    # vessel fields are provided, apply narrowband slow-time tones along the
    # corresponding centerlines before clutter and coarse synthetic Doppler.
    if (micro_vessels is not None and micro_vessels.size) or (
        alias_vessels is not None and alias_vessels.size
    ):
        Icube, vessel_meta = _inject_vessels_slowtime(
            Icube,
            micro_vessels=micro_vessels,
            alias_vessels=alias_vessels,
            prf_hz=prf_hz,
            f0_hz=g.f0,
            c0=g.c0,
        )
    clutter_beta_val = float(clutter_beta) if clutter_beta is not None else 0.0
    needs_masks = (
        (flow_alias_hz is not None and abs(flow_alias_hz) > 0.0)
        or (bg_alias_hz is not None and abs(bg_alias_hz) > 0.0)
        or clutter_beta_val > 0.0
        or (flow_doppler_min_hz is not None and flow_doppler_max_hz is not None)
    )
    default_flow_mask: np.ndarray | None = None
    default_bg_mask: np.ndarray | None = None
    if needs_masks:
        # Prefer explicit geometric defaults (e.g., Macé-derived H1/H0
        # masks) when provided; otherwise fall back to a simple circular
        # mask from _default_masks.
        if flow_mask_default is not None and bg_mask_default is not None:
            default_flow_mask = np.asarray(flow_mask_default, dtype=bool)
            default_bg_mask = np.asarray(bg_mask_default, dtype=bool)
        else:
            default_flow_mask, default_bg_mask = _default_masks(g, H, W)
    # Optional: scale the complex flow amplitude inside the default flow mask before
    # any synthetic Doppler, alias, clutter, or vibration injections. This provides
    # a simple knob to adjust flow SNR for regimes like HAB without altering the
    # underlying geometric masks.
    if flow_amp_scale is not None and default_flow_mask is not None:
        scale = float(flow_amp_scale)
        if not math.isclose(scale, 1.0):
            flow_mask_amp = default_flow_mask.astype(bool)
            if flow_mask_amp.any():
                cube_flat = Icube.reshape(T, -1)
                idx_flow_amp = flow_mask_amp.ravel().nonzero()[0]
                cube_flat[:, idx_flow_amp] *= scale
                Icube = cube_flat.reshape(Icube.shape)
    # Optional depth-varying flow amplitude profile: refine SNR for skull/OR
    # regimes by boosting superficial flow and gently attenuating deep flow
    # inside the geometric flow mask.
    if default_flow_mask is not None and (
        not math.isclose(flow_amp_shallow_scale, 1.0)
        or not math.isclose(flow_amp_mid_scale, 1.0)
        or not math.isclose(flow_amp_deep_scale, 1.0)
    ):
        z_frac = (np.arange(H, dtype=np.float32) + 0.5) / max(float(H), 1.0)
        mid_frac = float(flow_depth_mid_frac)
        deep_frac = float(flow_depth_deep_frac)
        mid_frac = float(np.clip(mid_frac, 0.0, 1.0))
        deep_frac = float(np.clip(deep_frac, mid_frac, 1.0))
        gains_1d = np.empty(H, dtype=np.float32)
        for yi, zf in enumerate(z_frac):
            if zf < mid_frac:
                gains_1d[yi] = float(flow_amp_shallow_scale)
            elif zf < deep_frac:
                gains_1d[yi] = float(flow_amp_mid_scale)
            else:
                gains_1d[yi] = float(flow_amp_deep_scale)
        flow_mask_amp = default_flow_mask.astype(bool)
        if flow_mask_amp.any():
            cube_flat = Icube.reshape(T, -1)
            idx_flow_amp = flow_mask_amp.ravel().nonzero()[0]
            gains_flat = np.repeat(gains_1d, W).astype(Icube.dtype, copy=False)
            cube_flat[:, idx_flow_amp] *= gains_flat[idx_flow_amp]
            Icube = cube_flat.reshape(Icube.shape)
    if (
        flow_doppler_min_hz is not None
        and flow_doppler_max_hz is not None
        and (flow_doppler_min_hz != 0 or flow_doppler_max_hz != 0)
    ):
        if default_flow_mask is None:
            default_flow_mask = mask_flow_default.copy()
        flow_mask_fd = default_flow_mask.copy()
        if flow_mask_depth_min_frac is not None:
            depth_min_fd = int(np.clip(flow_mask_depth_min_frac * H, 0, H))
            flow_mask_fd[:depth_min_fd, :] = False
        if flow_mask_depth_max_frac is not None:
            depth_max_fd = int(np.clip(flow_mask_depth_max_frac * H, 0, H))
            flow_mask_fd[depth_max_fd:, :] = False
        flow_idx = flow_mask_fd.ravel().nonzero()[0]
        if flow_idx.size:
            f_lo = float(flow_doppler_min_hz)
            f_hi = float(flow_doppler_max_hz)
            if f_hi <= f_lo:
                f_hi = f_lo
            rng_flow = np.random.default_rng(seed + 123)
            freqs = np.zeros(H * W, dtype=np.float32)
            # By default draw a mixture of flow Dopplers so that most energy
            # lies in Pf, but a minority leaks toward the guard band and very
            # low frequencies. This better matches heterogeneous microvascular
            # flows in skull/OR regimes while remaining compatible with the
            # original HAB design when f_lo/f_hi sit in [40,220] Hz.
            u = rng_flow.random(size=flow_idx.size)
            f_draw = np.empty(flow_idx.size, dtype=np.float32)
            # 70%: main microvascular band
            mask1 = u < 0.7
            if np.any(mask1):
                f_draw[mask1] = rng_flow.uniform(
                    max(f_lo, 40.0), min(f_hi, 200.0), size=int(np.sum(mask1))
                )
            # 20%: slightly faster components extending toward guard band
            mask2 = (u >= 0.7) & (u < 0.9)
            if np.any(mask2):
                f_draw[mask2] = rng_flow.uniform(
                    max(f_lo, 200.0), min(f_hi, 320.0), size=int(np.sum(mask2))
                )
            # 10%: very slow components near DC
            mask3 = ~(mask1 | mask2)
            if np.any(mask3):
                f_draw[mask3] = rng_flow.uniform(
                    max(f_lo, 10.0), min(f_hi, 40.0), size=int(np.sum(mask3))
                )
            # Optional per-pixel Doppler jitter.
            freqs_eff = f_draw.copy()
            # Doppler jitter parameter (fractional std) can be set via meta_extra
            # or regime presets by adjusting flow_doppler_min/max; for now we
            # keep the jitter implicit to avoid additional signature complexity.
            freqs[flow_idx] = freqs_eff
            t_idx = np.arange(T, dtype=np.float32)
            # Optional slow Doppler drift across the sequence can be modelled
            # by perturbing freqs_eff over time before forming the phase; we
            # keep the default implementation stationary here and rely on the
            # broadened mixture above for realism.
            phase = np.exp(
                1j * 2.0 * np.pi * (freqs[None, :] * t_idx[:, None] / float(prf_hz))
            ).astype(np.complex64, copy=False)
            cube_flat = Icube.reshape(T, -1)
            cube_flat *= phase
            Icube = cube_flat.reshape(Icube.shape)
            flow_doppler_meta = {
                "flow_doppler_min_hz": f_lo,
                "flow_doppler_max_hz": f_hi,
                "flow_doppler_pixels": int(flow_idx.size),
            }
    if clutter_beta_val > 0.0:
        target_snr_db = float(clutter_snr_db) if clutter_snr_db is not None else -6.0
        if default_bg_mask is None:
            default_bg_mask = mask_bg_default.copy()
        Icube, clutter_meta = _inject_temporal_clutter(
            Icube,
            default_bg_mask,
            beta=clutter_beta_val,
            snr_db=target_snr_db,
            depth_min_frac=float(clutter_depth_min_frac),
            depth_max_frac=float(clutter_depth_max_frac),
            seed=seed,
        )
    # Optional shallow guided-wave contaminant representing skull/table-guided
    # modes. Injected as a coherent tone near skull_guided_hz in the shallow
    # depth band, with lateral correlation given by skull_guided_lat_corr.
    if skull_guided_amp > 0.0 and skull_guided_hz > 0.0 and 0 < skull_guided_depth_max_frac <= 1.0:
        Ny_local, Nx_local = H, W
        z_frac = (np.arange(Ny_local, dtype=np.float32) + 0.5) / max(float(Ny_local), 1.0)
        shallow_mask = z_frac <= float(skull_guided_depth_max_frac)
        if np.any(shallow_mask):
            rng_guided = np.random.default_rng(seed + 777)
            lat = rng_guided.standard_normal(Nx_local).astype(np.float32)
            lat_corr = max(float(skull_guided_lat_corr), 1.0)
            kx = np.exp(-0.5 * (np.arange(Nx_local, dtype=np.float32) / lat_corr) ** 2).astype(
                np.float32
            )
            kx /= float(np.sum(kx))
            lat = convolve1d(lat, kx, mode="wrap")
            lat /= float(np.std(lat) + 1e-6)
            t_sec = np.arange(T, dtype=np.float32) / float(prf_hz)
            phase_t = 2.0 * np.pi * float(skull_guided_hz) * t_sec
            tone_t = np.exp(1j * phase_t).astype(np.complex64, copy=False)[:, None, None]
            lat_map = lat.reshape(1, 1, Nx_local).astype(np.complex64, copy=False)
            guided = float(skull_guided_amp) * tone_t * lat_map
            Icube[:, shallow_mask, :] += guided[:, : np.count_nonzero(shallow_mask), :]
    # Optional small residual rigid motion applied frame-wise after synthesis.
    # This models incomplete cancellation of bulk motion (e.g., after a prior
    # registration step) and introduces realistic frame-to-frame jitter.
    resid_std = float(resid_shift_std_px)
    if resid_std > 0.0:
        rng_resid = np.random.default_rng(seed + 888)
        for t_idx in range(T):
            dy = float(rng_resid.normal(scale=resid_std))
            dx = float(rng_resid.normal(scale=resid_std))
            if abs(dy) > 1e-6 or abs(dx) > 1e-6:
                Icube[t_idx] = _fft_shift_apply(Icube[t_idx], dy, dx)
    # Optional slow hemodynamic modulation of flow amplitude over time. This
    # models low-frequency vascular dynamics (respiration/cardiac coupled) on
    # top of the injected flow Doppler. It is applied only on the default
    # flow mask and is disabled when hemo_mod_amp <= 0.
    hemo_amp = float(hemo_mod_amp)
    if hemo_amp > 0.0 and default_flow_mask is not None and float(prf_hz) > 0.0:
        t_sec = np.arange(T, dtype=np.float32) / float(prf_hz)
        breath_T = max(float(hemo_mod_breath_period), 1e-3)
        card_T = max(float(hemo_mod_card_period), 1e-3)
        mod = (
            1.0
            + 0.5 * hemo_amp * np.sin(2.0 * np.pi * t_sec / breath_T)
            + 0.3 * hemo_amp * np.sin(2.0 * np.pi * t_sec / card_T)
        ).astype(np.float32)
        mod /= float(np.sqrt(np.mean(mod**2)) + 1e-6)
        cube_flat = Icube.reshape(T, -1)
        idx_flow = default_flow_mask.ravel().nonzero()[0]
        if idx_flow.size:
            cube_flat[:, idx_flow] *= mod[:, None].astype(Icube.dtype, copy=False)
            Icube = cube_flat.reshape(Icube.shape)
    if flow_alias_hz is not None and abs(flow_alias_hz) > 0.0:
        if default_flow_mask is None:
            default_flow_mask = mask_flow_default.copy()
        alias_mask = default_flow_mask.copy()
        if flow_alias_depth_min_frac is not None:
            depth_min = int(np.clip(flow_alias_depth_min_frac * H, 0, H))
            alias_mask[:depth_min, :] = False
        if flow_alias_depth_max_frac is not None:
            depth_max = int(np.clip(flow_alias_depth_max_frac * H, 0, H))
            alias_mask[depth_max:, :] = False
        alias_idx = alias_mask.ravel().nonzero()[0]
        if alias_idx.size:
            frac = float(np.clip(flow_alias_fraction, 0.0, 1.0))
            if 0.0 < frac < 0.999:
                rng_alias = np.random.default_rng(seed + 911)
                k = max(1, int(frac * alias_idx.size))
                select = rng_alias.choice(alias_idx.size, size=k, replace=False)
                mask = np.zeros_like(alias_mask.ravel(), dtype=bool)
                mask[alias_idx[select]] = True
                alias_mask = mask.reshape(alias_mask.shape)
                alias_idx = alias_mask.ravel().nonzero()[0]
            alias_freq = float(flow_alias_hz)
            jitter = float(flow_alias_jitter_hz or 0.0)
            if jitter > 0.0:
                rng_alias = np.random.default_rng(seed + 911)
                alias_freq += float(rng_alias.uniform(-jitter, jitter))
            phase = np.exp(
                1j
                * 2.0
                * np.pi
                * alias_freq
                / float(prf_hz)
                * np.arange(T, dtype=np.float32)[:, None]
            ).astype(np.complex64, copy=False)
            cube_flat = Icube.reshape(T, -1)
            cube_flat[:, alias_idx] = cube_flat[:, alias_idx] * phase
            # Optional alias amplitude scaling: enlarge or shrink the aliased
            # component relative to the underlying cube.
            if alias_amp_scale is not None and not math.isclose(alias_amp_scale, 1.0):
                cube_flat[:, alias_idx] *= float(alias_amp_scale)
            Icube = cube_flat.reshape(Icube.shape)
            alias_meta = {
                "flow_alias_hz": float(alias_freq),
                "flow_alias_fraction": float(np.clip(flow_alias_fraction, 0.0, 1.0)),
                "flow_alias_pixels": int(alias_idx.size),
            }
            if jitter > 0.0:
                alias_meta["flow_alias_jitter_hz"] = float(jitter)
            if flow_alias_depth_max_frac is not None:
                alias_meta["flow_alias_depth_max_frac"] = float(flow_alias_depth_max_frac)
            if flow_alias_depth_min_frac is not None:
                alias_meta["flow_alias_depth_min_frac"] = float(flow_alias_depth_min_frac)
    if bg_alias_hz is not None and abs(bg_alias_hz) > 0.0:
        if default_bg_mask is None:
            default_bg_mask = mask_bg_default.copy()
        alias_mask_bg = default_bg_mask.copy()
        if bg_alias_depth_min_frac is not None:
            depth_min_bg = int(np.clip(bg_alias_depth_min_frac * H, 0, H))
            alias_mask_bg[:depth_min_bg, :] = False
        if bg_alias_depth_max_frac is not None:
            depth_max_bg = int(np.clip(bg_alias_depth_max_frac * H, 0, H))
            alias_mask_bg[depth_max_bg:, :] = False
        alias_idx_bg = alias_mask_bg.ravel().nonzero()[0]
        if alias_idx_bg.size:
            frac_bg = float(np.clip(bg_alias_fraction, 0.0, 1.0))
            if 0.0 < frac_bg < 0.999:
                rng_bg = np.random.default_rng(seed + 917)
                k_bg = max(1, int(frac_bg * alias_idx_bg.size))
                select_bg = rng_bg.choice(alias_idx_bg.size, size=k_bg, replace=False)
                mask_flat_bg = np.zeros_like(alias_mask_bg.ravel(), dtype=bool)
                mask_flat_bg[alias_idx_bg[select_bg]] = True
                alias_mask_bg = mask_flat_bg.reshape(alias_mask_bg.shape)
                alias_idx_bg = alias_mask_bg.ravel().nonzero()[0]
            alias_freq_bg = float(bg_alias_hz)
            jitter_bg = float(bg_alias_jitter_hz or 0.0)
            if jitter_bg > 0.0:
                rng_bg = np.random.default_rng(seed + 917)
                alias_freq_bg += float(rng_bg.uniform(-jitter_bg, jitter_bg))
            phase_bg = np.exp(
                1j
                * 2.0
                * np.pi
                * alias_freq_bg
                / float(prf_hz)
                * np.arange(T, dtype=np.float32)[:, None]
            ).astype(np.complex64, copy=False)
            cube_flat = Icube.reshape(T, -1)
            cube_flat[:, alias_idx_bg] = cube_flat[:, alias_idx_bg] * phase_bg
            Icube = cube_flat.reshape(Icube.shape)
            bg_alias_meta = {
                "bg_alias_hz": float(alias_freq_bg),
                "bg_alias_fraction": float(np.clip(bg_alias_fraction, 0.0, 1.0)),
                "bg_alias_pixels": int(alias_idx_bg.size),
            }
            if jitter_bg > 0.0:
                bg_alias_meta["bg_alias_jitter_hz"] = float(jitter_bg)
            if bg_alias_depth_min_frac is not None:
                bg_alias_meta["bg_alias_depth_min_frac"] = float(bg_alias_depth_min_frac)
            if bg_alias_depth_max_frac is not None:
                bg_alias_meta["bg_alias_depth_max_frac"] = float(bg_alias_depth_max_frac)
    if vibration_hz is not None and float(vibration_amp) > 0.0:
        Icube, vibration_meta = _inject_global_vibration(
            Icube,
            prf_hz,
            freq_hz=float(vibration_hz),
            amp=float(vibration_amp),
            depth_min_frac=float(vibration_depth_min_frac),
            depth_decay_frac=float(vibration_depth_decay_frac),
        )
    alias_meta_for_band = alias_meta
    if alias_meta_for_band is None and bg_alias_meta is not None:
        alias_meta_for_band = {
            "flow_alias_hz": float(bg_alias_meta["bg_alias_hz"]),
            "flow_alias_fraction": float(bg_alias_meta["bg_alias_fraction"]),
            "flow_alias_pixels": int(bg_alias_meta["bg_alias_pixels"]),
        }
    band_ratio_spec = _adjust_alias_band_for_meta(band_ratio_spec, alias_meta_for_band)
    if aperture_phase_std is not None and float(aperture_phase_std) > 0.0:
        phase_meta = {
            "phase_std": float(aperture_phase_std),
            "phase_corr_len": float(aperture_phase_corr_len or 0.0),
        }
        if aperture_phase_seed is not None:
            phase_meta["phase_seed"] = int(aperture_phase_seed)
    if clutter_beta is not None and float(clutter_beta) > 0.0:
        clutter_meta = {
            "beta": float(clutter_beta),
            "snr_db": float(clutter_snr_db if clutter_snr_db is not None else -6.0),
            "depth_min_frac": float(clutter_depth_min_frac),
            "depth_max_frac": float(clutter_depth_max_frac),
        }

    # Optional slow-time window slice applied after synthesis + injections.
    # Note: this slices the realized clip (including injected clutter/alias)
    # so disjoint windows share the same underlying simulation.
    if slow_time_length is not None:
        full_T = int(Icube.shape[0])
        offset = int(slow_time_offset or 0)
        length = int(slow_time_length)
        if offset < 0:
            raise ValueError("slow_time_offset must be non-negative")
        if length <= 0:
            raise ValueError("slow_time_length must be positive")
        end = offset + length
        if end > full_T:
            raise ValueError(
                f"slow-time window [{offset}, {end}) exceeds total_frames={full_T}"
            )
        Icube = np.ascontiguousarray(Icube[offset:end], dtype=Icube.dtype)

    baseline_type_norm = (baseline_type or "svd").strip().lower()
    baseline_device = "cuda" if stap_device_resolved.lower().startswith("cuda") else "cpu"
    baseline_telemetry: dict | None = None
    baseline_filtered_cube: np.ndarray | None = None
    if baseline_type_norm == "mc_svd":
        if use_whitened_ratio:
            pd_base, baseline_telemetry, baseline_filtered_cube = _baseline_pd_mcsvd(
                Icube,
                reg_enable=reg_enable,
                reg_method=reg_method,
                reg_subpixel=max(1, int(reg_subpixel)),
                reg_reference=reg_reference,
                svd_rank=svd_rank,
                svd_energy_frac=svd_energy_frac,
                device=baseline_device,
                return_filtered_cube=True,
            )
        else:
            pd_base, baseline_telemetry = _baseline_pd_mcsvd(
                Icube,
                reg_enable=reg_enable,
                reg_method=reg_method,
                reg_subpixel=max(1, int(reg_subpixel)),
                reg_reference=reg_reference,
                svd_rank=svd_rank,
                svd_energy_frac=svd_energy_frac,
                device=baseline_device,
            )
    elif baseline_type_norm == "hosvd":
        if use_whitened_ratio:
            raise ValueError("Whitened band-ratio scoring requires mc_svd baseline.")
        pd_base, baseline_telemetry = _baseline_pd_hosvd(
            Icube,
            ranks=hosvd_ranks,
            energy_fracs=hosvd_energy_fracs,
            max_iters=hosvd_max_iters,
            spatial_downsample=hosvd_spatial_downsample,
            t_sub=hosvd_t_sub,
        )
    elif baseline_type_norm == "rpca" and rpca_enable:
        if use_whitened_ratio:
            raise ValueError("Whitened band-ratio scoring requires mc_svd baseline.")
        pd_base, baseline_telemetry = _baseline_pd_rpca(
            Icube,
            lambda_=rpca_lambda,
            max_iters=int(rpca_max_iters),
        )
    else:
        if use_whitened_ratio:
            raise ValueError("Whitened band-ratio scoring requires mc_svd baseline.")
        hp_modes = int(svd_rank) if svd_rank is not None else 1
        hp_modes = max(1, hp_modes)
        pd_base = _baseline_pd(Icube, hp_modes=hp_modes)
        baseline_telemetry = {
            "baseline_type": "svd",
            "svd_rank_removed": hp_modes,
        }

    H, W = pd_base.shape
    tile_count = _tile_count(pd_base.shape, tile_hw, tile_stride)
    br_spec = BandRatioSpec(
        flow_low_hz=float(band_ratio_spec["flow_low_hz"]),
        flow_high_hz=float(band_ratio_spec["flow_high_hz"]),
        alias_center_hz=float(band_ratio_spec["alias_center_hz"]),
        alias_width_hz=float(band_ratio_spec["alias_width_hz"]),
        alias_min_bins=band_ratio_spec.get("alias_min_bins"),
        alias_max_bins=band_ratio_spec.get("alias_max_bins"),
    )
    base_br_recorder: BandRatioRecorder | None = None
    stap_br_recorder: BandRatioRecorder | None = None
    if use_whitened_ratio:
        base_br_recorder = BandRatioRecorder(
            prf_hz,
            tile_count,
            br_spec,
            tapers=psd_tapers,
            bandwidth=psd_bandwidth,
        )
        stap_br_recorder = BandRatioRecorder(
            prf_hz,
            tile_count,
            br_spec,
            tapers=psd_tapers,
            bandwidth=psd_bandwidth,
        )

    # Default flow/background masks used for synthetic injections and as a
    # geometric prior for PD-based flow mask refinement. When explicit
    # defaults are provided, use them; otherwise fall back to a simple
    # circular mask from _default_masks.
    if flow_mask_default is not None and bg_mask_default is not None:
        mask_flow_default = np.asarray(flow_mask_default, dtype=bool)
        mask_bg_default = np.asarray(bg_mask_default, dtype=bool)
        if mask_flow_default.shape != (H, W) or mask_bg_default.shape != (H, W):
            raise ValueError(
                f"flow_mask_default/bg_mask_default shape mismatch: "
                f"{mask_flow_default.shape}, {mask_bg_default.shape} vs {(H, W)}"
            )
    else:
        mask_flow_default, mask_bg_default = _default_masks(g, H, W)
    flow_mask_mode_norm = (flow_mask_mode or "default").strip().lower()
    mask_flow, mask_bg, flow_mask_stats = _resolve_flow_mask(
        pd_base,
        mask_flow_default,
        mask_bg_default,
        mode=flow_mask_mode_norm,
        pd_quantile=flow_mask_pd_quantile,
        depth_min_frac=flow_mask_depth_min_frac,
        depth_max_frac=flow_mask_depth_max_frac,
        erode_iters=flow_mask_erode_iters,
        dilate_iters=flow_mask_dilate_iters,
        min_pixels=flow_mask_min_pixels,
        min_coverage_frac=flow_mask_min_coverage_fraction,
        union_with_default=flow_mask_union_default,
    )

    # Optional: in alias-injection regimes, treat the background alias depth band
    # as background even if PD-based masking would otherwise promote those pixels
    # into the flow mask. This is primarily used for the short-T pial-alias
    # scenario so that pial alias tiles are evaluated as H0 in ROC / telemetry.
    if flow_mask_suppress_alias_depth and bg_alias_hz is not None:
        if bg_alias_depth_min_frac is not None and bg_alias_depth_max_frac is not None:
            try:
                depth_min_alias = float(bg_alias_depth_min_frac)
                depth_max_alias = float(bg_alias_depth_max_frac)
            except Exception:
                depth_min_alias = 0.0
                depth_max_alias = 0.0
            if depth_max_alias > depth_min_alias and H > 0:
                r0 = int(np.clip(depth_min_alias * H, 0, H))
                r1 = int(np.clip(depth_max_alias * H, 0, H))
                if r1 > r0:
                    mask_flow[r0:r1, :] = False
                    mask_bg[r0:r1, :] = True
                    if flow_mask_stats is not None:
                        flow_mask_stats["suppress_alias_depth"] = 1.0

    # Resolve the conditional-STAP gating masks. These can differ from the
    # evaluation masks (mask_flow/mask_bg) so that we can (i) disable
    # conditional execution (full STAP) or (ii) run leakage ablations where
    # the conditional mask is derived from a disjoint window or randomized
    # while evaluation masks remain fixed.
    cond_enabled = bool(stap_conditional_enable)
    cond_tag = (stap_conditional_mask_tag or "").strip()
    if not cond_enabled:
        mask_flow_cond = None
        mask_bg_cond = None
        cond_source = "disabled"
    else:
        mask_flow_cond = mask_flow
        mask_bg_cond = mask_bg
        cond_source = "eval_mask"
        if stap_conditional_flow_mask is not None:
            mask_flow_cond = np.asarray(stap_conditional_flow_mask, dtype=bool)
            if mask_flow_cond.shape != (H, W):
                raise ValueError(
                    f"stap_conditional_flow_mask shape mismatch: {mask_flow_cond.shape} vs {(H, W)}"
                )
            if stap_conditional_bg_mask is not None:
                mask_bg_cond = np.asarray(stap_conditional_bg_mask, dtype=bool)
                if mask_bg_cond.shape != (H, W):
                    raise ValueError(
                        f"stap_conditional_bg_mask shape mismatch: {mask_bg_cond.shape} vs {(H, W)}"
                    )
            else:
                mask_bg_cond = (~mask_flow_cond).copy()
            cond_source = "override"

    if cond_tag:
        cond_source = f"{cond_source}:{cond_tag}"

    cond_mask_stats: dict[str, float | str] = {
        "enabled": 1.0 if cond_enabled else 0.0,
        "source": cond_source,
    }
    if mask_flow_cond is not None:
        cond_mask_stats["coverage_pixels"] = float(mask_flow_cond.mean())
        cov_cond, _coords_cond = _tile_coverages(mask_flow_cond, tile_hw, tile_stride)
        cond_mask_stats["coverage_tiles_any"] = float(np.mean(cov_cond > 0.0)) if cov_cond.size else 0.0
    else:
        cond_mask_stats["coverage_pixels"] = 0.0
        cond_mask_stats["coverage_tiles_any"] = 0.0

    if use_whitened_ratio and baseline_filtered_cube is not None and base_br_recorder is not None:
        _collect_band_ratio_from_cube(
            baseline_filtered_cube,
            base_br_recorder,
            mask_bg,
            tile_hw,
            tile_stride,
        )
        baseline_filtered_cube = None

    band_ratio_spec_meta = dict(band_ratio_spec)

    t_stap_start = time.time()
    pd_stap, stap_scores, stap_info = _stap_pd(
        Icube,
        tile_hw=tile_hw,
        stride=tile_stride,
        Lt=Lt,
        prf_hz=prf_hz,
        diag_load=diag_load,
        estimator=cov_estimator,
        huber_c=huber_c,
        mvdr_load_mode=mvdr_load_mode,
        mvdr_auto_kappa=mvdr_auto_kappa,
        constraint_ridge=constraint_ridge,
        fd_span_mode=fd_mode,
        fd_span_rel=fd_span_rel,
        fd_fixed_span_hz=fd_fixed_span_hz,
        constraint_mode=constraint_mode,
        grid_step_rel=grid_step_rel,
        min_pts=fd_min_pts,
        max_pts=fd_max_pts,
        fd_min_abs_hz=fd_min_abs_hz,
        msd_lambda=msd_lambda,
        msd_ridge=msd_ridge,
        msd_agg_mode=msd_agg_mode,
        msd_ratio_rho=msd_ratio_rho,
        motion_half_span_rel=motion_half_span_rel,
        msd_contrast_alpha=msd_contrast_alpha,
        debug_max_samples=stap_debug_samples,
        debug_tile_coords=stap_debug_tile_coords,
        stap_device=stap_device_resolved,
        tile_batch=tile_batch_size,
        pd_base_full=pd_base,
        mask_flow=mask_flow_cond,
        mask_bg=mask_bg_cond,
        ka_mode=ka_mode_norm if ka_active else "none",
        ka_prior_library=ka_prior_library if ka_active else None,
        ka_opts=ka_opts_dict if ka_active else None,
        alias_psd_select_enable=alias_psd_select_enable,
        alias_psd_select_ratio_thresh=alias_psd_select_ratio_thresh,
        alias_psd_select_bins=alias_psd_select_bins,
        psd_telemetry=psd_telemetry,
        psd_tapers=psd_tapers,
        psd_bandwidth=psd_bandwidth,
        band_ratio_recorder=stap_br_recorder,
        feasibility_mode=feas_mode,
        band_ratio_spec=band_ratio_spec_meta,
        tile_debug_limit=tile_debug_limit,
    )
    gate_mask_flow = stap_info.pop("_gate_mask_flow", None)
    gate_mask_bg = stap_info.pop("_gate_mask_bg", None)
    score_mode_raw = (score_mode or "msd").strip().lower()
    score_mode_alias = {
        "band_pd": "pd",
        "band": "pd",
        "band_ratio": "band_ratio",
        "band_ratio_whitened": "band_ratio",
    }
    score_mode_resolved = score_mode_alias.get(score_mode_raw, score_mode_raw)
    if score_mode_resolved not in {"msd", "pd", "band_ratio"}:
        raise ValueError(
            f"Unsupported score_mode '{score_mode}'. Expected 'msd', 'pd', or 'band_ratio'."
        )
    band_ratio_mode_effective = band_ratio_mode_norm
    if score_mode_raw == "band_ratio_whitened":
        band_ratio_mode_effective = "whitened"
    stap_info["stap_ms"] = float(1000.0 * (time.time() - t_stap_start))
    # stap_extract_ms / stap_batch_proc_ms / stap_post_ms are populated inside _stap_pd;
    # leave them as-is if present.
    stap_info["flow_mask_mode"] = flow_mask_mode_norm
    if flow_mask_stats is not None:
        stap_info["flow_mask_stats"] = flow_mask_stats
    stap_info["stap_conditional_mask"] = cond_mask_stats
    stap_info["band_ratio_mode_requested"] = band_ratio_mode_norm
    stap_info["band_ratio_mode_effective"] = band_ratio_mode_effective
    stap_info["band_ratio_bands_hz"] = dict(band_ratio_spec_meta)
    stap_info["band_ratio_flavor"] = (
        "whitened_mt_logratio" if use_whitened_ratio else "legacy_pd_ratio"
    )
    if use_whitened_ratio:
        mt_meta = {"tapers": int(psd_tapers), "bandwidth": float(psd_bandwidth)}
        stap_info["band_ratio_mt_params"] = mt_meta
        baseline_telemetry.setdefault("band_ratio_mt_params", mt_meta)
    base_tile_scores: np.ndarray | None = None
    stap_tile_scores: np.ndarray | None = None
    if use_whitened_ratio and base_br_recorder is not None and stap_br_recorder is not None:
        base_tile_scores, base_br_stats = base_br_recorder.finalize()
        stap_tile_scores, stap_br_stats = stap_br_recorder.finalize()
        # Always persist recorder summaries, even if one side is missing PSD
        # observations (e.g. PD-only / fast STAP paths may not wire the
        # post-filter callback for stap_br_recorder).
        stap_info["band_ratio_br_stats"] = stap_br_stats or {"count": 0}
        baseline_telemetry.setdefault("band_ratio_stats", base_br_stats or {"count": 0})

        # Construct the baseline (MC-SVD residual) band-ratio map whenever
        # baseline tile scores are available, independent of stap tile scores.
        if base_tile_scores is not None and base_tile_scores.size == tile_count:
            base_band_ratio_map = _tile_scores_to_map(
                base_tile_scores, pd_base.shape, tile_hw, tile_stride
            )
        else:
            base_band_ratio_map = np.ones_like(pd_base, dtype=np.float32)

        # Construct the STAP band-ratio map only when stap tile scores exist.
        if stap_tile_scores is not None and stap_tile_scores.size == tile_count:
            stap_band_ratio_map = _tile_scores_to_map(
                stap_tile_scores, pd_base.shape, tile_hw, tile_stride
            )
        else:
            stap_band_ratio_map = np.ones_like(pd_base, dtype=np.float32)
    else:
        denom_pd = np.maximum(pd_base, 1e-12)
        stap_band_ratio = pd_stap / denom_pd
        stap_band_ratio[pd_base <= 1e-12] = 0.0
        stap_band_ratio_map = stap_band_ratio.astype(np.float32, copy=False)
        base_band_ratio_map = np.ones_like(pd_base, dtype=np.float32)

    # Additional band telemetry maps (Phase 0 telemetry hardening).
    # These maps are derived from the baseline band-ratio recorder when
    # available; otherwise they fall back to neutral zeros.
    base_m_alias_map = (-base_band_ratio_map).astype(np.float32, copy=False)
    base_guard_frac_map = np.zeros_like(pd_base, dtype=np.float32)
    base_peak_freq_map = np.zeros_like(pd_base, dtype=np.float32)
    if use_whitened_ratio and base_br_recorder is not None:
        try:
            base_guard_frac_map = _tile_scores_to_map(
                np.asarray(base_br_recorder.rg_raw, dtype=np.float32),
                pd_base.shape,
                tile_hw,
                tile_stride,
            )
            base_peak_freq_map = _tile_scores_to_map(
                np.asarray(base_br_recorder.peak_freqs, dtype=np.float32),
                pd_base.shape,
                tile_hw,
                tile_stride,
            )
        except Exception:
            base_guard_frac_map = np.zeros_like(pd_base, dtype=np.float32)
            base_peak_freq_map = np.zeros_like(pd_base, dtype=np.float32)

    # Phase 1 (KA Contract v2): evaluate a label-free contract state and log it.
    # This is logging-only: it does not modify any score maps in Phase 1.
    ka_contract_v2_report: dict | None = None
    ka_contract_v2_inputs: dict | None = None
    if evaluate_ka_contract_v2 is not None:
        if (
            use_whitened_ratio
            and base_br_recorder is not None
            and base_tile_scores is not None
            and base_tile_scores.size == tile_count
        ):
            tile_cov_flow, tile_coords = _tile_coverages(mask_flow, tile_hw, tile_stride)
            th, tw = tile_hw
            # s_base must be aligned with the detector score convention
            # (higher = more flow evidence). For PD scoring, downstream ROC uses
            # S = -PD, so we use -PD here as well.
            if score_mode_resolved == "pd":
                score_map_for_contract = -pd_stap
            elif score_mode_resolved == "band_ratio":
                score_map_for_contract = stap_band_ratio_map
            else:
                score_map_for_contract = stap_scores
            s_base_tiles = np.zeros(tile_cov_flow.shape[0], dtype=np.float32)
            for idx, (y0, x0) in enumerate(tile_coords):
                s_base_tiles[idx] = float(
                    np.mean(score_map_for_contract[y0 : y0 + th, x0 : x0 + tw])
                )
            m_alias_tiles = (-np.asarray(base_tile_scores, dtype=np.float32)).astype(
                np.float32, copy=False
            )
            r_guard_tiles = np.asarray(base_br_recorder.rg_raw, dtype=np.float32)
            peak_freq_tiles = np.asarray(base_br_recorder.peak_freqs, dtype=np.float32)
            pf_peak_tiles = (peak_freq_tiles >= float(br_spec.flow_low_hz)) & (
                peak_freq_tiles <= float(br_spec.flow_high_hz)
            )
            valid_tiles = np.asarray(base_br_recorder.tile_filled, dtype=bool)
            ka_contract_v2_inputs = {
                "tile_cov_flow": tile_cov_flow,
                "tile_coords": tile_coords,
                "s_base_tiles": s_base_tiles,
                "m_alias_tiles": m_alias_tiles,
                "r_guard_tiles": r_guard_tiles,
                "pf_peak_tiles": pf_peak_tiles.astype(bool),
                "valid_tiles": valid_tiles,
            }
            ka_contract_v2_report = evaluate_ka_contract_v2(
                s_base=s_base_tiles,
                m_alias=m_alias_tiles,
                r_guard=r_guard_tiles,
                pf_peak=pf_peak_tiles,
                c_flow=tile_cov_flow,
                valid_mask=valid_tiles,
            )
        else:
            ka_contract_v2_report = evaluate_ka_contract_v2(
                s_base=None,
                m_alias=None,
                r_guard=None,
                c_flow=None,
            )

        try:
            stap_info["ka_contract_v2_state"] = str(ka_contract_v2_report.get("state"))
            stap_info["ka_contract_v2_reason"] = str(ka_contract_v2_report.get("reason"))
            ka_metrics = ka_contract_v2_report.get("metrics", {}) or {}
            for key in (
                "iqr_alias_bg",
                "guard_q90",
                "delta_bg_flow_median",
                "delta_tail",
                "p_shrink",
                "uplift_eligible",
            ):
                stap_info[f"ka_contract_v2_{key}"] = ka_metrics.get(key)
        except Exception:
            pass

    base_score_map = pd_base
    stap_maps_by_mode = {
        "msd": stap_scores,
        "pd": pd_stap,
        "band_ratio": stap_band_ratio_map,
    }
    stap_score_pool_map_default = stap_maps_by_mode[score_mode_resolved]
    base_pos_shared, base_neg_shared = _pool_scores(
        base_score_map, mask_flow, mask_bg, 20000, 60000, seed
    )
    if use_whitened_ratio:
        base_pos_ratio, base_neg_ratio = _pool_scores(
            base_band_ratio_map, mask_flow, mask_bg, 20000, 60000, seed + 6
        )
    else:
        base_pos_ratio = np.ones_like(base_pos_shared, dtype=np.float32)
        base_neg_ratio = np.ones_like(base_neg_shared, dtype=np.float32)
    stap_pos_msd, stap_neg_msd = _pool_scores(
        stap_scores, mask_flow, mask_bg, 20000, 60000, seed + 1
    )
    stap_pos_pd, stap_neg_pd = _pool_scores(pd_stap, mask_flow, mask_bg, 20000, 60000, seed + 2)
    stap_pos_band_ratio, stap_neg_band_ratio = _pool_scores(
        stap_band_ratio_map, mask_flow, mask_bg, 20000, 60000, seed + 5
    )

    score_pool_arrays = {
        "msd": {
            "base_pos": base_pos_shared,
            "base_neg": base_neg_shared,
            "stap_pos": stap_pos_msd,
            "stap_neg": stap_neg_msd,
        },
        "pd": {
            "base_pos": base_pos_shared,
            "base_neg": base_neg_shared,
            "stap_pos": stap_pos_pd,
            "stap_neg": stap_neg_pd,
        },
        "band_ratio": {
            "base_pos": base_pos_ratio,
            "base_neg": base_neg_ratio,
            "stap_pos": stap_pos_band_ratio,
            "stap_neg": stap_neg_band_ratio,
        },
    }

    base_score_map_default = (
        base_score_map if score_mode_resolved in {"msd", "pd"} else base_band_ratio_map
    )
    stap_score_map_default = stap_score_pool_map_default

    def _rank_consistency_from_mask(
        mask: np.ndarray | None,
        seed_offset: int,
    ) -> dict[str, float | int | None]:
        entry: dict[str, float | int | None] = {
            "pixels": 0,
            "concordance": None,
            "kendall_tau": None,
        }
        if mask is None:
            return entry
        mask_bool = np.asarray(mask, dtype=bool)
        hits = int(mask_bool.sum())
        entry["pixels"] = hits
        if hits < 2:
            return entry
        base_vals = base_score_map_default[mask_bool]
        stap_vals = stap_score_map_default[mask_bool]
        total_pairs = hits * (hits - 1) // 2
        max_pairs = min(50_000, total_pairs)
        prob = _rank_consistency_same_class(
            base_vals,
            stap_vals,
            max_pairs=max_pairs if max_pairs > 0 else 0,
            seed=seed + seed_offset,
        )
        if prob is None:
            return entry
        entry["concordance"] = prob
        entry["kendall_tau"] = 2.0 * prob - 1.0
        return entry

    def _score_transform_from_mask(mask: np.ndarray | None) -> dict[str, float | int | None]:
        entry: dict[str, float | int | None] = {
            "pixels": 0,
            "p10_ratio": None,
            "median_ratio": None,
            "p90_ratio": None,
        }
        if mask is None:
            return entry
        mask_bool = np.asarray(mask, dtype=bool)
        hits = int(mask_bool.sum())
        entry["pixels"] = hits
        if hits == 0:
            return entry
        base_vals = base_score_map_default[mask_bool]
        stap_vals = stap_score_map_default[mask_bool]
        valid = np.isfinite(base_vals) & np.isfinite(stap_vals)
        if not np.any(valid):
            return entry
        safe_base = np.maximum(base_vals[valid], 1e-9)
        ratios = stap_vals[valid] / safe_base
        ratios = ratios[np.isfinite(ratios)]
        if ratios.size == 0:
            return entry
        entry["median_ratio"] = float(np.median(ratios))
        if ratios.size >= 2:
            entry["p10_ratio"] = float(np.quantile(ratios, 0.10))
            entry["p90_ratio"] = float(np.quantile(ratios, 0.90))
        else:
            val = float(ratios[0])
            entry["p10_ratio"] = val
            entry["p90_ratio"] = val
        return entry

    rank_consistency_gated = {
        "flow": _rank_consistency_from_mask(gate_mask_flow, 31),
        "bg": _rank_consistency_from_mask(gate_mask_bg, 37),
    }
    score_transform_gated = {
        "flow": _score_transform_from_mask(gate_mask_flow),
        "bg": _score_transform_from_mask(gate_mask_bg),
    }
    ka_effective_runtime = bool(stap_info.get("ka_effective", True))
    ka_disable_reasons_runtime = list(stap_info.get("ka_disable_reasons", []))
    flow_transform_stats = score_transform_gated.get("flow", {})
    flow_rank_stats = rank_consistency_gated.get("flow", {})
    flow_pixels = int(flow_transform_stats.get("pixels") or 0)
    # Apply the gated-flow monotonicity guard. The original design only
    # evaluated monotonicity when we had a sufficiently large gated flow
    # sample; here we retain that behavior but add an additional sparse
    # guard so that obviously pathological transformations on a small
    # number of gated flow pixels are still caught.
    monotone_min_samples = 200
    stap_info["ka_gated_flow_pixels"] = int(flow_pixels)
    stap_info["ka_monotonicity_min_samples"] = int(monotone_min_samples)
    if ka_effective_runtime and flow_pixels >= monotone_min_samples:
        p10_ratio = flow_transform_stats.get("p10_ratio")
        p90_ratio = flow_transform_stats.get("p90_ratio")
        median_ratio = flow_transform_stats.get("median_ratio")
        ratio_spread = None
        if p90_ratio is not None and p10_ratio not in (None, 0.0) and p10_ratio is not None:
            denom = abs(p10_ratio)
            if denom > 0.0:
                ratio_spread = abs(p90_ratio) / denom
        tau_flow = flow_rank_stats.get("kendall_tau")
        disable_monotone = False
        if ratio_spread is not None and ratio_spread > 1e3:
            disable_monotone = True
        elif (
            tau_flow is not None
            and abs(tau_flow) < 0.1
            and median_ratio is not None
            and median_ratio > 10.0
        ):
            disable_monotone = True
        if disable_monotone:
            ka_effective_runtime = False
            ka_disable_reasons_runtime.append("gated_monotonicity")
    # Sparse-sample guard: if only a small number of gated flow pixels
    # exist but KA applies extremely large, non-monotone score changes
    # on them, mark KA as ineffective for this configuration.
    if ka_effective_runtime and 0 < flow_pixels < monotone_min_samples:
        median_ratio = flow_transform_stats.get("median_ratio")
        p10_ratio = flow_transform_stats.get("p10_ratio")
        tau_flow = flow_rank_stats.get("kendall_tau")
        disable_sparse = False
        if median_ratio is not None and median_ratio > 20.0:
            disable_sparse = True
        elif p10_ratio is not None and p10_ratio > 10.0:
            disable_sparse = True
        elif (
            tau_flow is not None
            and tau_flow < -0.2
            and median_ratio is not None
            and median_ratio > 5.0
        ):
            disable_sparse = True
        if disable_sparse:
            ka_effective_runtime = False
            ka_disable_reasons_runtime.append("gated_monotonicity_sparse")
    stap_info["ka_effective"] = bool(ka_effective_runtime)
    stap_info["ka_disable_reasons"] = ka_disable_reasons_runtime
    # Persist gated flow / background masks for telemetry-driven analysis.
    # These were removed from stap_info earlier (to compute the guards)
    # but are safe to store here after converting to plain Python lists.
    if gate_mask_flow is not None:
        stap_info["_gate_mask_flow"] = gate_mask_flow.astype(bool).tolist()
    if gate_mask_bg is not None:
        stap_info["_gate_mask_bg"] = gate_mask_bg.astype(bool).tolist()

    default_pool = score_pool_arrays[score_mode_resolved]
    stap_score_pool_map = stap_score_pool_map_default
    stap_pos_default = default_pool["stap_pos"]
    stap_neg_default = default_pool["stap_neg"]

    confirm2_s1, confirm2_s2 = _split_confirm2_pairs(stap_neg_default, seed=seed + 3)
    confirm2_pos1, confirm2_pos2 = _split_confirm2_pairs(stap_pos_default, seed=seed + 4)

    total_frames = Icube.shape[0]
    angles_per_set = [imgs.shape[0] for imgs in image_sets]
    f0_mhz = g.f0 / 1e6
    if dataset_name is None:
        if len(image_sets) == 1:
            dataset_name = f"pw_{f0_mhz:.1f}MHz_{angles_per_set[0]}ang_{total_frames}T_seed{seed}"
        else:
            dataset_name = (
                f"pw_{f0_mhz:.1f}MHz_{angles_per_set[0]}ang_{len(image_sets)}ens_"
                f"{total_frames}T_seed{seed}"
            )
    if dataset_suffix:
        suffix = str(dataset_suffix).strip()
        if suffix:
            dataset_name = f"{dataset_name}_{suffix}"

    bundle_dir = out_root / dataset_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}

    def _save(name: str, arr: np.ndarray) -> str:
        path = bundle_dir / f"{name}.npy"
        np.save(path, arr, allow_pickle=False)
        paths[name] = str(path)
        return str(path)

    pool_keys = ("base_pos", "base_neg", "stap_pos", "stap_neg")
    score_pool_files_rel: dict[str, dict[str, str]] = {}
    for mode, pools in score_pool_arrays.items():
        files: dict[str, str] = {}
        for key in pool_keys:
            arr = pools[key].astype(np.float32, copy=False)
            name = key if mode == score_mode_resolved else f"{key}_{mode}"
            fname = f"{name}.npy"
            _save(name, arr)
            files[key] = fname
            paths[f"{key}_{mode}"] = paths[name]
            if mode == score_mode_resolved and key not in paths:
                paths[key] = paths[name]
        score_pool_files_rel[mode] = files
    # Ensure canonical keys exist for the default mode
    for key in pool_keys:
        if key not in paths:
            paths[key] = paths[f"{key}_{score_mode_resolved}"]

    rank_consistency: dict[str, dict[str, float | None]] = {}
    default_pool = score_pool_arrays[score_mode_resolved]
    for cls_key in ("pos", "neg"):
        base_arr = default_pool[f"base_{cls_key}"]
        stap_arr = default_pool[f"stap_{cls_key}"]
        prob = _rank_consistency_same_class(base_arr, stap_arr)
        tau = None if prob is None else (2.0 * prob - 1.0)
        rank_consistency[cls_key] = {"concordance": prob, "kendall_tau": tau}

    # Optional score-space KA v1: shrink-only transform on PD-based scores using
    # a pre-trained risk model. We encode the transform by inflating pd_stap on
    # high-risk tiles (so that S = -pd_stap is shrunk) while leaving baseline
    # pd_base unchanged. This only applies for PD score_mode; sentinels ensure
    # we do not catastrophically reorder strong flow detections.
    score_model_json = ka_opts_dict.get("score_model_json")
    score_alpha_val = float(ka_opts_dict.get("score_alpha", 0.0))
    score_contract_v2_enable = bool(float(ka_opts_dict.get("score_contract_v2", 0.0)) > 0.0)
    score_contract_v2_force = bool(
        float(ka_opts_dict.get("score_contract_v2_force", 0.0)) > 0.0
    )
    score_contract_v2_applied = False

    # Phase 2: score-space KA v2 (contract-driven, shrink-only) in safety mode.
    # This uses the KA Contract v2 state machine and does not require a trained
    # model. By default we apply it only when the contract reports C1_SAFETY.
    # For ablations, an optional "force" mode applies the same shrink mapping
    # even when the contract would disable it.
    if (
        score_contract_v2_enable
        and derive_score_shrink_v2_tile_scales is not None
        and score_mode_resolved == "pd"
    ):
        if score_model_json and score_alpha_val > 0.0:
            stap_info["score_ka_v2_disabled_reason"] = "conflict_with_score_model_v1"
        elif ka_contract_v2_report is None or ka_contract_v2_inputs is None:
            stap_info["score_ka_v2_disabled_reason"] = "missing_contract_v2"
        else:
            state = str(ka_contract_v2_report.get("state") or "C0_OFF")
            reason = str(ka_contract_v2_report.get("reason") or "")
            stap_info["score_ka_v2_state"] = state
            stap_info["score_ka_v2_contract_reason"] = reason
            if not use_whitened_ratio:
                stap_info["score_ka_v2_disabled_reason"] = "requires_whitened_band_ratio"
            else:
                ka_metrics = ka_contract_v2_report.get("metrics", {}) or {}
                risk_mode = str(ka_metrics.get("risk_mode") or "alias").strip().lower()
                stap_info["score_ka_v2_risk_mode"] = risk_mode
                if risk_mode == "guard":
                    risk_tiles = ka_contract_v2_inputs["r_guard_tiles"]
                else:
                    risk_tiles = ka_contract_v2_inputs["m_alias_tiles"]
                shrink = None
                if score_contract_v2_force:
                    if derive_score_shrink_v2_tile_scales_forced is None:
                        stap_info["score_ka_v2_disabled_reason"] = "forced_helper_unavailable"
                    else:
                        stap_info["score_ka_v2_forced"] = True
                        shrink = derive_score_shrink_v2_tile_scales_forced(
                            report=ka_contract_v2_report,
                            s_base=ka_contract_v2_inputs["s_base_tiles"],
                            m_alias=risk_tiles,
                            c_flow=ka_contract_v2_inputs["tile_cov_flow"],
                            valid_mask=ka_contract_v2_inputs["valid_tiles"],
                            mode="safety",
                            risk_mode=risk_mode,
                        )
                else:
                    if state != "C1_SAFETY" or reason != "ok":
                        stap_info["score_ka_v2_disabled_reason"] = f"contract_{state}:{reason}"
                    else:
                        shrink = derive_score_shrink_v2_tile_scales(
                            report=ka_contract_v2_report,
                            s_base=ka_contract_v2_inputs["s_base_tiles"],
                            m_alias=risk_tiles,
                            c_flow=ka_contract_v2_inputs["tile_cov_flow"],
                            valid_mask=ka_contract_v2_inputs["valid_tiles"],
                            mode="safety",
                        )

                if shrink is None:
                    if "score_ka_v2_disabled_reason" not in stap_info:
                        stap_info["score_ka_v2_disabled_reason"] = "no_shrink_result"
                elif not shrink.get("apply"):
                    stap_info["score_ka_v2_disabled_reason"] = str(
                        shrink.get("reason") or "unknown"
                    )
                else:
                    scale_tiles = np.asarray(shrink["scale_tiles"], dtype=np.float32)
                    gated_tiles = np.asarray(shrink["gated_tiles"], dtype=bool)
                    tile_coords = ka_contract_v2_inputs["tile_coords"]
                    th, tw = tile_hw
                    # Build a per-pixel scale map by averaging overlaps, then
                    # restrict the action to the union of gated tiles.
                    scale_map = _tile_scores_to_map(scale_tiles, pd_base.shape, tile_hw, tile_stride)
                    gate_union = np.zeros_like(pd_stap, dtype=bool)
                    for idx, (y0, x0) in enumerate(tile_coords):
                        if idx < gated_tiles.size and gated_tiles[idx]:
                            gate_union[y0 : y0 + th, x0 : x0 + tw] = True
                    scale_final = np.ones_like(scale_map, dtype=np.float32)
                    scale_final[gate_union] = scale_map[gate_union].astype(np.float32, copy=False)

                    # Protected pixels: never modify flow mask or extreme-score pixels.
                    pd_stap_orig = pd_stap
                    s_base_pix = -pd_stap_orig
                    # IMPORTANT: never mutate `mask_flow` (used for evaluation and
                    # conditional execution). Use a copy for protected pixels.
                    prot_pix = np.asarray(mask_flow, dtype=bool).copy()
                    cfg = ka_contract_v2_report.get("config") or {}
                    q_hi = float(cfg.get("q_hi_protect", 0.995))
                    finite = np.isfinite(s_base_pix)
                    if finite.any():
                        thr_hi = float(np.quantile(s_base_pix[finite], q_hi))
                        prot_pix |= s_base_pix >= thr_hi
                    scale_final[prot_pix] = 1.0

                    pd_stap = (pd_stap_orig.astype(np.float32, copy=False) * scale_final).astype(
                        np.float32, copy=False
                    )
                    # Keep the exported default score map consistent when PD is
                    # the selected score_mode.
                    stap_score_pool_map = pd_stap
                    score_contract_v2_applied = True
                    stap_info["score_ka_v2_applied"] = True
                    stap_info["score_ka_v2_stats"] = shrink.get("stats", {})
                    scaled = scale_final > (1.0 + 1e-6)
                    stap_info["score_ka_v2_scaled_pixel_fraction"] = float(np.mean(scaled))
                    if scaled.any():
                        vals = scale_final[scaled].astype(np.float64, copy=False)
                        stap_info["score_ka_v2_scale_p50"] = float(np.median(vals))
                        stap_info["score_ka_v2_scale_p90"] = float(np.quantile(vals, 0.90))
                        stap_info["score_ka_v2_scale_max"] = float(np.max(vals))

    if score_model_json and score_alpha_val > 0.0 and score_mode_resolved == "pd" and not score_contract_v2_applied:
        try:
            import json as _json  # local alias to avoid shadowing

            with open(str(score_model_json), "r") as _f:
                score_model = _json.load(_f)
            w = np.asarray(score_model.get("weights", []), dtype=np.float64)
            b = float(score_model.get("bias", 0.0))
            mean = np.asarray(score_model.get("mean", []), dtype=np.float64)
            std = np.asarray(score_model.get("std", []), dtype=np.float64)
            if w.size == 0 or mean.size != w.size or std.size != w.size:
                raise RuntimeError("invalid score-space KA model configuration")

            H, W = pd_stap.shape
            depth = np.linspace(0.0, 1.0, H, endpoint=True)[:, None]
            depth_map = depth.repeat(W, axis=1)
            geom_flow = mask_flow_default.astype(np.float64)

            feats = np.stack(
                [
                    pd_stap.ravel(),
                    base_band_ratio_map.ravel(),
                    depth_map.ravel(),
                    geom_flow.ravel(),
                ],
                axis=1,
            )
            std_safe = std.copy()
            std_safe[std_safe == 0.0] = 1.0
            feats_std = (feats - mean) / std_safe
            logits = feats_std @ w + b
            logits = np.clip(logits, -20.0, 20.0)
            r = 1.0 / (1.0 + np.exp(-logits))
            r = r.reshape(H, W)

            pd_stap_orig = pd_stap.copy()
            # Inflate PD on high-risk tiles so that S = -PD is shrunk; ensure
            # the scale factor is >= 1 (shrink-only behavior in score space).
            scale = 1.0 + score_alpha_val * r
            pd_stap = pd_stap * scale

            # Simple rank-consistency sentinel on flow pixels to avoid
            # pathological reorderings.
            if mask_flow is not None and mask_flow.any():
                safe_flow = mask_flow.astype(bool)
                base_vals = -pd_stap_orig[safe_flow].ravel()
                stap_vals = -pd_stap[safe_flow].ravel()
                if base_vals.size >= 2 and stap_vals.size == base_vals.size:
                    prob = _rank_consistency_same_class(base_vals, stap_vals)
                    if prob is not None:
                        tau = 2.0 * prob - 1.0
                        stap_info["score_ka_flow_tau"] = float(tau)
                        if tau < 0.5:
                            stap_info["score_ka_disabled_reason"] = "flow_rank_tau_lt_0.5"
                            pd_stap = pd_stap_orig
            if score_mode_resolved == "pd":
                stap_score_pool_map = pd_stap
        except Exception as exc:  # pragma: no cover - best-effort safeguard
            stap_info["score_ka_disabled_reason"] = f"{type(exc).__name__}: {exc}"

    # Ensure the pooled PD score samples (stap_pos/stap_neg) reflect the final
    # pd_stap after any score-space KA transforms. Several analysis scripts
    # load these pools directly; keeping them in sync avoids confusing
    # "KA applied but ROC unchanged" artifacts.
    if score_mode_resolved == "pd":
        try:
            stap_pos_pd, stap_neg_pd = _pool_scores(
                pd_stap, mask_flow, mask_bg, 20000, 60000, seed + 2
            )
            score_pool_arrays["pd"]["stap_pos"] = stap_pos_pd
            score_pool_arrays["pd"]["stap_neg"] = stap_neg_pd
            _save("stap_pos", stap_pos_pd.astype(np.float32, copy=False))
            _save("stap_neg", stap_neg_pd.astype(np.float32, copy=False))

            base_pos_pd = score_pool_arrays["pd"]["base_pos"]
            base_neg_pd = score_pool_arrays["pd"]["base_neg"]
            prob = _rank_consistency_same_class(base_pos_pd, stap_pos_pd)
            tau = None if prob is None else (2.0 * prob - 1.0)
            rank_consistency["pos"] = {"concordance": prob, "kendall_tau": tau}
            prob = _rank_consistency_same_class(base_neg_pd, stap_neg_pd)
            tau = None if prob is None else (2.0 * prob - 1.0)
            rank_consistency["neg"] = {"concordance": prob, "kendall_tau": tau}
        except Exception as exc:  # pragma: no cover - best-effort safeguard
            stap_info["score_pool_refresh_error"] = f"{type(exc).__name__}: {exc}"

    # ---- PD-mode score convention (repository) ----
    # We persist PD maps as `pd_*.npy` and evaluate ROC by thresholding their
    # lower tail (equivalently, use the right-tail score S = -pd).
    #
    # In the Brain-* regimes used in this repo, H0 tiles often contain strong
    # alias/clutter energy, so *smaller* values of `pd_*.npy` can be more
    # flow-like; this is why scripts such as `scripts/hab_contract_check.py`
    # use `score = -pd` for PD-based ROC.
    #
    # For transparency and to avoid sign ambiguity, we also export the explicit
    # right-tail score maps `score_pd_*.npy := -pd_*.npy`.
    _save("pd_base", pd_base.astype(np.float32, copy=False))
    _save("pd_stap", pd_stap.astype(np.float32, copy=False))
    _save("score_pd_base", (-pd_base).astype(np.float32, copy=False))
    _save("score_pd_stap", (-pd_stap).astype(np.float32, copy=False))
    _save("base_band_ratio_map", base_band_ratio_map.astype(np.float32, copy=False))
    _save("base_m_alias_map", base_m_alias_map.astype(np.float32, copy=False))
    _save("base_guard_frac_map", base_guard_frac_map.astype(np.float32, copy=False))
    _save("base_peak_freq_map", base_peak_freq_map.astype(np.float32, copy=False))
    _save("stap_band_ratio_map", stap_band_ratio_map.astype(np.float32, copy=False))
    _save("base_score_map", base_score_map.astype(np.float32, copy=False))
    _save("stap_score_map", stap_scores.astype(np.float32, copy=False))
    _save("stap_score_pool_map", stap_score_pool_map.astype(np.float32, copy=False))
    _save("mask_flow", mask_flow.astype(np.bool_, copy=False))
    _save("mask_bg", mask_bg.astype(np.bool_, copy=False))
    if cond_enabled and mask_flow_cond is not None:
        _save("mask_flow_stap_gate", mask_flow_cond.astype(np.bool_, copy=False))
        if mask_bg_cond is not None:
            _save("mask_bg_stap_gate", mask_bg_cond.astype(np.bool_, copy=False))
    _save("confirm2_scores1", confirm2_s1)
    _save("confirm2_scores2", confirm2_s2)
    _save("confirm2_pos_scores1", confirm2_pos1)
    _save("confirm2_pos_scores2", confirm2_pos2)

    debug_files: list[str] = []
    debug_entries = stap_info.pop("debug_samples", [])
    if debug_entries:
        debug_dir = bundle_dir / "stap_debug"
        debug_dir.mkdir(exist_ok=True)
        for idx, sample in enumerate(debug_entries):
            path = debug_dir / f"tile_{idx}_y{sample['y0']}_x{sample['x0']}.npz"
            payload: dict[str, object] = {
                "tile": sample["tile"],
                "fd_grid": (
                    np.array(sample.get("fd_grid"), dtype=np.float32)
                    if sample.get("fd_grid") is not None
                    else None
                ),
                "fd_grid_initial": (
                    np.array(sample.get("fd_grid_initial"), dtype=np.float32)
                    if sample.get("fd_grid_initial") is not None
                    else None
                ),
                "band_fraction_tile": sample.get("band_fraction_tile"),
                "score_tile": sample.get("score_tile"),
                "flow_mu_ratio": sample.get("flow_mu_ratio"),
                "bg_var_inflation": sample.get("bg_var_inflation"),
                "flow_coverage": sample.get("flow_coverage"),
                "score_mode": sample.get("score_mode"),
                "msd_lambda": sample.get("msd_lambda"),
                "msd_ridge": sample.get("msd_ridge"),
                "msd_agg_mode": sample.get("msd_agg_mode"),
                "band_fraction_quantiles": sample.get("band_fraction_quantiles"),
                "score_quantiles": sample.get("score_quantiles"),
                "y0": sample["y0"],
                "x0": sample["x0"],
                "tile_index": sample.get("tile_index"),
                "diag_load": sample.get("diag_load"),
                "load_mode": sample.get("load_mode"),
                "constraint_ridge": sample.get("constraint_ridge"),
                "fd_min_abs_hz": sample.get("fd_min_abs_hz"),
                "fd_min_abs_applied": sample.get("fd_min_abs_applied"),
                "fd_symmetry_added": sample.get("fd_symmetry_added"),
                "fd_min_abs_fallback": sample.get("fd_min_abs_fallback"),
                "kc_flow": sample.get("kc_flow"),
                "kc_motion": sample.get("kc_motion"),
                "kc_flow_cap": sample.get("kc_flow_cap"),
                "kc_flow_cap_motion": sample.get("kc_flow_cap_motion"),
                "kc_flow_freqs": sample.get("kc_flow_freqs"),
                "psd_peak_hz": sample.get("psd_peak_hz"),
                "psd_peak_power": sample.get("psd_peak_power"),
                "psd_top_freqs_hz": sample.get("psd_top_freqs_hz"),
                "psd_top_power": sample.get("psd_top_power"),
                "psd_flow_to_dc_ratio": sample.get("psd_flow_to_dc_ratio"),
                "psd_power_dc": sample.get("psd_power_dc"),
                "psd_power_flow": sample.get("psd_power_flow"),
                "psd_power_flow_hz": sample.get("psd_power_flow_hz"),
                "psd_flow_freq_target": sample.get("psd_flow_freq_target"),
                "psd_flow_freq_power": sample.get("psd_flow_freq_power"),
                "psd_flow_freq_source": sample.get("psd_flow_freq_source"),
                "psd_flow_freq_source_resolved": sample.get("psd_flow_freq_source_resolved"),
                "psd_fundamental_ratio": sample.get("psd_fundamental_ratio"),
                "psd_flow_alias_ratio": sample.get("psd_flow_alias_ratio"),
                "psd_flow_alias": sample.get("psd_flow_alias"),
                "constraint_mode": sample.get("constraint_mode"),
                "ka_beta": sample.get("ka_beta"),
                "ka_mismatch": sample.get("ka_mismatch"),
                "ka_lambda_used": sample.get("ka_lambda_used"),
                "ka_sigma_min_raw": sample.get("ka_sigma_min_raw"),
                "ka_sigma_max_raw": sample.get("ka_sigma_max_raw"),
            }
            for key, value in sample.items():
                if key not in payload and value is not None:
                    payload[key] = value
            np.savez(path, **payload)
            debug_files.append(str(path))

    pd_stats_flat = {
        "baseline_flow_median": float(np.median(pd_base[mask_flow])) if mask_flow.any() else None,
        "baseline_bg_median": float(np.median(pd_base[mask_bg])) if mask_bg.any() else None,
        "stap_flow_median": float(np.median(pd_stap[mask_flow])) if mask_flow.any() else None,
        "stap_bg_median": float(np.median(pd_stap[mask_bg])) if mask_bg.any() else None,
    }
    if (
        pd_stats_flat["stap_flow_median"] is not None
        and pd_stats_flat["baseline_flow_median"] is not None
    ):
        pd_stats_flat["flow_ratio"] = float(
            pd_stats_flat["stap_flow_median"] / max(pd_stats_flat["baseline_flow_median"], 1e-12)
        )
    if (
        pd_stats_flat["stap_bg_median"] is not None
        and pd_stats_flat["baseline_bg_median"] is not None
    ):
        pd_stats_flat["bg_ratio"] = float(
            pd_stats_flat["stap_bg_median"] / max(pd_stats_flat["baseline_bg_median"], 1e-12)
        )

    def _map_stats(arr: np.ndarray) -> dict[str, float | None]:
        stats: dict[str, float | None] = {
            "flow_median": float(np.median(arr[mask_flow])) if mask_flow.any() else None,
            "bg_median": float(np.median(arr[mask_bg])) if mask_bg.any() else None,
        }
        if stats["flow_median"] is not None and stats["bg_median"] is not None:
            stats["flow_to_bg_ratio"] = float(
                float(stats["flow_median"]) / max(float(stats["bg_median"]), 1e-12)
            )
        return stats

    def _pool_stats(
        samples_pos: np.ndarray, samples_neg: np.ndarray
    ) -> dict[str, float | int | None]:
        stats: dict[str, float | int | None] = {
            "npos": int(samples_pos.size),
            "nneg": int(samples_neg.size),
            "flow_median": float(np.median(samples_pos)) if samples_pos.size else None,
            "bg_median": float(np.median(samples_neg)) if samples_neg.size else None,
            "flow_mean": float(np.mean(samples_pos)) if samples_pos.size else None,
            "bg_mean": float(np.mean(samples_neg)) if samples_neg.size else None,
        }
        if stats["flow_median"] is not None and stats["bg_median"] is not None:
            stats["flow_to_bg_ratio"] = float(
                float(stats["flow_median"]) / max(float(stats["bg_median"]), 1e-12)
            )
        return stats

    map_stats_by_mode = {
        "msd": _map_stats(stap_scores),
        "pd": _map_stats(pd_stap),
        "band_ratio": _map_stats(stap_band_ratio_map),
    }
    pool_stats_by_mode = {
        mode: _pool_stats(arrs["stap_pos"], arrs["stap_neg"])
        for mode, arrs in score_pool_arrays.items()
    }

    map_stats_default = map_stats_by_mode[score_mode_resolved]
    pool_stats_default = pool_stats_by_mode[score_mode_resolved]

    score_stats = {
        "mode": score_mode_resolved,
        "score_flow_median": map_stats_default.get("flow_median"),
        "score_bg_median": map_stats_default.get("bg_median"),
        "flow_to_bg_ratio": map_stats_default.get("flow_to_bg_ratio"),
        "pool_flow_median": pool_stats_default.get("flow_median"),
        "pool_bg_median": pool_stats_default.get("bg_median"),
        "pool_flow_to_bg_ratio": pool_stats_default.get("flow_to_bg_ratio"),
        "map_stats": map_stats_by_mode,
        "pool_stats": pool_stats_by_mode,
    }

    pd_stats_flat["pd_overlap_count_min"] = float(stap_info.get("pd_overlap_count_min", 0.0))
    pd_stats_flat["pd_overlap_count_max"] = float(stap_info.get("pd_overlap_count_max", 0.0))
    pd_stats_flat["pd_overlap_count_mean"] = float(stap_info.get("pd_overlap_count_mean", 0.0))
    pd_stats_flat["pd_overlap_any_zero"] = bool(stap_info.get("pd_overlap_any_zero", False))

    pd_stats_global = _pd_global_stats_dict(pd_base, pd_stap, mask_flow, mask_bg)
    pd_stats_combined = {**pd_stats_flat, **pd_stats_global}
    mask_flow_pd = _pd_derived_flow_mask(pd_stap, mask_flow)
    mask_flow_pd_path = bundle_dir / "mask_flow_pd.npy"
    np.save(mask_flow_pd_path, mask_flow_pd.astype(np.bool_, copy=False))
    paths["mask_flow_pd"] = str(mask_flow_pd_path)
    pd_stats_pdmask = _pd_mask_stats(pd_base, pd_stap, mask_flow_pd)
    pd_stats_combined["pd_mask"] = pd_stats_pdmask
    coverage_thresholds = (0.2, 0.5, 0.8)
    tile_cov_pd, tile_coords_pd = _tile_coverages(mask_flow_pd, tile_hw, tile_stride)
    pd_mask_cov_stats: dict[str, dict[str, float | int | None]] = {}
    for thr in coverage_thresholds:
        frac = float(np.mean(tile_cov_pd >= thr)) if tile_cov_pd.size else 0.0
        gate_mask = _mask_from_tile_thresholds(
            mask_flow_pd.shape, tile_hw, tile_coords_pd, tile_cov_pd, thr
        )
        mask_thr = mask_flow_pd & gate_mask
        stats_thr = _pd_mask_stats(pd_base, pd_stap, mask_thr)
        stats_thr["fraction"] = frac
        key = f"cov_ge_{int(thr * 100):02d}"
        pd_mask_cov_stats[key] = stats_thr
    pd_stats_combined["pd_mask_coverages"] = pd_mask_cov_stats

    pd_stats_path = bundle_dir / "pd_stats.json"
    with open(pd_stats_path, "w") as f:
        json.dump(pd_stats_combined, f, indent=2)
    paths["pd_stats"] = str(pd_stats_path)
    baseline_stats = pd_stats_global.get("baseline", {})
    stap_stats = pd_stats_global.get("stap", {})
    stap_info["flow_mu_base_actual"] = baseline_stats.get("flow_mean")
    stap_info["flow_mu_stap_actual"] = stap_stats.get("flow_mean")
    stap_info["flow_mu_ratio_actual"] = pd_stats_global.get("flow_ratio")
    stap_info["flow_pdmask_ratio_median"] = pd_stats_pdmask.get("ratio_median")
    stap_info["flow_pdmask_ratio_p10"] = pd_stats_pdmask.get("ratio_p10")
    stap_info["flow_pdmask_ratio_p90"] = pd_stats_pdmask.get("ratio_p90")
    stap_info["flow_pdmask_pixels"] = pd_stats_pdmask.get("pixels")
    stap_info["flow_pdmask_fraction"] = (
        float(pd_stats_pdmask.get("pixels") or 0) / float(mask_flow_pd.size)
        if mask_flow_pd.size
        else None
    )
    stap_info["flow_pdmask_median_base"] = pd_stats_pdmask.get("median_base")
    stap_info["flow_pdmask_median_stap"] = pd_stats_pdmask.get("median_map")
    stap_info["flow_pdmask_p10_base"] = pd_stats_pdmask.get("p10_base")
    stap_info["flow_pdmask_p10_stap"] = pd_stats_pdmask.get("p10_map")
    stap_info["flow_pdmask_p90_base"] = pd_stats_pdmask.get("p90_base")
    stap_info["flow_pdmask_p90_stap"] = pd_stats_pdmask.get("p90_map")
    for key, stats_cov in pd_mask_cov_stats.items():
        prefix = f"flow_pdmask_{key}"
        stap_info[f"{prefix}_fraction"] = stats_cov.get("fraction")
        stap_info[f"{prefix}_ratio_median"] = stats_cov.get("ratio_median")
        stap_info[f"{prefix}_ratio_p10"] = stats_cov.get("ratio_p10")
        stap_info[f"{prefix}_ratio_p90"] = stats_cov.get("ratio_p90")
        stap_info[f"{prefix}_median_base"] = stats_cov.get("median_base")
        stap_info[f"{prefix}_median_stap"] = stats_cov.get("median_map")
        stap_info[f"{prefix}_p10_base"] = stats_cov.get("p10_base")
        stap_info[f"{prefix}_p10_stap"] = stats_cov.get("p10_map")
        stap_info[f"{prefix}_p90_base"] = stats_cov.get("p90_base")
        stap_info[f"{prefix}_p90_stap"] = stats_cov.get("p90_map")
    stap_info["bg_var_base_actual"] = baseline_stats.get("bg_var")
    stap_info["bg_var_stap_actual"] = stap_stats.get("bg_var")
    stap_info["bg_var_ratio_actual"] = pd_stats_global.get("bg_var_ratio")
    stap_info["score_pool_default"] = score_mode_resolved
    if alias_meta:
        stap_info["flow_alias_pixels"] = alias_meta.get("flow_alias_pixels")
        stap_info["flow_alias_hz"] = alias_meta.get("flow_alias_hz")
    if clutter_meta:
        stap_info["temporal_clutter_beta"] = clutter_meta.get("beta")
        stap_info["temporal_clutter_snr_db_target"] = clutter_meta.get("snr_db_target")
        stap_info["temporal_clutter_snr_db"] = clutter_meta.get("snr_db_actual")
        stap_info["temporal_clutter_pixels"] = clutter_meta.get("n_pixels")
    stap_info["score_pool_stats"] = {
        mode: {
            "npos": stats.get("npos"),
            "nneg": stats.get("nneg"),
            "flow_median": stats.get("flow_median"),
            "bg_median": stats.get("bg_median"),
            "flow_to_bg_ratio": stats.get("flow_to_bg_ratio"),
        }
        for mode, stats in pool_stats_by_mode.items()
    }

    bundle_files: dict[str, str] = {}
    for key in (
        "pd_base",
        "pd_stap",
        "score_pd_base",
        "score_pd_stap",
        "mask_flow",
        "mask_bg",
        "base_band_ratio_map",
        "stap_band_ratio_map",
        "stap_score_map",
        "stap_score_pool_map",
        "base_score_map",
    ):
        if key in paths:
            bundle_files[key] = Path(paths[key]).name

    telemetry_combined = dict(baseline_telemetry or {})
    telemetry_combined.update(stap_info)
    # Expose key PD statistics in stap_fallback_telemetry for all baseline types.
    # These are already stored in pd_stats.json; here we mirror them for convenience.
    if "baseline_flow_median" not in telemetry_combined:
        telemetry_combined["baseline_flow_median"] = pd_stats_flat.get("baseline_flow_median")
    if "baseline_bg_median" not in telemetry_combined:
        telemetry_combined["baseline_bg_median"] = pd_stats_flat.get("baseline_bg_median")
    if "stap_flow_median" not in telemetry_combined:
        telemetry_combined["stap_flow_median"] = pd_stats_flat.get("stap_flow_median")
    if "stap_bg_median" not in telemetry_combined:
        telemetry_combined["stap_bg_median"] = pd_stats_flat.get("stap_bg_median")
    # Also mirror global ratios if not already reported by the baseline.
    if "flow_ratio" not in telemetry_combined:
        telemetry_combined["flow_ratio"] = pd_stats_global.get("flow_ratio")
    if "bg_var_ratio" not in telemetry_combined:
        telemetry_combined["bg_var_ratio"] = pd_stats_global.get("bg_var_ratio")

    meta = {
        "sim_geom": asdict(g),
        "angles_deg_sets": angle_values,
        "dt_per_angle": dt_sets,
        "pulses_per_set": int(pulses_per_set),
        "total_frames": int(total_frames),
        "prf_hz": float(prf_hz),
        "f0_hz": float(g.f0),
        "tile_hw": tuple(int(v) for v in tile_hw),
        "tile_stride": int(tile_stride),
        "stap_debug_tile_coords": (
            [[int(y0), int(x0)] for y0, x0 in stap_debug_tile_coords]
            if stap_debug_tile_coords
            else []
        ),
        "Lt": int(Lt),
        "diag_load": float(diag_load),
        "cov_estimator": cov_estimator,
        "huber_c": float(huber_c),
        "fd_config": {
            "mode": fd_mode,
            "span_rel": list(fd_span_rel),
            "fixed_span_hz": fd_fixed_span_hz,
            "grid_step_rel": float(grid_step_rel),
            "min_pts": int(fd_min_pts),
            "max_pts": int(fd_max_pts),
        },
        "mvdr_config": {
            "load_mode": mvdr_load_mode,
            "auto_kappa_target": float(mvdr_auto_kappa),
            "constraint_ridge": float(constraint_ridge),
            "diag_load": float(diag_load),
        },
        "constraint_mode": constraint_mode,
        "msd_config": {
            "lambda": None if msd_lambda is None else float(msd_lambda),
            "ridge": float(msd_ridge),
            "agg": msd_agg_mode,
            "ratio_rho": float(msd_ratio_rho),
            "motion_half_span_rel": (
                None if motion_half_span_rel is None else float(motion_half_span_rel)
            ),
            "contrast_alpha": None if msd_contrast_alpha is None else float(msd_contrast_alpha),
        },
        "ka_config": {
            "mode": ka_mode_norm if ka_active else "none",
            "beta_bounds": list(ka_beta_bounds),
            "kappa_target": float(ka_kappa),
            "alpha": None if ka_alpha is None else float(ka_alpha),
            "prior_path": str(ka_prior_path) if ka_prior_path else None,
        },
        "score_pool_default": score_mode_resolved,
        "score_pool_files": score_pool_files_rel,
        "score_pool_stats": pool_stats_by_mode,
        "pd_mode": {
            "pd_files": {"base": "pd_base.npy", "stap": "pd_stap.npy"},
            "score_files": {"base": "score_pd_base.npy", "stap": "score_pd_stap.npy"},
            "roc_convention": "lower_tail_on_pd (equivalently right_tail_on_score=-pd)",
            "pd_base_definition": "pd_base[y,x]=(1/T)∑_t |X_base[t,y,x]|^2 for the chosen baseline-filtered cube",
            "pd_stap_definition": "pd_stap is the STAP PD-mode map produced by the temporal core; in this repo it is derived from baseline PD and per-tile flow-subspace energy fractions with explicit background invariance clamps",
        },
        "bundle_files": bundle_files,
        "stap_device": stap_info.get("stap_device"),
        "seed": int(seed),
        "stap_fallback_telemetry": telemetry_combined,
        "confirm2_pairs": {
            "n_pairs": int(confirm2_s1.size),
            "seed": int(seed + 2),
            "positive_pairs": int(confirm2_pos1.size),
        },
        "stap_debug_files": debug_files,
        # Merge flat medians and global stats into meta
        "pd_stats": pd_stats_combined,
        "score_stats": score_stats,
        "feasibility_mode": feas_mode,
        "rank_consistency": rank_consistency,
        "rank_consistency_gated": rank_consistency_gated,
        "score_transform_gated": score_transform_gated,
        "ka_contract_v2": ka_contract_v2_report,
    }
    if alias_meta:
        meta["flow_alias"] = alias_meta
    if bg_alias_meta:
        meta["bg_alias"] = bg_alias_meta
    if vibration_meta:
        meta["vibration"] = vibration_meta
    if flow_doppler_meta:
        meta["flow_doppler"] = flow_doppler_meta
    if phase_meta:
        meta["phase_screen"] = phase_meta
    if clutter_meta:
        meta["temporal_clutter"] = clutter_meta
    if meta_extra:
        meta.update(meta_extra)

    meta_path = bundle_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    paths["meta"] = str(meta_path)
    return paths


def _resolve_stap_device(device: str | None) -> str:
    if device is None:
        device = "auto"
    dev = str(device).lower()
    if dev in {"auto", "default"}:
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    if dev.startswith("cuda"):
        if torch is not None and torch.cuda.is_available():
            return device  # preserve cuda:idx if provided
        return "cpu"
    return "cpu"


def _bg_var_ratio(
    pd_map: np.ndarray | None,
    pd_base: np.ndarray | None,
    mask_bg: np.ndarray | None,
) -> tuple[float | None, float | None, float | None]:
    if pd_map is None or pd_base is None or mask_bg is None:
        return None, None, None
    mask = mask_bg.astype(bool, copy=False)
    if not mask.any():
        return None, None, None
    bg_base = pd_base[mask]
    bg_map = pd_map[mask]
    var_base = float(bg_base.var())
    var_map = float(bg_map.var())
    if var_base <= 0.0:
        return None, var_base, var_map
    ratio = float(var_map / max(var_base, 1e-12))
    return ratio, var_base, var_map


def _pd_global_stats_dict(
    pd_base: np.ndarray | None,
    pd_stap: np.ndarray | None,
    mask_flow: np.ndarray | None,
    mask_bg: np.ndarray | None,
) -> dict[str, object]:
    stats: dict[str, object] = {
        "baseline": {"flow_mean": None, "bg_var": None},
        "stap": {"flow_mean": None, "bg_var": None},
        "flow_ratio": None,
        "bg_var_ratio": None,
    }
    if pd_base is None or pd_stap is None or mask_flow is None or mask_bg is None:
        return stats
    mask_flow = mask_flow.astype(bool, copy=False)
    mask_bg = mask_bg.astype(bool, copy=False)
    if mask_flow.any():
        flow_base = pd_base[mask_flow]
        flow_stap = pd_stap[mask_flow]
        flow_mean_base = float(flow_base.mean())
        flow_mean_stap = float(flow_stap.mean())
        stats["baseline"]["flow_mean"] = flow_mean_base
        stats["stap"]["flow_mean"] = flow_mean_stap
        if flow_mean_base != 0.0:
            stats["flow_ratio"] = float(flow_mean_stap / max(flow_mean_base, 1e-12))
    ratio_bg, var_base, var_stap = _bg_var_ratio(pd_stap, pd_base, mask_bg)
    if ratio_bg is not None:
        stats["baseline"]["bg_var"] = var_base
        stats["stap"]["bg_var"] = var_stap
        stats["bg_var_ratio"] = ratio_bg
    return stats


def _tile_iter(shape: tuple[int, int], tile_hw: tuple[int, int], stride: int):
    H, W = shape
    th, tw = tile_hw
    for y0 in range(0, H - th + 1, stride):
        for x0 in range(0, W - tw + 1, stride):
            yield y0, x0


def _tile_coverages(mask: np.ndarray, tile_hw: tuple[int, int], stride: int):
    covs: list[float] = []
    coords: list[tuple[int, int]] = []
    th, tw = tile_hw
    for y0, x0 in _tile_iter(mask.shape, tile_hw, stride):
        tile = mask[y0 : y0 + th, x0 : x0 + tw]
        covs.append(float(tile.mean()))
        coords.append((y0, x0))
    return np.asarray(covs, dtype=np.float32), coords


def _tile_count(shape: tuple[int, int], tile_hw: tuple[int, int], stride: int) -> int:
    th, tw = tile_hw
    H, W = shape
    ny = max(0, (H - th) // stride + 1)
    nx = max(0, (W - tw) // stride + 1)
    return ny * nx


def _tile_scores_to_map(
    tile_scores: np.ndarray,
    shape: tuple[int, int],
    tile_hw: tuple[int, int],
    stride: int,
) -> np.ndarray:
    """Scatter per-tile scores back to the image grid with averaging on overlaps."""
    H, W = shape
    th, tw = tile_hw
    accum = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)
    idx = 0
    for y0 in range(0, H - th + 1, stride):
        for x0 in range(0, W - tw + 1, stride):
            if idx >= tile_scores.size:
                raise RuntimeError("Insufficient tile scores for map construction.")
            val = float(tile_scores[idx])
            accum[y0 : y0 + th, x0 : x0 + tw] += val
            counts[y0 : y0 + th, x0 : x0 + tw] += 1.0
            idx += 1
    if idx != tile_scores.size:
        raise RuntimeError("Excess tile scores provided for map construction.")
    mask = counts > 0.0
    output = np.zeros_like(accum, dtype=np.float32)
    output[mask] = accum[mask] / counts[mask]
    return output


def _rank_consistency_same_class(
    base_scores: np.ndarray,
    stap_scores: np.ndarray,
    *,
    max_pairs: int = 50_000,
    seed: int = 1337,
) -> float | None:
    base = np.asarray(base_scores, dtype=np.float64).ravel()
    stap = np.asarray(stap_scores, dtype=np.float64).ravel()
    n = base.size
    if n < 2:
        return None
    total_pairs = n * (n - 1) // 2
    agree = 0
    total = 0
    rng = np.random.default_rng(seed)
    if total_pairs <= max_pairs:
        for i in range(n):
            bi = base[i]
            si = stap[i]
            for j in range(i + 1, n):
                db = bi - base[j]
                ds = si - stap[j]
                if db == 0.0 or ds == 0.0:
                    continue
                total += 1
                if db * ds >= 0.0:
                    agree += 1
    else:
        tries = max_pairs
        while tries > 0:
            i = rng.integers(0, n)
            j = rng.integers(0, n - 1)
            if j >= i:
                j += 1
            db = base[i] - base[j]
            ds = stap[i] - stap[j]
            if db == 0.0 or ds == 0.0:
                tries -= 1
                continue
            total += 1
            if db * ds >= 0.0:
                agree += 1
            tries -= 1
    if total == 0:
        return None
    return float(agree / total)


def _ka_effective_status(
    ka_tile_count: int,
    base_reasons: Sequence[str],
    info_out: dict[str, object],
) -> tuple[bool, list[str], float | None]:
    reasons = list(base_reasons)
    pf_trace_ratio: float | None = None

    def _append(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    if ka_tile_count > 0:
        snr_flow_median = info_out.get("ka_median_snr_flow_ratio")
        if snr_flow_median is not None and float(snr_flow_median) < 0.9:
            _append("c1")
        noise_perp_median = info_out.get("ka_median_noise_perp_ratio")
        if noise_perp_median is not None and (
            float(noise_perp_median) < 0.5 or float(noise_perp_median) > 1.2
        ):
            _append("c2")
        pf_loaded_median = info_out.get("ka_pf_trace_loaded_median")
        pf_sample_median = info_out.get("ka_pf_trace_sample_median")
        if (
            pf_loaded_median is not None
            and pf_sample_median not in (None, 0.0)
            and abs(float(pf_sample_median)) > 0.0
        ):
            pf_trace_ratio = float(pf_loaded_median) / float(pf_sample_median)
            if not (0.8 <= pf_trace_ratio <= 1.2):
                _append("pf_trace")
        operator_feasible = info_out.get("ka_operator_feasible")
        if operator_feasible is False:
            _append("operator")
        pf_lambda_min = info_out.get("ka_pf_lambda_min")
        if pf_lambda_min is not None and float(pf_lambda_min) < 0.95:
            _append("operator_pf")
        perp_lambda_max = info_out.get("ka_perp_lambda_max")
        if perp_lambda_max is not None and float(perp_lambda_max) > 1.10:
            _append("operator_perp")
        mix_eps = info_out.get("ka_operator_mixing_epsilon")
        if mix_eps is not None and float(mix_eps) > 0.05:
            _append("operator_mix")

    effective = bool(ka_tile_count > 0 and not reasons)
    return effective, reasons, pf_trace_ratio


def _collect_band_ratio_from_cube(
    cube_T_hw: np.ndarray,
    recorder: BandRatioRecorder,
    mask_bg: np.ndarray | None,
    tile_hw: tuple[int, int],
    stride: int,
) -> None:
    """Compute per-tile slow-time series from a cube and feed the recorder."""
    T = cube_T_hw.shape[0]
    th, tw = tile_hw
    H, W = cube_T_hw.shape[1:]
    tile_idx = 0
    for y0 in range(0, H - th + 1, stride):
        for x0 in range(0, W - tw + 1, stride):
            tile = cube_T_hw[:, y0 : y0 + th, x0 : x0 + tw]
            series = np.mean(tile.reshape(T, -1), axis=1)
            tile_is_bg = False
            if mask_bg is not None:
                bg_tile = mask_bg[y0 : y0 + th, x0 : x0 + tw]
                tile_is_bg = bool(bg_tile.all())
            recorder.observe(tile_idx, series, tile_is_bg)
            tile_idx += 1


def _mask_from_tile_thresholds(
    shape: tuple[int, int],
    tile_hw: tuple[int, int],
    coords: Sequence[tuple[int, int]],
    coverages: np.ndarray,
    threshold: float,
) -> np.ndarray:
    th, tw = tile_hw
    mask = np.zeros(shape, dtype=bool)
    for cov, (y0, x0) in zip(coverages, coords, strict=False):
        if cov >= threshold:
            mask[y0 : y0 + th, x0 : x0 + tw] = True
    return mask


def _pd_derived_flow_mask(
    pd_map: np.ndarray,
    base_mask: np.ndarray,
    *,
    quantile: float = 0.97,
    depth_min_frac: float = 0.18,
    depth_max_frac: float = 0.85,
    erode_iters: int = 1,
    dilate_iters: int = 2,
) -> np.ndarray:
    mask = np.zeros_like(pd_map, dtype=bool)
    finite = np.isfinite(pd_map)
    if finite.any():
        q = float(np.clip(quantile, 0.0, 1.0))
        thresh = float(np.quantile(pd_map[finite], q))
        mask = pd_map >= thresh
        H = pd_map.shape[0]
        y0 = int(np.clip(depth_min_frac * H, 0, H))
        y1 = int(np.clip(depth_max_frac * H, y0 + 1, H))
        depth_gate = np.zeros_like(mask, dtype=bool)
        depth_gate[y0:y1, :] = True
        mask &= depth_gate
        if erode_iters > 0:
            mask = binary_erosion(mask, iterations=erode_iters)
        if dilate_iters > 0:
            mask = binary_dilation(mask, iterations=dilate_iters)
    return np.asarray(mask | base_mask, dtype=bool)


def _pd_mask_stats(
    pd_base: np.ndarray,
    pd_map: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int | None]:
    stats: dict[str, float | int | None] = {"pixels": int(mask.sum())}
    if not mask.any():
        stats.update(
            {
                "median_base": None,
                "median_map": None,
                "ratio_median": None,
                "ratio_p10": None,
                "ratio_p90": None,
            }
        )
        return stats
    base_vals = pd_base[mask]
    map_vals = pd_map[mask]
    stats["median_base"] = float(np.median(base_vals))
    stats["median_map"] = float(np.median(map_vals))
    stats["p10_base"] = float(np.percentile(base_vals, 10.0))
    stats["p90_base"] = float(np.percentile(base_vals, 90.0))
    stats["p10_map"] = float(np.percentile(map_vals, 10.0))
    stats["p90_map"] = float(np.percentile(map_vals, 90.0))
    denom_med = max(stats["median_base"], 1e-12)
    denom_p10 = max(stats["p10_base"], 1e-12)
    denom_p90 = max(stats["p90_base"], 1e-12)
    stats["ratio_median"] = float(stats["median_map"] / denom_med)
    stats["ratio_p10"] = float(stats["p10_map"] / denom_p10)
    stats["ratio_p90"] = float(stats["p90_map"] / denom_p90)
    return stats


# Test helper: simplified background-guard application
def _apply_bg_guard_simple(
    pd_avg: np.ndarray,
    pd_base: np.ndarray,
    mask_bg: np.ndarray,
    tile_bg_var_ratios: list[float],
    target_p90: float = 1.15,
    min_alpha: float = 0.6,
    metric: str = "tile_p90",
) -> tuple[np.ndarray, dict]:
    """Apply a simplified background-variance guard to pd_avg (test utility).

    Shrinks STAP PD toward baseline on the background mask if tile p90 (or global)
    BG variance inflation exceeds the target. Mirrors the math used in _stap_pd.
    """
    assert pd_avg.shape == pd_base.shape == mask_bg.shape
    vals = [float(v) for v in tile_bg_var_ratios if np.isfinite(v)]
    p90 = float(np.quantile(vals, 0.9)) if len(vals) >= 2 else (float(vals[0]) if vals else None)
    ratio_global, _, _ = _bg_var_ratio(pd_avg, pd_base, mask_bg)
    ratio_pre = ratio_global
    if ratio_global is not None:
        sel = ratio_global
    else:
        sel = p90 if metric != "global" else None
    alpha_used: float | None = None
    if sel is not None and sel > target_p90:
        for _ in range(20):
            alpha = float(np.sqrt(target_p90 / max(sel, 1e-12)))
            alpha = float(min(1.0, max(min_alpha, alpha)))
            if alpha >= 0.999:
                break
            pd_new = pd_base[mask_bg] + alpha * (pd_avg[mask_bg] - pd_base[mask_bg])
            pd_avg = pd_avg.copy()
            pd_avg[mask_bg] = pd_new.astype(pd_avg.dtype, copy=False)
            alpha_used = alpha
            ratio_global, _, _ = _bg_var_ratio(pd_avg, pd_base, mask_bg)
            sel = ratio_global if metric == "global" or ratio_global is not None else p90
            if ratio_global is not None and ratio_global <= target_p90 * (1.0 + 1e-3):
                break
    if alpha_used is None:
        ratio_global = ratio_pre
    elif ratio_global is not None:
        for _ in range(20):
            if ratio_global <= target_p90 * (1.0 + 1e-3):
                break
            gamma = float(np.sqrt(target_p90 / max(ratio_global, 1e-12)))
            gamma = float(min(1.0, max(0.0, gamma)))
            pd_new = pd_base[mask_bg] + gamma * (pd_avg[mask_bg] - pd_base[mask_bg])
            pd_avg = pd_avg.copy()
            pd_avg[mask_bg] = pd_new.astype(pd_avg.dtype, copy=False)
            alpha_used = gamma
            ratio_global, _, _ = _bg_var_ratio(pd_avg, pd_base, mask_bg)
    info = {
        "tile_bg_var_ratio_p90": p90,
        "bg_var_ratio_pre": ratio_pre,
        "bg_var_ratio_post": ratio_global,
        "bg_guard_alpha": alpha_used,
    }
    return pd_avg, info
