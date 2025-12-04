# pipeline/sim/synthetic.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

_HETEROGENEITY_LEVELS = {
    "low": 0.08,
    "medium": 0.18,
    "med": 0.18,
    "high": 0.32,
}


@dataclass(frozen=True)
class SimConfig:
    """
    Configuration parameters for synthetic score + PD simulations.

    These knobs heuristically mimic the impact of motion, heterogeneity,
    sampling cadence, and hardware faults on baseline vs STAP performance.
    """

    motion_amp_um: float = 30.0
    motion_freq_hz: float = 0.5
    prf_hz: float = 3000.0
    prf_jitter_pct: float = 0.0
    heterogeneity: str = "medium"
    skull_attenuation_db: float = 0.0
    scatter_density: float = 1.0
    sensor_dropout: float = 0.0
    angle_count: int = 9
    tile: Tuple[int, int] = (8, 8)
    tile_stride: int = 4
    seed: int = 0
    enable_motion_comp: bool = False
    doppler_grid: Tuple[float, ...] = (0.0,)
    steer_fuse: str = "max"
    angle_grouping: str = "none"
    steer_mode: str = "bank"
    tbd_enable: bool = False


def _heavy_tail_samples(
    rng: np.random.Generator,
    size: int,
    mean: float,
    std: float,
    tail_prob: float,
    tail_scale: float,
    tail_shape: float = 4.0,
) -> np.ndarray:
    """Gaussian body with Pareto tail for heavy contamination."""
    samples = rng.normal(loc=mean, scale=std, size=size)
    mask = rng.random(size) < np.clip(tail_prob, 0.0, 0.95)
    if np.any(mask):
        tail = (rng.pareto(tail_shape, mask.sum()) + 1.0) * tail_scale
        samples[mask] = mean + tail
    return samples


def _apply_dropout_to_maps(
    pd_base: np.ndarray,
    pd_stap: np.ndarray,
    rng: np.random.Generator,
    dropout_frac: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly attenuate columns/rows to mimic sensor dropout."""
    if dropout_frac <= 0.0:
        return pd_base, pd_stap, np.zeros(pd_base.shape[1], dtype=bool)
    W = pd_base.shape[1]
    drop_cols = rng.random(W) < np.clip(dropout_frac, 0.0, 0.9)
    if np.any(drop_cols):
        pd_base = pd_base.copy()
        pd_stap = pd_stap.copy()
        pd_base[:, drop_cols] *= 0.4
        pd_stap[:, drop_cols] *= 0.6
    return pd_base, pd_stap, drop_cols


def simulate_scores_and_pd(
    n_pos: int,
    n_neg: int,
    height: int,
    width: int,
    config: SimConfig,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Generate synthetic detector scores and PD maps for baseline vs STAP.

    Parameters
    ----------
    n_pos, n_neg : int
        Number of flow / null samples for score distributions.
    height, width : int
        Spatial dimensions for PD maps.
    config : SimConfig
        Scenario knobs controlling motion, heterogeneity, dropout, etc.
    rng : Optional[np.random.Generator]
        Optional RNG to reuse across calls; when None, derived from config.seed.

    Returns
    -------
    Tuple of (baseline_pos_scores, baseline_null_scores,
              stap_pos_scores, stap_null_scores,
              pd_map_baseline, pd_map_stap,
              mask_flow, mask_bg)
    """
    cfg = config
    rng = np.random.default_rng(cfg.seed) if rng is None else rng

    motion_norm = np.clip(cfg.motion_amp_um / 120.0, 0.0, 2.0)
    freq_norm = np.clip(cfg.motion_freq_hz / 1.2, 0.0, 2.5)
    dropout = np.clip(cfg.sensor_dropout, 0.0, 0.9)
    prf_norm = np.clip(3000.0 / max(cfg.prf_hz, 1.0), 0.25, 3.0)
    jitter = np.clip(cfg.prf_jitter_pct / 100.0, 0.0, 0.6)
    hetero_scale = _HETEROGENEITY_LEVELS.get(
        cfg.heterogeneity.lower(), _HETEROGENEITY_LEVELS["medium"]
    )
    scatter = np.clip(cfg.scatter_density, 0.3, 3.5)
    skull_factor = 10 ** (-cfg.skull_attenuation_db / 20.0)

    motion_penalty = motion_norm
    if cfg.enable_motion_comp:
        motion_penalty *= 0.5

    dropout_eff = dropout
    if cfg.angle_grouping.lower() != "none":
        dropout_eff *= 0.6

    doppler_grid = cfg.doppler_grid if len(cfg.doppler_grid) > 0 else (0.0,)
    steer_bonus = 0.0
    stap_std_scale = 1.0
    tail_prob_scale = 1.0
    tail_scale_scale = 1.0
    if len(doppler_grid) > 1:
        steer_bonus = 0.15 * motion_norm + 0.05 * (len(doppler_grid) - 1)
        dropout_eff *= 0.9
        stap_std_scale *= 0.9

    steer_mode = cfg.steer_mode.lower()
    if steer_mode == "lcmv":
        steer_bonus += 0.12 * motion_norm + 0.03 * max(len(doppler_grid) - 1, 0)
        stap_std_scale *= 0.9
        dropout_eff *= 0.92
        tail_prob_scale *= 0.9
        tail_scale_scale *= 0.92

    # --- Scores (positives)
    base_pos_mean = np.clip(
        1.6 - 0.75 * motion_penalty + 0.12 * (1.0 - dropout) + 0.08 * (1.0 - prf_norm),
        0.4,
        2.0,
    )
    base_pos_std = np.clip(0.8 + 0.25 * motion_penalty + 0.2 * jitter, 0.4, 1.6)
    stap_pos_mean = np.clip(
        2.3
        - 0.35 * motion_penalty
        + 0.2 * (1.0 - dropout_eff)
        + 0.15 * (prf_norm**-0.4 - 1.0)
        + steer_bonus,
        1.0,
        3.8,
    )
    stap_pos_std = np.clip(0.7 + 0.18 * jitter + 0.05 * motion_penalty, 0.35, 1.4)
    stap_pos_std *= stap_std_scale

    base_pos = rng.normal(loc=base_pos_mean, scale=base_pos_std, size=n_pos)
    stap_pos = rng.normal(loc=stap_pos_mean, scale=stap_pos_std, size=n_pos)
    stap_pos = np.maximum(stap_pos, base_pos + 0.15)

    # --- Scores (null)
    base_neg = _heavy_tail_samples(
        rng=rng,
        size=n_neg,
        mean=0.0,
        std=1.0 + 0.4 * motion_penalty + 0.3 * jitter,
        tail_prob=0.08 + 0.14 * motion_penalty + 0.06 * dropout,
        tail_scale=1.6 + 2.5 * motion_penalty + 1.4 * dropout,
    )
    stap_tail_prob = 0.035 + 0.05 * dropout_eff + 0.02 * max(motion_penalty - 0.4, 0.0)
    stap_tail_scale = 1.0 + 1.4 * motion_penalty + 0.9 * dropout_eff
    stap_tail_prob *= tail_prob_scale
    stap_tail_scale *= tail_scale_scale

    if cfg.tbd_enable:
        stap_tail_prob *= 0.75
        stap_tail_scale *= 0.85

    if len(doppler_grid) > 1:
        stap_tail_prob *= 0.7
        stap_tail_scale *= 0.85

    stap_neg = _heavy_tail_samples(
        rng=rng,
        size=n_neg,
        mean=0.0,
        std=0.85 + 0.25 * jitter + 0.15 * dropout_eff,
        tail_prob=stap_tail_prob,
        tail_scale=stap_tail_scale,
    )

    # Sensor dropout introduces extra clutter for baseline
    if dropout > 0.0:
        clutter = rng.gamma(shape=1.2 + 1.5 * dropout, scale=0.8 + 0.4 * dropout, size=n_neg)
        base_neg = base_neg + clutter
        stap_neg = stap_neg + 0.5 * clutter * (0.7 if len(doppler_grid) > 1 else 1.0)
        base_pos *= 1.0 - 0.18 * dropout
        stap_pos *= 1.0 - 0.1 * dropout_eff

    base_pos = np.clip(base_pos, 0.05, None).astype(np.float32)
    stap_pos = np.clip(stap_pos, 0.1, None).astype(np.float32)
    base_neg = np.clip(base_neg, -2.5, None).astype(np.float32)
    stap_neg = np.clip(stap_neg, -2.5, None).astype(np.float32)

    # --- PD maps
    pd_base = rng.gamma(
        shape=1.4 + 0.6 * scatter,
        scale=0.9 + 0.25 * motion_norm,
        size=(height, width),
    )
    pd_base *= skull_factor
    pd_base += rng.standard_normal((height, width)) * (hetero_scale + 0.05)

    yy, xx = np.ogrid[:height, :width]
    cy = height // 2
    cx = width // 2
    offset = int(np.clip(cfg.motion_amp_um / 6.0, -width // 5, width // 5))
    cx_flow = np.clip(cx + offset, width // 4, width - width // 4)
    ry = max(height // 5, 4)
    rx = max(width // 6, 4)
    mask_flow = ((yy - cy) ** 2) / (ry**2) + ((xx - cx_flow) ** 2) / (rx**2) <= 1.0
    mask_bg = ~mask_flow

    motion_wave = np.sin(2.0 * np.pi * freq_norm * (xx / max(width - 1, 1))) * (
        0.12 * motion_penalty
    )
    pd_base += motion_wave
    if cfg.prf_jitter_pct > 0.0:
        jitter_wave = np.sin(2.0 * np.pi * rng.random() + 2.0 * np.pi * yy / max(height - 1, 1))
        pd_base += jitter_wave * (0.08 + 0.35 * jitter)

    # Flow boost for STAP
    flow_gain = 1.6 + 0.3 * (1.0 - dropout) - 0.25 * motion_norm
    stap_gain = flow_gain * mask_flow.astype(np.float32)
    hetero_noise = rng.standard_normal((height, width)) * (0.04 + 0.25 * hetero_scale)
    pd_stap = pd_base * (0.78 - 0.05 * dropout) + stap_gain + hetero_noise
    pd_base = np.clip(pd_base, 0.0, None)
    pd_stap = np.clip(pd_stap, 0.0, None)

    # Inject scatterer hot-spots
    n_hotspots = max(1, int(4 * scatter))
    for _ in range(n_hotspots):
        y = rng.integers(0, height)
        x = rng.integers(0, width)
        patch = slice(max(0, y - 1), min(height, y + 2)), slice(max(0, x - 1), min(width, x + 2))
        pd_base[patch] += 0.4 * scatter
        pd_stap[patch] += 0.6 * scatter

    pd_base, pd_stap, drop_cols = _apply_dropout_to_maps(pd_base, pd_stap, rng, dropout)
    if np.any(drop_cols):
        # dropout columns degrade flow SNR further
        stap_pos *= 1.0 - 0.05 * drop_cols.mean()
        base_pos *= 1.0 - 0.1 * drop_cols.mean()

    return (
        base_pos.astype(np.float32),
        base_neg.astype(np.float32),
        stap_pos.astype(np.float32),
        stap_neg.astype(np.float32),
        pd_base.astype(np.float32),
        pd_stap.astype(np.float32),
        mask_flow.astype(bool),
        mask_bg.astype(bool),
    )
