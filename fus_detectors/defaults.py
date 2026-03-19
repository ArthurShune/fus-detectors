from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PublicDetectorDefaults:
    """Stable install-facing detector defaults shared across entrypoints."""

    tile_shape: tuple[int, int] = (8, 8)
    tile_stride: int = 3
    temporal_support: int = 8
    diag_load: float = 0.07
    covariance_estimator: str = "tyler_pca"
    huber_c: float = 5.0
    grid_step_rel: float = 0.20
    fd_span_rel: tuple[float, float] = (0.30, 1.10)
    min_frequency_bins: int = 9
    max_frequency_bins: int = 15
    min_flow_hz: float = 30.0
    msd_lambda: float | None = 0.05
    msd_ridge: float = 0.10
    msd_aggregation: str = "median"
    msd_ratio_rho: float = 0.05
    motion_half_span_rel: float | None = None
    whiten_gamma: float = 1.0
    chunk_size: int = 128


@dataclass(frozen=True)
class AdaptiveGuardDefaults:
    """Frozen label-free telemetry settings for the public adaptive rule."""

    flow_band_hz: tuple[float, float] = (30.0, 250.0)
    alias_center_hz: float = 575.0
    alias_width_hz: float = 175.0
    tapers: int = 3
    bandwidth: float = 2.0
    promote_threshold: float = 0.1453727245330811


@dataclass(frozen=True)
class ClinicalReplayDefaults:
    """Replay-only defaults that sit alongside the shared detector settings."""

    fd_span_mode: str = "fixed"
    fd_fixed_span_hz: float = 250.0
    constraint_mode: str = "exp+deriv"
    constraint_ridge: float = 0.18
    mvdr_load_mode: str = "auto"
    mvdr_auto_kappa: float = 120.0
    msd_contrast_alpha: float = 0.6
    band_ratio_mode: str = "whitened"
    time_window_length: int = 32
    snapshot_stride_env: str = "4"
    max_snapshots_env: str = "64"
    fast_path_env: str = "1"
    fast_pd_only_env: str = "1"


PUBLIC_DETECTOR_DEFAULTS = PublicDetectorDefaults()
ADAPTIVE_GUARD_DEFAULTS = AdaptiveGuardDefaults()
CLINICAL_REPLAY_DEFAULTS = ClinicalReplayDefaults()
