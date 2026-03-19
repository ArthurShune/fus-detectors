from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import torch

from pipeline.stap.temporal import pd_temporal_core_batched
from pipeline.stap.tiles import extract_tiles_3d, make_tile_specs, overlap_add

from .adaptive import choose_promoted_tiles, compute_guard_fraction_tiles
from .defaults import ADAPTIVE_GUARD_DEFAULTS, PUBLIC_DETECTOR_DEFAULTS

DetectorVariant = Literal["fixed", "adaptive", "whitened", "whitened_power"]

_VARIANT_ALIASES: dict[str, str] = {
    "fixed": "fixed",
    "fixed_statistic": "fixed",
    "unwhitened_ratio": "fixed",
    "raw": "fixed",
    "raw_ratio": "fixed",
    "adaptive": "adaptive",
    "adaptive_guard": "adaptive",
    "adaptive_guard_v1": "adaptive",
    "guard_promote": "adaptive",
    "whitened": "whitened",
    "fully_whitened": "whitened",
    "whitened_variant": "whitened",
    "msd_ratio": "whitened",
    "whitened_power": "whitened_power",
    "power": "whitened_power",
}

_CORE_VARIANTS: dict[str, str] = {
    "fixed": "unwhitened_ratio",
    "adaptive": "adaptive_guard",
    "whitened": "msd_ratio",
    "whitened_power": "whitened_power",
}


def _normalize_variant(variant: str) -> str:
    key = str(variant).strip().lower()
    try:
        return _VARIANT_ALIASES[key]
    except KeyError as exc:  # pragma: no cover - defensive
        supported = ", ".join(sorted(set(_VARIANT_ALIASES)))
        raise ValueError(f"Unsupported variant {variant!r}. Expected one of: {supported}.") from exc


def _as_complex64_cube(residual_cube: Any) -> np.ndarray:
    if hasattr(residual_cube, "detach") and hasattr(residual_cube, "cpu"):
        residual_cube = residual_cube.detach().cpu()
    if hasattr(residual_cube, "numpy"):
        residual_cube = residual_cube.numpy()
    cube = np.asarray(residual_cube)
    if cube.ndim != 3:
        raise ValueError(
            f"Expected residual cube with shape (T, H, W); received shape {tuple(cube.shape)}."
        )
    if not np.iscomplexobj(cube):
        raise ValueError("residual cube must be complex-valued IQ with shape (T, H, W).")
    return cube.astype(np.complex64, copy=False)


def _finite_median(values: list[float]) -> float | None:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return None
    return float(np.median(np.asarray(finite, dtype=np.float64)))


def _finite_mean(values: list[float]) -> float | None:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return None
    return float(np.mean(np.asarray(finite, dtype=np.float64)))


def _finite_quantile(values: np.ndarray, q: float) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.quantile(finite, q))


@dataclass(frozen=True)
class DetectorConfig:
    """Stable integration config for detector scoring on clutter-filtered residual cubes."""

    variant: DetectorVariant | str = "fixed"
    tile_shape: tuple[int, int] = PUBLIC_DETECTOR_DEFAULTS.tile_shape
    tile_stride: int = PUBLIC_DETECTOR_DEFAULTS.tile_stride
    temporal_support: int = PUBLIC_DETECTOR_DEFAULTS.temporal_support
    diag_load: float = PUBLIC_DETECTOR_DEFAULTS.diag_load
    covariance_estimator: str = PUBLIC_DETECTOR_DEFAULTS.covariance_estimator
    huber_c: float = PUBLIC_DETECTOR_DEFAULTS.huber_c
    grid_step_rel: float = PUBLIC_DETECTOR_DEFAULTS.grid_step_rel
    fd_span_rel: tuple[float, float] = PUBLIC_DETECTOR_DEFAULTS.fd_span_rel
    min_frequency_bins: int = PUBLIC_DETECTOR_DEFAULTS.min_frequency_bins
    max_frequency_bins: int = PUBLIC_DETECTOR_DEFAULTS.max_frequency_bins
    min_flow_hz: float = PUBLIC_DETECTOR_DEFAULTS.min_flow_hz
    msd_lambda: float | None = PUBLIC_DETECTOR_DEFAULTS.msd_lambda
    msd_ridge: float = PUBLIC_DETECTOR_DEFAULTS.msd_ridge
    msd_aggregation: str = PUBLIC_DETECTOR_DEFAULTS.msd_aggregation
    msd_ratio_rho: float = PUBLIC_DETECTOR_DEFAULTS.msd_ratio_rho
    motion_half_span_rel: float | None = PUBLIC_DETECTOR_DEFAULTS.motion_half_span_rel
    whiten_gamma: float = PUBLIC_DETECTOR_DEFAULTS.whiten_gamma
    adaptive_guard_flow_band_hz: tuple[float, float] = ADAPTIVE_GUARD_DEFAULTS.flow_band_hz
    adaptive_guard_alias_center_hz: float = ADAPTIVE_GUARD_DEFAULTS.alias_center_hz
    adaptive_guard_alias_width_hz: float = ADAPTIVE_GUARD_DEFAULTS.alias_width_hz
    adaptive_guard_tapers: int = ADAPTIVE_GUARD_DEFAULTS.tapers
    adaptive_guard_bandwidth: float = ADAPTIVE_GUARD_DEFAULTS.bandwidth
    adaptive_guard_promote_threshold: float = ADAPTIVE_GUARD_DEFAULTS.promote_threshold
    device: str | None = None
    chunk_size: int = PUBLIC_DETECTOR_DEFAULTS.chunk_size

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DetectorSummary:
    """Compact summary of one detector run for logging and reproducibility."""

    variant: str
    internal_variant: str
    total_tiles: int
    fast_path_tile_fraction: float | None
    median_band_fraction: float | None
    p90_band_fraction: float | None
    mean_score: float | None
    median_loaded_condition_number: float | None
    adaptive_guard_fraction_p90: float | None = None
    adaptive_promote_fraction: float | None = None
    adaptive_promoted_tiles: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DetectorResult:
    """Detector outputs for one clutter-filtered residual cube.

    `readout_map` is the detector-weighted PD-style readout for display and
    backwards-compatible pipeline wiring. `score_map` is the primary detector
    statistic used for thresholding and ROC-style evaluation.
    """

    config: DetectorConfig
    readout_map: np.ndarray
    score_map: np.ndarray
    summary: DetectorSummary
    tile_telemetry: list[dict[str, Any]] | None = None


def _validate_config(config: DetectorConfig, *, cube_shape: tuple[int, int, int]) -> None:
    t_len, height, width = cube_shape
    tile_h, tile_w = (int(config.tile_shape[0]), int(config.tile_shape[1]))
    if tile_h <= 0 or tile_w <= 0:
        raise ValueError(f"tile_shape must be positive; got {config.tile_shape}.")
    if tile_h > height or tile_w > width:
        raise ValueError(
            f"tile_shape {config.tile_shape} exceeds cube spatial shape {(height, width)}."
        )
    if int(config.tile_stride) <= 0:
        raise ValueError(f"tile_stride must be positive; got {config.tile_stride}.")
    if not 2 <= int(config.temporal_support) < int(t_len):
        raise ValueError(
            f"temporal_support must satisfy 2 <= Lt < T; got Lt={config.temporal_support}, T={t_len}."
        )
    if int(config.min_frequency_bins) <= 0 or int(config.max_frequency_bins) <= 0:
        raise ValueError("min_frequency_bins and max_frequency_bins must be positive.")
    if int(config.min_frequency_bins) > int(config.max_frequency_bins):
        raise ValueError(
            f"min_frequency_bins must be <= max_frequency_bins; got "
            f"{config.min_frequency_bins} > {config.max_frequency_bins}."
        )
    if int(config.chunk_size) <= 0:
        raise ValueError(f"chunk_size must be positive; got {config.chunk_size}.")
    if float(config.adaptive_guard_alias_width_hz) < 0.0:
        raise ValueError("adaptive_guard_alias_width_hz must be non-negative.")
    if int(config.adaptive_guard_tapers) <= 0:
        raise ValueError("adaptive_guard_tapers must be positive.")
    if float(config.adaptive_guard_bandwidth) < 1.0:
        raise ValueError("adaptive_guard_bandwidth must be >= 1.0.")


def _score_tile_batch(
    tile_batch: np.ndarray,
    *,
    prf_hz: float,
    config: DetectorConfig,
    internal_variant: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    tile_batch_t = torch.as_tensor(tile_batch, dtype=torch.complex64)
    band_batch, score_batch, info_batch = pd_temporal_core_batched(
        tile_batch_t,
        prf_hz=float(prf_hz),
        Lt=int(config.temporal_support),
        diag_load=float(config.diag_load),
        cov_train_trim_q=0.0,
        kappa_shrink=200.0,
        kappa_msd=200.0,
        cov_estimator=str(config.covariance_estimator),
        huber_c=float(config.huber_c),
        grid_step_rel=float(config.grid_step_rel),
        fd_span_rel=(float(config.fd_span_rel[0]), float(config.fd_span_rel[1])),
        min_pts=int(config.min_frequency_bins),
        max_pts=int(config.max_frequency_bins),
        fd_min_abs_hz=float(config.min_flow_hz),
        motion_half_span_rel=config.motion_half_span_rel,
        msd_ridge=float(config.msd_ridge),
        msd_agg_mode=str(config.msd_aggregation),
        msd_ratio_rho=float(config.msd_ratio_rho),
        msd_contrast_alpha=0.0,
        msd_lambda=config.msd_lambda,
        detector_variant=internal_variant,
        whiten_gamma=float(config.whiten_gamma),
        device=config.device,
        use_ref_cov=False,
        fd_span_mode="psd",
        flow_band_hz=None,
        return_torch=False,
    )
    return (
        np.asarray(band_batch, dtype=np.float32),
        np.asarray(score_batch, dtype=np.float32),
        [dict(info or {}) for info in info_batch],
    )


def _build_summary(
    tile_infos: list[dict[str, Any]],
    *,
    variant: str,
    internal_variant: str,
    guard_fraction_tiles: np.ndarray | None = None,
    promote_tiles: np.ndarray | None = None,
) -> DetectorSummary:
    return DetectorSummary(
        variant=str(variant),
        internal_variant=str(internal_variant),
        total_tiles=len(tile_infos),
        fast_path_tile_fraction=_finite_mean(
            [1.0 if bool(info.get("stap_fast_path_used")) else 0.0 for info in tile_infos]
        ),
        median_band_fraction=_finite_median(
            [float(info.get("band_fraction_median", np.nan)) for info in tile_infos]
        ),
        p90_band_fraction=_finite_median(
            [float(info.get("band_fraction_p90", np.nan)) for info in tile_infos]
        ),
        mean_score=_finite_mean([float(info.get("score_mean", np.nan)) for info in tile_infos]),
        median_loaded_condition_number=_finite_median(
            [float(info.get("cond_loaded", np.nan)) for info in tile_infos]
        ),
        adaptive_guard_fraction_p90=(
            _finite_quantile(np.asarray(guard_fraction_tiles, dtype=np.float32), 0.90)
            if guard_fraction_tiles is not None
            else None
        ),
        adaptive_promote_fraction=(
            float(np.mean(np.asarray(promote_tiles, dtype=np.float32)))
            if promote_tiles is not None and np.asarray(promote_tiles).size > 0
            else None
        ),
        adaptive_promoted_tiles=(
            int(np.count_nonzero(np.asarray(promote_tiles, dtype=bool)))
            if promote_tiles is not None
            else None
        ),
    )


def _score_nonadaptive(
    tile_cubes: list[np.ndarray],
    *,
    prf_hz: float,
    config: DetectorConfig,
    internal_variant: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    readout_tiles: list[np.ndarray] = []
    score_tiles: list[np.ndarray] = []
    tile_infos: list[dict[str, Any]] = []
    chunk = int(config.chunk_size)

    for start in range(0, len(tile_cubes), chunk):
        batch = np.stack(tile_cubes[start : start + chunk], axis=0).astype(np.complex64, copy=False)
        band_batch, score_batch, info_batch = _score_tile_batch(
            batch,
            prf_hz=float(prf_hz),
            config=config,
            internal_variant=internal_variant,
        )
        base_batch = np.mean(np.abs(batch) ** 2, axis=1, dtype=np.float32)
        readout_tiles.extend(list(base_batch * band_batch))
        score_tiles.extend(list(score_batch))
        tile_infos.extend(info_batch)
    return readout_tiles, score_tiles, tile_infos


def _score_adaptive(
    tile_cubes: list[np.ndarray],
    *,
    prf_hz: float,
    config: DetectorConfig,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[dict[str, Any]],
    np.ndarray,
    np.ndarray,
]:
    readout_tiles: list[np.ndarray] = []
    score_tiles: list[np.ndarray] = []
    tile_infos: list[dict[str, Any]] = []
    guard_chunks: list[np.ndarray] = []
    chunk = int(config.chunk_size)

    for start in range(0, len(tile_cubes), chunk):
        batch = np.stack(tile_cubes[start : start + chunk], axis=0).astype(np.complex64, copy=False)
        band_batch, score_batch, info_batch = _score_tile_batch(
            batch,
            prf_hz=float(prf_hz),
            config=config,
            internal_variant="unwhitened_ratio",
        )
        base_batch = np.mean(np.abs(batch) ** 2, axis=1, dtype=np.float32)
        guard_fraction = compute_guard_fraction_tiles(
            batch,
            prf_hz=float(prf_hz),
            flow_band_hz=tuple(float(v) for v in config.adaptive_guard_flow_band_hz),
            alias_center_hz=float(config.adaptive_guard_alias_center_hz),
            alias_width_hz=float(config.adaptive_guard_alias_width_hz),
            tapers=int(config.adaptive_guard_tapers),
            bandwidth=float(config.adaptive_guard_bandwidth),
        )
        readout_tiles.extend(list(base_batch * band_batch))
        score_tiles.extend(list(score_batch))
        guard_chunks.append(guard_fraction)
        for idx, info in enumerate(info_batch):
            info["adaptive_guard_fraction"] = float(guard_fraction[idx])
            tile_infos.append(info)

    guard_fraction_tiles = np.concatenate(guard_chunks, axis=0) if guard_chunks else np.zeros((0,), dtype=np.float32)
    promote_tiles = choose_promoted_tiles(
        guard_fraction_tiles,
        threshold=float(config.adaptive_guard_promote_threshold),
    )
    promoted_indices = np.flatnonzero(promote_tiles)

    for start in range(0, int(promoted_indices.size), chunk):
        idxs = promoted_indices[start : start + chunk]
        batch = np.stack([tile_cubes[int(idx)] for idx in idxs], axis=0).astype(np.complex64, copy=False)
        band_batch, score_batch, info_batch = _score_tile_batch(
            batch,
            prf_hz=float(prf_hz),
            config=config,
            internal_variant="msd_ratio",
        )
        base_batch = np.mean(np.abs(batch) ** 2, axis=1, dtype=np.float32)
        readout_batch = base_batch * band_batch
        for local_idx, tile_idx in enumerate(idxs.tolist()):
            readout_tiles[tile_idx] = readout_batch[local_idx]
            score_tiles[tile_idx] = score_batch[local_idx]
            promoted_info = dict(info_batch[local_idx] or {})
            promoted_info["adaptive_guard_fraction"] = float(guard_fraction_tiles[tile_idx])
            tile_infos[tile_idx] = promoted_info

    for tile_idx, info in enumerate(tile_infos):
        info["adaptive_promoted"] = bool(promote_tiles[tile_idx])
        info["adaptive_branch"] = "whitened" if bool(promote_tiles[tile_idx]) else "fixed"
        info["adaptive_guard_threshold"] = float(config.adaptive_guard_promote_threshold)

    return readout_tiles, score_tiles, tile_infos, guard_fraction_tiles, promote_tiles


def score_residual_cube(
    residual_cube: Any,
    *,
    prf_hz: float,
    config: DetectorConfig | None = None,
    return_tile_telemetry: bool = False,
) -> DetectorResult:
    """Score one clutter-filtered residual cube with the stable public API."""

    cfg = config or DetectorConfig()
    cube = _as_complex64_cube(residual_cube)
    _validate_config(cfg, cube_shape=tuple(int(x) for x in cube.shape))
    variant = _normalize_variant(cfg.variant)
    internal_variant = _CORE_VARIANTS[variant]

    _, height, width = cube.shape
    tile_h, tile_w = (int(cfg.tile_shape[0]), int(cfg.tile_shape[1]))
    specs = make_tile_specs(height, width, tile_h, tile_w, int(cfg.tile_stride), dtype=np.float32)
    tile_cubes = extract_tiles_3d(cube, specs)

    guard_fraction_tiles: np.ndarray | None = None
    promote_tiles: np.ndarray | None = None
    if variant == "adaptive":
        readout_tiles, score_tiles, tile_infos, guard_fraction_tiles, promote_tiles = _score_adaptive(
            tile_cubes,
            prf_hz=float(prf_hz),
            config=cfg,
        )
    else:
        readout_tiles, score_tiles, tile_infos = _score_nonadaptive(
            tile_cubes,
            prf_hz=float(prf_hz),
            config=cfg,
            internal_variant=internal_variant,
        )

    readout_map = overlap_add(readout_tiles, specs, height, width, dtype=np.float32)
    score_map = overlap_add(score_tiles, specs, height, width, dtype=np.float32)

    summary = _build_summary(
        tile_infos,
        variant=variant,
        internal_variant=internal_variant,
        guard_fraction_tiles=guard_fraction_tiles,
        promote_tiles=promote_tiles,
    )

    return DetectorResult(
        config=cfg,
        readout_map=readout_map,
        score_map=score_map,
        summary=summary,
        tile_telemetry=tile_infos if return_tile_telemetry else None,
    )


def score_residual_batch(
    residual_batch: Any,
    *,
    prf_hz: float,
    config: DetectorConfig | None = None,
    return_tile_telemetry: bool = False,
) -> list[DetectorResult]:
    """Score a batch of clutter-filtered residual cubes."""

    if hasattr(residual_batch, "detach") and hasattr(residual_batch, "cpu"):
        residual_batch = residual_batch.detach().cpu()
    if hasattr(residual_batch, "numpy"):
        residual_batch = residual_batch.numpy()
    batch = np.asarray(residual_batch)
    if batch.ndim != 4:
        raise ValueError(
            f"Expected residual batch with shape (B, T, H, W); received shape {tuple(batch.shape)}."
        )
    return [
        score_residual_cube(
            batch[idx],
            prf_hz=float(prf_hz),
            config=config,
            return_tile_telemetry=return_tile_telemetry,
        )
        for idx in range(int(batch.shape[0]))
    ]
