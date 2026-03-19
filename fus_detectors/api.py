from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import torch

from pipeline.stap.temporal import pd_temporal_core_batched
from pipeline.stap.tiles import extract_tiles_3d, make_tile_specs, overlap_add

DetectorVariant = Literal["fixed", "whitened", "whitened_power"]

_VARIANT_ALIASES: dict[str, str] = {
    "fixed": "unwhitened_ratio",
    "fixed_statistic": "unwhitened_ratio",
    "unwhitened_ratio": "unwhitened_ratio",
    "raw": "unwhitened_ratio",
    "raw_ratio": "unwhitened_ratio",
    "whitened": "msd_ratio",
    "fully_whitened": "msd_ratio",
    "whitened_variant": "msd_ratio",
    "msd_ratio": "msd_ratio",
    "whitened_power": "whitened_power",
    "power": "whitened_power",
}


def _normalize_variant(variant: str) -> str:
    key = str(variant).strip().lower()
    try:
        return _VARIANT_ALIASES[key]
    except KeyError as exc:  # pragma: no cover - defensive
        supported = ", ".join(sorted(set(_VARIANT_ALIASES)))
        raise ValueError(f"Unsupported variant {variant!r}. Expected one of: {supported}.") from exc


def _variant_label(internal_variant: str) -> str:
    if internal_variant == "unwhitened_ratio":
        return "fixed"
    if internal_variant == "msd_ratio":
        return "whitened"
    if internal_variant == "whitened_power":
        return "whitened_power"
    return internal_variant


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


@dataclass(frozen=True)
class DetectorConfig:
    """Public integration config for detector scoring on a clutter-filtered residual cube.

    The defaults are conservative drop-in settings for a fixed matched-subspace
    statistic. They intentionally avoid the larger script surface used for paper
    generation so external callers can depend on a narrow, documented API.
    """

    variant: DetectorVariant | str = "fixed"
    tile_shape: tuple[int, int] = (12, 12)
    tile_stride: int = 4
    temporal_support: int = 8
    diag_load: float = 1e-2
    covariance_estimator: str = "tyler_pca"
    huber_c: float = 4.5
    grid_step_rel: float = 0.12
    fd_span_rel: tuple[float, float] = (0.30, 1.10)
    min_frequency_bins: int = 3
    max_frequency_bins: int = 7
    min_flow_hz: float = 30.0
    msd_lambda: float | None = 0.05
    msd_ridge: float = 0.12
    msd_aggregation: str = "median"
    msd_ratio_rho: float = 0.05
    motion_half_span_rel: float | None = None
    whiten_gamma: float = 1.0
    device: str | None = None
    chunk_size: int = 128

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DetectorResult:
    """Detector outputs for one clutter-filtered residual cube.

    `readout_map` is the PD-style map after applying the detector's band-limited
    weighting, which is useful for display and backwards-compatible pipeline
    wiring. `score_map` is the primary right-tail detector statistic used for
    thresholding and ROC-style evaluation.
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
        list(info_batch),
    )


def score_residual_cube(
    residual_cube: Any,
    *,
    prf_hz: float,
    config: DetectorConfig | None = None,
    return_tile_telemetry: bool = False,
) -> DetectorResult:
    """Score one clutter-filtered residual cube with a stable public detector API.

    Parameters
    ----------
    residual_cube:
        Complex-valued clutter-filtered slow-time cube with shape `(T, H, W)`.
    prf_hz:
        Slow-time pulse repetition frequency for the residual cube.
    config:
        Public detector configuration. When omitted, uses the fixed matched-subspace
        default.
    return_tile_telemetry:
        When true, include per-tile telemetry dictionaries in the result.
    """

    cfg = config or DetectorConfig()
    cube = _as_complex64_cube(residual_cube)
    _validate_config(cfg, cube_shape=tuple(int(x) for x in cube.shape))
    internal_variant = _normalize_variant(cfg.variant)

    _, height, width = cube.shape
    tile_h, tile_w = (int(cfg.tile_shape[0]), int(cfg.tile_shape[1]))
    specs = make_tile_specs(height, width, tile_h, tile_w, int(cfg.tile_stride), dtype=np.float32)
    tile_cubes = extract_tiles_3d(cube, specs)

    readout_tiles: list[np.ndarray] = []
    score_tiles: list[np.ndarray] = []
    tile_infos: list[dict[str, Any]] = []
    chunk = int(cfg.chunk_size)

    for start in range(0, len(tile_cubes), chunk):
        batch = np.stack(tile_cubes[start : start + chunk], axis=0).astype(np.complex64, copy=False)
        band_batch, score_batch, info_batch = _score_tile_batch(
            batch,
            prf_hz=float(prf_hz),
            config=cfg,
            internal_variant=internal_variant,
        )
        base_batch = np.mean(np.abs(batch) ** 2, axis=1, dtype=np.float32)
        readout_batch = base_batch * band_batch
        readout_tiles.extend(readout_batch)
        score_tiles.extend(score_batch)
        tile_infos.extend(info_batch)

    readout_map = overlap_add(readout_tiles, specs, height, width, dtype=np.float32)
    score_map = overlap_add(score_tiles, specs, height, width, dtype=np.float32)

    summary = DetectorSummary(
        variant=_variant_label(internal_variant),
        internal_variant=internal_variant,
        total_tiles=len(specs),
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
    """Score a batch of clutter-filtered residual cubes.

    The public batch API deliberately returns one `DetectorResult` per cube so
    callers can log, persist, or post-process each run independently.
    """

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
