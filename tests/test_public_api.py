import numpy as np
import torch

from fus_detectors import DetectorConfig, score_residual_batch, score_residual_cube
from pipeline.stap.temporal import pd_temporal_core_batched
from pipeline.stap.tiles import extract_tiles_3d, make_tile_specs, overlap_add


def _tone_cube(
    t_len: int,
    height: int,
    width: int,
    *,
    prf_hz: float,
    fd_hz: float,
    amp: float = 1.0,
) -> np.ndarray:
    t = np.arange(t_len, dtype=np.float32)
    tone = amp * np.exp(1j * 2.0 * np.pi * fd_hz * t / prf_hz).astype(np.complex64)
    cube = 0.05 * (
        np.random.randn(t_len, height, width).astype(np.float32)
        + 1j * np.random.randn(t_len, height, width).astype(np.float32)
    )
    cube[:, height // 2, width // 2] += tone
    return cube.astype(np.complex64)


def _direct_core_maps(cube: np.ndarray, *, prf_hz: float, config: DetectorConfig) -> tuple[np.ndarray, np.ndarray]:
    internal_variant = {"fixed": "unwhitened_ratio", "fully_whitened": "msd_ratio"}.get(
        str(config.variant),
        str(config.variant),
    )
    _, height, width = cube.shape
    specs = make_tile_specs(
        height,
        width,
        int(config.tile_shape[0]),
        int(config.tile_shape[1]),
        int(config.tile_stride),
        dtype=np.float32,
    )
    tiles = extract_tiles_3d(cube, specs)
    batch = np.stack(tiles, axis=0).astype(np.complex64, copy=False)
    batch_t = torch.as_tensor(batch, dtype=torch.complex64)
    band_batch, score_batch, _ = pd_temporal_core_batched(
        batch_t,
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
    band_batch = np.asarray(band_batch, dtype=np.float32)
    score_batch = np.asarray(score_batch, dtype=np.float32)
    base_batch = np.mean(np.abs(batch) ** 2, axis=1, dtype=np.float32)
    readout_batch = base_batch * band_batch
    readout_map = overlap_add(list(readout_batch), specs, height, width, dtype=np.float32)
    score_map = overlap_add(list(score_batch), specs, height, width, dtype=np.float32)
    return readout_map, score_map


def test_public_fixed_api_matches_direct_core():
    cube = _tone_cube(14, 8, 8, prf_hz=2500.0, fd_hz=350.0, amp=1.5)
    config = DetectorConfig(
        variant="fixed",
        tile_shape=(4, 4),
        tile_stride=4,
        temporal_support=6,
        diag_load=0.07,
        covariance_estimator="scm",
        huber_c=5.0,
        grid_step_rel=0.1,
        fd_span_rel=(0.2, 0.8),
        min_frequency_bins=3,
        max_frequency_bins=9,
        min_flow_hz=0.0,
        msd_lambda=0.05,
        msd_ridge=0.10,
        msd_aggregation="median",
        msd_ratio_rho=0.05,
        device="cpu",
        chunk_size=8,
    )

    result = score_residual_cube(cube, prf_hz=2500.0, config=config, return_tile_telemetry=True)
    readout_direct, score_direct = _direct_core_maps(cube, prf_hz=2500.0, config=config)

    np.testing.assert_allclose(result.readout_map, readout_direct, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(result.score_map, score_direct, rtol=1e-6, atol=1e-7)
    assert result.summary.variant == "fixed"
    assert result.summary.internal_variant == "unwhitened_ratio"
    assert result.tile_telemetry is not None
    assert len(result.tile_telemetry) > 0


def test_public_batch_api_returns_one_result_per_cube():
    cube_a = _tone_cube(12, 8, 8, prf_hz=2500.0, fd_hz=300.0, amp=1.2)
    cube_b = _tone_cube(12, 8, 8, prf_hz=2500.0, fd_hz=420.0, amp=1.0)
    batch = np.stack([cube_a, cube_b], axis=0)
    config = DetectorConfig(
        variant="whitened",
        tile_shape=(4, 4),
        tile_stride=4,
        temporal_support=6,
        diag_load=0.07,
        covariance_estimator="scm",
        huber_c=5.0,
        grid_step_rel=0.1,
        fd_span_rel=(0.2, 0.8),
        min_frequency_bins=3,
        max_frequency_bins=9,
        min_flow_hz=0.0,
        msd_lambda=0.05,
        msd_ridge=0.10,
        msd_aggregation="median",
        msd_ratio_rho=0.05,
        device="cpu",
        chunk_size=8,
    )

    results = score_residual_batch(batch, prf_hz=2500.0, config=config)

    assert len(results) == 2
    assert all(result.readout_map.shape == (8, 8) for result in results)
    assert all(result.score_map.shape == (8, 8) for result in results)
    assert all(result.summary.internal_variant == "msd_ratio" for result in results)


def test_public_api_rejects_real_input():
    cube = np.zeros((12, 8, 8), dtype=np.float32)
    try:
        score_residual_cube(cube, prf_hz=2500.0)
    except ValueError as exc:
        assert "complex-valued" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected score_residual_cube to reject real-valued input.")
