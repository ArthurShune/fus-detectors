import numpy as np
import pytest

from sim.kwave.common import _baseline_pd, _stap_pd, _stap_pd_tile_lcmv


def _synthetic_cube(T: int, h: int, w: int, prf: float, fd: float, amp: float = 1.0) -> np.ndarray:
    t = np.arange(T, dtype=np.float32)
    tone = amp * np.exp(1j * 2.0 * np.pi * fd * t / prf).astype(np.complex64)
    cube = 0.05 * (
        np.random.randn(T, h, w).astype(np.float32)
        + 1j * np.random.randn(T, h, w).astype(np.float32)
    )
    cube[:, h // 2, w // 2] += tone
    return cube.astype(np.complex64)


def test_pd_rescale_not_exceed_baseline():
    np.random.seed(0)
    T, h, w = 32, 8, 8
    prf = 3200.0
    cube = _synthetic_cube(T, h, w, prf, fd=400.0, amp=1.5)
    pd_base = _baseline_pd(cube, hp_modes=1)

    pd_map, score_map, info = _stap_pd(
        cube,
        tile_hw=(8, 8),
        stride=8,
        Lt=3,
        prf_hz=prf,
        diag_load=1e-2,
        estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="absolute",
        mvdr_auto_kappa=40.0,
        constraint_ridge=0.12,
        fd_span_mode="fixed",
        fd_span_rel=(0.2, 0.9),
        fd_fixed_span_hz=None,
        grid_step_rel=0.08,
        min_pts=5,
        max_pts=7,
        msd_lambda=2e-2,
        msd_ridge=0.10,
        msd_agg_mode="trim10",
        pd_base_full=pd_base,
        stap_device="cpu",
    )

    assert np.all(np.isfinite(pd_map))
    assert np.all(pd_map <= pd_base + 1e-6)
    assert info["median_band_fraction_q50"] is not None
    assert score_map.shape == pd_map.shape


def test_invalid_msd_agg_falls_back_to_mean():
    np.random.seed(1)
    T, h, w = 24, 4, 4
    prf = 2800.0
    cube = _synthetic_cube(T, h, w, prf, fd=350.0)
    band_frac_tile, score_tile, info, _ = _stap_pd_tile_lcmv(
        cube,
        prf_hz=prf,
        diag_load=1e-2,
        cov_estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="absolute",
        constraint_ridge=0.1,
        msd_lambda=2e-2,
        msd_ridge=0.10,
        msd_agg_mode="invalid",
        fd_span_mode="fixed",
        fd_span_rel=(0.2, 0.8),
        fd_fixed_span_hz=None,
        grid_step_rel=0.08,
        min_pts=5,
        max_pts=7,
        capture_debug=False,
        device="cpu",
    )
    assert info["msd_agg_mode"] == "mean"
    assert np.all(np.isfinite(band_frac_tile))
    assert np.all(np.isfinite(score_tile))
