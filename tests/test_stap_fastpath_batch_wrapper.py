import numpy as np
import pytest

try:
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore


@pytest.mark.skipif(torch is None, reason="torch not available")
def test_stap_pd_tile_batch_fastpath_accepts_tensor_without_numpy_roundtrip():
    """
    Regression guard for Sprint 1:

    `_stap_pd_tile_lcmv_batch` is used by the tiled STAP pipeline. When the caller
    already provides a torch tensor (often already on GPU), we must not force a
    CPU NumPy round-trip before running the batched fast path.

    This test checks that the fast path produces identical outputs when invoked
    with a NumPy batch vs a torch batch (same values), which would not hold if
    dtype/device handling diverged.
    """
    from sim.kwave.common import _stap_pd_tile_lcmv_batch

    rng = np.random.default_rng(0)
    B, T, h, w = 3, 12, 4, 4
    prf_hz = 2500.0
    Lt = 6
    cube_np = (rng.standard_normal((B, T, h, w)) + 1j * rng.standard_normal((B, T, h, w))).astype(
        np.complex64
    )
    cube_t = torch.as_tensor(cube_np)

    kwargs = dict(
        prf_hz=prf_hz,
        diag_load=0.07,
        cov_estimator="scm",
        huber_c=5.0,
        grid_step_rel=0.1,
        fd_span_rel=(0.2, 0.8),
        min_pts=3,
        max_pts=9,
        capture_debug=False,
        device="cpu",
        ka_mode="none",
        Lt_fixed=Lt,
        enable_fast_path=True,
    )

    band_np, score_np, _, _ = _stap_pd_tile_lcmv_batch(cube_np, **kwargs)
    band_t, score_t, _, _ = _stap_pd_tile_lcmv_batch(cube_t, **kwargs)

    assert band_np.shape == (B, h, w)
    assert score_np.shape == (B, h, w)
    assert band_t.shape == (B, h, w)
    assert score_t.shape == (B, h, w)

    np.testing.assert_allclose(band_t, band_np, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(score_t, score_np, rtol=1e-6, atol=1e-7)


@pytest.mark.skipif(torch is None, reason="torch not available")
def test_fastpath_whiten_gamma_zero_matches_unwhitened_ratio():
    from sim.kwave.common import _stap_pd_tile_lcmv_batch

    rng = np.random.default_rng(1)
    B, T, h, w = 2, 14, 4, 4
    prf_hz = 2500.0
    cube_np = (rng.standard_normal((B, T, h, w)) + 1j * rng.standard_normal((B, T, h, w))).astype(
        np.complex64
    )

    kwargs = dict(
        prf_hz=prf_hz,
        diag_load=0.07,
        cov_estimator="scm",
        huber_c=5.0,
        grid_step_rel=0.1,
        fd_span_rel=(0.2, 0.8),
        min_pts=3,
        max_pts=9,
        capture_debug=False,
        device="cpu",
        ka_mode="none",
        Lt_fixed=6,
        enable_fast_path=True,
        msd_lambda=0.05,
        msd_ridge=0.10,
        msd_agg_mode="median",
        msd_ratio_rho=0.05,
    )

    band_gamma0, score_gamma0, info_gamma0, _ = _stap_pd_tile_lcmv_batch(
        cube_np,
        detector_variant="msd_ratio",
        whiten_gamma=0.0,
        **kwargs,
    )
    band_unwhitened, score_unwhitened, info_unwhitened, _ = _stap_pd_tile_lcmv_batch(
        cube_np,
        detector_variant="unwhitened_ratio",
        **kwargs,
    )

    np.testing.assert_allclose(band_gamma0, band_unwhitened, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(score_gamma0, score_unwhitened, rtol=1e-6, atol=1e-7)
    assert all(float(item["whiten_gamma"]) == pytest.approx(0.0) for item in info_gamma0)
    assert all(float(item["whiten_gamma"]) == pytest.approx(0.0) for item in info_unwhitened)


@pytest.mark.skipif(torch is None, reason="torch not available")
def test_fastpath_fractional_whitening_smoke():
    from sim.kwave.common import _stap_pd_tile_lcmv_batch

    rng = np.random.default_rng(2)
    B, T, h, w = 2, 14, 4, 4
    prf_hz = 2500.0
    cube_np = (rng.standard_normal((B, T, h, w)) + 1j * rng.standard_normal((B, T, h, w))).astype(
        np.complex64
    )

    band_mid, score_mid, info_mid, _ = _stap_pd_tile_lcmv_batch(
        cube_np,
        prf_hz=prf_hz,
        diag_load=0.07,
        cov_estimator="scm",
        huber_c=5.0,
        grid_step_rel=0.1,
        fd_span_rel=(0.2, 0.8),
        min_pts=3,
        max_pts=9,
        capture_debug=False,
        device="cpu",
        ka_mode="none",
        Lt_fixed=6,
        enable_fast_path=True,
        msd_lambda=0.05,
        msd_ridge=0.10,
        msd_agg_mode="median",
        msd_ratio_rho=0.05,
        detector_variant="msd_ratio",
        whiten_gamma=0.5,
    )

    assert band_mid.shape == (B, h, w)
    assert score_mid.shape == (B, h, w)
    assert np.isfinite(band_mid).all()
    assert np.isfinite(score_mid).all()
    assert all(float(item["whiten_gamma"]) == pytest.approx(0.5) for item in info_mid)
