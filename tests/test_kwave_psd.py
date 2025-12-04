import numpy as np
import pytest

from sim.kwave.common import _doppler_psd_summary, _stap_pd_tile_lcmv

pytest.importorskip("torch")


def _make_coherent_cube(freq_hz: float, prf_hz: float, T: int = 32, h: int = 4, w: int = 4):
    t = np.arange(T, dtype=np.float32) / prf_hz
    carrier = np.exp(1j * 2.0 * np.pi * freq_hz * t, dtype=np.complex64)
    cube = carrier[:, None, None] * np.ones((T, h, w), dtype=np.complex64)
    noise = 0.02 * (
        np.random.default_rng(0).standard_normal(size=(T, h, w))
        + 1j * np.random.default_rng(1).standard_normal(size=(T, h, w))
    ).astype(np.complex64)
    return cube + noise


def test_doppler_psd_summary_identifies_primary_tone():
    prf_hz = 3000.0
    freq_hz = 600.0
    cube = _make_coherent_cube(freq_hz=freq_hz, prf_hz=prf_hz)

    summary = _doppler_psd_summary(cube, prf_hz, targets_hz=(0.0, freq_hz))

    assert summary["psd_power_flow_hz"] == pytest.approx(freq_hz, rel=0.01)
    assert summary["psd_power_flow"] > summary["psd_power_dc"]
    assert summary["psd_flow_to_dc_ratio"] > 1.0


def test_stap_tile_uses_psd_override_and_keeps_symmetry():
    prf_hz = 3000.0
    freq_hz = 600.0
    cube = _make_coherent_cube(freq_hz=freq_hz, prf_hz=prf_hz)

    band_frac_tile, score_tile, info, _ = _stap_pd_tile_lcmv(
        cube,
        prf_hz=prf_hz,
        diag_load=1e-2,
        cov_estimator="tyler_pca",
        huber_c=5.0,
        mvdr_load_mode="auto",
        mvdr_auto_kappa=50.0,
        constraint_ridge=0.1,
        fd_span_mode="psd",
        fd_span_rel=(0.2, 1.3),
        fd_fixed_span_hz=None,
        constraint_mode="exp+deriv",
        grid_step_rel=0.08,
        min_pts=3,
        max_pts=7,
        fd_min_abs_hz=200.0,
        msd_lambda=0.04,
        msd_ridge=0.12,
        msd_agg_mode="median",
        msd_ratio_rho=0.05,
        motion_half_span_rel=0.15,
        msd_contrast_alpha=0.7,
        capture_debug=False,
        device="cpu",
        cube_tensor=None,
        ka_mode="none",
        ka_prior_library=None,
        ka_opts=None,
        Lt_fixed=5,
    )

    assert info["fd_grid_source"] == "psd_override"
    assert info["psd_flow_freq_target"] == pytest.approx(freq_hz, rel=1e-2)
    assert info["kc_flow_freqs"] >= 2
    assert info["kc_flow"] >= 4  # exp+deriv keeps two tones plus derivatives
    assert info.get("motion_contrast_disabled") is True
    assert info["score_mode"] == "msd"
    # sanity check that energy is not zeroed-out
    assert np.all(band_frac_tile >= 0.0)
    assert np.isfinite(score_tile).all()
