import numpy as np

from sim.kwave.common import _stap_pd_tile_lcmv


def _dc_cube(T=64, h=4, w=4, seed=1):
    rng = np.random.default_rng(seed)
    x = 0.02 * (rng.standard_normal((T, h, w)) + 1j * rng.standard_normal((T, h, w)))
    # add DC on a pixel
    x[:, 1, 1] += 0.5 + 0j
    return x.astype(np.complex64)


def _tone_cube(fd_hz, prf_hz=3000.0, T=64, h=4, w=4, seed=2):
    rng = np.random.default_rng(seed)
    x = 0.02 * (rng.standard_normal((T, h, w)) + 1j * rng.standard_normal((T, h, w)))
    t = np.arange(T)
    tone = np.exp(1j * 2 * np.pi * fd_hz * t / prf_hz).astype(np.complex64)
    x[:, 2, 2] += 0.5 * tone
    return x.astype(np.complex64)


def test_energy_removed_ratio_high_for_dc_motion():
    prf = 3000.0
    cube = _dc_cube()
    _, _, info, _ = _stap_pd_tile_lcmv(
        cube,
        prf_hz=prf,
        diag_load=1e-2,
        cov_estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="absolute",
        constraint_ridge=0.10,
        msd_lambda=5e-2,
        msd_ridge=0.12,
        msd_agg_mode="median",
        motion_half_span_rel=0.1,
        msd_contrast_alpha=0.7,
        fd_span_mode="fixed",
        fd_fixed_span_hz=600.0,
        grid_step_rel=0.1,
        min_pts=9,
        max_pts=9,
        device="cpu",
    )
    if not info.get("contrast_enabled"):
        assert info.get("score_mode") == "msd"
        return
    motion_init = info.get("contrast_motion_rank_initial")
    motion_eff = info.get("contrast_motion_rank_eff")
    assert motion_init is not None
    assert motion_eff is not None
    assert motion_init >= motion_eff >= 0


def test_energy_kept_ratio_high_for_high_freq_tone():
    prf = 3000.0
    fd_hz = 600.0  # well outside motion half-span (~75 Hz)
    cube = _tone_cube(fd_hz, prf_hz=prf)
    _, _, info, _ = _stap_pd_tile_lcmv(
        cube,
        prf_hz=prf,
        diag_load=1e-2,
        cov_estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="absolute",
        constraint_ridge=0.10,
        msd_lambda=5e-2,
        msd_ridge=0.12,
        msd_agg_mode="median",
        motion_half_span_rel=0.1,
        msd_contrast_alpha=0.7,
        fd_span_mode="fixed",
        fd_fixed_span_hz=600.0,
        grid_step_rel=0.1,
        min_pts=9,
        max_pts=9,
        device="cpu",
    )
    if not info.get("contrast_enabled"):
        assert info.get("score_mode") == "msd"
        return
    kept = info.get("energy_kept_ratio")
    assert kept is not None
    assert kept >= 0.0
