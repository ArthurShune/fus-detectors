import numpy as np

from sim.kwave.common import _stap_pd_tile_lcmv


def _cube(T=48, h=4, w=4, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((T, h, w)) + 1j * rng.standard_normal((T, h, w))).astype(
        np.complex64
    )


def test_kc_flow_motion_counts_sum_to_kc_and_nonzero():
    cube = _cube()
    prf = 3000.0
    # Use a fixed span so grid is deterministic
    # PRF/Lt = 750 Hz (Lt=4). step_rel=0.1 => ~75 Hz steps
    # fixed span 600 Hz => grid ~ [-600..600], min_pts=max_pts=9 => Kc=9
    band, score, info, _ = _stap_pd_tile_lcmv(
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
        msd_ratio_rho=0.0,
        motion_half_span_rel=0.1,  # ~75 Hz motion band
        msd_contrast_alpha=0.7,
        fd_span_mode="fixed",
        fd_fixed_span_hz=600.0,
        grid_step_rel=0.1,
        min_pts=9,
        max_pts=9,
        device="cpu",
    )
    assert info.get("score_mode") == "msd"
    kc = int(info["band_Kc"])
    assert kc == int(info["kc_flow"])
    assert kc == 3
