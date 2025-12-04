import numpy as np

from sim.kwave.common import _stap_pd_tile_lcmv


def _random_cube(seed: int = 0):
    rng = np.random.default_rng(seed)
    T, h, w = 32, 4, 4
    return (rng.standard_normal((T, h, w)) + 1j * rng.standard_normal((T, h, w))).astype(
        np.complex64
    )


def test_msd_ratio_ridge_tames_extreme_scores():
    cube = _random_cube()
    base_args = dict(
        prf_hz=3000.0,
        diag_load=1e-2,
        cov_estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="absolute",
        constraint_ridge=0.10,
        msd_lambda=5e-2,
        msd_ridge=0.12,
        msd_agg_mode="median",
        fd_span_mode="psd",
        fd_span_rel=(0.30, 0.90),
        grid_step_rel=0.12,
        min_pts=3,
        max_pts=5,
        device="cpu",
    )

    _, score_default, info_default, _ = _stap_pd_tile_lcmv(cube, msd_ratio_rho=0.0, **base_args)
    _, score_ridged, info_ridged, _ = _stap_pd_tile_lcmv(cube, msd_ratio_rho=0.2, **base_args)

    assert info_default["score_mode"] == "msd"
    assert info_ridged["score_mode"] == "msd"
    assert float(np.nanmax(score_ridged)) <= float(np.nanmax(score_default)) + 1e-6
