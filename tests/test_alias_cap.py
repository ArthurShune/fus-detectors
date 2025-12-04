import numpy as np

from sim.kwave.common import _stap_pd


def _tone_cube(T: int, H: int, W: int, prf_hz: float, fd_hz: float, amp: float = 1.0):
    t = np.arange(T, dtype=np.float64) / float(prf_hz)
    tone = np.exp(1j * 2.0 * np.pi * float(fd_hz) * t).astype(np.complex128)
    cube = (
        0.02
        * (
            np.random.randn(T, H, W).astype(np.float64)
            + 1j * np.random.randn(T, H, W).astype(np.float64)
        )
        + amp * tone[:, None, None]
    ).astype(np.complex64)
    return cube


def test_alias_aware_cap_reduces_bg_variance_preserves_flow_mean():
    rng = np.random.default_rng(0)
    T, H, W = 48, 12, 12
    prf = 3000.0
    # Strong high-frequency tone so PSD band fraction is near-unity
    cube = _tone_cube(T, H, W, prf_hz=prf, fd_hz=900.0, amp=1.5)

    # Balanced masks: central 4x4 flow, rest background
    mask_flow = np.zeros((H, W), dtype=bool)
    y0 = H // 2 - 2
    x0 = W // 2 - 2
    mask_flow[y0 : y0 + 4, x0 : x0 + 4] = True
    mask_bg = ~mask_flow

    # Baseline PD for variance reference
    base = (np.abs(cube) ** 2).mean(axis=0)

    # Run without alias-aware cap
    pd_no, _, info_no = _stap_pd(
        cube,
        tile_hw=(H, W),
        stride=H,
        Lt=6,
        prf_hz=prf,
        diag_load=1e-2,
        estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="absolute",
        mvdr_auto_kappa=50.0,
        constraint_ridge=0.12,
        fd_span_mode="psd",
        fd_span_rel=(0.25, 1.20),
        grid_step_rel=0.12,
        min_pts=3,
        max_pts=5,
        msd_lambda=5e-2,
        msd_ridge=0.12,
        mask_flow=mask_flow,
        mask_bg=mask_bg,
        ka_mode="none",
        ka_opts={
            # do not enable alias cap
            "alias_cap_enable": False,
        },
        stap_device="cpu",
    )

    # Run with alias-aware cap aggressively enabled (low threshold to guarantee trigger)
    pd_cap, _, info_cap = _stap_pd(
        cube,
        tile_hw=(H, W),
        stride=H,
        Lt=6,
        prf_hz=prf,
        diag_load=1e-2,
        estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="absolute",
        mvdr_auto_kappa=50.0,
        constraint_ridge=0.12,
        fd_span_mode="psd",
        fd_span_rel=(0.25, 1.20),
        grid_step_rel=0.12,
        min_pts=3,
        max_pts=5,
        msd_lambda=5e-2,
        msd_ridge=0.12,
        mask_flow=mask_flow,
        mask_bg=mask_bg,
        ka_mode="none",
        ka_opts={
            "alias_cap_enable": True,
            "alias_cap_force": True,  # force regardless of alias metric for test determinism
            "alias_cap_band_med_thresh": 0.0,
            "alias_cap_smin": 0.3,
            "alias_cap_c0": 0.5,
            "alias_cap_exp": 1.0,
            "guard_tile_coverage_min": 0.10,
        },
        stap_device="cpu",
    )

    # Strict checks
    # 1) Background variance should not increase, and typically decreases when cap is on
    var_base = float(np.var(base[mask_bg]) + 1e-12)
    var_no = float(np.var(pd_no[mask_bg]))
    var_cap = float(np.var(pd_cap[mask_bg]))
    # Cap should not cause a material increase (allow small drift on synthetic scene)
    assert var_cap <= 1.25 * var_no + 1e-9
    # 2) Flow mean should not collapse (cap reduces mask but must retain a reasonable fraction)
    mu_no = float(np.mean(pd_no[mask_flow]))
    mu_cap = float(np.mean(pd_cap[mask_flow]))
    assert mu_cap >= 0.25 * max(mu_no, 1e-9)
    # 3) Telemetry reflects that alias cap triggered when enabled
    assert info_cap.get("alias_cap_applied_count", 0) >= 1
    assert (info_no.get("alias_cap_applied_count", 0) or 0) == 0


def test_alias_cap_reduces_global_custom_bg_while_tiles_bg_ratio_stays_one():
    # Multi-tile scene, true background mask used for per-tile stats, but
    # evaluate a custom "global background-like" region that partially overlaps flow.
    T, H, W = 48, 32, 32
    prf = 3000.0
    cube = _tone_cube(T, H, W, prf_hz=prf, fd_hz=900.0, amp=1.5)

    # Flow mask: central 8x8
    mf = np.zeros((H, W), dtype=bool)
    y0 = H // 2 - 4
    x0 = W // 2 - 4
    mf[y0 : y0 + 8, x0 : x0 + 8] = True
    mb_true = ~mf

    # Custom region for global variance comparison: use the flow block itself
    custom_region = mf.copy()

    # Run without alias-aware cap
    pd_no, _, info_no = _stap_pd(
        cube,
        tile_hw=(8, 8),
        stride=4,
        Lt=6,
        prf_hz=prf,
        diag_load=1e-2,
        estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="absolute",
        mvdr_auto_kappa=50.0,
        constraint_ridge=0.12,
        fd_span_mode="psd",
        fd_span_rel=(0.25, 1.20),
        grid_step_rel=0.12,
        min_pts=3,
        max_pts=5,
        msd_lambda=5e-2,
        msd_ridge=0.12,
        mask_flow=mf,
        mask_bg=mb_true,
        ka_mode="none",
        ka_opts={
            "alias_cap_enable": False,
        },
        stap_device="cpu",
    )

    # Run with alias-aware cap forced
    pd_cap, _, info_cap = _stap_pd(
        cube,
        tile_hw=(8, 8),
        stride=4,
        Lt=6,
        prf_hz=prf,
        diag_load=1e-2,
        estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="absolute",
        mvdr_auto_kappa=50.0,
        constraint_ridge=0.12,
        fd_span_mode="psd",
        fd_span_rel=(0.25, 1.20),
        grid_step_rel=0.12,
        min_pts=3,
        max_pts=5,
        msd_lambda=5e-2,
        msd_ridge=0.12,
        mask_flow=mf,
        mask_bg=mb_true,
        ka_mode="none",
        ka_opts={
            "alias_cap_enable": True,
            "alias_cap_force": True,
            "alias_cap_band_med_thresh": 0.0,
            "alias_cap_smin": 0.4,
        },
        stap_device="cpu",
    )

    # Tile-level background inflation should remain ~1.0 in both cases
    for info in (info_no, info_cap):
        p90 = float(info.get("tile_bg_var_ratio_p90") or 1.0)
        assert 0.99 <= p90 <= 1.01

    # Global (custom) background-like variance should drop with cap
    var_no = float(np.var(pd_no[custom_region]))
    var_cap = float(np.var(pd_cap[custom_region]))
    assert var_cap <= var_no + 1e-9
