import numpy as np

from sim.kwave.common import _baseline_pd, _stap_pd


def _tone_cube(
    T: int, H: int, W: int, prf_hz: float, fd_hz: float, amp: float = 1.0
) -> np.ndarray:
    t = np.arange(T, dtype=np.float32)
    tone = amp * np.exp(1j * 2.0 * np.pi * fd_hz * t / prf_hz).astype(np.complex64)
    cube = 0.05 * (
        np.random.randn(T, H, W).astype(np.float32)
        + 1j * np.random.randn(T, H, W).astype(np.float32)
    )
    # Place the tone at the center to avoid edge-only activation
    cy, cx = H // 2, W // 2
    cube[:, cy, cx] += tone
    return cube.astype(np.complex64)


def test_pd_overlap_monotone_under_rescale():
    # Overlapping tiles: stride < tile dims
    np.random.seed(0)
    T, H, W = 32, 24, 24
    tile_hw = (8, 8)
    stride = 4  # overlapping
    prf = 3000.0
    cube = _tone_cube(T, H, W, prf_hz=prf, fd_hz=400.0, amp=1.5)
    pd_base = _baseline_pd(cube, hp_modes=1)

    pd_map, _, _ = _stap_pd(
        cube,
        tile_hw=tile_hw,
        stride=stride,
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

    # Elementwise guardrail holds even with overlap-add averaging
    assert np.all(pd_map <= pd_base + 1e-6)


def test_pd_nonoverlap_monotone_under_rescale():
    # Non-overlapping tiles: stride == tile dims
    np.random.seed(1)
    T, H, W = 32, 24, 24
    tile_hw = (8, 8)
    stride = 8  # non-overlapping
    prf = 3000.0
    cube = _tone_cube(T, H, W, prf_hz=prf, fd_hz=400.0, amp=1.5)
    pd_base = _baseline_pd(cube, hp_modes=1)

    pd_map, _, _ = _stap_pd(
        cube,
        tile_hw=tile_hw,
        stride=stride,
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

    # Elementwise guardrail holds for non-overlapping tiling as well
    assert np.all(pd_map <= pd_base + 1e-6)
