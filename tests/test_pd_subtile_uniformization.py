import numpy as np

from sim.kwave.common import _baseline_pd, _stap_pd


def test_subtile_background_clamping_preserves_baseline_partial_tile():
    np.random.seed(1)
    T, H, W = 32, 24, 24
    prf = 3200.0
    t = np.arange(T, dtype=np.float32)
    # Background noise + global tone
    cube = (
        0.02
        * (
            np.random.randn(T, H, W).astype(np.float32)
            + 1j * np.random.randn(T, H, W).astype(np.float32)
        )
    ).astype(np.complex64)
    tone = np.exp(1j * 2.0 * np.pi * 450.0 * t / prf).astype(np.complex64)
    cube += 0.2 * tone[:, None, None]

    pd_base = _baseline_pd(cube, hp_modes=1)

    # Define a small central flow mask to induce partial tile overlap
    mask_flow = np.zeros((H, W), dtype=bool)
    cy, cx, r = H // 2, W // 2, 4
    mask_flow[cy - r : cy + r, cx - r : cx + r] = True
    mask_bg = ~mask_flow

    pd_map, _, info = _stap_pd(
        cube,
        tile_hw=(12, 12),
        stride=6,
        Lt=4,
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
        grid_step_rel=0.1,
        min_pts=5,
        max_pts=7,
        msd_lambda=2e-2,
        msd_ridge=0.10,
        msd_agg_mode="median",
        pd_base_full=pd_base,
        mask_flow=mask_flow,
        mask_bg=mask_bg,
        stap_device="cpu",
        debug_max_samples=1,
    )

    # Background PD must match baseline exactly (subtile clamping), so var ratio is 1
    var_ratio = float(np.var(pd_map[mask_bg]) / (np.var(pd_base[mask_bg]) + 1e-12))
    assert abs(var_ratio - 1.0) < 1e-3

    # Also verify at least one debug tile reports subtile clamping when overlapping flow
    sample = info["debug_samples"][0]
    assert sample.get("subtile_background_uniformized") is not None
