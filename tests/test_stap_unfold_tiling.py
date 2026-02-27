import numpy as np
import pytest

try:
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore


@pytest.mark.skipif(torch is None, reason="torch not available")
def test_stap_pd_unfold_tiling_matches_legacy_fastpath(monkeypatch):
    """
    Sprint 2 regression guard:

    When STAP fast path is used, enabling unfold/fold tiling should preserve the
    PD and score maps produced by `_stap_pd` (within numerical tolerance).
    """
    from sim.kwave.common import _baseline_pd, _stap_pd

    rng = np.random.default_rng(0)
    T, H, W = 24, 16, 18
    prf = 2500.0
    cube = (rng.standard_normal((T, H, W)) + 1j * rng.standard_normal((T, H, W))).astype(
        np.complex64
    )
    pd_base = _baseline_pd(cube, hp_modes=1)

    mask_flow = np.zeros((H, W), dtype=bool)
    mask_flow[5:9, 6:12] = True
    mask_bg = ~mask_flow

    kwargs = dict(
        tile_hw=(4, 4),
        stride=3,
        Lt=8,
        prf_hz=prf,
        diag_load=0.07,
        estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="auto",
        mvdr_auto_kappa=50.0,
        constraint_ridge=0.10,
        fd_span_mode="psd",
        fd_span_rel=(0.30, 1.10),
        grid_step_rel=0.12,
        min_pts=3,
        max_pts=9,
        fd_min_abs_hz=0.0,
        msd_lambda=0.07,
        msd_ridge=0.10,
        msd_agg_mode="trim10",
        msd_ratio_rho=0.05,
        motion_half_span_rel=0.1,
        msd_contrast_alpha=0.6,
        debug_max_samples=0,
        stap_device="cpu",
        tile_batch=64,
        pd_base_full=pd_base,
        mask_flow=mask_flow,
        mask_bg=mask_bg,
        conditional_enable=False,
        ka_mode="none",
        ka_prior_library=None,
        ka_opts=None,
        alias_psd_select_enable=False,
        psd_telemetry=False,
        feasibility_mode="legacy",
        band_ratio_spec=None,
        band_ratio_recorder=None,
    )

    monkeypatch.setenv("STAP_FAST_PATH", "1")

    # Legacy fast-path tiling (NumPy raster + batching).
    monkeypatch.delenv("STAP_TILING_UNFOLD", raising=False)
    pd_legacy, score_legacy, info_legacy = _stap_pd(cube, **kwargs)

    # Unfold/fold tiling (new path).
    monkeypatch.setenv("STAP_TILING_UNFOLD", "1")
    pd_unfold, score_unfold, info_unfold = _stap_pd(cube, **kwargs)

    np.testing.assert_allclose(pd_unfold, pd_legacy, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(score_unfold, score_legacy, rtol=2e-5, atol=2e-6)

    assert bool(info_unfold.get("stap_unfold_tiling_used")) is True
    assert bool(info_legacy.get("stap_unfold_tiling_used")) is False

    assert int(info_unfold["total_tiles"]) == int(info_legacy["total_tiles"])
    assert int(info_unfold.get("stap_tiles_skipped_flow0", 0)) == int(
        info_legacy.get("stap_tiles_skipped_flow0", 0)
    )
