import numpy as np

from sim.kwave.common import (
    _baseline_pd,
    _ka_gate_decision,
    _stap_pd,
    _stap_pd_tile_lcmv,
)


def _print_metrics(tag: str, **metrics: float) -> None:
    parts: list[str] = []
    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            parts.append(f"{key}={float(value):.6f}")
        elif isinstance(value, (int, np.integer)):
            parts.append(f"{key}={int(value)}")
        else:
            parts.append(f"{key}={value}")
    print(f"[{tag}] " + " ".join(parts))


def _tone_cube(
    T: int, h: int, w: int, prf_hz: float, fd_hz: float, amp: float = 1.0
) -> np.ndarray:
    t = np.arange(T, dtype=np.float32)
    tone = amp * np.exp(1j * 2.0 * np.pi * fd_hz * t / prf_hz).astype(np.complex64)
    cube = 0.05 * (
        np.random.randn(T, h, w).astype(np.float32)
        + 1j * np.random.randn(T, h, w).astype(np.float32)
    )
    cube[:, h // 2, w // 2] += tone
    return cube.astype(np.complex64)


def _dual_tone_cube(
    T: int,
    h: int,
    w: int,
    prf_hz: float,
    fd_primary_hz: float,
    fd_alias_hz: float,
    amp_primary: float = 1.0,
    amp_alias: float = 0.6,
) -> np.ndarray:
    np.random.seed(123)
    t = np.arange(T, dtype=np.float32)
    tone_primary = amp_primary * np.exp(1j * 2.0 * np.pi * fd_primary_hz * t / prf_hz).astype(
        np.complex64
    )
    tone_alias = amp_alias * np.exp(1j * 2.0 * np.pi * fd_alias_hz * t / prf_hz).astype(
        np.complex64
    )
    cube = 0.05 * (
        np.random.randn(T, h, w).astype(np.float32)
        + 1j * np.random.randn(T, h, w).astype(np.float32)
    )
    cube[:, h // 2, w // 2] += tone_primary + tone_alias
    return cube.astype(np.complex64)


def test_tile_pd_highlights_signal_pixel():
    np.random.seed(0)
    T, h, w = 24, 4, 4
    prf = 3000.0
    cube = _tone_cube(T, h, w, prf, fd_hz=400.0, amp=2.0)

    band_frac_tile, score_tile, info, debug = _stap_pd_tile_lcmv(
        cube,
        prf_hz=prf,
        diag_load=1e-2,
        cov_estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="absolute",
        constraint_ridge=0.12,
        msd_lambda=2e-2,
        msd_ridge=0.10,
        fd_span_mode="fixed",
        fd_fixed_span_hz=600.0,
        grid_step_rel=0.08,
        min_pts=5,
        max_pts=5,
        capture_debug=True,
        device="cpu",
    )

    signal_val = band_frac_tile[h // 2, w // 2]
    bg_vals = np.delete(band_frac_tile.flatten(), h // 2 * w + w // 2)
    bg_p90 = float(np.percentile(bg_vals, 90))
    _print_metrics(
        "tile_signal_pixel",
        signal_val=float(signal_val),
        bg_p90=bg_p90,
        band_mean=float(np.mean(band_frac_tile)),
        band_std=float(np.std(band_frac_tile)),
        score_mean=float(np.mean(score_tile)),
        diag_load=float(info.get("diag_load") or 0.0),
        gram_diag_med=float(info.get("gram_diag_median") or 0.0),
    )
    assert signal_val >= 0.95 * np.percentile(bg_vals, 90)
    assert 0.0 <= signal_val <= 1.0
    assert "diag_load" in info and info["diag_load"] is not None
    assert "score_mean" in info and info["score_mean"] is not None
    assert "gram_diag_median" in info
    assert info["band_nan_count"] == 0 and info["band_inf_count"] == 0
    assert info["score_nan_count"] == 0 and info["score_inf_count"] == 0
    assert np.isfinite(info["band_fraction_q90"])
    assert np.isfinite(info["score_q90"])
    assert debug is not None
    assert "score_tile" in debug and "band_fraction_tile" in debug


def test_tile_fallback_on_invalid_load_mode():
    T, h, w = 16, 4, 4
    prf = 2000.0
    cube = _tone_cube(T, h, w, prf, fd_hz=250.0, amp=1.0)

    _, _, info, _ = _stap_pd_tile_lcmv(
        cube,
        prf_hz=prf,
        diag_load=1e-2,
        cov_estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="unsupported",
        constraint_ridge=0.10,
        msd_lambda=2e-2,
        msd_ridge=0.10,
        fd_span_mode="fixed",
        fd_fixed_span_hz=400.0,
        grid_step_rel=0.1,
        min_pts=5,
        max_pts=5,
        capture_debug=False,
        device="cpu",
    )
    _print_metrics(
        "tile_invalid_load_mode",
        resolved_mode=info.get("load_mode"),
        score_mode=info.get("score_mode"),
        total_tiles=int(info.get("total_tiles", 0)),
        band_nan=int(info.get("band_nan_count", 0)),
        score_nan=int(info.get("score_nan_count", 0)),
    )
    assert info["load_mode"] == "capon"
    assert info["score_mode"] in {"none", "msd"}


def test_stap_gamma_zero_matches_unwhitened_ratio(monkeypatch):
    np.random.seed(7)
    T, h, w = 24, 8, 8
    prf = 3000.0
    cube = _tone_cube(T, h, w, prf, fd_hz=400.0, amp=1.5)
    pd_base = _baseline_pd(cube, hp_modes=1)
    monkeypatch.setenv("STAP_FAST_PATH", "1")

    mask_flow = np.zeros((h, w), dtype=bool)
    mask_flow[3:5, 3:5] = True
    mask_bg = ~mask_flow

    kwargs = dict(
        tile_hw=(4, 4),
        stride=4,
        Lt=4,
        prf_hz=prf,
        diag_load=1e-2,
        estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="absolute",
        mvdr_auto_kappa=40.0,
        constraint_ridge=0.12,
        fd_span_rel=(0.2, 0.8),
        fd_fixed_span_hz=None,
        constraint_mode="exp+deriv",
        msd_lambda=2e-2,
        msd_ridge=0.10,
        msd_agg_mode="trim10",
        msd_ratio_rho=0.0,
        fd_span_mode="psd",
        grid_step_rel=0.1,
        min_pts=5,
        max_pts=7,
        pd_base_full=pd_base,
        mask_flow=mask_flow,
        mask_bg=mask_bg,
        stap_device="cpu",
    )

    band_gamma0, score_gamma0, info_gamma0 = _stap_pd(
        cube,
        detector_variant="msd_ratio",
        whiten_gamma=0.0,
        **kwargs,
    )
    band_unwhitened, score_unwhitened, info_unwhitened = _stap_pd(
        cube,
        detector_variant="unwhitened_ratio",
        whiten_gamma=1.0,
        **kwargs,
    )

    np.testing.assert_allclose(band_gamma0, band_unwhitened, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(score_gamma0, score_unwhitened, rtol=1e-6, atol=1e-7)
    assert float(info_gamma0["whiten_gamma"]) == 0.0
    assert float(info_unwhitened["whiten_gamma"]) == 0.0


def test_tile_aggregator_collects_telemetry(tmp_path):
    np.random.seed(1)
    T, H, W = 20, 8, 8
    prf = 2500.0
    cube = _tone_cube(T, H, W, prf, fd_hz=300.0, amp=1.5)
    pd_base = _baseline_pd(cube, hp_modes=1)

    mask_flow = np.zeros((H, W), dtype=bool)
    mask_flow[3:5, 3:5] = True
    mask_bg = ~mask_flow

    pd_map, score_map, info = _stap_pd(
        cube,
        tile_hw=(4, 4),
        stride=4,
        Lt=4,
        prf_hz=prf,
        diag_load=1e-2,
        estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="absolute",
        mvdr_auto_kappa=40.0,
        constraint_ridge=0.15,
        fd_span_mode="fixed",
        fd_span_rel=(0.2, 0.8),
        fd_fixed_span_hz=None,
        grid_step_rel=0.1,
        min_pts=5,
        max_pts=7,
        msd_lambda=2e-2,
        msd_ridge=0.10,
        debug_max_samples=1,
        pd_base_full=pd_base,
        mask_flow=mask_flow,
        mask_bg=mask_bg,
        stap_device="cpu",
    )

    _print_metrics(
        "tile_aggregator",
        total_tiles=int(info["total_tiles"]),
        median_flow_mu_ratio=float(info["median_flow_mu_ratio"] or 0.0),
        median_bg_var=float(info["median_bg_var_inflation"] or 0.0),
        median_diag_load=float(info["median_diag_load"] or 0.0),
        score_mode_msd=int(info["score_mode_histogram"].get("msd", 0)),
        debug_samples=len(info.get("debug_samples") or []),
    )
    assert pd_map.shape == (H, W)
    assert score_map.shape == (H, W)
    assert info["total_tiles"] > 0
    assert info["median_flow_mu_ratio"] is not None
    assert info["median_bg_var_inflation"] is not None
    hist = info["score_mode_histogram"]
    assert isinstance(hist, dict)
    assert sum(hist.values()) == info["total_tiles"]
    assert hist.get("msd", 0) > 0
    assert 1e-5 <= info["median_diag_load"] <= 1e-1
    assert np.isfinite(info["median_cond_loaded"])
    assert info["median_grid_step_hz"] is not None and info["median_grid_step_hz"] > 0.0
    assert info["median_band_fraction_q50"] is not None
    assert info["median_score_q50"] is not None
    assert info["total_band_nan"] == 0
    assert info["total_score_nan"] == 0
    debug_entries = info["debug_samples"]
    assert debug_entries
    sample = debug_entries[0]
    assert "score_tile" in sample
    assert "flow_mu_ratio" in sample
    assert "band_fraction_tile" in sample


def test_tile_alias_ratio_records_high_frequency():
    np.random.seed(4)
    T, h, w = 48, 4, 4
    prf = 2400.0
    Lt = 4
    fundamental = prf / Lt
    alias_freq = fundamental * 2.2
    cube = _dual_tone_cube(
        T,
        h,
        w,
        prf_hz=prf,
        fd_primary_hz=fundamental,
        fd_alias_hz=alias_freq,
        amp_primary=1.0,
        amp_alias=0.8,
    )

    _, _, info, _ = _stap_pd_tile_lcmv(
        cube,
        prf_hz=prf,
        diag_load=1e-2,
        cov_estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="absolute",
        mvdr_auto_kappa=30.0,
        constraint_ridge=0.12,
        fd_span_mode="psd",
        fd_span_rel=(0.3, 1.1),
        grid_step_rel=0.08,
        min_pts=5,
        max_pts=7,
        msd_lambda=2e-2,
        msd_ridge=0.08,
        Lt_fixed=Lt,
        capture_debug=False,
        device="cpu",
    )
    alias_ratio = info.get("psd_flow_alias_ratio")
    assert alias_ratio is not None and np.isfinite(alias_ratio)
    assert alias_ratio > 1.5


def test_contrast_pd_matches_flow_only_when_flow_preserved():
    np.random.seed(2)
    T, H, W = 24, 4, 4
    prf = 2800.0
    cube = _tone_cube(T, H, W, prf, fd_hz=400.0, amp=1.0)
    pd_base = _baseline_pd(cube, hp_modes=1)

    common_args = dict(
        tile_hw=(4, 4),
        stride=4,
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
        grid_step_rel=0.08,
        min_pts=5,
        max_pts=7,
        msd_lambda=2e-2,
        msd_ridge=0.10,
        pd_base_full=pd_base,
        stap_device="cpu",
    )

    pd_plain, _, _ = _stap_pd(
        cube,
        msd_contrast_alpha=None,
        **common_args,
    )
    pd_contrast, _, info_contrast = _stap_pd(
        cube,
        msd_contrast_alpha=0.2,
        motion_half_span_rel=0.12,
        msd_ratio_rho=0.05,
        **common_args,
    )

    assert (
        info_contrast.get("contrast_tile_fraction") is None
        or info_contrast["contrast_tile_fraction"] >= 0.0
    )
    ratio = pd_contrast / np.maximum(pd_plain, 1e-8)
    assert np.all(ratio > 0.45)
    assert np.all(ratio < 1.35)


def test_tile_fallback_counts():
    T, h, w = 16, 4, 4
    prf = 2200.0
    cube = np.zeros((T, h, w), dtype=np.complex64)
    band_frac_tile, _, info_tile, _ = _stap_pd_tile_lcmv(
        cube,
        prf_hz=prf,
        diag_load=1e-2,
        cov_estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="unsupported",
        constraint_ridge=0.0,
        msd_lambda=2e-2,
        msd_ridge=0.0,
        fd_span_mode="fixed",
        fd_fixed_span_hz=400.0,
        grid_step_rel=0.1,
        min_pts=5,
        max_pts=5,
        capture_debug=False,
        device="cpu",
    )
    _print_metrics(
        "tile_fallback_counts_tile",
        load_mode=info_tile.get("load_mode"),
        msd_agg=info_tile.get("msd_agg_mode"),
        band_nan=int(info_tile.get("band_nan_count", 0)),
        score_nan=int(info_tile.get("score_nan_count", 0)),
        band_max=float(np.max(band_frac_tile)),
    )
    assert np.all(np.isfinite(band_frac_tile))
    assert info_tile["load_mode"] == "capon"
    assert info_tile["msd_agg_mode"] in {"mean", "median", "trim10"}

    pd_map, score_map, info = _stap_pd(
        cube,
        tile_hw=(4, 4),
        stride=4,
        Lt=4,
        prf_hz=prf,
        diag_load=1e-2,
        estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="unsupported",
        mvdr_auto_kappa=40.0,
        constraint_ridge=0.0,
        fd_span_mode="fixed",
        fd_span_rel=(0.2, 0.8),
        grid_step_rel=0.1,
        min_pts=5,
        max_pts=5,
        msd_lambda=2e-2,
        msd_ridge=0.0,
        debug_max_samples=0,
        stap_device="cpu",
    )
    _print_metrics(
        "tile_fallback_counts_scene",
        total_tiles=int(info.get("total_tiles", 0)),
        load_mode=info.get("load_mode"),
        diag_load=float(info.get("median_diag_load") or 0.0),
        score_mode_hist=str(info.get("score_mode_histogram")),
    )
    assert np.all(np.isfinite(pd_map))
    assert np.all(np.isfinite(score_map))


def test_bg_variance_guardrail():
    np.random.seed(2)
    T, h, w = 32, 12, 12
    prf = 3200.0
    cube = _tone_cube(T, h, w, prf, fd_hz=450.0, amp=1.2)
    pd_base = _baseline_pd(cube, hp_modes=1)
    mask_flow = np.zeros((h, w), dtype=bool)
    mask_flow[h // 2 - 1 : h // 2 + 1, w // 2 - 1 : w // 2 + 1] = True
    mask_bg = ~mask_flow

    pd_map, _, info = _stap_pd(
        cube,
        tile_hw=(8, 8),
        stride=4,
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
        grid_step_rel=0.12,
        min_pts=3,
        max_pts=5,
        msd_lambda=5e-2,
        msd_ridge=0.12,
        debug_max_samples=0,
        pd_base_full=pd_base,
        mask_flow=mask_flow,
        mask_bg=mask_bg,
        stap_device="cpu",
    )
    assert info["total_tiles"] > 0
    # Background variance should not inflate beyond a small tolerance under PD rescaling
    bg_ratio = np.var(pd_map[mask_bg]) / (np.var(pd_base[mask_bg]) + 1e-9)
    assert bg_ratio <= 1.10
    stap_flow_mu = float(np.mean(pd_map[mask_flow]))
    stap_bg_mu = float(np.mean(pd_map[mask_bg]))
    base_flow_mu = float(np.mean(pd_base[mask_flow]))
    base_bg_mu = float(np.mean(pd_base[mask_bg]))
    flow_gain = stap_flow_mu / (base_flow_mu + 1e-9)
    bg_gain = stap_bg_mu / (base_bg_mu + 1e-9)
    _print_metrics(
        "tile_bg_guardrail",
        bg_ratio=bg_ratio,
        flow_gain=flow_gain,
        bg_gain=bg_gain,
        stap_flow_mu=stap_flow_mu,
        stap_bg_mu=stap_bg_mu,
        base_flow_mu=base_flow_mu,
        base_bg_mu=base_bg_mu,
        max_pd=float(np.max(pd_map)),
        max_base=float(np.max(pd_base)),
    )
    # Flow region should retain a substantial fraction of its baseline energy
    assert flow_gain >= 0.50
    # Background should not inflate materially relative to baseline
    assert bg_gain <= 1.05
    # Flow gain should not drop far below background gain (allow small clamp slack)
    assert flow_gain + 0.40 >= bg_gain
    # Elementwise guardrail: STAP PD should not exceed baseline PD (rescaled by band fraction)
    assert np.all(pd_map <= pd_base + 1e-6)


def test_background_tiles_uniformized_to_baseline():
    np.random.seed(0)
    T, H, W = 32, 24, 24
    prf = 3200.0
    t = np.arange(T, dtype=np.float32)
    cube = (
        0.02
        * (
            np.random.randn(T, H, W).astype(np.float32)
            + 1j * np.random.randn(T, H, W).astype(np.float32)
        )
    ).astype(np.complex64)
    tone = np.exp(1j * 2.0 * np.pi * 450.0 * t / prf).astype(np.complex64)
    cube += 0.3 * tone[:, None, None]
    pd_base = _baseline_pd(cube, hp_modes=1)
    mask_flow = np.zeros((H, W), dtype=bool)
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

    assert info["background_uniformized_count"] == info["total_tiles"]
    np.testing.assert_allclose(pd_map, pd_base, rtol=1e-6, atol=1e-6)
    debug_sample = info["debug_samples"][0]
    assert debug_sample["background_uniformized"]
    assert np.allclose(debug_sample["band_fraction_tile"], 1.0, atol=1e-6)
    assert np.allclose(debug_sample["score_tile"], 0.0, atol=1e-6)


def test_low_flow_coverage_caps_band_fraction() -> None:
    rng = np.random.default_rng(3)
    T, H, W = 16, 8, 8
    cube = (rng.standard_normal((T, H, W)) + 1j * rng.standard_normal((T, H, W))).astype(
        np.complex64
    )
    mask_flow = np.zeros((H, W), dtype=bool)
    mask_flow[0, 0] = True
    mask_bg = ~mask_flow
    pd_map, _, info = _stap_pd(
        cube,
        tile_hw=(H, W),
        stride=H,
        Lt=4,
        prf_hz=3000.0,
        diag_load=1e-2,
        estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="absolute",
        mvdr_auto_kappa=40.0,
        constraint_ridge=0.12,
        fd_span_mode="fixed",
        fd_span_rel=(0.3, 0.5),
        fd_fixed_span_hz=600.0,
        grid_step_rel=0.12,
        min_pts=3,
        max_pts=3,
        msd_lambda=5e-2,
        msd_ridge=0.1,
        mask_flow=mask_flow,
        mask_bg=mask_bg,
        stap_device="cpu",
        ka_mode="none",
        ka_opts={
            "coverage_cap_enable": True,
        },
    )
    assert info["total_tiles"] == 1
    assert info["band_fraction_capped_count"] == 1
    cap_scale = info["median_band_fraction_cap_scale"]
    assert cap_scale is not None and cap_scale < 1.0
    assert np.all(pd_map[mask_bg] <= (pd_map.max() + 1e-6))


def test_edge_background_uniformization_clamps_to_base() -> None:
    np.random.seed(3)
    T, H, W = 32, 24, 24
    prf = 3500.0
    cube = _tone_cube(T, H, W, prf_hz=prf, fd_hz=700.0, amp=1.5)
    pd_base = _baseline_pd(cube, hp_modes=1)

    mask_flow = np.zeros((H, W), dtype=bool)
    mask_flow[8:16, 8:16] = True
    mask_bg = ~mask_flow

    pd_map, _, info = _stap_pd(
        cube,
        tile_hw=(12, 12),
        stride=8,
        Lt=6,
        prf_hz=prf,
        diag_load=1e-2,
        estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="absolute",
        mvdr_auto_kappa=40.0,
        constraint_ridge=0.12,
        fd_span_mode="psd",
        fd_span_rel=(0.25, 1.20),
        fd_fixed_span_hz=None,
        grid_step_rel=0.12,
        min_pts=3,
        max_pts=5,
        msd_lambda=5e-2,
        msd_ridge=0.12,
        pd_base_full=pd_base,
        mask_flow=mask_flow,
        mask_bg=mask_bg,
        stap_device="cpu",
    )

    edges = np.zeros((H, W), dtype=bool)
    edges[[0, -1], :] = True
    edges[:, [0, -1]] = True
    edge_bg = edges & mask_bg
    diff = float(np.max(np.abs(pd_map[edge_bg] - pd_base[edge_bg])))
    assert diff <= 1e-9, f"edge background mismatch {diff}"


def test_kc_cap_and_band_fraction_and_grid_step():
    # Ensure Kc respects Lt and band fraction median is not ~1
    T, h, w = 28, 6, 6
    prf = 2800.0
    rng = np.random.default_rng(0)
    cube = (rng.standard_normal((T, h, w)) + 1j * rng.standard_normal((T, h, w))).astype(
        np.complex64
    )

    band_frac, score_tile, info, _ = _stap_pd_tile_lcmv(
        cube,
        prf_hz=prf,
        diag_load=1e-2,
        cov_estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="absolute",
        constraint_ridge=0.12,
        msd_lambda=5e-2,
        msd_ridge=0.12,
        msd_agg_mode="trim10",
        fd_span_mode="psd",
        fd_fixed_span_hz=None,
        grid_step_rel=0.12,
        min_pts=3,
        max_pts=5,
        capture_debug=False,
        device="cpu",
    )
    assert not info.get("fallback", False)
    # Kc <= Lt and odd
    Lt = info.get("Lt", 0)
    Kc = info.get("band_Kc", 0)
    assert Kc % 2 == 1 and Kc <= max(1, Lt)
    # Band fraction median < 0.95 (not identity-like)
    assert float(np.median(band_frac)) < 0.95
    # Grid step close to grid_step_rel * prf/Lt
    fd_grid = np.asarray(info.get("fd_grid", []), dtype=np.float64)
    step_obs: float | None = None
    if fd_grid.size >= 3 and Lt > 0:
        diffs = np.diff(fd_grid)
        step_obs = float(np.median(np.abs(diffs)))
        # Sanity: positive and not exceeding a generous bound
        assert step_obs > 0.0
        assert step_obs <= 2.0 * (prf / float(Lt))
    _print_metrics(
        "tile_kc_cap",
        Lt=Lt,
        band_Kc=Kc,
        band_median=float(np.median(band_frac)),
        band_mean=float(np.mean(band_frac)),
        step_obs=step_obs if step_obs is not None else "n/a",
        fd_grid_len=int(fd_grid.size),
    )


def test_band_fraction_tightens_when_reducing_max_pts():
    # Compare band fraction median with max_pts=9 vs 5 (same Lt); tighter grid should reduce r
    T, h, w = 28, 6, 6
    prf = 2800.0
    rng = np.random.default_rng(1)
    cube = (rng.standard_normal((T, h, w)) + 1j * rng.standard_normal((T, h, w))).astype(
        np.complex64
    )

    def run(max_pts):
        band_frac, _, info, _ = _stap_pd_tile_lcmv(
            cube,
            prf_hz=prf,
            diag_load=1e-2,
            cov_estimator="scm",
            huber_c=5.0,
            mvdr_load_mode="absolute",
            constraint_ridge=0.12,
            msd_lambda=5e-2,
            msd_ridge=0.12,
            msd_agg_mode="trim10",
            fd_span_mode="psd",
            grid_step_rel=0.12,
            min_pts=3,
            max_pts=max_pts,
            capture_debug=False,
            device="cpu",
        )
        assert not info.get("fallback", False)
        return float(np.median(band_frac)), info

    r9, info9 = run(9)
    r5, info5 = run(5)
    _print_metrics(
        "tile_band_fraction_comparison",
        r9=r9,
        r5=r5,
        kc9=int(info9.get("band_Kc", 0)),
        kc5=int(info5.get("band_Kc", 0)),
        total_tiles9=int(info9.get("total_tiles", 0)),
        total_tiles5=int(info5.get("total_tiles", 0)),
    )
    # The Kc cap should reduce or keep Kc same
    assert info5.get("band_Kc", 0) <= info9.get("band_Kc", 0)
    # Both medians should be bounded away from 1
    assert r9 < 0.99 and r5 < 0.99


def test_stap_pd_with_ka_blend_outputs_metrics():
    np.random.seed(3)
    T, h, w = 24, 8, 8
    prf = 2800.0
    cube = _tone_cube(T, h, w, prf, fd_hz=320.0, amp=1.4)
    mask_flow = np.zeros((h, w), dtype=bool)
    mask_flow[3:5, 3:5] = True
    mask_bg = ~mask_flow

    _, score_map, info = _stap_pd(
        cube,
        tile_hw=(4, 4),
        stride=4,
        Lt=4,
        prf_hz=prf,
        diag_load=1e-2,
        estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="auto",
        mvdr_auto_kappa=40.0,
        constraint_ridge=0.12,
        fd_span_mode="psd",
        fd_span_rel=(0.30, 1.10),
        fd_fixed_span_hz=None,
        grid_step_rel=0.10,
        min_pts=5,
        max_pts=7,
        msd_lambda=2e-2,
        msd_ridge=0.10,
        msd_agg_mode="median",
        msd_ratio_rho=0.05,
        motion_half_span_rel=None,
        msd_contrast_alpha=0.0,
        pd_base_full=_baseline_pd(cube, hp_modes=1),
        mask_flow=mask_flow,
        mask_bg=mask_bg,
        stap_device="cpu",
        ka_mode="analytic",
        ka_opts={"beta_bounds": (0.05, 0.50), "kappa_target": 35.0},
    )

    _print_metrics(
        "tile_ka_blend_metrics",
        total_tiles=int(info.get("total_tiles", 0)),
        ka_tile_count=int(info.get("ka_tile_count", 0)),
        ka_mode=info.get("ka_mode"),
        ka_median_beta=float(info.get("ka_median_beta") or 0.0),
        ka_median_lambda=float(info.get("ka_median_lambda_used") or 0.0),
        score_mean=float(np.mean(score_map)),
        score_std=float(np.std(score_map)),
    )
    assert np.all(np.isfinite(score_map))
    assert info["ka_mode"] == "analytic"
    assert info["ka_tile_count"] == info["total_tiles"]
    assert info["ka_median_beta"] is not None
    assert info["ka_median_lambda_used"] is not None
    assert info["ka_median_sigma_min_raw"] is not None
    assert info["ka_median_sigma_max_raw"] is not None
    assert info["ka_median_mismatch"] is not None


def test_stap_pd_reports_pf_trace_telemetry_when_equalized():
    rng = np.random.default_rng(7)
    T, h, w = 32, 8, 8
    cube = (rng.standard_normal((T, h, w)) + 1j * rng.standard_normal((T, h, w))).astype(
        np.complex64
    )

    _, _, info = _stap_pd(
        cube,
        tile_hw=(4, 4),
        stride=4,
        Lt=5,
        prf_hz=3000.0,
        diag_load=1e-2,
        estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="auto",
        mvdr_auto_kappa=40.0,
        constraint_ridge=0.12,
        fd_span_mode="psd",
        fd_span_rel=(0.30, 1.10),
        grid_step_rel=0.10,
        min_pts=5,
        max_pts=7,
        msd_lambda=2e-2,
        msd_ridge=0.10,
        msd_agg_mode="trim10",
        msd_ratio_rho=0.05,
        motion_half_span_rel=None,
        msd_contrast_alpha=0.0,
        stap_device="cpu",
        ka_mode="analytic",
        ka_opts={
            "beta_bounds": (0.05, 0.50),
            "kappa_target": 35.0,
            "equalize_pf_trace": True,
            "beta_directional": True,
            "ridge_split": True,
        },
    )

    assert info.get("ka_pf_trace_equalized_fraction") == 1.0
    assert info.get("ka_pf_trace_alpha_median") is not None
    assert info.get("ka_pf_trace_alpha_valid_fraction") is not None
    assert info.get("ka_pf_trace_alpha_invalid_count") == 0


def test_stap_pd_ka_library_mismatch_disables():
    np.random.seed(4)
    T, h, w = 20, 6, 6
    prf = 2600.0
    cube = _tone_cube(T, h, w, prf, fd_hz=280.0, amp=1.0)
    prior = np.ones((2, 3), dtype=np.complex64)  # non-square -> mismatch triggers disable

    _, _, info = _stap_pd(
        cube,
        tile_hw=(3, 3),
        stride=3,
        Lt=4,
        prf_hz=prf,
        diag_load=1e-2,
        estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="auto",
        mvdr_auto_kappa=30.0,
        constraint_ridge=0.12,
        fd_span_mode="psd",
        fd_span_rel=(0.30, 1.10),
        grid_step_rel=0.10,
        min_pts=5,
        max_pts=5,
        msd_lambda=2e-2,
        msd_ridge=0.10,
        msd_agg_mode="median",
        msd_ratio_rho=0.05,
        motion_half_span_rel=None,
        msd_contrast_alpha=0.0,
        stap_device="cpu",
        ka_mode="library",
        ka_prior_library=prior,
        ka_opts={"beta_bounds": (0.05, 0.50), "kappa_target": 35.0},
    )

    _print_metrics(
        "tile_ka_library_mismatch",
        ka_mode=info.get("ka_mode"),
        ka_tile_count=int(info.get("ka_tile_count", 0)),
        mismatch_flag=info.get("ka_prior_mismatch"),
    )
    assert info["ka_mode"] == "none"
    assert info["ka_tile_count"] == 0


def test_ka_gamma_trace_ratio_recorded() -> None:
    np.random.seed(12)
    T, h, w = 20, 4, 4
    prf = 2500.0
    cube = _tone_cube(T, h, w, prf, fd_hz=350.0, amp=1.5)
    Lt = 4
    ka_prior = np.eye(Lt, dtype=np.complex64)
    _, _, info, _ = _stap_pd_tile_lcmv(
        cube,
        prf_hz=prf,
        diag_load=1e-2,
        cov_estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="absolute",
        constraint_ridge=0.12,
        msd_lambda=2e-2,
        msd_ridge=0.10,
        fd_span_mode="fixed",
        fd_fixed_span_hz=500.0,
        grid_step_rel=0.08,
        min_pts=5,
        max_pts=7,
        device="cpu",
        ka_mode="library",
        ka_prior_library=ka_prior,
        ka_opts={"kappa_target": 40.0},
        Lt_fixed=Lt,
    )
    assert info.get("gamma_flow") is not None
    assert info.get("gamma_perp") is not None
    assert np.isfinite(info["gamma_flow"]) and np.isfinite(info["gamma_perp"])


def test_ka_gate_decision_checks_all_signals() -> None:
    gate = _ka_gate_decision(
        alias_ratio=1.05,
        flow_cov=0.05,
        depth_frac=0.1,
        alias_rmin=1.10,
        flow_cov_min=0.2,
        depth_min_frac=0.2,
        depth_max_frac=0.9,
        pd_metric=0.8,
        pd_min=0.9,
        pd_q_lo=None,
        pd_q_hi=None,
        reg_psr=7.0,
        reg_psr_max=6.0,
    )
    gate_ok, alias_ok, flow_ok, depth_ok, pd_ok, reg_ok = gate
    assert gate_ok is False
    assert alias_ok is False
    assert flow_ok is False
    assert depth_ok is False
    assert pd_ok is False
    assert reg_ok is False

    gate2 = _ka_gate_decision(
        alias_ratio=1.5,
        flow_cov=0.6,
        depth_frac=0.4,
        alias_rmin=1.10,
        flow_cov_min=0.2,
        depth_min_frac=0.2,
        depth_max_frac=0.9,
        pd_metric=1.2,
        pd_min=0.9,
        pd_q_lo=None,
        pd_q_hi=None,
        reg_psr=4.0,
        reg_psr_max=6.0,
    )
    assert all(gate2)


def test_ka_gate_blocks_and_allows_tiles() -> None:
    np.random.seed(21)
    T, h, w = 18, 4, 4
    prf = 2400.0
    cube = _tone_cube(T, h, w, prf, fd_hz=320.0, amp=1.2)
    Lt = 4
    ka_prior = np.eye(Lt, dtype=np.complex64)
    base_gate = {
        "enable": True,
        "alias_rmin": 0.0,
        "flow_cov_min": 0.3,
        "depth_min_frac": 0.0,
        "depth_max_frac": 1.0,
        "pd_min": None,
        "reg_psr_max": None,
    }
    blocked_ctx = dict(base_gate)
    blocked_ctx["context"] = {
        "flow_cov": 0.1,
        "depth_frac": 0.5,
        "pd_metric": 1.0,
        "pd_norm": 1.0,
        "reg_psr": None,
        "tile_has_flow": True,
        "tile_is_bg": False,
    }
    _, _, info_blocked, _ = _stap_pd_tile_lcmv(
        cube,
        prf_hz=prf,
        diag_load=1e-2,
        cov_estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="absolute",
        constraint_ridge=0.10,
        msd_lambda=2e-2,
        msd_ridge=0.10,
        fd_span_mode="fixed",
        fd_fixed_span_hz=450.0,
        grid_step_rel=0.08,
        min_pts=5,
        max_pts=7,
        device="cpu",
        ka_mode="library",
        ka_prior_library=ka_prior,
        ka_opts={"kappa_target": 40.0},
        Lt_fixed=Lt,
        ka_gate=blocked_ctx,
    )
    assert info_blocked.get("ka_gate_ok") is False
    assert info_blocked.get("ka_mode") == "none"

    allow_ctx = dict(base_gate)
    allow_ctx["context"] = {
        "flow_cov": 0.8,
        "depth_frac": 0.4,
        "pd_metric": 1.0,
        "pd_norm": 1.0,
        "reg_psr": None,
        "tile_has_flow": True,
        "tile_is_bg": False,
    }
    _, _, info_allowed, _ = _stap_pd_tile_lcmv(
        cube,
        prf_hz=prf,
        diag_load=1e-2,
        cov_estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="absolute",
        constraint_ridge=0.10,
        msd_lambda=2e-2,
        msd_ridge=0.10,
        fd_span_mode="fixed",
        fd_fixed_span_hz=450.0,
        grid_step_rel=0.08,
        min_pts=5,
        max_pts=7,
        device="cpu",
        ka_mode="library",
        ka_prior_library=ka_prior,
        ka_opts={"kappa_target": 40.0},
        Lt_fixed=Lt,
        ka_gate=allow_ctx,
    )
    assert info_allowed.get("ka_gate_ok") is True
    assert info_allowed.get("ka_mode") == "library"


def test_updated_gating_works_without_flow_mask() -> None:
    np.random.seed(3)
    T, h, w = 16, 4, 4
    prf = 2000.0
    cube = _tone_cube(T, h, w, prf, fd_hz=250.0, amp=1.0)
    ka_prior = np.eye(4, dtype=np.complex64)
    ka_opts = {
        "ka_gate_enable": True,
        "ka_gate_alias_rmin": 0.0,
        "ka_gate_flow_cov_min": 0.0,
        "ka_gate_pd_min": 0.0,
        "ka_gate_depth_min_frac": 0.0,
        "ka_gate_depth_max_frac": 1.0,
    }

    _, _, info_legacy = _stap_pd(
        cube,
        tile_hw=(4, 4),
        stride=4,
        Lt=4,
        prf_hz=prf,
        diag_load=1e-2,
        estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="absolute",
        mvdr_auto_kappa=40.0,
        constraint_ridge=0.10,
        mask_flow=np.zeros((h, w), dtype=bool),
        mask_bg=np.ones((h, w), dtype=bool),
        ka_mode="library",
        ka_prior_library=ka_prior,
        ka_opts=ka_opts,
        feasibility_mode="legacy",
    )
    assert info_legacy.get("ka_gate_fraction_total") in (None, 0.0)

    _, _, info_updated = _stap_pd(
        cube,
        tile_hw=(4, 4),
        stride=4,
        Lt=4,
        prf_hz=prf,
        diag_load=1e-2,
        estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="absolute",
        mvdr_auto_kappa=40.0,
        constraint_ridge=0.10,
        mask_flow=np.zeros((h, w), dtype=bool),
        mask_bg=np.ones((h, w), dtype=bool),
        ka_mode="library",
        ka_prior_library=ka_prior,
        ka_opts=ka_opts,
        feasibility_mode="updated",
    )
    assert info_updated["feasibility_mode"] == "updated"
    frac = info_updated.get("ka_gate_fraction_total")
    assert frac is None or frac >= 0.0
