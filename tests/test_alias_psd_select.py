import numpy as np
import pytest

from sim.kwave.common import _STAP_AVAILABLE, _stap_pd

pytest.importorskip("torch")
if not _STAP_AVAILABLE:
    pytest.skip("STAP pipeline not available", allow_module_level=True)


def _two_tone_cube(
    T: int,
    H: int,
    W: int,
    prf_hz: float,
    f0_hz: float,
    f_alias_hz: float,
    mask_flow: np.ndarray,
    amp0: float = 1.0,
    amp_alias: float = 0.7,
) -> np.ndarray:
    t = np.arange(T, dtype=np.float64) / float(prf_hz)
    sig0 = amp0 * np.exp(1j * 2.0 * np.pi * f0_hz * t)
    sig1 = amp_alias * np.exp(1j * 2.0 * np.pi * f_alias_hz * t)
    base = 0.02 * (np.random.randn(T, H, W) + 1j * np.random.randn(T, H, W))
    cube = base.reshape(T, -1)
    flow_vec = mask_flow.reshape(-1)
    bg_vec = ~flow_vec
    cube[:, flow_vec] += sig0[:, None]
    cube[:, bg_vec] += sig1[:, None]
    return cube.reshape(T, H, W).astype(np.complex64)


def test_alias_psd_selection_improves_flow_over_bg_when_enabled():
    # Synthetic setup tied to Lt bin spacing (bin = prf/Lt)
    T, H, W = 48, 12, 12
    Lt = 8
    prf = 3000.0
    bin_hz = prf / Lt  # 375 Hz
    f0 = 1 * bin_hz
    f_alias = 3 * bin_hz

    mask_flow = np.zeros((H, W), dtype=bool)
    mask_flow[H // 2 - 2 : H // 2 + 2, W // 2 - 2 : W // 2 + 2] = True
    mask_bg = ~mask_flow

    np.random.seed(0)
    cube = _two_tone_cube(
        T,
        H,
        W,
        prf_hz=prf,
        f0_hz=f0,
        f_alias_hz=f_alias,
        mask_flow=mask_flow,
        amp0=0.6,
        amp_alias=1.2,
    )
    pd_base = (np.abs(cube) ** 2).mean(axis=0).astype(np.float32)

    common_kwargs = dict(
        tile_hw=(H, W),
        stride=H,
        Lt=Lt,
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
        pd_base_full=pd_base,
        mask_flow=mask_flow,
        mask_bg=mask_bg,
        stap_device="cpu",
        debug_max_samples=1,
    )

    # Baseline (no selection)
    pd_no, _, info_no = _stap_pd(cube, **common_kwargs)

    # With alias-aware PSD selection (future implementation should support this flag and telemetry)
    pd_sel, _, info_sel = _stap_pd(
        cube,
        alias_psd_select_enable=True,
        alias_psd_select_ratio_thresh=1.1,
        alias_psd_select_bins=1,
        **common_kwargs,
    )

    # If selection not implemented yet, skip rather than mask the failure
    debug = None
    if isinstance(info_sel.get("debug_samples"), list) and info_sel["debug_samples"]:
        debug = info_sel["debug_samples"][0]
    kept_key = None
    for k in ("psd_kept_bin_count", "band_kept_bin_count", "psd_kept_freqs_hz"):
        if debug is not None and k in debug:
            kept_key = k
            break
    if kept_key is None:
        pytest.skip("alias-aware PSD selection telemetry not present (implementation pending)")
    assert info_sel.get("psd_alias_select_fraction", 0.0) > 0.0

    # Strict checks
    # 1) Flow over background mean improves with selection
    flow_ratio_no = float(np.mean(pd_no[mask_flow]) / (np.mean(pd_no[mask_bg]) + 1e-12))
    flow_ratio_sel = float(np.mean(pd_sel[mask_flow]) / (np.mean(pd_sel[mask_bg]) + 1e-12))
    assert flow_ratio_sel >= 1.5 * flow_ratio_no
    assert flow_ratio_sel >= 0.2

    # 2) Selection is narrow: kept bins are few (<=3) compared to Lt
    if kept_key == "psd_kept_freqs_hz":
        kept_count_sel = int(len(debug[kept_key]))
    else:
        kept_count_sel = int(debug[kept_key])
    assert kept_count_sel <= 3

    # 3) Background invariance remains (var ratio ~1 within tolerance)
    var_no = float(np.var(pd_no[mask_bg]) + 1e-12)
    var_sel = float(np.var(pd_sel[mask_bg]) + 1e-12)
    assert var_sel <= 1.05 * var_no
