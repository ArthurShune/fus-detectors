import numpy as np

from sim.kwave.common import _resolve_flow_mask


def test_flow_mask_pd_auto_respects_threshold_and_union():
    pd = np.zeros((16, 16), dtype=np.float32)
    pd[4:8, 4:8] = 10.0  # strong flow core
    mask_flow_default = np.zeros_like(pd, dtype=bool)
    mask_flow_default[-2:, :] = True  # default ROI bottom band
    mask_bg_default = ~mask_flow_default

    # Without union, PD-based mask should keep only the hot block
    mask_flow, mask_bg, stats = _resolve_flow_mask(
        pd,
        mask_flow_default,
        mask_bg_default,
        mode="pd_auto",
        pd_quantile=1.0,
        depth_min_frac=0.0,
        depth_max_frac=1.0,
        dilate_iters=0,
        erode_iters=0,
        min_pixels=2,
        min_coverage_frac=0.0,
        union_with_default=False,
    )
    assert stats["pd_auto_used"] == 1.0
    assert mask_flow.sum() == 16
    assert np.array_equal(mask_bg, mask_bg_default & (~mask_flow))
    assert "pd_auto_reason" not in stats or not stats.get("pd_auto_failed")

    # With union enabled, ensure default ROI pixels are included
    mask_flow_union, _, stats_union = _resolve_flow_mask(
        pd,
        mask_flow_default,
        mask_bg_default,
        mode="pd_auto",
        pd_quantile=1.0,
        depth_min_frac=0.0,
        depth_max_frac=1.0,
        dilate_iters=0,
        erode_iters=0,
        min_pixels=2,
        min_coverage_frac=0.0,
        union_with_default=True,
    )
    assert stats_union["union_applied"] == 1.0
    assert mask_flow_union.sum() == mask_flow.sum() + mask_flow_default.sum()
    assert np.all(mask_flow_union[-2:, :])  # default band blended in


def test_flow_mask_pd_auto_falls_back_when_coverage_too_small():
    pd = np.zeros((16, 16), dtype=np.float32)
    pd[0, 0] = 5.0  # single hot pixel
    mask_flow_default = np.zeros_like(pd, dtype=bool)
    mask_flow_default[2:, :] = True
    mask_bg_default = ~mask_flow_default

    mask_flow, _, stats = _resolve_flow_mask(
        pd,
        mask_flow_default,
        mask_bg_default,
        mode="pd_auto",
        pd_quantile=1.0,
        depth_min_frac=0.0,
        depth_max_frac=1.0,
        dilate_iters=0,
        erode_iters=0,
        min_pixels=10,  # force fallback
        min_coverage_frac=0.05,
        union_with_default=False,
    )
    assert stats.get("pd_auto_failed") == 1.0
    assert stats.get("pd_auto_reason") in {"min_pixels", "min_coverage"}
    assert stats["pd_auto_used"] == 0.0
    assert np.array_equal(mask_flow, mask_flow_default)
