import numpy as np

from sim.kwave.common import _apply_bg_guard_simple


def test_bg_guard_simple_shrinks_variance():
    rng = np.random.default_rng(0)
    H, W = 32, 32
    # Baseline PD: unit-variance Gaussian field
    pd_base = rng.standard_normal((H, W)).astype(np.float32)
    # STAP PD: inflate BG variance by adding stronger noise on BG
    mask_bg = np.zeros((H, W), dtype=bool)
    mask_bg[:, : W // 2] = True  # left half as background
    pd_avg = pd_base.copy()
    pd_avg[mask_bg] = pd_base[mask_bg] + 2.0 * rng.standard_normal(mask_bg.sum()).astype(
        np.float32
    )

    # Construct tile-level BG var ratios above the cap (ensure p90 > target)
    tile_ratios = [1.4, 1.3, 1.25, 1.2, 1.35, 1.1, 1.05, 1.0]

    var_base = float(np.var(pd_base[mask_bg]))
    var_stap = float(np.var(pd_avg[mask_bg]))
    ratio_pre = var_stap / max(var_base, 1e-12)
    assert ratio_pre > 1.15

    pd_guarded, info = _apply_bg_guard_simple(
        pd_avg=pd_avg,
        pd_base=pd_base,
        mask_bg=mask_bg,
        tile_bg_var_ratios=tile_ratios,
        target_p90=1.15,
        min_alpha=0.0,
        metric="global",
    )

    assert info["bg_guard_alpha"] is not None
    assert 0.0 <= float(info["bg_guard_alpha"]) <= 1.0

    var_post = float(np.var(pd_guarded[mask_bg]))
    ratio_post = var_post / max(var_base, 1e-12)
    # Post-guard ratio should be reduced significantly
    assert ratio_post < ratio_pre
    assert ratio_post <= 0.85 * ratio_pre
    assert np.isclose(info["bg_var_ratio_pre"], ratio_pre, rtol=1e-6)
    assert np.isclose(info["bg_var_ratio_post"], ratio_post, rtol=1e-6)


def test_bg_guard_uses_global_metric_even_when_tiles_safe():
    rng = np.random.default_rng(1)
    H, W = 32, 32
    pd_base = rng.standard_normal((H, W)).astype(np.float32)
    mask_bg = np.zeros((H, W), dtype=bool)
    mask_bg[:, : W // 2] = True
    pd_avg = pd_base.copy()
    pd_avg[mask_bg] = pd_base[mask_bg] + 4.0 * rng.standard_normal(mask_bg.sum()).astype(
        np.float32
    )

    tile_ratios = [1.05 for _ in range(16)]  # below target
    pd_guarded, info = _apply_bg_guard_simple(
        pd_avg=pd_avg,
        pd_base=pd_base,
        mask_bg=mask_bg,
        tile_bg_var_ratios=tile_ratios,
        target_p90=1.15,
        min_alpha=0.7,
        metric="tile_p90",
    )

    assert (
        info["bg_guard_alpha"] is not None
    ), "Global ratio should trigger guard even if tiles safe"
    ratio_post = info["bg_var_ratio_post"]
    assert ratio_post is not None and ratio_post < info["bg_var_ratio_pre"]
    assert ratio_post <= 1.15 * 1.001
