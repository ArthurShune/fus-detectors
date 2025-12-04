import numpy as np
import pytest

from eval.acceptance import DetectorDataset, Masks, evaluate_performance


def _make_masks(size: int = 4) -> Masks:
    mask_flow = np.zeros((size, size), dtype=bool)
    mask_flow[: size // 2, : size // 2] = True
    mask_bg = ~mask_flow
    return Masks(mask_flow=mask_flow, mask_bg=mask_bg)


def test_pd_stats_validation_passes():
    masks = _make_masks()
    pd_base = np.ones((4, 4), dtype=np.float32)
    pd_stap = np.full((4, 4), 0.8, dtype=np.float32)
    base_stats = {
        "flow_mean": float(pd_base[masks.mask_flow].mean()),
        "bg_var": float(pd_base[masks.mask_bg].var()),
    }
    stap_stats = {
        "flow_mean": float(pd_stap[masks.mask_flow].mean()),
        "bg_var": float(pd_stap[masks.mask_bg].var()),
    }
    base = DetectorDataset(
        scores_pos=np.ones(8), scores_null=np.ones(8), pd_map=pd_base, pd_stats=base_stats
    )
    stap = DetectorDataset(
        scores_pos=np.ones(8), scores_null=np.ones(8), pd_map=pd_stap, pd_stats=stap_stats
    )
    perf = evaluate_performance(base, stap, masks)
    assert np.isfinite(perf["pd_snr_baseline_db"])


def test_pd_stats_validation_raises_on_mismatch():
    masks = _make_masks()
    pd_base = np.ones((4, 4), dtype=np.float32)
    pd_stap = np.full((4, 4), 0.5, dtype=np.float32)
    wrong_stats = {"flow_mean": 123.0, "bg_var": 0.0}
    base = DetectorDataset(
        scores_pos=np.ones(8), scores_null=np.ones(8), pd_map=pd_base, pd_stats=wrong_stats
    )
    stap = DetectorDataset(
        scores_pos=np.ones(8), scores_null=np.ones(8), pd_map=pd_stap, pd_stats=wrong_stats
    )
    with pytest.raises(ValueError):
        evaluate_performance(base, stap, masks)
