from pathlib import Path

import numpy as np

from scripts.analyze_coverage_roc import (
    BundleData,
    _compute_tile_coverages,
    _tile_gate,
    summarize_bundle,
)


def test_tile_gate_selects_expected_tiles():
    mask_flow = np.zeros((4, 4), dtype=bool)
    mask_flow[0, 0] = True
    mask_flow[0, 1] = True
    mask_flow[1, 0] = True
    tile_hw = (2, 2)
    stride = 2
    covs, coords = _compute_tile_coverages(mask_flow, tile_hw, stride)
    assert covs.shape == (4,)
    gate, selected = _tile_gate(covs, coords, 0.5, mask_flow.shape, tile_hw)
    assert selected == 1
    assert gate[:2, :2].all()
    assert not gate[2:, 2:].any()


def test_summarize_bundle_reports_coverage_slices():
    mask_flow = np.zeros((4, 4), dtype=bool)
    mask_flow[0, 0] = True
    mask_flow[0, 1] = True
    mask_flow[1, 0] = True
    mask_bg = ~mask_flow
    stap_map = np.zeros((4, 4), dtype=np.float32)
    base_map = np.zeros((4, 4), dtype=np.float32)
    flow_vals_stap = np.array([2.0, 1.5, 1.2], dtype=np.float32)
    flow_vals_base = np.array([0.9, 0.45, 0.3], dtype=np.float32)
    stap_map[mask_flow] = flow_vals_stap
    base_map[mask_flow] = flow_vals_base
    base_map[mask_bg] = 0.6
    stap_map[mask_bg] = 0.3
    telemetry = {
        "tile_flow_coverage_p50": 0.75,
        "tile_flow_coverage_p90": 0.75,
        "flow_cov_ge_20_fraction": 0.25,
        "flow_cov_ge_50_fraction": 0.25,
        "flow_cov_ge_80_fraction": 0.0,
    }
    meta = {
        "tile_hw": [2, 2],
        "tile_stride": 2,
        "stap_fallback_telemetry": telemetry,
    }
    bundle = BundleData(
        name="toy",
        path=Path("/tmp"),
        meta=meta,
        mask_flow=mask_flow,
        mask_bg=mask_bg,
        stap_map=stap_map,
        base_map=base_map,
        tile_hw=(2, 2),
        stride=2,
    )
    summary = summarize_bundle(
        bundle, thresholds=[0.2, 0.6], fpr_target=1e-3, pauc_max=1e-2, extra_targets=[1e-4]
    )
    assert len(summary.coverage_results) == 2
    for row in summary.coverage_results:
        assert row.n_flow > 0
        assert row.n_bg > 0
        assert row.auc_base >= 0 and row.auc_stap >= 0
        assert row.fpr_floor > 0
        assert row.pauc_base >= 0 and row.pauc_stap >= 0
        assert row.tpr_extra and abs(row.tpr_extra[0]["fpr_target"] - 1e-4) < 1e-12
