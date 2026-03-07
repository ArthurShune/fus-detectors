from __future__ import annotations

from scripts.real_motion_proxy_telemetry import summarize_rows


def test_summarize_rows_groups_and_threshold():
    rows = [
        {"kind": "shin", "reg_shift_rms": 0.01, "reg_shift_p90": 0.02, "reg_psr_median": 100.0},
        {"kind": "gammex_along", "reg_shift_rms": 0.5, "reg_shift_p90": 1.8, "reg_psr_median": 10.0},
        {"kind": "gammex_along", "reg_shift_rms": 0.6, "reg_shift_p90": 2.3, "reg_psr_median": 11.0},
    ]
    out = summarize_rows(rows, threshold=2.0)
    assert out["by_kind"]["shin"]["n"] == 1
    assert out["by_kind"]["shin"]["fraction_reg_shift_p90_gt_threshold"] == 0.0
    assert out["by_kind"]["gammex_along"]["n"] == 2
    assert out["by_kind"]["gammex_along"]["reg_shift_p90_max"] == 2.3
    assert out["by_kind"]["gammex_along"]["fraction_reg_shift_p90_gt_threshold"] == 0.5
    assert out["overall"]["reg_shift_p90_max"] == 2.3
