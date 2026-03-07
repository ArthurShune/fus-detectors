from __future__ import annotations

from scripts.simus_real_envelope_check import summarize_rows


def test_summarize_rows_reports_outside_fraction():
    rows = [
        {"motion_scale": 0.25, "reg_shift_p90": 2.1, "within_real_reg_shift_envelope": False},
        {"motion_scale": 0.25, "reg_shift_p90": 2.2, "within_real_reg_shift_envelope": False},
        {"motion_scale": 1.0, "reg_shift_p90": 1.5, "within_real_reg_shift_envelope": True},
    ]
    out = summarize_rows(rows, real_max=1.93, real_q95=1.57)
    assert out["simus_reg_shift_p90_min"] == 1.5
    assert out["fraction_within_real_reg_shift_envelope"] == 1.0 / 3.0
    assert out["summary_by_motion_scale"]["0.25"]["fraction_within_real_reg_shift_envelope"] == 0.0
    assert out["summary_by_motion_scale"]["1.0"]["fraction_within_real_reg_shift_envelope"] == 1.0
