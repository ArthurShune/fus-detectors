from __future__ import annotations

from scripts.simus_motion_anchor_scan import summarize_rows


def test_summarize_rows_tracks_real_envelope_crossing():
    rows = [
        {"profile": "A", "seed": 1, "motion_scale": 0.05, "reg_shift_p90": 1.0, "motion_disp_rms_px": 0.4, "within_real_reg_shift_envelope": True},
        {"profile": "A", "seed": 1, "motion_scale": 0.10, "reg_shift_p90": 1.5, "motion_disp_rms_px": 0.5, "within_real_reg_shift_envelope": True},
        {"profile": "A", "seed": 1, "motion_scale": 0.20, "reg_shift_p90": 2.1, "motion_disp_rms_px": 0.7, "within_real_reg_shift_envelope": False},
    ]
    out = summarize_rows(rows, real_reg_shift_p90_max=1.93)
    assert out["A"]["max_motion_scale_within_real_envelope"] == 0.1
    assert out["A"]["min_motion_scale_above_real_envelope"] == 0.2
    assert out["A"]["real_reg_shift_p90_max"] == 1.93
