def test_simus_motion_calibration_summarizes_best_matches():
    from scripts.simus_motion_calibration import parse_sim_key, summarize_calibration

    table_rows = [
        {
            "key": "sim_simus_clinintraop_pf_v1_motionx0_phasex0_seed21",
            "kind": "sim",
            "motion_disp_rms_px": "0.0",
            "phase_rms_rad": "0.0",
            "flow_malias_q50": "-1.0",
            "bg_malias_q50": "-0.5",
            "flow_fpeak_q50": "23.4",
            "bg_fpeak_q50": "23.4",
            "flow_coh1_q50": "0.98",
            "bg_coh1_q50": "0.95",
        },
        {
            "key": "sim_simus_clinintraop_pf_v1_motionx0p25_phasex0p25_seed21",
            "kind": "sim",
            "motion_disp_rms_px": "1.0",
            "phase_rms_rad": "0.2",
            "flow_malias_q50": "0.3",
            "bg_malias_q50": "0.4",
            "flow_fpeak_q50": "328.1",
            "bg_fpeak_q50": "421.8",
            "flow_coh1_q50": "0.14",
            "bg_coh1_q50": "0.15",
        },
    ]
    delta_rows = [
        {"sim_key": "sim_simus_clinintraop_pf_v1_motionx0_phasex0_seed21", "ref_key": "shin_ref", "ref_kind": "shin", "mean_abs_delta_selected": "2.0"},
        {"sim_key": "sim_simus_clinintraop_pf_v1_motionx0p25_phasex0p25_seed21", "ref_key": "shin_ref", "ref_kind": "shin", "mean_abs_delta_selected": "5.0"},
        {"sim_key": "sim_simus_clinintraop_pf_v1_motionx0_phasex0_seed21", "ref_key": "gammex_ref", "ref_kind": "gammex", "mean_abs_delta_selected": "9.0"},
        {"sim_key": "sim_simus_clinintraop_pf_v1_motionx0p25_phasex0p25_seed21", "ref_key": "gammex_ref", "ref_kind": "gammex", "mean_abs_delta_selected": "3.0"},
    ]

    parsed = parse_sim_key("sim_simus_clinintraop_pf_v1_motionx0p25_phasex0p25_seed21")
    assert parsed["profile_slug"] == "clinintraop_pf_v1"
    assert parsed["motion_scale"] == 0.25
    rows, details = summarize_calibration(table_rows=table_rows, delta_rows=delta_rows)

    assert any(r["summary_type"] == "ref_best_sim" and r["ref_key"] == "shin_ref" and r["motion_scale"] == 0.0 for r in rows)
    assert any(r["summary_type"] == "ref_best_sim" and r["ref_key"] == "gammex_ref" and r["motion_scale"] == 0.25 for r in rows)
    best = details["best_by_profile_ref_kind"]["clinintraop_pf_v1:shin"]
    assert best["motion_scale"] == 0.0
