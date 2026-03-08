import json


def test_simus_v2_anchor_envelopes_build_envelopes():
    from scripts import simus_v2_anchor_envelopes as mod

    rows = [
        {
            "anchor_kind": "shin",
            "anchor_mode": "iq",
            "case_key": "a",
            "flow_malias_q50": 0.1,
            "bg_malias_q50": -0.2,
            "flow_fpeak_q50": 100.0,
            "bg_fpeak_q50": 40.0,
            "flow_coh1_q50": 0.9,
            "bg_coh1_q50": 0.4,
            "svd_flow_cum_r1": 0.5,
            "svd_flow_cum_r2": 0.7,
            "svd_flow_cum_r5": 0.9,
            "svd_flow_cum_r10": 0.95,
            "svd_bg_cum_r1": 0.6,
            "svd_bg_cum_r2": 0.8,
            "svd_bg_cum_r5": 0.92,
            "svd_bg_cum_r10": 0.97,
            "reg_shift_rms": 0.1,
            "reg_shift_p90": 0.2,
            "reg_psr_median": 8.0,
        },
        {
            "anchor_kind": "shin",
            "anchor_mode": "iq",
            "case_key": "b",
            "flow_malias_q50": 0.3,
            "bg_malias_q50": -0.1,
            "flow_fpeak_q50": 110.0,
            "bg_fpeak_q50": 35.0,
            "flow_coh1_q50": 0.95,
            "bg_coh1_q50": 0.35,
            "svd_flow_cum_r1": 0.55,
            "svd_flow_cum_r2": 0.75,
            "svd_flow_cum_r5": 0.91,
            "svd_flow_cum_r10": 0.96,
            "svd_bg_cum_r1": 0.65,
            "svd_bg_cum_r2": 0.82,
            "svd_bg_cum_r5": 0.93,
            "svd_bg_cum_r10": 0.98,
            "reg_shift_rms": 0.2,
            "reg_shift_p90": 0.3,
            "reg_psr_median": 9.0,
        },
        {
            "anchor_kind": "mace_phase2",
            "anchor_mode": "functional_readout",
            "case_key": "scan1",
            "pd_pauc": 0.05,
            "br_pauc": 0.04,
            "gate_kept_frac": 0.8,
            "pd_fp_at_tpr": 100.0,
            "gated_pd_fp_at_tpr": 80.0,
            "frac_pf_peak_pos": 0.2,
            "frac_pf_peak_neg": 0.1,
            "median_log_alias_pos": 0.01,
            "median_log_alias_neg": 0.005,
        },
    ]
    env = mod.build_envelopes(rows)
    assert env["shin"]["n_rows"] == 2
    assert env["shin"]["metrics"]["flow_fpeak_q50"]["q50"] == 105.0
    assert env["mace_phase2"]["anchor_mode"] == "functional_readout"
    assert env["mace_phase2"]["metrics"]["pd_pauc"]["q50"] == 0.05


def test_simus_v2_anchor_envelopes_mace_report(tmp_path):
    from scripts import simus_v2_anchor_envelopes as mod

    report = tmp_path / "mace.csv"
    report.write_text(
        "scan_name,plane_idx,pd_pauc,br_pauc,gate_kept_frac,pd_fp_at_tpr,gated_pd_fp_at_tpr,frac_pf_peak_pos,frac_pf_peak_neg,median_log_alias_pos,median_log_alias_neg\n"
        "scan1,0,0.05,0.04,0.8,100,80,0.2,0.1,0.01,0.005\n",
        encoding="utf-8",
    )
    rows = mod.summarize_mace_phase2_report(report)
    assert rows[0]["anchor_kind"] == "mace_phase2"
    assert rows[0]["anchor_mode"] == "functional_readout"
    assert rows[0]["pd_pauc"] == 0.05
