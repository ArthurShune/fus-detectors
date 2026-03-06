import pytest


def test_simus_sanity_link_flatten_and_deltas_smoke():
    from scripts.simus_sanity_link import build_delta_rows, flatten_summary_row

    sim_report = {
        "meta": {"shape": [64, 32, 32], "prf_hz": 1500.0},
        "summary": {
            "flow": {
                "n_tiles": 10,
                "malias": {"q50": -0.3},
                "fpeak_hz": {"q50": 120.0},
                "coh1": {"q50": 0.75},
            },
            "bg": {
                "n_tiles": 14,
                "malias": {"q50": -1.0},
                "fpeak_hz": {"q50": 40.0},
                "coh1": {"q50": 0.92},
            },
        },
        "svd": {
            "flow": {"cum_r1": 0.55, "cum_r2": 0.70},
            "bg": {"cum_r1": 0.80, "cum_r2": 0.90},
        },
    }
    ref_report = {
        "meta": {"shape": [64, 32, 32], "prf_hz": 1000.0},
        "summary": {
            "flow": {
                "n_tiles": 11,
                "malias": {"q50": -0.1},
                "fpeak_hz": {"q50": 135.0},
                "coh1": {"q50": 0.68},
            },
            "bg": {
                "n_tiles": 12,
                "malias": {"q50": -0.8},
                "fpeak_hz": {"q50": 35.0},
                "coh1": {"q50": 0.88},
            },
        },
        "svd": {
            "flow": {"cum_r1": 0.50, "cum_r2": 0.65},
            "bg": {"cum_r1": 0.76, "cum_r2": 0.87},
        },
    }

    sim_row = flatten_summary_row("sim_case", sim_report, kind="sim", motion_disp_rms_px=1.2, phase_rms_rad=0.4)
    ref_row = flatten_summary_row("shin_case", ref_report, kind="shin")
    deltas = build_delta_rows([sim_row, ref_row], metrics=["flow_malias_q50", "flow_fpeak_q50", "svd_bg_cum_r1"])

    assert sim_row["kind"] == "sim"
    assert sim_row["motion_disp_rms_px"] == 1.2
    assert sim_row["phase_rms_rad"] == 0.4
    assert ref_row["kind"] == "shin"
    assert len(deltas) == 1
    assert deltas[0]["sim_key"] == "sim_case"
    assert deltas[0]["ref_key"] == "shin_case"
    assert deltas[0]["delta_flow_malias_q50"] == pytest.approx(-0.2)
    assert deltas[0]["delta_flow_fpeak_q50"] == pytest.approx(-15.0)
    assert deltas[0]["delta_svd_bg_cum_r1"] == pytest.approx(0.04)
    assert deltas[0]["mean_abs_delta_selected"] == pytest.approx((0.2 + 15.0 + 0.04) / 3.0)
