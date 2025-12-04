from sim.kwave.common import _ka_effective_status


def test_ka_effective_all_conditions_satisfied() -> None:
    info = {
        "ka_median_snr_flow_ratio": 0.95,
        "ka_median_noise_perp_ratio": 1.0,
        "ka_pf_trace_loaded_median": 1.0,
        "ka_pf_trace_sample_median": 1.0,
    }
    effective, reasons, ratio = _ka_effective_status(
        ka_tile_count=10,
        base_reasons=[],
        info_out=info,
    )
    assert effective
    assert reasons == []
    assert ratio == 1.0


def test_ka_effective_disables_on_c1_violation() -> None:
    info = {
        "ka_median_snr_flow_ratio": 0.4,
        "ka_median_noise_perp_ratio": 1.0,
        "ka_pf_trace_loaded_median": 1.0,
        "ka_pf_trace_sample_median": 1.0,
    }
    effective, reasons, ratio = _ka_effective_status(
        ka_tile_count=8,
        base_reasons=[],
        info_out=info,
    )
    assert not effective
    assert "c1" in reasons
    assert ratio == 1.0


def test_ka_effective_disables_on_pf_trace_ratio() -> None:
    info = {
        "ka_median_snr_flow_ratio": 0.95,
        "ka_median_noise_perp_ratio": 1.0,
        "ka_pf_trace_loaded_median": 3.0,
        "ka_pf_trace_sample_median": 1.0,
    }
    effective, reasons, ratio = _ka_effective_status(
        ka_tile_count=4,
        base_reasons=[],
        info_out=info,
    )
    assert not effective
    assert "pf_trace" in reasons
    assert ratio == 3.0


def test_ka_effective_disables_on_operator_flag() -> None:
    info = {
        "ka_median_snr_flow_ratio": 0.95,
        "ka_median_noise_perp_ratio": 1.0,
        "ka_pf_trace_loaded_median": 1.0,
        "ka_pf_trace_sample_median": 1.0,
        "ka_operator_feasible": False,
    }
    effective, reasons, ratio = _ka_effective_status(
        ka_tile_count=6,
        base_reasons=[],
        info_out=info,
    )
    assert not effective
    assert "operator" in reasons
