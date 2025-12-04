from sim.kwave.common import _auto_band_ratio_spec


def test_auto_band_ratio_spec_clamps_alias_and_separates_bands():
    spec = _auto_band_ratio_spec(
        prf_hz=1500.0,
        Lt=4,
        flow_alias_hz=900.0,
        flow_alias_fraction=0.6,
    )
    flow_low = spec["flow_low_hz"]
    flow_high = spec["flow_high_hz"]
    alias_center = spec["alias_center_hz"]
    alias_width = spec["alias_width_hz"]
    assert flow_low >= 0.0
    assert flow_high > flow_low
    nyquist = 0.5 * 1500.0
    alias_low = alias_center - 0.5 * alias_width
    alias_high = alias_center + 0.5 * alias_width
    assert alias_high <= nyquist
    assert alias_low > flow_high
    assert alias_center > flow_high


def test_auto_band_ratio_spec_lt8_discrete_guard():
    spec = _auto_band_ratio_spec(
        prf_hz=1500.0,
        Lt=8,
        flow_alias_hz=562.5,
    )
    flow_low = spec["flow_low_hz"]
    flow_high = spec["flow_high_hz"]
    alias_center = spec["alias_center_hz"]
    alias_width = spec["alias_width_hz"]
    alias_low = alias_center - 0.5 * alias_width
    alias_high = alias_center + 0.5 * alias_width
    bin_hz = 1500.0 / 8.0
    guard = bin_hz  # one bin guard
    assert flow_low >= 0.0
    assert flow_high - flow_low <= 2 * bin_hz
    assert alias_low >= flow_high + guard
    assert alias_high <= 0.5 * 1500.0 + 1e-6
    assert spec["alias_min_bins"] >= 2
