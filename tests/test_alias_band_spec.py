from sim.kwave.common import _adjust_alias_band_for_meta


def test_alias_band_adjusts_center_and_width():
    base_spec = {"alias_center_hz": 900.0, "alias_width_hz": 15.0}
    alias_meta = {"flow_alias_hz": 950.0, "flow_alias_jitter_hz": 20.0}
    adjusted = _adjust_alias_band_for_meta(base_spec, alias_meta)
    assert adjusted["alias_center_hz"] == 950.0
    # width should expand to cover jitter plus padding
    assert adjusted["alias_width_hz"] >= 25.0


def test_alias_band_no_alias_meta_returns_original():
    base_spec = {"alias_center_hz": 880.0, "alias_width_hz": 10.0}
    adjusted = _adjust_alias_band_for_meta(base_spec, None)
    assert adjusted == base_spec
