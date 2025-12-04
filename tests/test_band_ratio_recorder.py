import numpy as np

from sim.kwave.common import (
    BandRatioRecorder,
    BandRatioSpec,
    _tile_scores_to_map,
)


def _make_series(prf_hz: float, freq_hz: float, amp: float, length: int) -> np.ndarray:
    t = np.arange(length, dtype=np.float32)
    phase = 2.0 * np.pi * freq_hz / prf_hz
    tone = amp * np.exp(1j * phase * t)
    return tone.astype(np.complex64, copy=False)


def test_band_ratio_recorder_alias_suppression() -> None:
    prf = 1000.0
    spec = BandRatioSpec(
        flow_low_hz=120.0,
        flow_high_hz=200.0,
        alias_center_hz=400.0,
        alias_width_hz=10.0,
    )
    recorder = BandRatioRecorder(prf, tile_count=2, spec=spec, tapers=3, bandwidth=2.0)
    length = 256
    rng = np.random.default_rng(1234)
    noise = 0.05 * (rng.standard_normal(length) + 1j * rng.standard_normal(length))
    series_bg = (
        _make_series(prf, 150.0, 1.0, length) + _make_series(prf, 400.0, 2.5, length) + noise
    )
    series_flow = (
        _make_series(prf, 150.0, 2.0, length) + _make_series(prf, 400.0, 0.1, length) + noise
    )
    recorder.observe(0, series_bg, tile_is_bg=True)
    recorder.observe(1, series_flow, tile_is_bg=False)
    scores, tele = recorder.finalize()
    assert scores[1] > scores[0]
    assert tele["br_bg_tiles"] == 1
    assert tele["br_flow_bins"] >= 1
    assert tele["br_alias_bins"] >= 1


def test_tile_scores_to_map_average() -> None:
    H, W = 8, 8
    tile_hw = (4, 4)
    stride = 4
    scores = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    br_map = _tile_scores_to_map(scores, (H, W), tile_hw, stride)
    assert br_map.shape == (H, W)
    # Each quadrant should match the corresponding tile score
    assert np.allclose(br_map[0:4, 0:4], 1.0)
    assert np.allclose(br_map[0:4, 4:8], 2.0)
    assert np.allclose(br_map[4:8, 0:4], 3.0)
    assert np.allclose(br_map[4:8, 4:8], 4.0)
