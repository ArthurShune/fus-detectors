import numpy as np

from sim.kwave.common import _psd_peak_and_span_hz


def _synthetic_cube(freq_hz: float, prf_hz: float, T: int = 64) -> np.ndarray:
    t = np.arange(T, dtype=np.float32)
    tone = np.exp(1j * 2.0 * np.pi * freq_hz * t / prf_hz, dtype=np.complex64)
    noise = 0.05 * (
        np.random.randn(T, 2, 2).astype(np.float32)
        + 1j * np.random.randn(T, 2, 2).astype(np.float32)
    )
    cube = noise
    cube[:, 0, 0] += tone
    return cube.astype(np.complex64)


def test_psd_peak_matches_injected_frequency():
    prf = 4000.0
    freq = 500.0
    cube = _synthetic_cube(freq, prf)
    f_peak, span = _psd_peak_and_span_hz(cube, prf_hz=prf, min_rel=0.05, max_rel=0.45)
    assert f_peak is not None and span is not None
    assert abs(f_peak - freq) < 80.0
    assert span > 0.0


def test_psd_returns_none_for_short_sequences():
    cube = np.zeros((8, 2, 2), dtype=np.complex64)
    f_peak, span = _psd_peak_and_span_hz(cube, prf_hz=3000.0)
    assert f_peak is None and span is None


def test_psd_handles_noise_only_cube():
    rng = np.random.default_rng(3)
    cube = (rng.standard_normal((64, 4, 4)) + 1j * rng.standard_normal((64, 4, 4))).astype(
        np.complex64
    )
    f_peak, span = _psd_peak_and_span_hz(cube, prf_hz=3000.0, energy_q=0.95)
    if f_peak is not None:
        assert abs(f_peak) <= 0.49 * 3000.0
        assert span is not None and span > 0.0
