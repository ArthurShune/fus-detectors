import numpy as np

from sim.kwave.common import _multi_taper_psd


def test_multi_taper_psd_identifies_tone_peak():
    prf = 1000.0
    n = 256
    freq = 150.0
    t = np.arange(n)
    tone = np.exp(1j * 2 * np.pi * freq * t / prf)
    freqs, psd = _multi_taper_psd(tone, prf, tapers=3, bandwidth=2.0)
    peak = float(freqs[int(np.argmax(psd))])
    assert abs(peak - freq) < 5.0
    assert psd.max() > 10 * psd.mean()


def test_multi_taper_psd_requires_data():
    try:
        _multi_taper_psd(np.array([], dtype=np.complex64), 1000.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for empty input")
