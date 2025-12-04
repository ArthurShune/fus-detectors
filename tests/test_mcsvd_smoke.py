"""Smoke test for the MC-SVD baseline utilities."""

from __future__ import annotations

import numpy as np

from sim.kwave.common import _baseline_pd_mcsvd


def _fft_shift(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Deterministic sub-pixel shift using Fourier shift theorem."""
    h, w = img.shape
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.fftfreq(w)[None, :]
    phase = np.exp(-2j * np.pi * (ky * dy + kx * dx))
    return np.fft.ifft2(np.fft.fft2(img) * phase).astype(np.complex64)


def test_mcsvd_registration_and_projection_smoke() -> None:
    rng = np.random.default_rng(0)
    t, h, w = 64, 48, 48

    # Build a low-rank tissue component plus a weaker flow component.
    tissue = (rng.standard_normal((t, 2)) @ rng.standard_normal((2, h * w))).reshape(t, h, w)
    tissue = tissue.astype(np.complex64)
    flow = (rng.standard_normal((t, h, w)) + 1j * rng.standard_normal((t, h, w))).astype(
        np.complex64
    ) * 0.1
    cube = tissue + flow

    # Inject deterministic sub-pixel motion so registration has work to do.
    shifts = []
    for idx in range(1, t):
        dy = 0.6 * np.sin(idx / t)
        dx = -0.9 * np.cos(idx / t)
        cube[idx] = _fft_shift(cube[idx], dy, dx)
        shifts.append(np.hypot(dy, dx))

    pd_map, telemetry = _baseline_pd_mcsvd(
        cube, reg_enable=True, reg_reference="first", svd_rank=2, reg_subpixel=4
    )

    assert pd_map.shape == (h, w)
    assert np.isfinite(pd_map).all()
    assert telemetry["baseline_type"] == "mc_svd"
    assert telemetry["svd_rank_removed"] == 2
    assert telemetry["reg_enable"] is True
    expected_shift = float(np.sqrt(np.mean(np.square(shifts))))
    # Registration should observe non-zero motion roughly matching injected motion.
    assert telemetry["reg_shift_rms"] > 0.1
    assert telemetry["reg_shift_rms"] <= expected_shift + 0.5
    assert telemetry["reg_shift_p90"] >= telemetry["reg_shift_rms"]
