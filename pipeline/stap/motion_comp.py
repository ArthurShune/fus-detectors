# pipeline/stap/motion_comp.py
from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.fft import fft2, ifft2

Array = np.ndarray


def _phase_correlation(ref: Array, img: Array) -> Tuple[float, float]:
    """
    Estimate the sub-pixel (dy, dx) shift that registers `img` to `ref`.

    Parameters
    ----------
    ref, img : 2D arrays (real-valued magnitudes recommended)

    Returns
    -------
    dy, dx : float
        Signed shifts in pixels (positive = downward / rightward).
    """
    A = np.asarray(ref, dtype=np.float64)
    B = np.asarray(img, dtype=np.float64)
    if A.shape != B.shape:
        raise ValueError("Images for phase correlation must share the same shape.")

    F = fft2(A)
    G = fft2(B)
    R = F * np.conj(G)
    denom = np.abs(R) + 1e-12
    R /= denom
    r = np.real(ifft2(R))

    max_y, max_x = np.unravel_index(np.argmax(r), r.shape)
    H, W = r.shape
    cy = max_y if max_y <= H // 2 else max_y - H
    cx = max_x if max_x <= W // 2 else max_x - W

    def _subpixel(coord: int, axis_len: int, axis: int) -> float:
        """Parabolic refinement around the peak along `axis` (0 -> y, 1 -> x)."""

        def sample_y(idx: int) -> float:
            return float(r[(idx % H, max_x)])

        def sample_x(idx: int) -> float:
            return float(r[(max_y, idx % W)])

        sample = sample_y if axis == 0 else sample_x

        y1 = sample(coord - 1)
        y2 = sample(coord)
        y3 = sample(coord + 1)
        denom_ = y1 - 2.0 * y2 + y3
        if abs(denom_) < 1e-12:
            return 0.0
        delta = 0.5 * (y1 - y3) / denom_
        return float(np.clip(delta, -0.5, 0.5))

    dy_ref = _subpixel(max_y, H, axis=0)
    dx_ref = _subpixel(max_x, W, axis=1)
    return float(cy + dy_ref), float(cx + dx_ref)


def _fourier_shift_2d(img: Array, dy: float, dx: float) -> Array:
    """
    Apply a sub-pixel shift to a 2D array via the Fourier shift theorem.

    Works for complex- or real-valued arrays. The dtype of the input is preserved.
    """
    H, W = img.shape
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.fftfreq(W)[None, :]
    phase = np.exp(-2j * np.pi * (ky * dy + kx * dx))
    F = fft2(img)
    shifted = ifft2(F * phase)
    return shifted.astype(img.dtype, copy=False)


def register_tile_slowtime(
    tile_T_hw: Array,
    ref_strategy: str = "median",
) -> tuple[Array, Array]:
    """
    Register a space-time tile by estimating per-frame shifts.

    Parameters
    ----------
    tile_T_hw : (T, h, w) complex or real array
    ref_strategy : {"median", "first"}
        Reference frame to align to. "median" improves robustness to outliers.

    Returns
    -------
    registered : ndarray
        Registered tile with the same shape/dtype as the input.
    shifts : ndarray, shape (T, 2)
        Per-frame (dy, dx) shifts applied (frame 0 shift is (0,0)).
    """
    if tile_T_hw.ndim != 3:
        raise ValueError("Expected tile_T_hw with shape (T, h, w)")
    T, h, w = tile_T_hw.shape
    if T < 1:
        raise ValueError("Tile must contain at least one frame.")

    mag = np.abs(tile_T_hw)
    if ref_strategy not in {"median", "first"}:
        raise ValueError("ref_strategy must be 'median' or 'first'")
    ref = np.median(mag, axis=0) if ref_strategy == "median" else mag[0]

    registered = np.empty_like(tile_T_hw)
    shifts = np.zeros((T, 2), dtype=np.float32)

    for t in range(T):
        if t == 0 and ref_strategy == "first":
            dy, dx = 0.0, 0.0
        else:
            dy, dx = _phase_correlation(ref, mag[t])
        shifts[t] = (dy, dx)
        registered[t] = _fourier_shift_2d(tile_T_hw[t], dy, dx)

    return registered, shifts
