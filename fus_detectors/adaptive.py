from __future__ import annotations

import numpy as np
from scipy.signal.windows import dpss


def multi_taper_psd_batch(
    series_batch: np.ndarray,
    prf_hz: float,
    *,
    tapers: int,
    bandwidth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate non-negative-frequency PSDs for a batch of complex series."""

    batch = np.asarray(series_batch, dtype=np.complex64)
    if batch.ndim != 2:
        raise ValueError(
            f"Expected complex series batch with shape (B, T); received {tuple(batch.shape)}."
        )
    if batch.shape[1] <= 0:
        raise ValueError("Empty series batch passed to multi_taper_psd_batch.")
    tapers = max(1, int(tapers))
    bandwidth = max(float(bandwidth), 1.0)

    taper_bank = dpss(batch.shape[1], bandwidth, Kmax=tapers, sym=False).astype(
        np.float32, copy=False
    )
    tapered = batch[:, None, :] * taper_bank[None, :, :]
    spectrum = np.fft.fft(tapered, axis=-1)
    power = np.abs(spectrum) ** 2
    psd_full = power.mean(axis=1)

    freqs_full = np.fft.fftfreq(batch.shape[1], d=1.0 / float(prf_hz))
    nyquist_freq = 0.5 * float(prf_hz)
    pos_mask = (freqs_full >= 0.0) | np.isclose(freqs_full, -nyquist_freq, atol=1e-6)
    freqs = np.abs(freqs_full[pos_mask]).astype(np.float32, copy=False)
    psd = psd_full[:, pos_mask].astype(np.float32, copy=False)
    return freqs, psd


def compute_guard_fraction_tiles(
    tile_batch: np.ndarray,
    *,
    prf_hz: float,
    flow_band_hz: tuple[float, float],
    alias_center_hz: float,
    alias_width_hz: float,
    tapers: int,
    bandwidth: float,
) -> np.ndarray:
    """Compute per-tile guard-band contamination from coherent tile-mean series."""

    tiles = np.asarray(tile_batch, dtype=np.complex64)
    if tiles.ndim != 4:
        raise ValueError(
            f"Expected tile batch with shape (B, T, H, W); received {tuple(tiles.shape)}."
        )

    series_batch = np.mean(tiles.reshape(tiles.shape[0], tiles.shape[1], -1), axis=2)
    freqs, psd = multi_taper_psd_batch(
        series_batch,
        prf_hz=float(prf_hz),
        tapers=int(tapers),
        bandwidth=float(bandwidth),
    )

    flow_low = float(min(flow_band_hz))
    flow_high = float(max(flow_band_hz))
    alias_half = float(max(alias_width_hz, 0.0))
    alias_low = float(alias_center_hz - alias_half)
    alias_high = float(alias_center_hz + alias_half)

    mask_f = (freqs >= flow_low) & (freqs <= flow_high)
    mask_a = (freqs >= alias_low) & (freqs <= alias_high)
    mask_g = np.zeros_like(mask_f, dtype=bool)
    if alias_low > flow_high:
        mask_g = (freqs >= flow_high) & (freqs <= alias_low)
        mask_g[mask_f | mask_a] = False

    if not np.any(mask_g):
        return np.zeros((tiles.shape[0],), dtype=np.float32)

    total = np.sum(psd, axis=1, dtype=np.float64) + 1e-12
    guard_energy = np.sum(psd[:, mask_g], axis=1, dtype=np.float64)
    return np.asarray(guard_energy / total, dtype=np.float32)


def choose_promoted_tiles(
    guard_fraction_tiles: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    """Apply the frozen tile-mean guard threshold used by the adaptive variant."""

    guard = np.asarray(guard_fraction_tiles, dtype=np.float32).reshape(-1)
    finite = np.isfinite(guard)
    promote = np.zeros_like(guard, dtype=bool)
    promote[finite] = guard[finite] >= float(threshold)
    return promote
