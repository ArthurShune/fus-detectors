from __future__ import annotations

import numpy as np


def choose_lt_from_coherence(
    tile_T_hw: np.ndarray,
    lt_candidates: tuple[int, ...] = (3, 4, 5, 6, 7, 8),
    method: str = "pd",
    corr_thresh: float = 0.5,
) -> int:
    """
    Select an Lt (slow-time taps) based on temporal coherence within a tile.

    Parameters
    ----------
    tile_T_hw : (T, h, w) array
        Tile cube over slow-time T with spatial axes h, w.
    lt_candidates : tuple of int
        Allowed Lt values (assumed sorted). Must contain at least one element.
    method : {"pd", "iq"}
        "pd" uses mean power vs time, "iq" uses mean real IQ samples.
    corr_thresh : float
        Absolute correlation threshold that marks coherence loss.

    Returns
    -------
    Lt : int
        Chosen slow-time taps from the provided candidate set.
    """
    if tile_T_hw.ndim != 3:
        raise ValueError("tile_T_hw must be (T, h, w)")
    if not lt_candidates:
        raise ValueError("lt_candidates must contain at least one value")

    data = np.asarray(tile_T_hw)
    T = data.shape[0]
    if T < 2:
        return lt_candidates[0]

    def _target(series: np.ndarray) -> int:
        seq = series - series.mean()
        ac_full = np.correlate(seq, seq, mode="full")
        ac = ac_full[T - 1 :]
        if ac[0] <= 0:
            return lt_candidates[0]
        rho = ac / (ac[0] + 1e-12)
        for lag in range(1, min(T, 64)):
            if abs(rho[lag]) < corr_thresh:
                return lag + 3
        return min(8, T // 3 if T >= 3 else 1) + 3

    method_l = method.lower()
    targets: list[int] = []

    if method_l in {"pd", "both"}:
        series_pd = np.mean(np.abs(data) ** 2, axis=(1, 2)).astype(np.float64, copy=False)
        targets.append(_target(series_pd))
        if method_l == "pd":
            # Also consider IQ coherence for conservative choice.
            series_iq = np.mean(data.real, axis=(1, 2)).astype(np.float64, copy=False)
            targets.append(_target(series_iq))
    if method_l in {"iq", "both"}:
        series_iq = np.mean(data.real, axis=(1, 2)).astype(np.float64, copy=False)
        targets.append(_target(series_iq))

    if not targets:
        raise ValueError("method must be 'pd', 'iq', or 'both'")

    target = max(targets)
    for cand in lt_candidates:
        if cand >= target:
            return cand
    return lt_candidates[-1]
