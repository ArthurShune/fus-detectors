# pipeline/detect/cfar.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import uniform_filter

EPS = 1e-12


# =========================== Temporal smoothing ==============================


def moving_average_pd(pd_T_hw: ArrayLike, W: int) -> np.ndarray:
    """
    Compute temporal moving-average of power-doppler sequence.

    Parameters
    ----------
    pd_T_hw : array-like, shape (T, H, W)
        PD(t, y, x) sequence; typically PD(t) = |w^H x_t|^2.
    W : int
        Window length (>=1). Output length is T' = T - W + 1.

    Returns
    -------
    pd_smooth_Tp_hw : np.ndarray, shape (T - W + 1, H, W)
        Smoothed PD(t) = mean over last W frames.
    """
    pd = np.asarray(pd_T_hw, dtype=np.float64)
    if pd.ndim != 3:
        raise ValueError("pd_T_hw must be (T,H,W)")
    T, H, Wd = pd.shape
    if W < 1 or W > T:
        raise ValueError(f"W must be in [1, T]. Got W={W}, T={T}")
    # cumulative sum along time, then sliding mean
    c = np.cumsum(pd, axis=0, dtype=np.float64)
    out = c[W - 1 :] - np.concatenate([np.zeros_like(pd[:1]), c[:-W]], axis=0)
    out /= float(W)
    return out.astype(np.float32)


# =========================== Robust null statistics ==========================


@dataclass(frozen=True)
class NullStats:
    """
    Robust null statistics for CFAR-like normalization.
    The score we use is S = PD / mu_hat, so E[S]≈1 under null if mu_hat≈E[PD].
    """

    mu: np.ndarray  # (H, W) robust mean under null
    m2: np.ndarray  # (H, W) robust second moment under null, E[PD^2]
    var: np.ndarray  # (H, W) robust variance under null, m2 - mu^2
    n_samples: int  # total time samples used


def _median_of_means(x: np.ndarray, blocks: int) -> np.ndarray:
    """
    Compute median-of-means along the first axis of x.

    x : (N, D1, D2, ...)
    Returns : (D1, D2, ...)
    """
    N = x.shape[0]
    b = int(blocks)
    if b < 2:
        return x.mean(axis=0)
    # trim to multiple of b
    M = (N // b) * b
    if M < b:
        return x.mean(axis=0)
    x = x[:M]
    m = M // b
    xb = x.reshape(b, m, *x.shape[1:]).mean(axis=1)  # (b, ...)
    # median across blocks
    return np.median(xb, axis=0)


def robust_null_stats_from_pd(
    pd_null_list: Iterable[np.ndarray],
    blocks: int = 20,
) -> NullStats:
    """
    Estimate robust per-pixel (H,W) mean and variance of PD under null
    using median-of-means across time.

    Parameters
    ----------
    pd_null_list : iterable of arrays (T_i, H, W)
        One or more null sequences (e.g., calibration tiles/time).
    blocks : int
        Number of blocks for median-of-means (default 20).

    Returns
    -------
    NullStats
    """
    # Stack time along axis 0
    seqs = [np.asarray(a, dtype=np.float64) for a in pd_null_list]
    if len(seqs) == 0:
        raise ValueError("pd_null_list is empty.")
    H, W = seqs[0].shape[1:]
    if not all(s.shape[1:] == (H, W) for s in seqs):
        raise ValueError("All sequences must share the same (H,W).")
    pd_all = np.concatenate(seqs, axis=0)  # (N, H, W)
    N = pd_all.shape[0]
    # robust mean and second moment
    mu_hat = _median_of_means(pd_all, blocks=blocks)
    m2_hat = _median_of_means(pd_all**2, blocks=blocks)
    var_hat = np.maximum(m2_hat - mu_hat**2, EPS)
    return NullStats(
        mu=mu_hat.astype(np.float32),
        m2=m2_hat.astype(np.float32),
        var=var_hat.astype(np.float32),
        n_samples=int(N),
    )


# =============================== Scores ======================================


def score_from_null_mean(pd_T_hw: ArrayLike, null: NullStats, W: int) -> np.ndarray:
    """
    Compute CFAR-style score S = (temporal-mean PD) / mu_hat, per time step.

    Parameters
    ----------
    pd_T_hw : (T,H,W) PD sequence.
    null : NullStats with mu (H,W).
    W : moving average window length.

    Returns
    -------
    S_Tp_hw : (T - W + 1, H, W) scores
    """
    pd_smooth = moving_average_pd(pd_T_hw, W=W)  # (T',H,W)
    mu = np.asarray(null.mu, dtype=pd_smooth.dtype)  # (H,W)
    return (pd_smooth / np.maximum(mu[None, ...], EPS)).astype(np.float32)


# ========================= Optional spatial CA-CFAR ===========================


def ca_cfar_local_mean(
    pd_hw: ArrayLike,
    guard: int = 2,
    train: int = 6,
    mode: str = "reflect",
) -> np.ndarray:
    """
    Cell-Averaging CFAR local mean estimate over a ring (square) around CUT.

    Parameters
    ----------
    pd_hw : (H,W) array
        Instantaneous (or smoothed) PD map.
    guard : int
        Half-size of guard window (excluded region) around CUT.
    train : int
        Half-size of training window; the ring is [-(g+t), +(g+t)] \\ [-g, +g].
    mode : str
        Boundary mode for uniform_filter (reflect by default).

    Returns
    -------
    mu_hw : (H,W) local mean excluding guard window.
    """
    PD = np.asarray(pd_hw, dtype=np.float64)
    H, W = PD.shape
    win_big = 2 * (guard + train) + 1
    win_small = 2 * guard + 1

    sum_big = uniform_filter(PD, size=win_big, mode=mode) * (win_big**2)
    sum_small = uniform_filter(PD, size=win_small, mode=mode) * (win_small**2)
    ring_sum = sum_big - sum_small
    M_ring = (win_big**2) - (win_small**2)
    mu = ring_sum / max(M_ring, 1)
    return mu.astype(np.float32)


def score_cafar(pd_hw: ArrayLike, guard: int = 2, train: int = 6) -> np.ndarray:
    """
    Compute spatial CA-CFAR score S = PD / mu_local for a single frame.
    """
    PD = np.asarray(pd_hw, dtype=np.float32)
    mu = ca_cfar_local_mean(PD, guard=guard, train=train)
    return (PD / np.maximum(mu, EPS)).astype(np.float32)
