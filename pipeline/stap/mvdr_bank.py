# pipeline/stap/mvdr_bank.py
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from ..gpu.linalg import get_device, to_tensor
from .fuse import lse_fuse
from .mvdr import apply_weights_to_snapshots, mvdr_weights, steering_vector


def choose_fd_grid_auto(
    prf_hz: float,
    Lt: int,
    median_shift_px: float,
    base_span_rel: float = 0.25,
    max_span_rel: float = 1.60,
    step_rel: float = 0.08,
    min_pts: int = 5,
    max_pts: int = 21,
) -> list[float]:
    """
    Pick a symmetric Doppler grid (Hz) driven by registration jitter magnitude.

    Parameters
    ----------
    prf_hz : float
        Pulse-repetition frequency.
    Lt : int
        Slow-time taps in the snapshots (>=1).
    median_shift_px : float
        Median |shift| (pixels) from motion registration.
    base_span_rel : float
        Minimum fractional span (relative to PRF/Lt) at zero jitter.
    max_span_rel : float
        Maximum fractional span once jitter saturates.
    step_rel : float
        Grid spacing relative to PRF/Lt.
    min_pts, max_pts : int
        Clamp the number of Doppler lines (forces odd count).
    """
    if Lt <= 0:
        raise ValueError("Lt must be positive for Doppler grid selection.")
    if min_pts < 1 or max_pts < min_pts:
        raise ValueError("min_pts must be >=1 and <= max_pts.")
    if step_rel <= 0:
        raise ValueError("step_rel must be positive.")

    norm = min(1.0, max(0.0, median_shift_px / 0.6))
    span_rel = base_span_rel + norm * (max_span_rel - base_span_rel)
    denom = max(Lt, 1)
    span_hz = span_rel * (prf_hz / denom)
    step_hz = step_rel * (prf_hz / denom)
    step_hz = max(step_hz, span_hz / (max_pts - 1) if max_pts > 1 else step_hz)

    lines = 2 * int(np.ceil(span_hz / step_hz)) + 1
    lines = int(np.clip(lines, min_pts, max_pts))
    if lines % 2 == 0:
        lines = min(lines + 1, max_pts) if lines < max_pts else lines - 1
        lines = max(lines, min_pts if min_pts % 2 == 1 else min_pts + 1)

    grid = np.linspace(-span_hz, span_hz, lines, dtype=np.float64)
    return grid.tolist()


def build_steering_bank(
    h: int,
    w: int,
    Lt: int,
    prf_hz: float,
    fd_grid: Sequence[float],
    device: Optional[str] = None,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """
    Construct a bank of steering vectors over the specified Doppler offsets.

    Parameters
    ----------
    h, w : int
        Spatial tile dimensions.
    Lt : int
        Number of slow-time taps used in the snapshots.
    prf_hz : float
        Pulse-repetition frequency (Hz).
    fd_grid : sequence of float
        Doppler frequency offsets (Hz).
    device : str, optional
        Target device for the returned tensor.
    dtype : torch.dtype, optional
        Complex dtype for the steering vectors.

    Returns
    -------
    bank : torch.Tensor, shape (F, M)
        Stack of steering vectors (M = h * w * Lt). F = len(fd_grid).
    """
    if len(fd_grid) == 0:
        raise ValueError("fd_grid must contain at least one Doppler offset.")
    dev = get_device(device)
    bank = [
        steering_vector(
            h=h,
            w=w,
            Lt=Lt,
            fd_hz=float(fd),
            prf_hz=prf_hz,
            device=str(dev),
            dtype=dtype,
        )
        for fd in fd_grid
    ]
    return torch.stack(bank, dim=0)


def mvdr_glrt_bank(
    R: torch.Tensor | np.ndarray,
    X: torch.Tensor | np.ndarray,
    s_bank: torch.Tensor | np.ndarray,
    diag_load: float = 1e-3,
    fuse: str = "max",
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply MVDR across a steering bank and fuse responses (GLRT-style).

    Parameters
    ----------
    R : (M, M) covariance (numpy or torch).
    X : (M, N) snapshots.
    s_bank : (F, M) steering bank.
    diag_load : float
        Diagonal loading factor passed to mvdr_weights.
    fuse : {"max", "sos", "lse"}
        Fusion rule across the bank: max power, sum-of-squares, or log-sum-exp.
    device : str, optional
        Device to run the computation on.

    Returns
    -------
    scores : torch.Tensor, shape (F, N)
        Per-steer power |w_f^H x_t|^2.
    fused : torch.Tensor, shape (N,)
        Fused statistic across the bank.
    """
    Rt = to_tensor(R, device=device)
    if Rt.dim() != 2:
        raise ValueError("R must have shape (M, M).")
    Xt = to_tensor(X, device=Rt.device, dtype=Rt.dtype)
    if Xt.dim() != 2 or Xt.shape[0] != Rt.shape[0]:
        raise ValueError("X must be (M, N) matching the dimension of R.")
    sb = to_tensor(s_bank, device=Rt.device, dtype=Rt.dtype)
    if sb.dim() != 2 or sb.shape[1] != Rt.shape[0]:
        raise ValueError("s_bank must be (F, M) matching the dimension of R.")

    F = sb.shape[0]
    scores = []
    for f in range(F):
        steer = sb[f]
        res = mvdr_weights(Rt, steer, diag_load=diag_load, device=str(Rt.device), dtype=Rt.dtype)
        w = res.w.squeeze(0)
        y = apply_weights_to_snapshots(w, Xt, device=str(Rt.device), dtype=Rt.dtype)
        scores.append(torch.real(y.conj() * y))
    S = torch.stack(scores, dim=0)

    fuse_lower = fuse.lower()
    if fuse_lower in {"max", "maxglrt"}:
        fused = torch.amax(S, dim=0)
    elif fuse_lower in {"sos", "sum", "glrt"}:
        fused = torch.sum(S, dim=0)
    elif fuse_lower in {"lse", "soft", "softmax"}:
        fused = lse_fuse(S, tau=2.0)
    else:
        raise ValueError("fuse must be 'max', 'sos', or 'lse'.")
    return S, fused
