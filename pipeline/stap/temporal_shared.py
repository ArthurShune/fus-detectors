"""
Shared temporal STAP helpers used by both the slow and fast paths.

These are deliberately lightweight wrappers around existing primitives in
`pipeline.stap.temporal`. They centralize control-plane logic (Hankel
construction, robust covariance, shrinkage, lambda conditioning) so both paths
can call the same code without duplicating math.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch not available
    torch = None  # type: ignore

try:
    # Reuse existing utilities to avoid diverging behavior.
    from pipeline.stap.temporal import robust_covariance, split_fd_grid_by_motion
except Exception:  # pragma: no cover - during partial imports
    robust_covariance = None  # type: ignore
    split_fd_grid_by_motion = None  # type: ignore


def build_temporal_hankels_batch(
    cube_B_T_hw: np.ndarray | "torch.Tensor",
    Lt: int,
    *,
    center: bool = True,
    device: str | None = None,
    dtype: "torch.dtype" = None,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """
    Batched Hankel construction and pooled SCM covariance.

    Parameters
    ----------
    cube_B_T_hw : (B,T,H,W) array/tensor (B can be 1 for slow path).
    Lt : int
        Temporal aperture.
    center : bool
        Subtract per-row mean across Hankel columns.
    """
    if torch is None:  # pragma: no cover - torch unavailable
        raise ImportError("torch is required for build_temporal_hankels_batch")
    x = torch.as_tensor(cube_B_T_hw)
    if dtype is not None:
        x = x.to(dtype=dtype)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    x = x.to(device)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    B, T, h, w = x.shape
    if Lt < 2 or Lt >= T:
        raise ValueError(f"Need 2 <= Lt < T (got Lt={Lt}, T={T})")

    # torch.unfold returns (B, N, Lt, h, w); permute to (B, Lt, N, h, w)
    S = x.unfold(dimension=1, size=Lt, step=1)
    S = S.permute(0, 2, 1, 3, 4).contiguous()
    if center:
        S = S - S.mean(dim=2, keepdim=True)

    # SCM pooled per tile: flatten spatial + snapshot dims
    S_flat = S.permute(0, 1, 3, 4, 2).contiguous().view(B, Lt, -1)
    R_scm = torch.matmul(S_flat, S_flat.conj().transpose(-2, -1)) / float(S_flat.shape[-1])
    return S, R_scm


def robust_temporal_cov_batch(
    X_B_Lt_K: "torch.Tensor",
    *,
    estimator: str = "huber",
    huber_c: float = 5.0,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> Tuple["torch.Tensor", List[Dict[str, float]]]:
    """
    Batched robust covariance; thin wrapper around `robust_covariance` per tile.
    """
    if torch is None or robust_covariance is None:  # pragma: no cover
        raise ImportError("torch/robust_covariance required for robust_temporal_cov_batch")
    if X_B_Lt_K.ndim != 3:
        raise ValueError(f"Expected (B,Lt,K) input, got shape {tuple(X_B_Lt_K.shape)}")
    R_list: List[torch.Tensor] = []
    info_list: List[Dict[str, float]] = []
    B = X_B_Lt_K.shape[0]
    for b in range(B):
        Rb, tel = robust_covariance(
            X_B_Lt_K[b],
            method=estimator,
            huber_c=huber_c,
            max_iter=max_iter,
            tol=tol,
        )
        Rb_t = torch.as_tensor(Rb, device=X_B_Lt_K.device, dtype=X_B_Lt_K.dtype)
        R_list.append(Rb_t)
        tel_dict: Dict[str, float] = {}
        if hasattr(tel, "__dict__"):
            for k, v in tel.__dict__.items():
                if isinstance(v, (int, float)):
                    tel_dict[k] = float(v)
        info_list.append(tel_dict)
    R_stack = torch.stack(R_list, dim=0)
    return R_stack, info_list


def shrinkage_alpha_for_kappa_batch(
    R_B_Lt_Lt: "torch.Tensor",
    *,
    kappa_target: float = 200.0,
) -> "torch.Tensor":
    """
    Per-tile shrinkage alpha toward mu*I using the same heuristic as
    `_shrinkage_alpha_for_kappa`, but implemented directly to avoid
    circular imports.
    """
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required for shrinkage_alpha_for_kappa_batch")
    if R_B_Lt_Lt.ndim != 3:
        raise ValueError(f"Expected (B,Lt,Lt) tensor, got shape {tuple(R_B_Lt_Lt.shape)}")
    if kappa_target <= 1.0:
        return torch.zeros((R_B_Lt_Lt.shape[0],), device=R_B_Lt_Lt.device, dtype=torch.float32)
    # Eigenvalues per tile: (B,Lt)
    evals = torch.linalg.eigvalsh(R_B_Lt_Lt).real
    mu = evals.mean(dim=1)
    e_min = evals.min(dim=1).values
    e_max = evals.max(dim=1).values
    num = e_max - float(kappa_target) * e_min
    den = (float(kappa_target) - 1.0) * mu + (e_max - float(kappa_target) * e_min)
    alpha = torch.where(den > 0.0, num / den, torch.zeros_like(den))
    alpha_clamped = torch.clamp(alpha, 0.0, 1.0)
    return alpha_clamped.to(dtype=torch.float32)


def conditioned_lambda_batch(
    R_B_Lt_Lt: "torch.Tensor",
    base_lambda: float | Sequence[float],
    kappa_target: float,
    *,
    eps: float = 1e-8,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    Batched λ conditioning: find λ ≥ base_lambda s.t. κ(R+λI) ≤ kappa_target.

    Returns (lambda_out_B, kappa_out_B, lambda_needed_B).
    """
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required for conditioned_lambda_batch")
    if R_B_Lt_Lt.ndim != 3:
        raise ValueError(f"Expected (B,Lt,Lt) tensor, got shape {tuple(R_B_Lt_Lt.shape)}")
    kappa_target = max(float(kappa_target), 1.0 + eps)
    base = torch.as_tensor(base_lambda, device=R_B_Lt_Lt.device, dtype=torch.float64)
    if base.numel() == 1:
        base = base.expand(R_B_Lt_Lt.shape[0])
    base = base.to(dtype=torch.float64)

    # Eigenvalues per tile
    herm = 0.5 * (R_B_Lt_Lt + R_B_Lt_Lt.conj().transpose(-2, -1))
    evals = torch.linalg.eigvalsh(herm).real
    evals_clamped = torch.clamp(evals, min=0.0)
    ev_min = evals_clamped[:, 0]
    ev_max = evals_clamped[:, -1]

    denom = torch.clamp(torch.as_tensor(kappa_target - 1.0), min=eps)
    lam_needed = torch.where(
        ev_min > 0.0,
        torch.clamp((ev_max - kappa_target * ev_min) / denom, min=0.0),
        torch.clamp(ev_max / denom, min=0.0),
    )
    lam_out = torch.maximum(base, lam_needed)

    ev_min_loaded = ev_min + lam_out
    ev_max_loaded = ev_max + lam_out
    kappa_out = ev_max_loaded / torch.clamp(ev_min_loaded, min=eps)
    return lam_out.to(R_B_Lt_Lt.dtype), kappa_out.to(torch.float32), lam_needed.to(torch.float32)


def build_fd_grid_span(
    prf_hz: float,
    Lt: int,
    fd_span_rel: Tuple[float, float],
    grid_step_rel: float,
    fd_min_abs_hz: float,
    *,
    min_pts: int = 3,
    max_pts: int = 21,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Simple span-based Doppler grid builder (mirrors fast-path default logic).

    Returns
    -------
    fd_grid : np.ndarray
        1-D grid of frequencies (Hz), symmetric and odd length.
    meta : dict
        span_hz, grid_step_hz, lines.
    """
    base = float(prf_hz) / max(float(Lt), 1.0)
    span_rel_low, span_rel_high = fd_span_rel
    span_rel = max(abs(span_rel_low), abs(span_rel_high))

    # For Lt=8 in k-Wave brain fUS–like regimes, align the fast path span with
    # the slow-path behavior: use a single off-DC bin (±base) and a 3-point grid.
    if Lt == 8 and max_pts <= 5:
        span_hz = base
        step_hz = base
        lines = 3
    else:
        span_hz = max(float(fd_min_abs_hz), span_rel * base)
        step_hz = max(1e-6, float(grid_step_rel) * base)
        lines = max(min_pts, int(2 * math.ceil(span_hz / step_hz) + 1))
        lines = int(min(lines, max_pts))
        if lines % 2 == 0:
            lines = lines + 1 if lines + 1 <= max_pts else max(lines - 1, min_pts | 1)
    fd_grid = np.linspace(-span_hz, span_hz, lines, dtype=np.float64)
    meta = {"span_hz": float(span_hz), "grid_step_hz": float(step_hz), "lines": int(lines)}
    return fd_grid, meta


__all__ = [
    "build_temporal_hankels_batch",
    "robust_temporal_cov_batch",
    "shrinkage_alpha_for_kappa_batch",
    "conditioned_lambda_batch",
    "build_fd_grid_span",
    "split_fd_grid_by_motion",
]


__all__ = [
    "build_temporal_hankels_batch",
    "robust_temporal_cov_batch",
    "shrinkage_alpha_for_kappa_batch",
    "conditioned_lambda_batch",
    "split_fd_grid_by_motion",
]
