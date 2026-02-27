"""
Shared temporal STAP helpers used by both the slow and fast paths.

These are deliberately lightweight wrappers around existing primitives in
`pipeline.stap.temporal`. They centralize control-plane logic (Hankel
construction, robust covariance, shrinkage, lambda conditioning) so both paths
can call the same code without duplicating math.
"""

from __future__ import annotations

import os
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch not available
    torch = None  # type: ignore

try:
    # Import directly to avoid circular imports (`temporal.py` depends on this module).
    from pipeline.stap.robust_cov import robust_covariance
except Exception:  # pragma: no cover - during partial imports
    robust_covariance = None  # type: ignore


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

    # torch.unfold appends the window dimension at the end:
    #   (B, N, h, w, Lt) for an input (B, T, h, w) unfolded along dim=1.
    # Reorder to the canonical (B, Lt, N, h, w) layout used throughout STAP.
    S = x.unfold(dimension=1, size=Lt, step=1)
    S = S.permute(0, 4, 1, 2, 3).contiguous()

    # Optional training snapshot subsampling to limit effective support, mirroring
    # the slow-path behavior. Controlled via env vars:
    #   STAP_SNAPSHOT_STRIDE : integer stride along the N axis (>=1).
    #   STAP_MAX_SNAPSHOTS   : integer cap on N after striding.
    stride_env = os.getenv("STAP_SNAPSHOT_STRIDE", "").strip()
    max_env = os.getenv("STAP_MAX_SNAPSHOTS", "").strip()
    try:
        stride = int(stride_env) if stride_env else 1
    except ValueError:
        stride = 1
    if stride < 1:
        stride = 1
    # Only apply striding when it still leaves at least two Hankel columns.
    # This avoids accidentally collapsing very short ensembles (e.g., T=17, Lt=16 -> N=2)
    # down to N=1, which can destabilize covariance estimation and regress ROC.
    if stride > 1 and int(S.shape[2]) > int(stride):
        S = S[:, :, ::stride, :, :].contiguous()
    try:
        max_snaps = int(max_env) if max_env else None
    except ValueError:
        max_snaps = None
    if max_snaps is not None and max_snaps > 0 and S.shape[2] > max_snaps:
        N = int(S.shape[2])
        idx = torch.linspace(
            0, N - 1, steps=int(max_snaps), device=S.device, dtype=torch.long
        )
        S = S.index_select(2, idx).contiguous()
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
    return_eigs: bool = False,
) -> "torch.Tensor" | Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
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

    def _eig_extrema_lanczos(
        A_B_N_N: "torch.Tensor",
        *,
        iters: int,
        eps: float,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Approximate (ev_min, ev_max) for a batch of Hermitian matrices via Lanczos.

        Deterministic: uses an all-ones initial vector (no RNG / seed dependence).
        """
        B, N = int(A_B_N_N.shape[0]), int(A_B_N_N.shape[1])
        m = max(1, min(int(iters), N))
        q = torch.ones((B, N), device=A_B_N_N.device, dtype=A_B_N_N.dtype)
        q = q / torch.linalg.norm(q, dim=1, keepdim=True).clamp_min(eps)
        q_prev = torch.zeros_like(q)
        q_next = torch.empty_like(q)
        beta_prev = torch.zeros((B,), device=A_B_N_N.device, dtype=torch.float32)

        alphas = torch.empty((B, m), device=A_B_N_N.device, dtype=torch.float32)
        betas = torch.empty((B, max(0, m - 1)), device=A_B_N_N.device, dtype=torch.float32)

        # Reusable buffers: reduce per-iteration allocations and dtype casts.
        w3 = torch.empty((B, N, 1), device=A_B_N_N.device, dtype=A_B_N_N.dtype)
        tmp = torch.empty((B, N), device=A_B_N_N.device, dtype=A_B_N_N.dtype)
        for j in range(m):
            # w = A @ q
            torch.bmm(A_B_N_N, q.unsqueeze(-1), out=w3)
            w = w3.squeeze(-1)
            if j > 0:
                # w -= beta_prev * q_prev
                torch.mul(q_prev, beta_prev.view(B, 1), out=tmp)
                w.sub_(tmp)

            # alpha_j = Re{q^H w} (real for Hermitian A).
            q_ri = torch.view_as_real(q)
            w_ri = torch.view_as_real(w)
            alpha_j = torch.sum(q_ri[..., 0] * w_ri[..., 0] + q_ri[..., 1] * w_ri[..., 1], dim=1)
            alphas[:, j] = alpha_j

            # w -= alpha_j * q
            torch.mul(q, alpha_j.view(B, 1), out=tmp)
            w.sub_(tmp)
            if j < m - 1:
                beta_j = torch.linalg.norm(w, dim=1)
                betas[:, j] = beta_j
                beta_safe = beta_j.clamp_min(eps)
                torch.div(w, beta_safe.view(B, 1), out=q_next)
                q_prev, q, q_next = q, q_next, q_prev
                beta_prev = beta_j

        T = torch.diag_embed(alphas)  # (B,m,m)
        if m > 1:
            idx = torch.arange(m - 1, device=T.device)
            T[:, idx, idx + 1] = betas
            T[:, idx + 1, idx] = betas
        evals = torch.linalg.eigvalsh(T)
        ev_min = evals[:, 0].to(dtype=torch.float32)
        ev_max = evals[:, -1].to(dtype=torch.float32)
        return ev_min, ev_max

    # Eigen-extrema per tile: (B,). In practice, numerical non-Hermitian drift
    # (or near-repeated eigenvalues) can trigger non-convergence in batched
    # eigensolvers. Hermitize and provide a fast Lanczos mode for large Lt on CUDA.
    herm = 0.5 * (R_B_Lt_Lt + R_B_Lt_Lt.conj().transpose(-2, -1))
    mode_env = os.getenv("STAP_SHRINK_EIG_MODE", "").strip().lower()
    if not mode_env or mode_env == "auto":
        mode = "lanczos" if herm.is_cuda and int(herm.shape[1]) >= 32 else "eigvalsh"
    elif mode_env in {"eigvalsh", "exact"}:
        mode = "eigvalsh"
    elif mode_env in {"lanczos", "lanczos_minmax"}:
        mode = "lanczos"
    else:
        raise ValueError(
            f"Unknown STAP_SHRINK_EIG_MODE='{mode_env}'. Expected auto|eigvalsh|lanczos."
        )

    ev_min: torch.Tensor
    ev_max: torch.Tensor
    if mode == "lanczos":
        iters_env = os.getenv("STAP_SHRINK_LANCZOS_ITERS", "").strip()
        try:
            iters = int(iters_env) if iters_env else 16
        except Exception:
            iters = 16
        iters = max(2, int(iters))
        eps = float(os.getenv("STAP_SHRINK_LANCZOS_EPS", "").strip() or 1e-12)
        ev_min, ev_max = _eig_extrema_lanczos(herm, iters=iters, eps=eps)
        # Trace/Lt is equal to mean(eigs) for Hermitian matrices; clamp to keep
        # the shrinkage defined even if tiny numerical negatives are present.
        mu = torch.real(torch.diagonal(herm, dim1=-2, dim2=-1)).mean(dim=1).to(dtype=torch.float32)
        mu = torch.clamp(mu, min=0.0)
        e_min = torch.clamp(ev_min, min=0.0)
        e_max = torch.clamp(ev_max, min=0.0)
    else:
        try:
            evals = torch.linalg.eigvalsh(herm).real
        except Exception:
            B, Lt = herm.shape[0], herm.shape[1]
            eye = torch.eye(Lt, device=herm.device, dtype=herm.dtype)
            evals_list = []
            # Jitter ladder is scaled by mu so it is roughly unitless across tiles.
            jitter_mults = (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)
            for b in range(B):
                Rb = herm[b]
                mu_b = torch.real(torch.diagonal(Rb, dim1=-2, dim2=-1)).mean()
                mu_b_safe = torch.clamp(mu_b, min=0.0) + 1e-12
                evals_b = None
                for mult in jitter_mults:
                    try:
                        evals_b = torch.linalg.eigvalsh(Rb + (float(mult) * mu_b_safe) * eye).real
                        break
                    except Exception:
                        evals_b = None
                if evals_b is None:
                    # Last resort: diagonal approximation (keeps the method defined).
                    evals_b = torch.real(torch.diagonal(Rb, dim1=-2, dim2=-1)).to(dtype=torch.float32)
                evals_list.append(evals_b)
            evals = torch.stack(evals_list, dim=0)

        ev_min = evals.min(dim=1).values.to(dtype=torch.float32)
        ev_max = evals.max(dim=1).values.to(dtype=torch.float32)

        evals_clamped = torch.clamp(evals, min=0.0)
        mu = evals_clamped.mean(dim=1).to(dtype=torch.float32)
        e_min = evals_clamped.min(dim=1).values.to(dtype=torch.float32)
        e_max = evals_clamped.max(dim=1).values.to(dtype=torch.float32)

    num = e_max - float(kappa_target) * e_min
    den = (float(kappa_target) - 1.0) * mu + (e_max - float(kappa_target) * e_min)
    alpha = torch.where(den > 0.0, num / den, torch.zeros_like(den))
    alpha_clamped = torch.clamp(alpha, 0.0, 1.0).to(dtype=torch.float32)
    if not return_eigs:
        return alpha_clamped
    return alpha_clamped, ev_min.to(dtype=torch.float32), ev_max.to(dtype=torch.float32)


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
    try:
        evals = torch.linalg.eigvalsh(herm).real
        evals_clamped = torch.clamp(evals, min=0.0)
        ev_min = evals_clamped[:, 0]
        ev_max = evals_clamped[:, -1]
    except Exception:
        # Fall back to per-tile jittered eigensolves; last resort is diagonal min/max.
        B, Lt = herm.shape[0], herm.shape[1]
        eye = torch.eye(Lt, device=herm.device, dtype=herm.dtype)
        jitter_mults = (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)
        ev_min_list = []
        ev_max_list = []
        for b in range(B):
            Rb = herm[b]
            diag = torch.real(torch.diagonal(Rb, dim1=-2, dim2=-1))
            mu_b = diag.mean()
            mu_b_safe = torch.clamp(mu_b, min=0.0) + 1e-12
            e_min_b = None
            e_max_b = None
            for mult in jitter_mults:
                try:
                    evals_b = torch.linalg.eigvalsh(Rb + (float(mult) * mu_b_safe) * eye).real
                    evals_b = torch.clamp(evals_b, min=0.0)
                    e_min_b = evals_b.min()
                    e_max_b = evals_b.max()
                    break
                except Exception:
                    e_min_b = None
                    e_max_b = None
            if e_min_b is None or e_max_b is None:
                diag_clamped = torch.clamp(diag, min=0.0)
                e_min_b = diag_clamped.min()
                e_max_b = diag_clamped.max()
            ev_min_list.append(e_min_b)
            ev_max_list.append(e_max_b)
        ev_min = torch.stack(ev_min_list, dim=0).to(dtype=torch.float64)
        ev_max = torch.stack(ev_max_list, dim=0).to(dtype=torch.float64)

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


def build_fd_grid_flow_band(
    prf_hz: float,
    Lt: int,
    flow_band_hz: Tuple[float, float],
    *,
    motion_half_span_hz: float = 0.0,
    min_pts: int = 3,
    max_pts: int = 21,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Build a symmetric Doppler grid spanning a fixed flow band.

    This mirrors the slow-path `fd_span_mode="flow_band"` behavior: it does not
    depend on per-tile PSD, and it avoids collapsing to a single DFT bin when Lt
    is short by choosing multiple tones across the band.
    """
    flow_low_hz, flow_high_hz = flow_band_hz
    flow_low_hz = float(flow_low_hz)
    flow_high_hz = float(flow_high_hz)
    if flow_high_hz < flow_low_hz:
        flow_low_hz, flow_high_hz = flow_high_hz, flow_low_hz

    nyquist = 0.5 * float(prf_hz)
    upper = min(float(flow_high_hz), nyquist + 1e-6)

    max_total = int(max_pts) if max_pts is not None else 0
    if max_total <= 0:
        max_total = 3
    max_allowed = int(Lt) if int(Lt) % 2 == 1 else max(int(Lt) - 1, 1)
    total = min(max_total, max_allowed)
    total = max(total, int(min_pts))
    total = min(total, max_allowed)
    if total >= 3 and total % 2 == 0:
        # Prefer odd length; keep within [min_pts, max_allowed].
        if total > int(min_pts):
            total -= 1
        elif total + 1 <= max_allowed:
            total += 1

    if total <= 1:
        center = float(np.clip(0.5 * (flow_low_hz + upper), 0.0, upper))
        fd_grid = np.asarray([0.0] if center <= 0.0 else [-center, center], dtype=np.float64)
        span_hz = float(np.max(np.abs(fd_grid))) if fd_grid.size else 0.0
        return fd_grid, {"span_hz": span_hz, "grid_step_hz": 0.0, "lines": int(fd_grid.size)}

    n_pos = max(1, (int(total) - 1) // 2) if total >= 3 else 0
    low_eff = float(flow_low_hz)
    try:
        if motion_half_span_hz is not None and float(motion_half_span_hz) > 0.0:
            low_eff = max(low_eff, float(motion_half_span_hz) + 1e-3)
    except Exception:
        pass

    if upper <= 0.0 or n_pos <= 0:
        mags: list[float] = []
    elif upper < flow_low_hz:
        mags = [float(np.clip(0.5 * (flow_low_hz + flow_high_hz), 0.0, upper))]
    elif n_pos == 1:
        mags = [0.5 * (low_eff + upper)]
    else:
        if upper < low_eff:
            mags = [float(np.clip(0.5 * (flow_low_hz + flow_high_hz), 0.0, upper))]
        else:
            mags = np.linspace(low_eff, upper, n_pos, dtype=np.float32).tolist()
    mags = [float(f) for f in mags if np.isfinite(f) and float(f) > 0.0]
    mags = sorted({float(f) for f in mags})
    if not mags:
        fd_grid = np.asarray([0.0], dtype=np.float64)
    else:
        fd_grid = np.asarray(sorted([-f for f in mags] + [0.0] + mags), dtype=np.float64)
    span_hz = float(np.max(np.abs(fd_grid))) if fd_grid.size else 0.0
    grid_step_hz = float(np.median(np.diff(np.sort(np.unique(np.abs(fd_grid)))))) if mags and len(mags) > 1 else 0.0
    meta = {"span_hz": span_hz, "grid_step_hz": grid_step_hz, "lines": int(fd_grid.size)}
    return fd_grid, meta


__all__ = [
    "build_temporal_hankels_batch",
    "robust_temporal_cov_batch",
    "shrinkage_alpha_for_kappa_batch",
    "conditioned_lambda_batch",
    "build_fd_grid_span",
    "build_fd_grid_flow_band",
]
