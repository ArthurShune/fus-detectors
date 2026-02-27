from __future__ import annotations

from contextlib import ExitStack
from contextlib import nullcontext
from contextlib import contextmanager
import contextvars
import os
import math
import time
import warnings
from dataclasses import asdict
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch

from pipeline.gpu.linalg import cholesky_solve_hermitian, to_tensor
from pipeline.stap.lcmv import LCMVResult, lcmv_weights
from pipeline.stap.robust_cov import robust_covariance
from pipeline.stap.temporal_shared import (
    build_fd_grid_flow_band,
    build_fd_grid_span,
    build_temporal_hankels_batch,
    robust_temporal_cov_batch,
    shrinkage_alpha_for_kappa_batch,
)
from pipeline.stap.temporal_shared import (
    conditioned_lambda_batch as conditioned_lambda_batch_shared,
)

try:  # Optional; only used when profiling is enabled.
    from torch.profiler import record_function as _record_function
except Exception:  # pragma: no cover - profiler optional
    _record_function = None


class _StapCudaStageTimer:
    def __init__(self, enabled: bool):
        self.enabled = bool(enabled) and torch.cuda.is_available()
        self._events: list[tuple[str, "torch.cuda.Event", "torch.cuda.Event"]] = []

    @contextmanager
    def stage(self, name: str):
        if not self.enabled:
            yield
            return
        stream = torch.cuda.current_stream()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        try:
            yield
        finally:
            end.record(stream)
            self._events.append((str(name), start, end))

    def summary_ms(self) -> dict[str, float]:
        if not self.enabled or not self._events:
            return {}
        torch.cuda.synchronize()
        out: dict[str, float] = {}
        for name, start, end in self._events:
            out[name] = out.get(name, 0.0) + float(start.elapsed_time(end))
        return out


_STAP_CUDA_STAGE_TIMER: contextvars.ContextVar[_StapCudaStageTimer | None] = contextvars.ContextVar(
    "STAP_CUDA_STAGE_TIMER", default=None
)

_STAP_TILE_STATISTIC_WARNED = False


def _warn_tile_statistic_experimental() -> None:
    global _STAP_TILE_STATISTIC_WARNED
    if _STAP_TILE_STATISTIC_WARNED:
        return
    _STAP_TILE_STATISTIC_WARNED = True
    warnings.warn(
        "Tile-statistic / cov-only STAP scoring is EXPERIMENTAL and is NOT ROC-equivalent to the "
        "manuscript detector. It replaces per-snapshot nonlinear MSD aggregation with a "
        "ratio-of-means approximation (mean(f) != f(mean)) and is known to catastrophically "
        "regress strict-FPR ROC on Twinkling/Gammex. Do not use for paper results or production configs.",
        UserWarning,
        stacklevel=2,
    )


@contextmanager
def stap_cuda_event_timing(*, enabled: bool | None = None):
    """
    Enable per-stage CUDA timing via CUDA events.

    This is an opt-in profiling helper: it does not change algorithm outputs.
    Stage boundaries are defined by existing `with _prof_ctx("stap:...")` markers.
    """
    if enabled is None:
        enabled = os.getenv("STAP_CUDA_EVENT_TIMING", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    timer = _StapCudaStageTimer(enabled=bool(enabled))
    token = _STAP_CUDA_STAGE_TIMER.set(timer)
    try:
        yield timer
    finally:
        _STAP_CUDA_STAGE_TIMER.reset(token)


@contextmanager
def _prof_ctx(name: str):
    prof_enabled = os.getenv("STAP_PROFILE_MARKERS", "").strip().lower() in {"1", "true", "yes", "on"}
    timer = _STAP_CUDA_STAGE_TIMER.get()
    cuda_enabled = (timer is not None) and bool(timer.enabled)
    if (not prof_enabled or _record_function is None) and not cuda_enabled:
        yield
        return
    with ExitStack() as stack:
        if prof_enabled and _record_function is not None:
            stack.enter_context(_record_function(str(name)))
        if cuda_enabled and timer is not None:
            stack.enter_context(timer.stage(str(name)))
        yield


stap_stage_ctx = _prof_ctx


def _power_iteration_max(R: torch.Tensor, num_iters: int = 3) -> float:
    """Approximate the largest eigenvalue of Hermitian R via power iteration."""
    Lt = R.shape[0]
    vec = torch.ones((Lt,), dtype=R.dtype, device=R.device)
    for _ in range(max(1, num_iters)):
        vec = R @ vec
        norm = torch.linalg.norm(vec)
        if norm < 1e-12:
            return 0.0
        vec = vec / norm
    lam = torch.dot(vec.conj(), R @ vec).real
    return float(torch.clamp(lam, min=0.0).item())


def _gershgorin_min_lb(R: torch.Tensor) -> float:
    """Lower bound on smallest eigenvalue via Gershgorin discs."""
    diag = torch.real(torch.diagonal(R))
    off_sum = torch.sum(torch.abs(R), dim=-1) - torch.abs(torch.diagonal(R))
    lb = torch.min(diag - off_sum)
    return float(torch.clamp(lb, min=0.0).item())


def conditioned_lambda(
    R: torch.Tensor,
    lam_requested: float = 0.0,
    *,
    kappa_target: float = 40.0,
    safety_factor: float = 1.5,
) -> tuple[float, float, float]:
    """
    Compute diagonal loading λ so κ(R + λI) ≤ kappa_target.

    Returns
    -------
    lam_final : float
        Loading to apply.
    sigma_max : float
        Estimated largest eigenvalue of R.
    sigma_min_lb : float
        Lower bound on smallest eigenvalue via Gershgorin.
    """
    kappa_target = max(float(kappa_target), 1.01)
    lam_requested = max(float(lam_requested), 0.0)
    sigma_max = _power_iteration_max(R)
    sigma_min_lb = _gershgorin_min_lb(R)
    if sigma_min_lb <= 0.0:
        lam_needed = sigma_max / (kappa_target - 1.0)
    else:
        lam_needed = max(0.0, (sigma_max - kappa_target * sigma_min_lb) / (kappa_target - 1.0))
    lam_final = max(lam_requested, lam_needed)
    lam_final *= safety_factor
    return lam_final, sigma_max, sigma_min_lb


def _trace_normalize(R: torch.Tensor) -> torch.Tensor:
    tr = torch.real(torch.trace(R))
    if tr <= 0:
        return R
    return R * (R.shape[-1] / tr)


def projector_from_tones(C: torch.Tensor, gram_ridge: float = 1e-6) -> torch.Tensor:
    """
    Build Hermitian projector onto span(C) using the Gram inverse with small ridge.

    C: (Lt, Kc) complex
    Returns Pf: (Lt, Lt) Hermitian idempotent (approximately with ridge)
    """
    if C is None or C.numel() == 0:
        raise ValueError("projector_from_tones: empty constraint matrix")
    G = C.conj().transpose(-2, -1) @ C
    if gram_ridge > 0.0:
        G = G + float(gram_ridge) * torch.eye(G.shape[-1], dtype=C.dtype, device=C.device)
    W, _ = cholesky_solve_hermitian(G, C.conj().transpose(-2, -1), jitter_init=1e-10, max_tries=3)
    Pf = C @ W
    Pf = 0.5 * (Pf + Pf.conj().transpose(-2, -1))
    return Pf


def equalize_pf_trace(
    R_loaded: torch.Tensor,
    Rt_sample: torch.Tensor,
    Pf: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, float, float, float]:
    """
    Scale the Pf-block of R_loaded so Tr(Pf R_loaded Pf) matches Tr(Pf Rt_sample Pf).

    Preserves the complement block exactly; returns a Hermitian matrix.

    Parameters
    ----------
    R_loaded : (Lt,Lt) complex Hermitian-like
    Rt_sample : (Lt,Lt) complex Hermitian-like (sample covariance)
    Pf : (Lt,Lt) Hermitian projector (~idempotent)
    eps : small positive to guard divides

    Returns
    -------
    R_equalized : (Lt,Lt) complex Hermitian
    """
    # Cast to a common dtype/device
    dtype = R_loaded.dtype
    device = R_loaded.device
    Rl = to_tensor(R_loaded, device=device, dtype=dtype)
    Rt = to_tensor(Rt_sample, device=device, dtype=dtype)
    Pfh = to_tensor(Pf, device=device, dtype=dtype)

    # Ensure Hermitian inputs for stable traces
    Rl = 0.5 * (Rl + Rl.conj().transpose(-2, -1))
    Rt = 0.5 * (Rt + Rt.conj().transpose(-2, -1))
    Pfh = 0.5 * (Pfh + Pfh.conj().transpose(-2, -1))

    tr_loaded = torch.real(torch.trace(Pfh @ Rl @ Pfh))
    tr_sample = torch.real(torch.trace(Pfh @ Rt @ Pfh))

    if not torch.isfinite(tr_loaded):
        return Rl
    den = tr_loaded + torch.tensor(eps, dtype=tr_loaded.dtype, device=tr_loaded.device)

    # Compute positive scaling and rebuild the Pf block
    # alpha is real-valued scaling; clamp on real scalar then cast
    alpha_val = torch.clamp(tr_sample / den, min=0.0)
    alpha = alpha_val.to(dtype)
    alpha_scalar = float(alpha_val.detach().cpu().real.item())
    tr_loaded_scalar = float(tr_loaded.detach().cpu().real.item())
    tr_sample_scalar = float(tr_sample.detach().cpu().real.item())
    R_pf = Pfh @ Rl @ Pfh
    R_perp = Rl - R_pf
    R_eq = R_perp + alpha * R_pf
    R_eq = 0.5 * (R_eq + R_eq.conj().transpose(-2, -1))
    return R_eq, alpha_scalar, tr_loaded_scalar, tr_sample_scalar


_EQUALIZE_PF_TRACE_FN = equalize_pf_trace

_FEASIBILITY_MODES = {"legacy", "updated", "blend"}


def _normalize_feasibility_mode(mode: str | None) -> str:
    if mode is None:
        return "legacy"
    mode_clean = str(mode).strip().lower()
    if mode_clean not in _FEASIBILITY_MODES:
        raise ValueError(
            f"Unsupported feasibility_mode '{mode}'. Expected one of {_FEASIBILITY_MODES}."
        )
    return mode_clean


def _shrinkage_alpha_for_kappa(R: torch.Tensor, kappa_target: float) -> float:
    """
    Choose alpha in [0,1] so sliding R toward mu*I yields cond <= kappa_target.
    Mirrors sim/kwave/common._shrinkage_alpha_for_kappa.
    """
    if kappa_target <= 1.0:
        return 0.0
    evals = torch.linalg.eigvalsh(R).real
    mu = float(torch.mean(evals).item())
    e_min = float(torch.min(evals).item())
    e_max = float(torch.max(evals).item())
    num = e_max - kappa_target * e_min
    den = (kappa_target - 1.0) * mu + (e_max - kappa_target * e_min)
    if den <= 0.0:
        return 0.0
    alpha = num / den
    return float(np.clip(alpha, 0.0, 1.0))


def apply_ridge_split(R_base: torch.Tensor, Pf: torch.Tensor, lam_abs: float) -> torch.Tensor:
    """Apply absolute ridge only on the complement of Pf.

    R_base: (Lt,Lt) Hermitian-like, Pf: (Lt,Lt) projector, lam_abs ≥ 0
    """
    eye = torch.eye(R_base.shape[-1], dtype=R_base.dtype, device=R_base.device)
    P_perp = eye - Pf
    return R_base + float(max(lam_abs, 0.0)) * P_perp


def _directional_traces(
    Rhat: torch.Tensor, R0: torch.Tensor, Pf: torch.Tensor
) -> tuple[float, float]:
    eye = torch.eye(Rhat.shape[-1], dtype=Rhat.dtype, device=Rhat.device)
    P_perp = eye - Pf
    num_f = torch.real(torch.trace(Pf @ R0))
    den_f = torch.real(torch.trace(Pf @ Rhat))
    num_p = torch.real(torch.trace(P_perp @ R0))
    den_p = torch.real(torch.trace(P_perp @ Rhat))
    retain_f = float((num_f / (den_f + 1e-12)).item())
    shrink_p = float((num_p / (den_p + 1e-12)).item())
    return retain_f, shrink_p


def _band_extrema(R: torch.Tensor, Pf: torch.Tensor) -> dict[str, float | int | None]:
    """Return extremal eigenvalues of R restricted to Pf / P_perp."""
    R_h = 0.5 * (R + R.conj().transpose(-2, -1))
    Qf, Qp, rf = _orthonormal_basis_from_projector(Pf)
    stats: dict[str, float | int | None] = {
        "pf_rank": rf,
        "perp_rank": int(R.shape[-1] - rf),
        "pf_min": None,
        "pf_max": None,
        "pf_mean": None,
        "perp_min": None,
        "perp_max": None,
        "perp_mean": None,
    }
    if rf > 0:
        band_f = Qf.conj().transpose(-2, -1) @ R_h @ Qf
        band_f = 0.5 * (band_f + band_f.conj().transpose(-2, -1))
        evals_f = torch.linalg.eigvalsh(band_f).real
        stats["pf_min"] = float(evals_f.min().item())
        stats["pf_max"] = float(evals_f.max().item())
        stats["pf_mean"] = float(evals_f.mean().item())
    if stats["perp_rank"] and stats["perp_rank"] > 0:
        band_p = Qp.conj().transpose(-2, -1) @ R_h @ Qp
        band_p = 0.5 * (band_p + band_p.conj().transpose(-2, -1))
        evals_p = torch.linalg.eigvalsh(band_p).real
        stats["perp_min"] = float(evals_p.min().item())
        stats["perp_max"] = float(evals_p.max().item())
        stats["perp_mean"] = float(evals_p.mean().item())
    return stats


def _estimate_noise_floor(
    R: torch.Tensor,
    projector: torch.Tensor,
    tail_frac: float = 0.4,
) -> float:
    """Estimate noise floor from the lower eigenvalues in the projected subspace."""
    R_block = projector @ R @ projector.conj().transpose(-2, -1)
    R_block = 0.5 * (R_block + R_block.conj().transpose(-2, -1))
    evals = torch.linalg.eigvalsh(R_block).real
    if evals.numel() == 0:
        return 0.0
    k = max(1, int(evals.numel() * float(tail_frac)))
    vals = torch.sort(evals).values[:k]
    return float(vals.mean().item())


def _flow_noise_ratios(
    R_sample: torch.Tensor,
    R_loaded: torch.Tensor,
    Pf: Optional[torch.Tensor],
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if Pf is None:
        return None, None, None, None
    Pf_eval = 0.5 * (Pf + Pf.conj().transpose(-2, -1))
    eye = torch.eye(Pf_eval.shape[-1], dtype=Pf_eval.dtype, device=Pf_eval.device)
    P_perp = eye - Pf_eval

    def _energy(mat: torch.Tensor, proj: torch.Tensor) -> float:
        block = proj @ mat @ proj.conj().transpose(-2, -1)
        block = 0.5 * (block + block.conj().transpose(-2, -1))
        val = torch.real(torch.trace(block))
        return float(val.item())

    flow_sample = _energy(R_sample, Pf_eval)
    flow_loaded = _energy(R_loaded, Pf_eval)
    noise_sample = _energy(R_sample, P_perp)
    noise_loaded = _energy(R_loaded, P_perp)
    eps = 1e-12
    if noise_sample <= eps:
        snr_base = None
    else:
        snr_base = flow_sample / noise_sample
    if noise_loaded <= eps:
        snr_loaded = None
    else:
        snr_loaded = flow_loaded / noise_loaded
    snr_ratio = None
    if snr_base and snr_base > eps and snr_loaded is not None:
        snr_ratio = snr_loaded / snr_base
    noise_ratio = None
    if noise_sample > eps:
        noise_ratio = noise_loaded / noise_sample
    return snr_ratio, noise_ratio, snr_loaded, snr_base


def _clip_prior_passband_safe(
    Rhat: torch.Tensor, R0: torch.Tensor, Pf: torch.Tensor
) -> torch.Tensor:
    """Clip prior so it cannot inflate passband or leak cross-terms.

    Returns R0_safe = P_perp R0 P_perp + Pf Rhat Pf, which preserves off-band structure
    from R0 and matches in-band energy to the sample covariance Rhat, with cross-terms
    zeroed. Hermitian by construction if inputs are Hermitian.
    """
    eye = torch.eye(R0.shape[-1], dtype=R0.dtype, device=R0.device)
    P_perp = eye - Pf
    # Project off-band portion of the prior, and drop cross-terms
    off = P_perp @ R0 @ P_perp.conj().transpose(-2, -1)
    on = Pf @ Rhat @ Pf.conj().transpose(-2, -1)
    R0_safe = off + on
    R0_safe = 0.5 * (R0_safe + R0_safe.conj().transpose(-2, -1))
    return R0_safe


def _orthonormal_basis_from_projector(
    Pf: torch.Tensor, tol: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return basis matrices Qf, Qp such that [Qf|Qp] is unitary."""
    Pf_herm = 0.5 * (Pf + Pf.conj().transpose(-2, -1))
    evals, vecs = torch.linalg.eigh(Pf_herm)
    mask_f = evals > 0.5
    rank_f = int(mask_f.sum().item())
    if rank_f == 0 or rank_f == Pf.shape[-1]:
        return vecs[:, mask_f], vecs[:, ~mask_f], rank_f
    Qf = vecs[:, mask_f]
    Qp = vecs[:, ~mask_f]
    return Qf, Qp, rank_f


def _apply_directional_strict_monotone(
    R_loaded: torch.Tensor,
    Rt_dev: torch.Tensor,
    Pf: torch.Tensor,
    lam_final: float,
    use_ridge_split: bool,
) -> tuple[torch.Tensor, bool]:
    """Project to Pf/P_perp blocks and shrink only the complement while preserving passband."""
    Lt = R_loaded.shape[-1]
    dtype = R_loaded.dtype
    device = R_loaded.device
    Pf = 0.5 * (Pf + Pf.conj().transpose(-2, -1))
    eye = torch.eye(Lt, dtype=dtype, device=device)
    P_perp = eye - Pf

    Qf, Qp, rf = _orthonormal_basis_from_projector(Pf)
    if rf == 0 or rf >= Lt:
        return R_loaded, False
    Q = torch.cat([Qf, Qp], dim=1)
    if Q.shape[-1] != Lt:
        # Numerical issues constructing U; fall back without strict handling
        return R_loaded, False

    Rt_herm = 0.5 * (Rt_dev + Rt_dev.conj().transpose(-2, -1))
    if use_ridge_split:
        R_base = apply_ridge_split(Rt_herm, Pf.to(dtype=dtype, device=device), lam_final)
    else:
        R_base = Rt_herm + float(lam_final) * eye
    R_loaded = 0.5 * (R_loaded + R_loaded.conj().transpose(-2, -1))

    tr_perp_base = torch.real(torch.trace(P_perp @ R_base @ P_perp.conj().transpose(-2, -1)))
    tr_perp_loaded = torch.real(torch.trace(P_perp @ R_loaded @ P_perp.conj().transpose(-2, -1)))
    denom = max(float(tr_perp_base.item()), 1e-12)
    ratio_target = min(1.0, float(tr_perp_loaded.item()) / denom)
    if ratio_target >= 1.0 - 1e-6:
        return R_base, True

    Q = torch.cat([Qf, Qp], dim=1)
    R_base_u = Q.conj().transpose(-2, -1) @ R_base @ Q
    # Partition into A (rf x rf), B (rf x rp), C (rp x rp)
    rp = Lt - rf
    A0 = R_base_u[:rf, :rf]
    B0 = R_base_u[:rf, rf:]
    C_base = R_base_u[rf:, rf:]

    try:
        A_inv_B = torch.linalg.solve(A0, B0)
    except RuntimeError:
        A_inv_B = torch.linalg.solve(A0 + 1e-6 * torch.eye(rf, dtype=dtype, device=device), B0)
    BHB = B0.conj().transpose(-2, -1) @ A_inv_B
    S = 0.5 * ((C_base - BHB) + (C_base - BHB).conj().transpose(-2, -1))
    evals = torch.linalg.eigvalsh(S).real
    min_eval = float(evals.min().item()) if evals.numel() > 0 else 0.0
    if min_eval < 1e-9:
        S = S + (1e-9 - min_eval) * torch.eye(rp, dtype=dtype, device=device)
    tr_S = float(torch.real(torch.trace(S)).item())
    tr_B = float(torch.real(torch.trace(BHB)).item())
    tr_total = tr_B + tr_S
    desired_trace = ratio_target * tr_total
    if tr_S <= 0.0:
        gamma = 0.0
    else:
        gamma = max(0.0, min(1.0, (desired_trace - tr_B) / (tr_S + 1e-12)))

    C_new = BHB + gamma * S
    R_u_new = R_base_u.clone()
    R_u_new[rf:, rf:] = C_new

    R_new = Q @ R_u_new @ Q.conj().transpose(-2, -1)
    R_new = 0.5 * (R_new + R_new.conj().transpose(-2, -1))
    return R_new, True


def choose_beta_directional(
    Rhat: torch.Tensor,
    R0: torch.Tensor,
    Pf: torch.Tensor,
    *,
    beta_max: float = 0.25,
    target_retain_f: float = 0.90,
    target_shrink_perp: float = 0.95,
    tol: float = 1e-3,
) -> tuple[float, dict]:
    """
    Search beta in [0,beta_max] to satisfy passband retention and off-band shrink.
    Returns (beta, info{retain_f, shrink_perp, pf_min_ratio, pf_max_ratio, perp_max_ratio}).
    """
    beta_max = float(max(0.0, beta_max))
    info: dict[str, float | None] = {
        "retain_f": float("nan"),
        "shrink_perp": float("nan"),
        "pf_min_ratio": None,
        "pf_max_ratio": None,
        "perp_max_ratio": None,
    }
    # If the prior is not lighter off-band than the sample, blending cannot shrink
    # the complement; fall back to zero beta and rely on ridge-split.
    rf0, sp0 = _directional_traces(Rhat, R0, Pf)
    if not (sp0 < 0.99):
        info.update({"retain_f": float(rf0), "shrink_perp": float(sp0)})
        return 0.0, info

    def _blend(beta: float) -> torch.Tensor:
        Rb = (1.0 - beta) * Rhat + beta * R0
        if Pf is not None:
            eye = torch.eye(Rhat.shape[-1], dtype=Rhat.dtype, device=Rhat.device)
            P_perp = eye - Pf
            on = Pf @ Rhat @ Pf.conj().transpose(-2, -1)
            off = P_perp @ Rb @ P_perp.conj().transpose(-2, -1)
            Rb = on + off
        return 0.5 * (Rb + Rb.conj().transpose(-2, -1))

    baseline_stats = _band_extrema(Rhat, Pf)
    pf_min_base = baseline_stats.get("pf_min")
    pf_max_base = baseline_stats.get("pf_max")
    perp_max_base = baseline_stats.get("perp_max")
    retain_low = target_retain_f
    retain_high = 1.1
    noise_cap = 1.2
    if pf_min_base is not None and pf_min_base <= 0.0:
        pf_min_base = None
    if pf_max_base is not None and pf_max_base <= 0.0:
        pf_max_base = None
    if perp_max_base is not None and perp_max_base <= 0.0:
        perp_max_base = None

    num_samples = max(3, int(max(beta_max / max(tol, 1e-3), 3)))
    beta_grid = [float(beta_max * i / (num_samples - 1)) for i in range(num_samples)]
    best_beta = 0.0
    best_stats = baseline_stats
    best_Rb = _blend(0.0)

    def _ratio(val: float | None, ref: float | None) -> float | None:
        if val is None or ref is None:
            return None
        if ref == 0.0:
            return None
        return float(val) / float(ref)

    for beta in beta_grid:
        Rb = _blend(beta)
        stats = _band_extrema(Rb, Pf)
        pf_min_ratio = _ratio(stats.get("pf_min"), pf_min_base)
        pf_max_ratio = _ratio(stats.get("pf_max"), pf_max_base)
        perp_max_ratio = _ratio(stats.get("perp_max"), perp_max_base)
        constraints_ok = True
        if pf_min_ratio is not None:
            constraints_ok = constraints_ok and pf_min_ratio >= retain_low
        if pf_max_ratio is not None:
            constraints_ok = constraints_ok and pf_max_ratio <= retain_high
        if perp_max_ratio is not None:
            constraints_ok = constraints_ok and perp_max_ratio <= noise_cap
        if not constraints_ok:
            continue
        if beta > best_beta + 1e-6:
            best_beta = beta
            best_stats = stats
            best_Rb = Rb
    Rb = best_Rb
    rf, sp = _directional_traces(Rhat, Rb, Pf)
    info.update(
        {
            "retain_f": rf,
            "shrink_perp": sp,
            "pf_min_ratio": _ratio(best_stats.get("pf_min"), pf_min_base),
            "pf_max_ratio": _ratio(best_stats.get("pf_max"), pf_max_base),
            "perp_max_ratio": _ratio(best_stats.get("perp_max"), perp_max_base),
        }
    )
    return float(best_beta), info


def _project_out_flow_from_prior(R0: torch.Tensor, Cf: Optional[torch.Tensor]) -> torch.Tensor:
    if Cf is None or Cf.numel() == 0:
        return R0
    G = Cf.conj().transpose(-2, -1) @ Cf
    eye = torch.eye(G.shape[-1], dtype=G.dtype, device=G.device)
    G_reg = G + 1e-6 * eye
    G_inv, _ = cholesky_solve_hermitian(G_reg, eye)
    P = Cf @ G_inv @ Cf.conj().transpose(-2, -1)
    eye = torch.eye(P.shape[-1], dtype=R0.dtype, device=R0.device)
    P_perp = eye - P
    projected = P_perp @ R0 @ P_perp.conj().transpose(-2, -1)
    return _trace_normalize(projected)


def _mismatch_score(R_sample: torch.Tensor, R_prior: torch.Tensor) -> float:
    diff = R_sample - R_prior
    num = torch.linalg.norm(diff, ord="fro") ** 2
    den = torch.linalg.norm(R_sample, ord="fro") ** 2 + 1e-12
    return float((num / den).real.item())


def _energy_match_to_sample(
    R_sample: torch.Tensor,
    R_prior: torch.Tensor,
    Cf: torch.Tensor,
) -> torch.Tensor:
    if Cf is None or Cf.numel() == 0:
        return R_prior
    Q, _ = torch.linalg.qr(Cf, mode="reduced")
    P = Q @ Q.conj().transpose(-2, -1)
    eye = torch.eye(R_sample.shape[-1], dtype=R_sample.dtype, device=R_sample.device)
    P_perp = eye - P
    sample_energy = torch.real(torch.trace(P_perp @ R_sample @ P_perp.conj().transpose(-2, -1)))
    prior_energy = torch.real(torch.trace(P_perp @ R_prior @ P_perp.conj().transpose(-2, -1)))
    scale = (sample_energy + 1e-12) / (prior_energy + 1e-12)
    return R_prior * scale


def _pf_projectors(Pf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    Pf = 0.5 * (Pf + Pf.conj().transpose(-2, -1))
    eye = torch.eye(Pf.shape[-1], dtype=Pf.dtype, device=Pf.device)
    P_perp = eye - Pf
    return Pf, P_perp


def _block_clip_covariance(R: torch.Tensor, Pf: torch.Tensor) -> torch.Tensor:
    Pf_h, P_perp = _pf_projectors(Pf)
    R = 0.5 * (R + R.conj().transpose(-2, -1))
    block_pf = Pf_h @ R @ Pf_h.conj().transpose(-2, -1)
    block_perp = P_perp @ R @ P_perp.conj().transpose(-2, -1)
    return block_pf + block_perp


def _generalized_band_metrics(
    R_sample: torch.Tensor,
    R_beta: torch.Tensor,
    Pf: torch.Tensor,
    Pa: Optional[torch.Tensor] = None,
    eps: float = 1e-9,
) -> dict[str, float]:
    Pf_h, P_perp = _pf_projectors(Pf)
    Qf, Qp, rf = _orthonormal_basis_from_projector(Pf_h)

    def _band_from_basis(mat: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        if basis.numel() == 0:
            return torch.zeros((0, 0), dtype=mat.dtype, device=mat.device)
        block = basis.conj().transpose(-2, -1) @ mat @ basis
        return 0.5 * (block + block.conj().transpose(-2, -1))

    Rf_sample = _band_from_basis(R_sample, Qf)
    Rf_beta = _band_from_basis(R_beta, Qf)
    Rp_sample = _band_from_basis(R_sample, Qp)
    Rp_beta = _band_from_basis(R_beta, Qp)

    def _gen_eval(
        Rs: torch.Tensor,
        Rb: torch.Tensor,
    ) -> tuple[float | None, float | None, float | None]:
        if Rs.numel() == 0 or Rb.numel() == 0:
            return None, None, None
        eye = torch.eye(Rb.shape[-1], dtype=Rb.dtype, device=Rb.device)
        L = torch.linalg.cholesky(Rb + eps * eye)
        sol = torch.cholesky_solve(Rs, L)
        sym = 0.5 * (sol + sol.conj().transpose(-2, -1))
        evals = torch.linalg.eigvalsh(sym).real
        evals = torch.clamp(evals, min=eps)
        lam_min = float(evals.min().item())
        lam_max = float(evals.max().item())
        lam_mean = float(evals.mean().item())
        return lam_min, lam_max, lam_mean

    pf_min, pf_max, pf_mean = _gen_eval(Rf_sample, Rf_beta)
    perp_min, perp_max, perp_mean = _gen_eval(Rp_sample, Rp_beta)

    stats = {
        "pf_min": pf_min,
        "pf_max": pf_max,
        "pf_mean": pf_mean,
        "perp_min": perp_min,
        "perp_max": perp_max,
        "perp_mean": perp_mean,
    }

    if Pa is not None:
        # If Alias projector is provided, compute metrics for Alias and Noise (Po) separately.
        # Po = I - Pf - Pa
        Pa_h = 0.5 * (Pa + Pa.conj().transpose(-2, -1))
        eye = torch.eye(R_sample.shape[-1], dtype=R_sample.dtype, device=R_sample.device)
        Po_h = eye - Pf_h - Pa_h

        # We need bases for Pa and Po to project the matrices
        Qa, _, ra = _orthonormal_basis_from_projector(Pa_h)
        Qo, _, ro = _orthonormal_basis_from_projector(Po_h)

        Ra_sample = _band_from_basis(R_sample, Qa)
        Ra_beta = _band_from_basis(R_beta, Qa)
        Ro_sample = _band_from_basis(R_sample, Qo)
        Ro_beta = _band_from_basis(R_beta, Qo)

        alias_min, alias_max, alias_mean = _gen_eval(Ra_sample, Ra_beta)
        noise_min, noise_max, noise_mean = _gen_eval(Ro_sample, Ro_beta)

        stats.update(
            {
                "alias_min": alias_min,
                "alias_max": alias_max,
                "alias_mean": alias_mean,
                "noise_min": noise_min,
                "noise_max": noise_max,
                "noise_mean": noise_mean,
            }
        )

    return stats


def _mixing_metric(
    R_sample: torch.Tensor,
    R_beta: torch.Tensor,
    Pf: torch.Tensor,
    eps: float = 1e-9,
) -> float:
    Pf_h, P_perp = _pf_projectors(Pf)
    eye = torch.eye(R_beta.shape[-1], dtype=R_beta.dtype, device=R_beta.device)
    L = torch.linalg.cholesky(R_beta + eps * eye)
    M = torch.cholesky_solve(R_sample, L)
    M = 0.5 * (M + M.conj().transpose(-2, -1))
    cross = P_perp @ M @ Pf_h
    num = torch.linalg.norm(cross, ord="fro")
    denom = torch.linalg.norm(P_perp @ M @ P_perp, ord="fro") * torch.linalg.norm(
        Pf_h @ M @ Pf_h, ord="fro"
    )
    if denom <= eps:
        return 0.0
    return float((num / denom).item())


_PF_MIN_TARGET = 0.95
_PERP_MAX_TARGET = 1.10
_MIX_TARGET = 0.05


def _repair_band_metrics(
    R_sample: torch.Tensor,
    R_beta: torch.Tensor,
    Pf: torch.Tensor,
    *,
    beta_init: float,
    max_passes: int = 3,
    pf_min_target: float = 0.95,
    perp_max_target: float = 1.10,
    mix_target: float = 0.05,
) -> tuple[torch.Tensor, float, dict[str, float]]:
    Pf_h, P_perp = _pf_projectors(Pf)
    R_current = 0.5 * (R_beta + R_beta.conj().transpose(-2, -1))
    beta_curr = float(beta_init)
    stats: dict[str, float] = {}
    repaired = False
    passes_used = 0

    def _split_blocks(mat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        block_pf = Pf_h @ mat @ Pf_h.conj().transpose(-2, -1)
        block_perp = P_perp @ mat @ P_perp.conj().transpose(-2, -1)
        return block_pf, block_perp

    for attempt in range(1, max_passes + 1):
        passes_used = attempt
        metrics = _generalized_band_metrics(R_sample, R_current, Pf)
        mix = _mixing_metric(R_sample, R_current, Pf)
        stats.update(
            {
                "pf_lambda_min": metrics.get("pf_min"),
                "perp_lambda_max": metrics.get("perp_max"),
                "mixing_epsilon": mix,
            }
        )

        pf_min = metrics.get("pf_min")
        perp_max = metrics.get("perp_max")
        adjusted = False

        if pf_min is not None and pf_min < pf_min_target:
            scale = max(0.2, min(5.0, pf_min / max(pf_min_target, 1e-9)))
            block_pf, block_perp = _split_blocks(R_current)
            block_pf = block_pf * float(scale)
            R_current = block_pf + block_perp
            adjusted = True

        if perp_max is not None and perp_max > perp_max_target:
            scale = max(1.0, min(5.0, perp_max / max(perp_max_target, 1e-9)))
            block_pf, block_perp = _split_blocks(R_current)
            block_perp = block_perp * float(scale)
            R_current = block_pf + block_perp
            adjusted = True

        if not adjusted and mix > mix_target:
            stats["repair_failed"] = True
            break

        if not adjusted and mix <= mix_target:
            repaired = True
            break

    stats["passes_used"] = passes_used
    if not repaired:
        stats.setdefault("repair_failed", True)
    return R_current, beta_curr, stats


def _conditioned_lambda_precise(
    R: torch.Tensor,
    lam_requested: float,
    kappa_target: float,
) -> float:
    kappa_target = max(float(kappa_target), 1.01)
    lam_requested = max(float(lam_requested), 0.0)
    R_cpu = R.detach().to(torch.complex128).to("cpu")
    herm = (R_cpu + R_cpu.conj().transpose(-2, -1)) * 0.5
    evals = torch.linalg.eigvalsh(herm).real
    evals = torch.clamp(evals, min=0.0)
    if evals.numel() == 0:
        return lam_requested
    ev_min = float(evals.min().item())
    ev_max = float(evals.max().item())
    if ev_max <= 0.0:
        lam_needed = 0.0
    else:
        denom = max(kappa_target - 1.0, 1e-6)
        lam_needed = max(0.0, (ev_max - kappa_target * ev_min) / denom)
    return max(lam_requested, lam_needed)


def _conditioned_lambda_split(
    R: torch.Tensor,
    Pf: torch.Tensor,
    lam_requested: float,
    kappa_target: float,
) -> float:
    """
    Find λ such that κ(R + λ (I - Pf)) ≤ kappa_target using bisection.

    Parameters
    ----------
    R : torch.Tensor
        Hermitian covariance estimate (Lt, Lt).
    Pf : torch.Tensor
        Projector onto flow subspace (Lt, Lt).
    lam_requested : float
        Minimum λ to honour (e.g. override or floor).
    kappa_target : float
        Desired upper bound on condition number.
    """
    kappa_target = max(float(kappa_target), 1.01)
    lam_requested = max(float(lam_requested), 0.0)
    R_cpu = R.detach().to(torch.complex128).to("cpu")
    Pf_cpu = Pf.detach().to(torch.complex128).to("cpu")
    eye = torch.eye(R_cpu.shape[-1], dtype=R_cpu.dtype, device=R_cpu.device)
    P_perp = eye - Pf_cpu
    lam_cap = float(lam_requested)

    def cond_val(lam: float) -> float:
        Rj = R_cpu + lam * P_perp
        herm = 0.5 * (Rj + Rj.conj().transpose(-2, -1))
        evals = torch.linalg.eigvalsh(herm).real
        if evals.numel() == 0:
            return 1.0
        ev_max = float(torch.max(evals).item())
        ev_min = float(torch.clamp(torch.min(evals), min=0.0).item())
        if ev_min <= 0.0:
            return float("inf")
        return ev_max / max(ev_min, 1e-12)

    lam_lo = lam_requested
    cond_lo = cond_val(lam_lo)
    if cond_lo <= kappa_target:
        return lam_lo

    lam_hi = max(lam_lo, 1e-9)
    cond_hi = cond_val(lam_hi)
    iter_expand = 0
    while cond_hi > kappa_target and iter_expand < 40:
        lam_hi *= 2.0
        cond_hi = cond_val(lam_hi)
        iter_expand += 1
    if cond_hi > kappa_target:
        lam_final = max(lam_hi, lam_requested)
        if lam_final > lam_cap:
            lam_final = lam_cap
        return lam_final

    for _ in range(50):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        cond_mid = cond_val(lam_mid)
        if cond_mid <= kappa_target:
            lam_hi = lam_mid
        else:
            lam_lo = lam_mid
        if lam_hi - lam_lo < max(1e-9, 0.05 * lam_hi):
            break
    lam_final = max(lam_hi, lam_requested)
    if lam_final > lam_cap:
        lam_final = lam_cap
    return lam_final


def ka_blend_covariance_temporal(
    R_sample: np.ndarray | torch.Tensor,
    R0_prior: np.ndarray | torch.Tensor,
    *,
    Cf_flow: Optional[np.ndarray | torch.Tensor] = None,
    Cf_alias: Optional[np.ndarray | torch.Tensor] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    beta_bounds: Tuple[float, float] = (0.05, 0.6),
    beta_max: Optional[float] = None,
    mismatch_tau: float = 1.0,
    kappa_target: float = 40.0,
    add_noise_floor_frac: float = 0.0,
    lambda_override: Optional[float] = None,
    # Optional directional safety knobs (default None: disabled)
    beta_directional: Optional[bool] = None,
    target_retain_f: Optional[float] = None,
    target_shrink_perp: Optional[float] = None,
    beta_directional_strict: Optional[bool] = None,
    ridge_split: Optional[bool] = None,
    lambda_override_split: Optional[float] = None,
    # Optional: equalize passband energy to the sample to preserve flow mean
    equalize_pf_trace: bool = False,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.complex64,
    feasibility_mode: str = "legacy",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    feas_mode = _normalize_feasibility_mode(feasibility_mode)
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    target_dtype = dtype

    Rt_cpu = to_tensor(R_sample, device="cpu", dtype=torch.complex128)
    R0_cpu = to_tensor(R0_prior, device="cpu", dtype=torch.complex128)
    Rt_dev = to_tensor(R_sample, device=target_device, dtype=target_dtype)
    Rt_dev_herm = 0.5 * (Rt_dev + Rt_dev.conj().transpose(-2, -1))
    Lt = Rt_cpu.shape[-1]
    trace_sample = torch.real(torch.trace(Rt_cpu))
    mu_sample = float(trace_sample.item()) / float(Lt) if trace_sample.abs().item() > 0 else 0.0

    ka_warning_msg: Optional[str] = None
    pf_trace_equalized = False
    Rt_n = _trace_normalize(Rt_cpu)
    R0_n = _trace_normalize(R0_cpu)
    R0_n_full = R0_n.clone()
    band_metrics: dict[str, float] | None = None
    mixing_metric_val: float | None = None

    Cf_cpu = None
    Pf_cpu = None
    Pa_cpu = None
    pf_rank = float("nan")
    Rt_comp = Rt_n
    R0_comp = R0_n
    if Cf_flow is not None:
        Cf_cpu = to_tensor(Cf_flow, device="cpu", dtype=torch.complex128)
        if Cf_cpu.numel() > 0:
            Cf_proj = Cf_cpu
            if Cf_proj.shape[-1] >= Lt:
                rank_target = max(1, Lt - 1)
                try:
                    evals, evecs = torch.linalg.eigh(Rt_n.to(torch.complex128))
                    v_flow = evecs[:, -1]
                except RuntimeError:
                    v_flow = torch.ones((Lt,), dtype=torch.complex128, device=Rt_n.device)
                scores = torch.abs(Cf_proj.conj().transpose(-2, -1) @ v_flow)
                top_idx = torch.argsort(scores, descending=True)[:rank_target]
                Cf_proj = Cf_proj[:, top_idx]
            Pf_cpu = projector_from_tones(Cf_proj)
            pf_rank = float(torch.real(torch.trace(Pf_cpu)).item())
            eye_cpu = torch.eye(Lt, dtype=Rt_n.dtype, device=Rt_n.device)
            P_perp = eye_cpu - Pf_cpu
            # Default prior safety: project out flow from prior (legacy mode)
            R0_proj = _project_out_flow_from_prior(R0_n, Cf_cpu)
            R0_proj = _energy_match_to_sample(Rt_n, R0_proj, Cf_cpu)
            R0_proj = _clip_prior_passband_safe(Rt_n, R0_proj, Pf_cpu)
            R0_n = R0_proj
            Rt_comp = P_perp @ Rt_n @ P_perp.conj().transpose(-2, -1)
            R0_comp = P_perp @ R0_proj @ P_perp.conj().transpose(-2, -1)

    if Cf_alias is not None:
        Cf_alias_cpu = to_tensor(Cf_alias, device="cpu", dtype=torch.complex128)
        if Cf_alias_cpu.numel() > 0:
            Pa_cpu = projector_from_tones(Cf_alias_cpu)

    # For the experimental blend feasibility mode, override the prior with a
    # simple band-scalar construction that inflates the alias band while
    # leaving the flow and residual bands approximately neutral. This keeps
    # the operator geometry easy to interpret while still leveraging the
    # learned Pf/Pa projectors.
    if feas_mode == "blend" and Pf_cpu is not None:
        eye_blend = torch.eye(Lt, dtype=Rt_n.dtype, device=Rt_n.device)
        Pf_proj = Pf_cpu.to(dtype=Rt_n.dtype, device=Rt_n.device)
        if Pa_cpu is not None:
            Pa_proj = Pa_cpu.to(dtype=Rt_n.dtype, device=Rt_n.device)
        else:
            Pa_proj = torch.zeros_like(eye_blend, dtype=Rt_n.dtype, device=Rt_n.device)
        Po_proj = eye_blend - Pf_proj - Pa_proj
        Po_proj = 0.5 * (Po_proj + Po_proj.conj().transpose(-2, -1))
        # Alias-band inflation factor: Pa receives higher variance in the
        # prior so that R_beta^{-1} effectively shrinks alias-dominant tiles.
        gamma_a = 3.0
        R0_blend_cpu = Pf_proj + gamma_a * Pa_proj + Po_proj
        R0_blend_cpu = 0.5 * (R0_blend_cpu + R0_blend_cpu.conj().transpose(-2, -1))
        R0_n = _trace_normalize(R0_blend_cpu)
        R0_n_full = R0_n.clone()

    sample_band_stats: dict[str, float | int | None] = {}
    prior_noise_floor = None
    sample_noise_floor = None
    if Pf_cpu is not None:
        sample_band_stats = _band_extrema(Rt_cpu, Pf_cpu)
        prior_noise_floor = _estimate_noise_floor(R0_n, P_perp)
        sample_noise_floor = _estimate_noise_floor(Rt_n, P_perp)

    mismatch = _mismatch_score(Rt_comp, R0_comp)
    mismatch_full = _mismatch_score(Rt_n, R0_n_full)
    mismatch_for_beta = mismatch_full if Cf_cpu is not None and Cf_cpu.numel() > 0 else mismatch

    if alpha is None:
        alpha_val = 0.0
    else:
        alpha_val = float(np.clip(alpha, 0.0, 1.0))

    # Commuting, band-block blend for updated feasibility: enforce gains per band directly.
    if feas_mode == "updated" and Pf_cpu is not None:
        # Promote Pf/Pa to true orthoprojectors to minimize mixing downstream.
        Qf, _, _ = _orthonormal_basis_from_projector(Pf_cpu)
        Pf_h = Qf @ Qf.conj().transpose(-2, -1)
        Pf_h = Pf_h.to(dtype=Rt_n.dtype, device=Rt_n.device)
        eye_cpu = torch.eye(Lt, dtype=Rt_n.dtype, device=Rt_n.device)
        if Pa_cpu is None:
            Pa_raw = torch.zeros_like(Pf_h, dtype=Rt_n.dtype, device=Rt_n.device)
        else:
            Qa, _, _ = _orthonormal_basis_from_projector(Pa_cpu)
            Pa_raw = Qa @ Qa.conj().transpose(-2, -1)
            Pa_raw = Pa_raw.to(dtype=Rt_n.dtype, device=Rt_n.device)
        # Orthogonalize alias projector against flow to reduce mixing.
        Pa_orth = (eye_cpu - Pf_h) @ Pa_raw @ (eye_cpu - Pf_h)
        Pa_orth = 0.5 * (Pa_orth + Pa_orth.conj().transpose(-2, -1))
        Qa_orth, _, _ = _orthonormal_basis_from_projector(Pa_orth)
        Pa_cpu_loc = Qa_orth @ Qa_orth.conj().transpose(-2, -1)
        Pa_cpu_loc = Pa_cpu_loc.to(dtype=Rt_n.dtype, device=Rt_n.device)
        Po_cpu = eye_cpu - Pf_h - Pa_cpu_loc
        Po_cpu = 0.5 * (Po_cpu + Po_cpu.conj().transpose(-2, -1))

        # Target gains (OC-2): keep flow/noise at 1, inflate alias.
        s_f = 1.0
        s_o = 1.0
        # Slightly stronger alias inflation for pial/alias-heavy regimes.
        s_a = 1.5 if Pa_cpu is not None else 1.0

        # Build M = sf^-2 Pf + sa^-2 Pa + so^-2 Po (commuting with all bands).
        inv_sf2 = 1.0 / (s_f * s_f)
        inv_sa2 = 1.0 / (s_a * s_a)
        inv_so2 = 1.0 / (s_o * s_o)
        M = inv_sf2 * Pf_h + inv_sa2 * Pa_cpu_loc + inv_so2 * Po_cpu
        M = 0.5 * (M + M.conj().transpose(-2, -1))

        # Compute R_hat^{1/2}
        evals, evecs = torch.linalg.eigh(Rt_n.to(torch.complex128))
        evals_clamped = torch.clamp(evals.real, min=1e-6)
        Rh_sqrt = (evecs * evals_clamped.sqrt()) @ evecs.conj().transpose(-2, -1)

        # R_beta = Rh^{1/2} M Rh^{1/2}
        R_loaded_cpu = Rh_sqrt @ M @ Rh_sqrt
        R_loaded_cpu = 0.5 * (R_loaded_cpu + R_loaded_cpu.conj().transpose(-2, -1))

        # Optional bandwise isotropy (reset traces per band to stay isotropic).
        def _reset_band(P, mat):
            r = torch.real(torch.trace(P)).clamp(min=1.0)
            tr = torch.real(torch.trace(P @ mat @ P))
            mu = (tr / r).to(mat.dtype)
            return mu * P

        M_reset = _reset_band(Pf_h, M) + _reset_band(Pa_cpu_loc, M) + _reset_band(Po_cpu, M)
        M_reset = 0.5 * (M_reset + M_reset.conj().transpose(-2, -1))
        R_loaded_cpu = Rh_sqrt @ M_reset @ Rh_sqrt
        R_loaded_cpu = 0.5 * (R_loaded_cpu + R_loaded_cpu.conj().transpose(-2, -1))

        # Ensure positive definiteness with a small diagonal lift (commuting with bands).
        trace_loaded = torch.real(torch.trace(R_loaded_cpu))
        eps_pd = float(max(1e-6, 1e-3 * (trace_loaded.item() / max(float(Lt), 1.0))))
        R_loaded_cpu = R_loaded_cpu + eps_pd * eye_cpu
        R_loaded_cpu = 0.5 * (R_loaded_cpu + R_loaded_cpu.conj().transpose(-2, -1))

        R_loaded = R_loaded_cpu.to(device=target_device, dtype=target_dtype)

        design_metrics = {
            "pf_min": s_f * s_f,
            "pf_max": s_f * s_f,
            "pf_mean": s_f * s_f,
            "perp_min": s_o * s_o,
            "perp_max": s_o * s_o,
            "perp_mean": s_o * s_o,
            "alias_min": (s_a * s_a) if Pa_cpu is not None else None,
            "alias_max": (s_a * s_a) if Pa_cpu is not None else None,
            "alias_mean": (s_a * s_a) if Pa_cpu is not None else None,
            "noise_min": s_o * s_o,
            "noise_max": s_o * s_o,
            "noise_mean": s_o * s_o,
        }
        design_mix = 0.0

        band_metrics = _generalized_band_metrics(Rt_cpu, R_loaded_cpu, Pf_h, Pa=Pa_cpu_loc)
        mixing_metric_val = (
            design_mix if design_mix is not None else _mixing_metric(Rt_cpu, R_loaded_cpu, Pf_h)
        )
        R_metrics = R_loaded_cpu
        sigma_vals = torch.linalg.eigvalsh(R_loaded_cpu).real
        sigma_min_raw = (
            float(torch.clamp(sigma_vals.min(), min=0.0).item()) if sigma_vals.numel() else 0.0
        )
        sigma_max_raw = float(sigma_vals.max().item()) if sigma_vals.numel() else 0.0

        # Retain/shrink metrics (total effect)
        retain_f_total = None
        shrink_perp_total = None
        try:
            Pf_eval = Pf_h.to(R_loaded_cpu.dtype)
            Icpu = torch.eye(Lt, dtype=R_loaded_cpu.dtype, device=R_loaded_cpu.device)
            Pperp_cpu = Icpu - Pf_eval
            num_f = torch.real(torch.trace(Pf_eval @ R_loaded_cpu))
            den_f = torch.real(torch.trace(Pf_eval @ Rt_n))
            retain_f_total = float((num_f / (den_f + 1e-12)).item())
            num_p = torch.real(
                torch.trace(Pperp_cpu @ R_loaded_cpu @ Pperp_cpu.conj().transpose(-2, -1))
            )
            den_p = torch.real(torch.trace(Pperp_cpu @ Rt_n @ Pperp_cpu.conj().transpose(-2, -1)))
            shrink_perp_total = float((num_p / (den_p + 1e-12)).item())
        except Exception:
            pass
        # Fallback: derive from band means.
        if retain_f_total is None and band_metrics is not None:
            pf_mean_val = band_metrics.get("pf_mean")
            perp_mean_val = band_metrics.get("perp_mean")
            if pf_mean_val is not None and perp_mean_val not in (None, 0):
                retain_f_total = float(pf_mean_val) / float(perp_mean_val)
        if shrink_perp_total is None and band_metrics is not None:
            perp_mean_val = band_metrics.get("perp_mean")
            if perp_mean_val is not None:
                shrink_perp_total = float(perp_mean_val)

        # Optional ranks to pass downstream
        try:
            details_pf_rank = int(round(torch.real(torch.trace(Pf_h)).item()))
        except Exception:
            details_pf_rank = None
        try:
            details_alias_rank = int(round(torch.real(torch.trace(Pa_cpu_loc)).item()))
        except Exception:
            details_alias_rank = None

        lambda_strategy = "commuting" if Pa_cpu is not None else "identity"

        details: Dict[str, float | int | None] = {
            "ka_mode": "commuting_band",
            "feasibility_mode": feas_mode,
            "alpha": alpha_val,
            "beta": 0.0,
            "mismatch": float(mismatch),
            "mismatch_full": float(mismatch_full),
            "mismatch_beta_metric": float(mismatch_for_beta),
            "lambda_req": 0.0,
            "lambda_used": 0.0,
            "lambda_floor": 0.0,
            "lambda_guard_factor": 1.0,
            "ridge_split": False,
            "sigma_min_raw": sigma_min_raw,
            "sigma_max_raw": sigma_max_raw,
            "sigma_min_lb": sigma_min_raw,
            "sigma_max_lb": sigma_max_raw,
            "kappa_target": float(kappa_target),
            "lambda_strategy": lambda_strategy,
            "retain_f_total": retain_f_total,
            "shrink_perp_total": shrink_perp_total,
            "pf_rank": details_pf_rank,
            "alias_rank": details_alias_rank,
            "ka_pf_rank": details_pf_rank,
            "ka_alias_rank": details_alias_rank,
            "ka_alias_gain_target": s_a,
            "operator_feasible": bool(
                (
                    (design_metrics or {}).get("pf_min", band_metrics.get("pf_min", 0.0))
                    >= _PF_MIN_TARGET
                )
                and (
                    (design_metrics or {}).get(
                        "perp_max", band_metrics.get("perp_max", float("inf"))
                    )
                    <= _PERP_MAX_TARGET
                )
                and (mixing_metric_val is not None and mixing_metric_val <= _MIX_TARGET)
            ),
            "mixing_epsilon": mixing_metric_val if mixing_metric_val is not None else None,
        }
        if design_metrics is not None:
            details.update(
                {
                    "pf_lambda_min": design_metrics.get("pf_min"),
                    "pf_lambda_max": design_metrics.get("pf_max"),
                    "pf_lambda_mean": design_metrics.get("pf_mean"),
                    "perp_lambda_min": design_metrics.get("perp_min"),
                    "perp_lambda_max": design_metrics.get("perp_max"),
                    "perp_lambda_mean": design_metrics.get("perp_mean"),
                    "ka_alias_lambda_min": design_metrics.get("alias_min"),
                    "ka_alias_lambda_max": design_metrics.get("alias_max"),
                    "ka_alias_lambda_mean": design_metrics.get("alias_mean"),
                    "ka_noise_lambda_min": design_metrics.get("noise_min"),
                    "ka_noise_lambda_max": design_metrics.get("noise_max"),
                    "ka_noise_lambda_mean": design_metrics.get("noise_mean"),
                }
            )
        if band_metrics is not None:
            details.update(
                {
                    "sample_alias_lambda_min": band_metrics.get("alias_min"),
                    "sample_alias_lambda_max": band_metrics.get("alias_max"),
                    "sample_alias_lambda_mean": band_metrics.get("alias_mean"),
                    "sample_noise_lambda_min": band_metrics.get("noise_min"),
                    "sample_noise_lambda_max": band_metrics.get("noise_max"),
                    "sample_noise_lambda_mean": band_metrics.get("noise_mean"),
                    "sample_pf_lambda_min": band_metrics.get("pf_min"),
                    "sample_pf_lambda_max": band_metrics.get("pf_max"),
                    "sample_pf_lambda_mean": band_metrics.get("pf_mean"),
                    "sample_perp_lambda_min": band_metrics.get("perp_min"),
                    "sample_perp_lambda_max": band_metrics.get("perp_max"),
                    "sample_perp_lambda_mean": band_metrics.get("perp_mean"),
                }
            )
        # Expose SNR/noise ratios for downstream C1/C2 checks
        snr_flow_out = retain_f_total
        noise_perp_out = shrink_perp_total
        if band_metrics is not None:
            pf_mean_val = band_metrics.get("pf_mean")
            perp_mean_val = band_metrics.get("perp_mean")
            if snr_flow_out is None and pf_mean_val is not None and perp_mean_val not in (None, 0):
                snr_flow_out = float(pf_mean_val) / float(perp_mean_val)
            if noise_perp_out is None and perp_mean_val is not None:
                noise_perp_out = float(perp_mean_val)
        if snr_flow_out is not None:
            details["snr_flow_ratio"] = float(snr_flow_out)
        if noise_perp_out is not None:
            details["noise_perp_ratio"] = float(noise_perp_out)
        # Populate sample band stats and noise floors if Pa available
        if Pa_cpu is not None:
            alias_stats = _band_extrema(Rt_cpu, Pa_cpu_loc)
            sample_band_stats["alias_min"] = alias_stats.get("pf_min")
            sample_band_stats["alias_max"] = alias_stats.get("pf_max")
            sample_band_stats["alias_mean"] = alias_stats.get("pf_mean")
            noise_stats = _band_extrema(Rt_cpu, Po_cpu)
            sample_band_stats["noise_min"] = noise_stats.get("pf_min")
            sample_band_stats["noise_max"] = noise_stats.get("pf_max")
            sample_band_stats["noise_mean"] = noise_stats.get("pf_mean")
            prior_noise_floor = _estimate_noise_floor(R0_n, Po_cpu)
            sample_noise_floor = _estimate_noise_floor(Rt_n, Po_cpu)

        return R_loaded, details

    # Simple R_beta blend mode: interpolate between sample and prior
    # covariances in the normalized space and return R_beta directly.
    if feas_mode == "blend" and Pf_cpu is not None:
        beta_val = float(beta) if beta is not None else 0.3
        beta_val = float(np.clip(beta_val, beta_bounds[0], beta_bounds[1]))
        R_beta = (1.0 - beta_val) * Rt_n + beta_val * R0_n_full
        R_beta = 0.5 * (R_beta + R_beta.conj().transpose(-2, -1))
        # Ensure positive definiteness with a small diagonal lift.
        evals_beta = torch.linalg.eigvalsh(R_beta).real
        trace_beta = torch.real(torch.trace(R_beta))
        eps_pd = 0.0
        if evals_beta.numel():
            lam_min = float(torch.clamp(evals_beta.min(), min=0.0).item())
            if lam_min <= 0.0:
                eps_pd = max(
                    1e-6,
                    1e-3 * float(trace_beta.item()) / max(float(Lt), 1.0),
                )
        if eps_pd > 0.0:
            eye_cpu = torch.eye(Lt, dtype=R_beta.dtype, device=R_beta.device)
            R_beta = R_beta + eps_pd * eye_cpu
            R_beta = 0.5 * (R_beta + R_beta.conj().transpose(-2, -1))
        R_loaded_cpu = R_beta
        R_loaded = R_loaded_cpu.to(device=target_device, dtype=target_dtype)
        # Band metrics and mixing for telemetry.
        band_metrics = _generalized_band_metrics(Rt_cpu, R_loaded_cpu, Pf_cpu, Pa=Pa_cpu)
        mixing_metric_val = _mixing_metric(Rt_cpu, R_loaded_cpu, Pf_cpu)
        sigma_vals = torch.linalg.eigvalsh(R_loaded_cpu).real
        sigma_min_raw = (
            float(torch.clamp(sigma_vals.min(), min=0.0).item()) if sigma_vals.numel() else 0.0
        )
        sigma_max_raw = float(sigma_vals.max().item()) if sigma_vals.numel() else 0.0
        # Simple retain/shrink estimates from band means.
        retain_f_total = None
        shrink_perp_total = None
        if band_metrics is not None:
            pf_mean_val = band_metrics.get("pf_mean")
            perp_mean_val = band_metrics.get("perp_mean")
            if pf_mean_val is not None and perp_mean_val not in (None, 0):
                retain_f_total = float(pf_mean_val) / float(perp_mean_val)
            if perp_mean_val is not None:
                shrink_perp_total = float(perp_mean_val)
        try:
            details_pf_rank = int(round(torch.real(torch.trace(Pf_cpu)).item()))
        except Exception:
            details_pf_rank = None
        try:
            details_alias_rank = (
                int(round(torch.real(torch.trace(Pa_cpu)).item())) if Pa_cpu is not None else None
            )
        except Exception:
            details_alias_rank = None
        details: Dict[str, float | int | None] = {
            "ka_mode": "blend_Rbeta",
            "feasibility_mode": feas_mode,
            "alpha": alpha_val,
            "beta": beta_val,
            "mismatch": float(mismatch),
            "mismatch_full": float(mismatch_full),
            "mismatch_beta_metric": float(mismatch_for_beta),
            "lambda_req": 0.0,
            "lambda_used": 0.0,
            "lambda_floor": 0.0,
            "lambda_guard_factor": 1.0,
            "ridge_split": False,
            "sigma_min_raw": sigma_min_raw,
            "sigma_max_raw": sigma_max_raw,
            "sigma_min_lb": sigma_min_raw,
            "sigma_max_lb": sigma_max_raw,
            "kappa_target": float(kappa_target),
            "lambda_strategy": "blend",
            "retain_f_total": retain_f_total,
            "shrink_perp_total": shrink_perp_total,
            "pf_rank": details_pf_rank,
            "alias_rank": details_alias_rank,
            "ka_pf_rank": details_pf_rank,
            "ka_alias_rank": details_alias_rank,
            "ka_alias_gain_target": None,
            "operator_feasible": True,
            "mixing_epsilon": mixing_metric_val if mixing_metric_val is not None else None,
        }
        if band_metrics is not None:
            details.update(
                {
                    "pf_lambda_min": band_metrics.get("pf_min"),
                    "pf_lambda_max": band_metrics.get("pf_max"),
                    "pf_lambda_mean": band_metrics.get("pf_mean"),
                    "perp_lambda_min": band_metrics.get("perp_min"),
                    "perp_lambda_max": band_metrics.get("perp_max"),
                    "perp_lambda_mean": band_metrics.get("perp_mean"),
                    "ka_alias_lambda_min": band_metrics.get("alias_min"),
                    "ka_alias_lambda_max": band_metrics.get("alias_max"),
                    "ka_alias_lambda_mean": band_metrics.get("alias_mean"),
                    "ka_noise_lambda_min": band_metrics.get("noise_min"),
                    "ka_noise_lambda_max": band_metrics.get("noise_max"),
                    "ka_noise_lambda_mean": band_metrics.get("noise_mean"),
                }
            )
        return R_loaded, details

    mu = torch.real(torch.trace(Rt_n)) / float(Lt)
    eye_cpu = torch.eye(Lt, dtype=Rt_n.dtype, device=Rt_n.device)
    R_mu = (1.0 - alpha_val) * Rt_n + alpha_val * mu * eye_cpu
    Pf_dev_eval = (
        Pf_cpu.to(device=target_device, dtype=target_dtype) if Pf_cpu is not None else None
    )
    beta_low, beta_high = map(float, beta_bounds)
    if beta_max is not None:
        try:
            beta_high = float(beta_max)
        except (TypeError, ValueError):
            beta_high = float(beta_bounds[1])
        beta_high = max(beta_low, min(beta_high, float(beta_bounds[1])))
    beta_val: float
    retain_f = float("nan")
    shrink_perp = float("nan")
    prior_clipped_passband = False
    prior_clipped_perp = False
    beta_dir = False
    R_metrics_preload: Optional[torch.Tensor] = None
    beta_dir_req: Optional[float] = None
    s_perp_used: Optional[float] = None
    # Optional directional beta with passband retention
    # ka_opts controls: beta_directional(bool), target_retain_f, target_shrink_perp
    # For backward compatibility, default is legacy beta schedule.
    default_dir = (Pf_cpu is not None) and (feas_mode != "updated")
    beta_directional = bool(beta_directional) if beta_directional is not None else default_dir
    target_retain_f = float(target_retain_f) if target_retain_f is not None else 0.95
    target_shrink_perp = float(target_shrink_perp) if target_shrink_perp is not None else 0.90
    use_ridge_split = bool(ridge_split) if ridge_split is not None else default_dir
    if feas_mode == "updated":
        use_ridge_split = False
    beta_directional_strict = (
        bool(beta_directional_strict) if beta_directional_strict is not None else default_dir
    )

    if beta is None and beta_directional and Pf_cpu is not None:
        num_f_R0 = torch.real(torch.trace(Pf_cpu @ R0_n_full))
        num_f_Rh = torch.real(torch.trace(Pf_cpu @ Rt_n))
        R0_clip = _clip_prior_passband_safe(Rt_n, R0_n_full, Pf_cpu)
        prior_clipped_passband = bool(num_f_R0 > num_f_Rh + 1e-9)
        Icpu = torch.eye(Lt, dtype=Rt_n.dtype, device=Rt_n.device)
        P_perp = Icpu - Pf_cpu
        off = P_perp @ R0_clip @ P_perp.conj().transpose(-2, -1)
        off = 0.5 * (off + off.conj().transpose(-2, -1))
        on = Pf_cpu @ Rt_n @ Pf_cpu.conj().transpose(-2, -1)
        t_hat_perp = torch.real(torch.trace(P_perp @ Rt_n @ P_perp.conj().transpose(-2, -1)))
        t0_perp = torch.real(torch.trace(off))
        sample_perp_val = float(t_hat_perp.item())
        prior_perp_val = float(t0_perp.item())
        safe_target_rel = max(0.0, min(float(target_shrink_perp), 0.995) - 1e-3)
        target_perp_val = safe_target_rel * sample_perp_val
        scale_perp = 1.0
        if sample_perp_val <= 0.0:
            off = torch.zeros_like(off)
            prior_perp_val = 0.0
            scale_perp = 0.0
            prior_clipped_perp = True
        elif prior_perp_val > target_perp_val >= 0.0:
            scale_perp = target_perp_val / max(prior_perp_val, 1e-12)
            off = off * float(scale_perp)
            prior_perp_val = float(target_perp_val)
            prior_clipped_perp = True
        R0_clip = off + on
        R0_clip = 0.5 * (R0_clip + R0_clip.conj().transpose(-2, -1))
        s_perp_used = float(scale_perp)

        if sample_perp_val > 0.0:
            t0_perp_eff = torch.real(
                torch.trace(P_perp @ R0_clip @ P_perp.conj().transpose(-2, -1))
            )
            r_eff = float((t0_perp_eff / (t_hat_perp + 1e-12)).item())
        else:
            r_eff = 0.0

        if abs(r_eff - 1.0) < 1e-6:
            beta_candidate = float(beta_high)
        else:
            beta_candidate = float((target_shrink_perp - 1.0) / (r_eff - 1.0))
        beta_dir_req = float(beta_candidate)
        beta_val = float(np.clip(beta_candidate, 0.0, beta_high))
        Rb_tmp = (1.0 - beta_val) * Rt_n + beta_val * R0_clip
        rf_tmp, sp_tmp = _directional_traces(Rt_n, Rb_tmp, Pf_cpu)

        if beta_val > 0.0 and (
            rf_tmp < target_retain_f - 1e-3
            or rf_tmp > 1.0 + 1e-3
            or sp_tmp > target_shrink_perp + 1e-3
        ):
            beta_safe, _ = choose_beta_directional(
                Rt_n,
                R0_clip,
                Pf_cpu,
                beta_max=min(beta_high, beta_val),
                target_retain_f=target_retain_f,
                target_shrink_perp=target_shrink_perp,
            )
            beta_val = float(np.clip(beta_safe, 0.0, min(beta_high, beta_val)))
            Rb_tmp = (1.0 - beta_val) * Rt_n + beta_val * R0_clip
            rf_tmp, sp_tmp = _directional_traces(Rt_n, Rb_tmp, Pf_cpu)

        retain_f = float(rf_tmp)
        shrink_perp = float(sp_tmp)
        beta_dir = True
        R0_n_use = R0_clip
        R_metrics_preload = Rb_tmp
    else:
        if beta is None:
            beta_auto = beta_high * float(np.exp(-mismatch_for_beta / max(mismatch_tau, 1e-6)))
            beta_val = float(np.clip(beta_auto, beta_low, beta_high))
        else:
            beta_val = float(np.clip(beta, beta_low, beta_high))
        R0_n_use = R0_n

    R_blend = (1.0 - beta_val) * R_mu + beta_val * R0_n_use
    R_metrics = R_metrics_preload if beta_dir and R_metrics_preload is not None else R_blend
    if add_noise_floor_frac > 0.0:
        R_blend = R_blend + float(add_noise_floor_frac) * mu * eye_cpu

    if Pf_cpu is not None:
        R_pf = Pf_cpu @ R_blend @ Pf_cpu.conj().transpose(-2, -1)
        tr_pf_blend = torch.real(torch.trace(R_pf))
        tr_pf_sample = torch.real(torch.trace(Pf_cpu @ Rt_n @ Pf_cpu.conj().transpose(-2, -1)))
        if tr_pf_blend > tr_pf_sample * (1.0 + 1e-3):
            scale_pf = (tr_pf_sample / (tr_pf_blend + 1e-12)).to(R_blend.dtype)
            R_perp = R_blend - R_pf
            R_blend = R_perp + scale_pf * R_pf

    trace_target = torch.real(torch.trace(Rt_cpu))
    trace_blend = torch.real(torch.trace(R_blend))
    if trace_blend > 0:
        scale_back = float((trace_target / trace_blend).real.item())
        if beta_dir and Pf_cpu is not None:
            Icpu = torch.eye(Lt, dtype=R_blend.dtype, device=R_blend.device)
            Pperp = Icpu - Pf_cpu.to(R_blend.dtype)
            tr_perp_blend = torch.real(
                torch.trace(Pperp @ R_blend @ Pperp.conj().transpose(-2, -1))
            )
            tr_perp_sample = torch.real(
                torch.trace(Pperp @ Rt_cpu @ Pperp.conj().transpose(-2, -1))
            )
            num = float(tr_perp_sample.item() + 1e-12)
            den = float(tr_perp_blend.item() + 1e-12)
            scale_max = num / den
            scale_back = min(scale_back, scale_max)
        R_blend = R_blend * scale_back

    herm = 0.5 * (R_blend + R_blend.conj().transpose(-2, -1))
    sigma_vals = torch.linalg.eigvalsh(herm).real
    if sigma_vals.numel() > 0:
        sigma_min_raw = float(torch.clamp(sigma_vals.min(), min=0.0).item())
        sigma_max_raw = float(sigma_vals.max().item())
    else:
        sigma_min_raw = 0.0
        sigma_max_raw = 0.0

    lambda_floor_factor = 1.45 if (Cf_cpu is not None and Cf_cpu.numel() > 0) else 0.05
    if Cf_cpu is not None and Cf_cpu.numel() > 0 and lambda_override is None:
        lambda_floor_factor = 2.0
    lam_floor_nominal = max(lambda_floor_factor * mu_sample, 2e-2)
    lam_floor = lam_floor_nominal
    if lambda_override is not None:
        lam_floor = min(lam_floor, float(lambda_override))
    if lambda_override_split is not None:
        lam_floor = min(lam_floor, float(lambda_override_split))
    lam_precise = _conditioned_lambda_precise(
        R_blend,
        0.0,
        kappa_target=float(kappa_target),
    )
    lambda_req = float(lam_precise)
    lam_request = max(lam_precise, lam_floor)
    override_global = float(lambda_override) if lambda_override is not None else None
    override_split = float(lambda_override_split) if lambda_override_split is not None else None
    if override_global is not None and Cf_cpu is not None and Cf_cpu.numel() > 0:
        override_global = min(override_global, lam_floor_nominal * 1.05)
    if override_global is not None and override_split is None:
        use_ridge_split = False
    if override_global is not None:
        lam_request = max(lam_request, override_global)
    if override_split is not None:
        lam_request = max(lam_request, override_split)

    if use_ridge_split and Pf_cpu is not None:
        if override_split is not None:
            lam_final = max(override_split, lam_floor)
            lambda_strategy = "split_override"
        elif override_global is not None:
            lam_final = max(override_global, lam_floor)
            lambda_strategy = "split_override"
        else:
            lam_final = _conditioned_lambda_split(
                R_blend,
                Pf_cpu,
                lam_request,
                float(kappa_target),
            )
            lam_final = max(lam_final, lam_floor)
            lambda_strategy = "split"
    else:
        if feas_mode == "updated":
            lam_final = max(lam_request, lam_floor)
            if override_global is not None:
                lam_final = max(override_global, lam_floor)
            lambda_strategy = "identity"
        elif override_global is not None:
            lam_final = max(override_global, lam_floor)
            lambda_strategy = "global_override"
        else:
            lam_final = max(lam_request, lam_floor)
            lambda_strategy = "global"

    if lambda_override is None and Cf_cpu is not None and Cf_cpu.numel() > 0:
        extra_guard = 0.06 * (1.0 + float(mismatch_for_beta))
        lam_final = max(lam_final, lam_floor + extra_guard)

    lambda_guard_factor = 1.0
    if lambda_override is None and Cf_cpu is not None and Cf_cpu.numel() > 0:
        lambda_guard_factor = 1.08
        lam_final = max(lam_final, lambda_guard_factor * lam_floor_nominal)

    R_blend_dev = R_blend.to(device=target_device, dtype=target_dtype)
    eye_dev = torch.eye(Lt, dtype=target_dtype, device=target_device)
    if use_ridge_split and Pf_cpu is not None:
        Pf_dev = Pf_cpu.to(device=target_device, dtype=target_dtype)
        # Add primary ridge on the complement, and a small safety ridge on Pf
        R_tmp = apply_ridge_split(R_blend_dev, Pf_dev, lam_final)
        R_loaded = R_tmp
    else:
        R_loaded = R_blend_dev + float(lam_final) * eye_dev

    operator_feasible = True
    repair_stats: dict[str, float] = {}
    if Pf_cpu is not None:
        R_loaded_cpu = R_loaded.detach().to(torch.complex128).to("cpu")
        R_loaded_cpu, beta_val, repair_stats = _repair_band_metrics(
            Rt_cpu,
            R_loaded_cpu,
            Pf_cpu,
            beta_init=beta_val,
            pf_min_target=_PF_MIN_TARGET,
            perp_max_target=_PERP_MAX_TARGET,
            mix_target=_MIX_TARGET,
        )
        band_metrics = _generalized_band_metrics(Rt_cpu, R_loaded_cpu, Pf_cpu, Pa=Pa_cpu)
        mixing_metric_val = _mixing_metric(Rt_cpu, R_loaded_cpu, Pf_cpu)
        R_loaded = R_loaded_cpu.to(device=target_device, dtype=target_dtype)
        if (
            repair_stats.get("repair_failed")
            or band_metrics.get("pf_min") is None
            or band_metrics["pf_min"] < _PF_MIN_TARGET
            or band_metrics.get("perp_max") is None
            or band_metrics["perp_max"] > _PERP_MAX_TARGET
            or mixing_metric_val > _MIX_TARGET
        ):
            operator_feasible = False
    else:
        repair_stats = {"passes_used": 0}

    # Directional post-load safeguard: ensure off-band block does not exceed baseline
    # when using directional KA. Apply a similarity transform D = Pf + sqrt(a) P_perp
    # to scale only the complement block and associated cross-terms.
    if beta_dir and Pf_cpu is not None:
        Pf_dev_local = (
            Pf_dev_eval
            if Pf_dev_eval is not None
            else Pf_cpu.to(device=target_device, dtype=target_dtype)
        )
        I_dev = torch.eye(Lt, dtype=target_dtype, device=target_device)
        Pperp_dev = I_dev - Pf_dev_local
        # Baseline with the same loading on the complement
        # Baseline built from the sample covariance with the same loading
        if use_ridge_split:
            R_base = apply_ridge_split(Rt_dev_herm, Pf_dev_local, lam_final)
        else:
            R_base = Rt_dev_herm + float(lam_final) * I_dev

        tr_perp_curr = torch.real(
            torch.trace(Pperp_dev @ R_loaded @ Pperp_dev.conj().transpose(-2, -1))
        )
        tr_perp_base = torch.real(
            torch.trace(Pperp_dev @ R_base @ Pperp_dev.conj().transpose(-2, -1))
        )
        if tr_perp_curr > tr_perp_base * (1.0 + 1e-6):
            num = (tr_perp_base + 1e-12).to(R_loaded.dtype)
            den = (tr_perp_curr + 1e-12).to(R_loaded.dtype)
            a = torch.clamp(num / den, max=1.0)
            # Similarity transform with D = Pf + sqrt(a) P_perp
            D = Pf_dev_local + torch.sqrt(a) * Pperp_dev
            R_loaded = D @ R_loaded @ D
            R_loaded = 0.5 * (R_loaded + R_loaded.conj().transpose(-2, -1))

    # (optional) Passband energy lock: match Pf trace of loaded to sample to preserve flow mean
    pf_trace_alpha = None
    pf_trace_loaded_trace = None
    pf_trace_sample_trace = None
    if equalize_pf_trace and Pf_cpu is not None:
        Pf_dev_local = (
            Pf_dev_eval
            if Pf_dev_eval is not None
            else Pf_cpu.to(device=target_device, dtype=target_dtype)
        )
        try:
            (
                R_loaded,
                pf_trace_alpha,
                pf_trace_loaded_trace,
                pf_trace_sample_trace,
            ) = _EQUALIZE_PF_TRACE_FN(R_loaded, Rt_dev_herm, Pf_dev_local)
            pf_trace_equalized = True
        except Exception as exc_pf:
            ka_warning_msg = f"pf_equalize_failed:{exc_pf}"

    directional_strict_applied = False
    if beta_dir and beta_directional_strict and Pf_cpu is not None:
        Pf_dev = Pf_cpu.to(device=target_device, dtype=target_dtype)
        R_loaded, directional_strict_applied = _apply_directional_strict_monotone(
            R_loaded,
            Rt_dev,
            Pf_dev,
            float(lam_final),
            use_ridge_split,
        )

    if Pf_cpu is not None:
        R_loaded_cpu = R_loaded.detach().to(torch.complex128).to("cpu")
        band_metrics = _generalized_band_metrics(Rt_cpu, R_loaded_cpu, Pf_cpu, Pa=Pa_cpu)
        mixing_metric_val = _mixing_metric(Rt_cpu, R_loaded_cpu, Pf_cpu)

        # Compute baseline spectra for Alias and Noise if Pa is present
        if Pa_cpu is not None:
            # Alias Band
            alias_stats = _band_extrema(Rt_cpu, Pa_cpu)
            sample_band_stats["alias_min"] = alias_stats.get("pf_min")
            sample_band_stats["alias_max"] = alias_stats.get("pf_max")
            sample_band_stats["alias_mean"] = alias_stats.get("pf_mean")

            # Noise Band (Po = I - Pf - Pa)
            eye_cpu = torch.eye(Lt, dtype=Rt_cpu.dtype, device=Rt_cpu.device)
            Pf_h, _ = _pf_projectors(Pf_cpu)
            Pa_h, _ = _pf_projectors(Pa_cpu)
            Po_cpu = eye_cpu - Pf_h - Pa_h
            noise_stats = _band_extrema(Rt_cpu, Po_cpu)
            sample_band_stats["noise_min"] = noise_stats.get("pf_min")
            sample_band_stats["noise_max"] = noise_stats.get("pf_max")
            sample_band_stats["noise_mean"] = noise_stats.get("pf_mean")

            # Update noise floor estimation to use Po
            prior_noise_floor = _estimate_noise_floor(R0_n, Po_cpu)
            sample_noise_floor = _estimate_noise_floor(Rt_n, Po_cpu)

    R_lam = R_loaded

    # Compute directional metrics (β-only vs total after loading)
    retain_f_beta_out = None
    shrink_perp_beta_out = None
    if beta_dir:
        retain_f_beta_out = retain_f
        shrink_perp_beta_out = shrink_perp

    retain_f_total = None
    shrink_perp_total = None
    try:
        if Pf_cpu is not None:
            R_metric_eval = (
                R_metrics_preload if (beta_dir and R_metrics_preload is not None) else R_metrics
            )
            Rb_cpu = R_metric_eval.detach().to(torch.complex128).to("cpu")
            Rh_cpu = Rt_n.detach().to(torch.complex128).to("cpu")
            Icpu = torch.eye(Lt, dtype=Rb_cpu.dtype, device=Rb_cpu.device)
            Pf_eval = Pf_cpu.to(Rb_cpu.dtype)
            Pperp_cpu = Icpu - Pf_eval
            num_f = torch.real(torch.trace(Pf_eval @ Rb_cpu))
            den_f = torch.real(torch.trace(Pf_eval @ Rh_cpu))
            retain_f_total = float((num_f / (den_f + 1e-12)).item())
            num_p = torch.real(
                torch.trace(Pperp_cpu @ Rb_cpu @ Pperp_cpu.conj().transpose(-2, -1))
            )
            den_p = torch.real(
                torch.trace(Pperp_cpu @ Rh_cpu @ Pperp_cpu.conj().transpose(-2, -1))
            )
            shrink_perp_total = float((num_p / (den_p + 1e-12)).item())
    except Exception:
        pass

    details: Dict[str, float] = {
        "ka_mode": "blend",
        "alpha": alpha_val,
        "beta": beta_val,
        "mismatch": float(mismatch),
        "mismatch_full": float(mismatch_full),
        "mismatch_beta_metric": float(mismatch_for_beta),
        "lambda_req": float(lambda_req),
        "lambda_used": float(lam_final),
        "lambda_floor": float(lam_floor_nominal),
        "lambda_guard_factor": float(lambda_guard_factor),
        "sigma_min_raw": sigma_min_raw,
        "sigma_max_raw": sigma_max_raw,
        "sigma_min_lb": sigma_min_raw,
        "sigma_max_lb": sigma_max_raw,
        "kappa_target": float(kappa_target),
        "lambda_strategy": lambda_strategy,
        # Directional metrics
        "retain_f_beta": float(retain_f_beta_out) if retain_f_beta_out is not None else None,
        "shrink_perp_beta": (
            float(shrink_perp_beta_out) if shrink_perp_beta_out is not None else None
        ),
        "retain_f_total": float(retain_f_total) if retain_f_total is not None else None,
        "shrink_perp_total": float(shrink_perp_total) if shrink_perp_total is not None else None,
        "beta_req": float(beta_dir_req) if beta_dir_req is not None else None,
        "s_perp_used": float(s_perp_used) if s_perp_used is not None else None,
        "operator_feasible": bool(operator_feasible),
        "repair_passes": float(repair_stats.get("passes_used", 0)),
        "mixing_epsilon": mixing_metric_val if mixing_metric_val is not None else None,
    }
    if band_metrics is not None:
        details.update(
            {
                "pf_lambda_min": band_metrics.get("pf_min"),
                "pf_lambda_max": band_metrics.get("pf_max"),
                "pf_lambda_mean": band_metrics.get("pf_mean"),
                "perp_lambda_min": band_metrics.get("perp_min"),
                "perp_lambda_max": band_metrics.get("perp_max"),
                "perp_lambda_mean": band_metrics.get("perp_mean"),
            }
        )
        alias_min = band_metrics.get("alias_min")
        alias_max = band_metrics.get("alias_max")
        alias_mean = band_metrics.get("alias_mean")
        noise_min = band_metrics.get("noise_min")
        noise_max = band_metrics.get("noise_max")
        noise_mean = band_metrics.get("noise_mean")
        if alias_min is not None:
            details["ka_alias_lambda_min"] = alias_min
        if alias_max is not None:
            details["ka_alias_lambda_max"] = alias_max
        if alias_mean is not None:
            details["ka_alias_lambda_mean"] = alias_mean
        if noise_min is not None:
            details["ka_noise_lambda_min"] = noise_min
        if noise_max is not None:
            details["ka_noise_lambda_max"] = noise_max
        if noise_mean is not None:
            details["ka_noise_lambda_mean"] = noise_mean
    if mixing_metric_val is not None:
        details["mixing_epsilon"] = mixing_metric_val
    details["beta_directional"] = bool(beta_dir)
    details["directional_strict"] = bool(directional_strict_applied and beta_dir)
    if prior_clipped_passband:
        details["prior_clipped_passband"] = True
    if prior_clipped_perp:
        details["prior_clipped_perp"] = True
    if lambda_override is not None:
        details["lambda_override"] = float(lam_final)
    if lambda_override_split is not None:
        details["lambda_override_split"] = float(lambda_override_split)
    if Cf_cpu is not None and Cf_cpu.numel() > 0:
        overlap = torch.linalg.norm(Cf_cpu.conj().transpose(-2, -1) @ R0_cpu @ Cf_cpu, ord="fro")
        details["flow_overlap_frob"] = float(overlap.real.item())
        details["pf_rank"] = pf_rank
    details["ridge_split"] = bool(use_ridge_split)
    trace_return = torch.real(torch.trace(R_lam))
    trace_ratio = None
    details["trace_return_pre"] = float(trace_return.real.item())
    allow_trace_scale = not (beta_dir and directional_strict_applied)
    trace_scale_lock_reason: Optional[str] = None
    if not allow_trace_scale and beta_dir and directional_strict_applied:
        trace_scale_lock_reason = "directional_strict"
    if pf_trace_equalized:
        allow_trace_scale = False
        trace_scale_lock_reason = "pf_equalize"
    details["trace_scale_lock_reason"] = trace_scale_lock_reason
    if trace_sample.abs().item() > 0.0 and allow_trace_scale:
        ratio_val = (trace_return / trace_sample).real
        trace_ratio = float(ratio_val.item())
        if trace_return.real < trace_sample.real:
            scale_return = (trace_sample / trace_return).real
            R_lam = R_lam * scale_return.to(dtype=R_lam.dtype)
            details["trace_scaled"] = True
        else:
            details["trace_scaled"] = False
    if trace_ratio is not None:
        details["trace_ratio"] = trace_ratio

    snr_ratio, noise_ratio, snr_loaded_val, snr_base_val = _flow_noise_ratios(
        Rt_dev_herm,
        R_lam,
        Pf_dev_eval,
    )
    details["snr_flow_ratio"] = snr_ratio
    details["noise_perp_ratio"] = noise_ratio
    details["snr_flow_loaded"] = snr_loaded_val
    details["snr_flow_base"] = snr_base_val
    details["trace_sample"] = float(trace_sample.real.item())
    details["pf_trace_equalized"] = bool(pf_trace_equalized)
    if pf_trace_alpha is not None:
        details["pf_trace_alpha"] = float(pf_trace_alpha)
    if pf_trace_loaded_trace is not None:
        details["pf_trace_trace_loaded"] = float(pf_trace_loaded_trace)
    if pf_trace_sample_trace is not None:
        details["pf_trace_trace_sample"] = float(pf_trace_sample_trace)
    if sample_band_stats:
        details["sample_pf_lambda_min"] = sample_band_stats.get("pf_min")
        details["sample_pf_lambda_max"] = sample_band_stats.get("pf_max")
        details["sample_pf_lambda_mean"] = sample_band_stats.get("pf_mean")
        details["sample_perp_lambda_min"] = sample_band_stats.get("perp_min")
        details["sample_perp_lambda_max"] = sample_band_stats.get("perp_max")
        details["sample_perp_lambda_mean"] = sample_band_stats.get("perp_mean")
        if sample_band_stats.get("alias_min") is not None:
            details["sample_alias_lambda_min"] = sample_band_stats.get("alias_min")
        if sample_band_stats.get("alias_max") is not None:
            details["sample_alias_lambda_max"] = sample_band_stats.get("alias_max")
        if sample_band_stats.get("alias_mean") is not None:
            details["sample_alias_lambda_mean"] = sample_band_stats.get("alias_mean")
        if sample_band_stats.get("noise_min") is not None:
            details["sample_noise_lambda_min"] = sample_band_stats.get("noise_min")
        if sample_band_stats.get("noise_max") is not None:
            details["sample_noise_lambda_max"] = sample_band_stats.get("noise_max")
        if sample_band_stats.get("noise_mean") is not None:
            details["sample_noise_lambda_mean"] = sample_band_stats.get("noise_mean")
    if sample_noise_floor is not None:
        details["sample_po_noise_floor"] = float(sample_noise_floor)
    if prior_noise_floor is not None:
        details["prior_po_noise_floor"] = float(prior_noise_floor)
    if ka_warning_msg:
        details["warning"] = str(ka_warning_msg)
    details["feasibility_mode"] = feas_mode
    return R_lam, details


def ka_prior_temporal_from_psd(
    Lt: int,
    prf_hz: float,
    f_peaks_hz: Sequence[float] = (0.0,),
    width_bins: int = 1,
    add_deriv: bool = True,
    noise_floor_frac: float = 0.05,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """
    Fast analytic KA prior composed of exponentials around specified Doppler peaks and
    optional linear trend.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    t = np.arange(Lt, dtype=np.float64) / float(prf_hz)
    cols: List[np.ndarray] = []
    for f in f_peaks_hz:
        omega = 2.0 * np.pi * float(f)
        base = np.exp(1j * omega * t)
        cols.append(base[:, None])
        for k in range(1, max(1, width_bins) + 1):
            df = k * (prf_hz / max(Lt, 1))
            cols.append(np.exp(1j * 2.0 * np.pi * (float(f) + df) * t)[:, None])
            cols.append(np.exp(1j * 2.0 * np.pi * (float(f) - df) * t)[:, None])
    if add_deriv:
        deriv = (t - t.mean()) / (t.std() + 1e-6)
        cols.append(deriv.astype(np.complex128)[:, None])
    if not cols:
        return torch.eye(Lt, dtype=dtype, device=device)
    C_np = np.concatenate(cols, axis=1).astype(np.complex128, copy=False)
    C = torch.as_tensor(C_np, dtype=torch.complex128, device="cpu")
    R0 = C @ C.conj().transpose(-2, -1)
    mu = torch.real(torch.trace(R0)) / float(Lt)
    R0 = R0 + float(noise_floor_frac) * mu * torch.eye(Lt, dtype=R0.dtype, device=R0.device)
    R0 = _trace_normalize(R0)
    return R0.to(device=device, dtype=dtype)


def _solve_lower_triangular(L: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Solve L X = B for lower-triangular L."""
    # torch.linalg.solve_triangular since torch>=1.9
    return torch.linalg.solve_triangular(L, B, upper=False)


def _exp_col(Lt: int, omega: float, *, device: str, dtype: torch.dtype) -> torch.Tensor:
    m = torch.arange(Lt, device=device, dtype=torch.float32)
    return torch.exp(1j * omega * m).to(dtype=dtype).reshape(-1, 1)


@torch.no_grad()
def build_motion_basis_temporal(
    Lt: int,
    prf_hz: float,
    *,
    width_bins: int = 1,
    include_dc: bool = True,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """
    Construct a low-order temporal basis capturing motion/low-frequency clutter.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cols: List[torch.Tensor] = []
    if include_dc:
        cols.append(torch.ones((Lt, 1), dtype=dtype, device=device))
    for k in range(1, max(1, width_bins) + 1):
        omega = 2.0 * np.pi * (k / max(Lt, 1))
        cols.append(_exp_col(Lt, omega, device=device, dtype=dtype))
        cols.append(_exp_col(Lt, -omega, device=device, dtype=dtype))
    if not cols:
        return torch.zeros((Lt, 0), dtype=dtype, device=device)
    return torch.cat(cols, dim=1)


@torch.no_grad()
def project_out_motion_whitened(
    R_t: torch.Tensor | np.ndarray,
    S_Lt_N_hw: torch.Tensor | np.ndarray,
    C_motion: torch.Tensor,
    *,
    lam_abs: float = 3e-2,
    kappa_target: float = 40.0,
    R0_prior: Optional[np.ndarray | torch.Tensor] = None,
    Cf_flow: Optional[np.ndarray | torch.Tensor] = None,
    ka_opts: Optional[Dict[str, float]] = None,
    ka_details: Optional[List[Dict[str, float]]] = None,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.complex64,
    return_metrics: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Whiten the temporal stacks then project out the motion subspace.

    Returns the whitened, motion-null data and the Cholesky factor used.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    Rt = to_tensor(R_t, device=device, dtype=dtype)
    S = to_tensor(S_Lt_N_hw, device=device, dtype=dtype)
    Lt_dim = Rt.shape[0]
    Cm = to_tensor(C_motion, device=device, dtype=dtype)
    Cf = to_tensor(Cf_flow, device=device, dtype=dtype) if Cf_flow is not None else None
    Cf_cols = int(Cf.shape[-1]) if Cf is not None and Cf.numel() > 0 else 0

    metrics: Dict[str, float] = {}
    lam_final = float(lam_abs)
    sigma_max = sigma_min_lb = None

    _guard_eps = 0.0
    _ratio_rho_guard = 0.0
    if ka_opts:
        _ratio_rho_guard = float(ka_opts.get("ratio_rho", ka_opts.get("null_tail_ratio_rho", 0.0)))
    if R0_prior is not None:
        if ka_opts:
            _guard_eps = float(ka_opts.get("null_tail_guard_eps", 0.0))
        opts: Dict[str, float] = {
            "kappa_target": float(kappa_target),
        }
        if ka_opts:
            opts.update(
                {
                    k: v
                    for k, v in ka_opts.items()
                    if v is not None
                    and k
                    not in {
                        "null_tail_guard_eps",
                        "null_tail_ratio_rho",
                        "flow_guard_ratio_thresh",
                        "flow_guard_drop",
                    }
                }
            )
        Cf_np = Cf_flow
        if Cf_np is None and Cm.numel() > 0:
            Cf_np = Cm.cpu().numpy()
        R_lam, ka_info = ka_blend_covariance_temporal(
            R_sample=Rt,
            R0_prior=R0_prior,
            Cf_flow=Cf_np,
            device=device,
            dtype=dtype,
            **opts,
        )
        if ka_details is not None:
            ka_details.append(ka_info)
        lam_final = float(ka_info.get("lambda_used", lam_abs))
        sigma_max = ka_info.get("sigma_max_raw")
        sigma_min_lb = ka_info.get("sigma_min_raw")
        L = torch.linalg.cholesky(R_lam)
        metrics.update(
            {
                "ka_mode": ka_info.get("ka_mode"),
                "ka_beta": ka_info.get("beta"),
                "ka_alpha": ka_info.get("alpha"),
                "ka_mismatch": ka_info.get("mismatch"),
                "ka_lambda_used": lam_final,
                "ka_sigma_min_raw": sigma_min_lb,
                "ka_sigma_max_raw": sigma_max,
            }
        )
    else:
        eye = torch.eye(Lt_dim, dtype=dtype, device=device)
        lam_final, sigma_max, sigma_min_lb = conditioned_lambda(
            Rt.to(torch.complex128) if Rt.dtype.is_floating_point else Rt,
            lam_abs,
            kappa_target=kappa_target,
        )
        lam_final = float(lam_final)
        R_loaded = Rt + lam_final * eye
        L = torch.linalg.cholesky(R_loaded)

    S_flat = S.reshape(Lt_dim, -1)
    Sw_flat = torch.linalg.solve_triangular(L, S_flat, upper=False)

    if Cm.shape[-1] > 0:
        Cm_w = torch.linalg.solve_triangular(L, Cm, upper=False)
        S0w_flat = Sw_flat.clone()
        # Flow-preserving projector
        eye = torch.eye(Lt_dim, dtype=dtype, device=device)
        Pf = torch.zeros((Lt_dim, Lt_dim), dtype=dtype, device=device)
        if Cf_cols > 0:
            Cf_w = torch.linalg.solve_triangular(L, Cf, upper=False)
            Gf = Cf_w.conj().transpose(-2, -1) @ Cf_w
            eps = torch.trace(Gf).real / max(Gf.shape[-1], 1)
            Gf = Gf + (1e-6 * float(eps)) * torch.eye(Gf.shape[-1], dtype=dtype, device=device)
            Pf = Cf_w @ torch.linalg.solve(Gf, Cf_w.conj().transpose(-2, -1))
        P_perp = eye - Pf
        Cm_eff = P_perp @ Cm_w
        # QR with rank truncation
        if torch.allclose(Cm_eff, torch.zeros_like(Cm_eff)):
            U = torch.zeros((Lt_dim, 0), dtype=dtype, device=device)
        else:
            U, R_motion = torch.linalg.qr(Cm_eff, mode="reduced")
            diag = torch.abs(torch.diagonal(R_motion))
            keep = (diag > 1e-6).nonzero(as_tuple=False).flatten()
            if keep.numel() < R_motion.shape[0]:
                U = U[:, : keep.numel()]
        motion_rank_eff = int(U.shape[1])
        motion_rank_initial = int(Cm.shape[-1])
        if motion_rank_eff > 0:
            motion_coeff = U.conj().transpose(-2, -1) @ S0w_flat
            E_motion = torch.sum(motion_coeff.conj() * motion_coeff).real
            Sw_flat = S0w_flat - U @ motion_coeff
            motion = float(E_motion.item())
        else:
            motion = 0.0
        E_total = torch.sum(S0w_flat.conj() * S0w_flat).real
        total = float(E_total.item())
        removed_ratio = motion / max(total, 1e-30)
        metrics = {
            "energy_total": total,
            "energy_motion": motion,
            "energy_removed_ratio": float(removed_ratio),
            "energy_kept_ratio": float(max(0.0, 1.0 - removed_ratio)),
            "motion_rank_initial": motion_rank_initial,
            "motion_rank_eff": motion_rank_eff,
            "flow_rank": Cf_cols,
        }
    else:
        total_energy = torch.sum(Sw_flat.conj() * Sw_flat).real
        metrics = {
            "energy_total": float(total_energy.item()),
            "energy_motion": 0.0,
            "energy_removed_ratio": 0.0,
            "energy_kept_ratio": 1.0,
            "motion_rank_initial": 0,
            "motion_rank_eff": 0,
            "flow_rank": Cf_cols,
        }
    metrics.update(
        {
            "lambda_requested": float(lam_abs),
            "lambda_final": lam_final,
            "sigma_max_lb": sigma_max,
            "sigma_min_lb": sigma_min_lb,
            "kappa_target": float(kappa_target),
        }
    )

    Sw = Sw_flat.reshape(S.shape[0], *S.shape[1:])
    if return_metrics:
        return Sw, L, metrics
    return Sw, L


def split_fd_grid_by_motion(
    fd_grid_hz: np.ndarray | List[float],
    motion_half_span_hz: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split Doppler grid into motion (|f| <= span) and flow (|f| > span) subsets.
    """
    fd = np.asarray(fd_grid_hz, dtype=float)
    if fd.size == 0:
        return fd, fd
    mask_motion = np.abs(fd) <= float(motion_half_span_hz + 1e-12)
    return fd[mask_motion], fd[~mask_motion]


def _select_flow_grid(
    fd_flow: np.ndarray,
    max_tones: int,
    *,
    min_tones: int = 3,
) -> np.ndarray:
    """Return a subset of fd_flow with odd count ≤ max_tones and ≥ min_tones when possible."""
    if fd_flow.size == 0:
        return fd_flow
    max_tones = max(1, int(max_tones))
    if max_tones % 2 == 0:
        max_tones -= 1
    if max_tones < 1:
        max_tones = 1
    selected = fd_flow
    if fd_flow.size > max_tones:
        order = np.argsort(np.abs(fd_flow))
        selected = fd_flow[order[:max_tones]]
        selected = np.sort(selected)
    if selected.size < min_tones and fd_flow.size >= min_tones:
        # take the closest min_tones frequencies around zero
        order = np.argsort(np.abs(fd_flow))
        need = min(fd_flow.size, min_tones | 1)
        if need % 2 == 0:
            need = max(1, need - 1)
        selected = np.sort(fd_flow[order[:need]])
    return selected


@torch.no_grad()
def band_energy_on_whitened(
    Sw_Lt_N_hw: torch.Tensor,
    L_chol: torch.Tensor,
    prf_hz: float,
    fd_grid_hz: np.ndarray | List[float],
    *,
    ridge: float = 0.10,
    basis_mode: Literal["exp", "exp+deriv"] = "exp+deriv",
) -> torch.Tensor:
    """
    Compute per-snapshot band energy in the whitened temporal domain.
    """
    device = Sw_Lt_N_hw.device
    dtype = Sw_Lt_N_hw.dtype
    Lt_dim, N, h, w = Sw_Lt_N_hw.shape
    fd = np.asarray(fd_grid_hz, dtype=float)
    if fd.size == 0:
        return torch.zeros((N, h, w), dtype=torch.float32, device=device)

    Ct = bandpass_constraints_temporal(
        Lt=Lt_dim,
        prf_hz=prf_hz,
        fd_grid_hz=fd.tolist(),
        device=device,
        dtype=dtype,
        mode=basis_mode,
    )
    Cw = torch.linalg.solve_triangular(L_chol, Ct, upper=False)

    gram = Cw.conj().transpose(-2, -1) @ Cw
    if ridge > 0.0:
        gram = gram + float(ridge) * torch.eye(gram.shape[-1], dtype=dtype, device=device)

    Sw_flat = Sw_Lt_N_hw.reshape(Lt_dim, N * h * w)
    z = Cw.conj().transpose(-2, -1) @ Sw_flat
    proj_coeffs, _ = cholesky_solve_hermitian(gram, z, jitter_init=1e-10, max_tries=3)
    T_band = torch.sum(z.conj() * proj_coeffs, dim=0).real
    return T_band.reshape(N, h, w)


@torch.no_grad()
def msd_contrast_score_batched(
    R_t: torch.Tensor | np.ndarray,
    S_Lt_N_hw: torch.Tensor | np.ndarray,
    *,
    prf_hz: float,
    fd_grid_hz: np.ndarray | List[float],
    motion_half_span_hz: float,
    lam_abs: float = 3e-2,
    ridge: float = 0.10,
    agg: Literal["mean", "median", "trim10"] = "trim10",
    contrast_alpha: float = 0.7,
    basis_mode: Literal["exp", "exp+deriv"] = "exp+deriv",
    ratio_rho: float = 0.0,
    R0_prior: Optional[np.ndarray | torch.Tensor] = None,
    ka_opts: Optional[Dict[str, float]] = None,
    ka_details: Optional[List[Dict[str, float]]] = None,
    return_details: bool = False,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.complex64,
    eps: float = 1e-10,
    return_band_energy: bool = False,
) -> Tuple:
    """
    Project out motion subspace, compute flow/motion band fractions, and
    return a contrast score map along with the aggregated fractions.
    """
    return _msd_contrast_core_single(
        R_t,
        S_Lt_N_hw,
        prf_hz=prf_hz,
        fd_grid_hz=fd_grid_hz,
        motion_half_span_hz=motion_half_span_hz,
        lam_abs=lam_abs,
        ridge=ridge,
        agg=agg,
        contrast_alpha=contrast_alpha,
        basis_mode=basis_mode,
        ratio_rho=ratio_rho,
        R0_prior=R0_prior,
        ka_opts=ka_opts,
        ka_details=ka_details,
        return_details=return_details,
        device=device,
        dtype=dtype,
        eps=eps,
        return_band_energy=return_band_energy,
    )


def msd_contrast_core_whitened_batched(
    R_B_Lt_Lt: torch.Tensor,
    S_B_Lt_N_hw: torch.Tensor,
    *,
    prf_hz: float,
    fd_grid_hz: Sequence[float] | np.ndarray,
    motion_half_span_hz: float,
    lam_abs: float | torch.Tensor,
    ridge: float = 0.10,
    agg: Literal["mean", "median", "trim10"] = "trim10",
    contrast_alpha: float = 0.7,
    basis_mode: Literal["exp", "exp+deriv"] = "exp+deriv",
    ratio_rho: float = 0.0,
    R0_prior: Optional[np.ndarray | torch.Tensor] = None,
    ka_opts: Optional[Dict[str, float]] = None,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.complex64,
    eps: float = 1e-10,
    return_band_energy: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[dict], Optional[torch.Tensor]]:
    """
    Thin batched wrapper around _msd_contrast_core_single.

    Intended as a drop-in for both slow (B=1) and fast paths while we build
    a fully batched core. Keeps parity with existing single-tile logic.
    """
    if device is None:
        device = S_B_Lt_N_hw.device
    fd_grid = np.asarray(fd_grid_hz, dtype=float).tolist()
    lam_tensor = torch.as_tensor(lam_abs, device=device).flatten()
    lam_tensor = lam_tensor.real
    scores: List[torch.Tensor] = []
    r_flows: List[torch.Tensor] = []
    r_motions: List[torch.Tensor] = []
    dets: List[dict] = []
    tb_flow_list: List[torch.Tensor] = []
    B = S_B_Lt_N_hw.shape[0]
    for b in range(B):
        lam_b = float(lam_tensor[min(b, lam_tensor.numel() - 1)].item())
        out = _msd_contrast_core_single(
            R_B_Lt_Lt[b],
            S_B_Lt_N_hw[b],
            prf_hz=prf_hz,
            fd_grid_hz=fd_grid,
            motion_half_span_hz=motion_half_span_hz,
            lam_abs=lam_b,
            ridge=ridge,
            agg=agg,
            contrast_alpha=contrast_alpha,
            basis_mode=basis_mode,
            ratio_rho=ratio_rho,
            R0_prior=R0_prior,
            ka_opts=ka_opts,
            ka_details=None,
            return_details=True,
            device=device,
            dtype=dtype,
            eps=eps,
            return_band_energy=return_band_energy,
        )
        if return_band_energy:
            score_tile, r_flow_tile, r_motion_tile, det, Tb_flow_tile = out  # type: ignore[misc]
            tb_flow_list.append(Tb_flow_tile)
        else:
            score_tile, r_flow_tile, r_motion_tile, det = out  # type: ignore[misc]
        scores.append(score_tile)
        r_flows.append(r_flow_tile)
        r_motions.append(r_motion_tile)
        dets.append(det)
    score_stack = torch.stack(scores, dim=0)
    r_flow_stack = torch.stack(r_flows, dim=0)
    r_motion_stack = torch.stack(r_motions, dim=0)
    tb_flow_stack = (
        torch.stack(tb_flow_list, dim=0) if return_band_energy and tb_flow_list else None
    )
    return score_stack, r_flow_stack, r_motion_stack, dets, tb_flow_stack


def _msd_contrast_core_single(
    R_t: torch.Tensor | np.ndarray,
    S_Lt_N_hw: torch.Tensor | np.ndarray,
    *,
    prf_hz: float,
    fd_grid_hz: np.ndarray | List[float],
    motion_half_span_hz: float,
    lam_abs: float = 3e-2,
    ridge: float = 0.10,
    agg: Literal["mean", "median", "trim10"] = "trim10",
    contrast_alpha: float = 0.7,
    basis_mode: Literal["exp", "exp+deriv"] = "exp+deriv",
    ratio_rho: float = 0.0,
    R0_prior: Optional[np.ndarray | torch.Tensor] = None,
    ka_opts: Optional[Dict[str, float]] = None,
    ka_details: Optional[List[Dict[str, float]]] = None,
    return_details: bool = False,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.complex64,
    eps: float = 1e-10,
    return_band_energy: bool = False,
) -> Tuple:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    Rt = to_tensor(R_t, device=device, dtype=dtype)
    S = to_tensor(S_Lt_N_hw, device=device, dtype=dtype)
    Lt_dim = Rt.shape[0]

    fd_full = np.asarray(fd_grid_hz, dtype=float)
    contrast_enabled = motion_half_span_hz > 0.0 and contrast_alpha > 0.0
    min_flow_tones = 3
    ka_log: List[Dict[str, float]] = []

    if contrast_enabled and fd_full.size > 0:
        fd_motion_raw, fd_flow_raw = split_fd_grid_by_motion(fd_full, motion_half_span_hz)
        motion_basis = build_motion_basis_temporal(
            Lt=Lt_dim,
            prf_hz=prf_hz,
            width_bins=1,
            include_dc=True,
            device=device,
            dtype=dtype,
        )
        motion_rank = motion_basis.shape[1]
        max_flow = max(1, Lt_dim - motion_rank)
        fd_flow = _select_flow_grid(fd_flow_raw, max_flow, min_tones=min_flow_tones)
        if fd_flow.size < min_flow_tones:
            # disable contrast for this call
            contrast_enabled = False
            fd_motion = np.array([], dtype=float)
            fd_flow = fd_full
            motion_basis = torch.zeros((Lt_dim, 0), dtype=dtype, device=device)
        else:
            fd_motion = fd_motion_raw
    else:
        contrast_enabled = False
        fd_motion = np.array([], dtype=float)
        fd_flow = fd_full
        motion_basis = torch.zeros((Lt_dim, 0), dtype=dtype, device=device)

    Cf_flow_tensor = None
    if fd_flow.size > 0:
        Cf_flow_tensor = bandpass_constraints_temporal(
            Lt=Lt_dim,
            prf_hz=prf_hz,
            fd_grid_hz=fd_flow.tolist(),
            device=device,
            dtype=dtype,
            mode=basis_mode,
        )

    result = project_out_motion_whitened(
        Rt,
        S,
        motion_basis,
        lam_abs=lam_abs,
        kappa_target=40.0,
        R0_prior=R0_prior,
        Cf_flow=Cf_flow_tensor,
        ka_opts=ka_opts,
        ka_details=ka_log if ka_details is not None else None,
        device=device,
        dtype=dtype,
        return_metrics=True,
    )
    Sw, L, motion_metrics = result

    Tb_flow = band_energy_on_whitened(Sw, L, prf_hz, fd_flow, ridge=ridge, basis_mode=basis_mode)
    Tb_motion = band_energy_on_whitened(
        Sw, L, prf_hz, fd_motion, ridge=ridge, basis_mode=basis_mode
    )
    sw_pow = torch.sum(Sw.conj() * Sw, dim=0).real

    r_flow = torch.clamp(Tb_flow / torch.clamp(sw_pow, min=eps), min=0.0, max=1.0)
    r_motion = torch.clamp(Tb_motion / torch.clamp(sw_pow, min=eps), min=0.0, max=1.0)

    r_flow_agg = aggregate_over_snapshots(r_flow, mode=agg)
    r_motion_agg = aggregate_over_snapshots(r_motion, mode=agg)

    ratio_denom = torch.clamp(sw_pow - Tb_flow + float(ratio_rho) * sw_pow, min=eps)
    ratio_nhw = torch.clamp(Tb_flow / ratio_denom, min=0.0)
    ratio_agg = aggregate_over_snapshots(ratio_nhw, mode=agg)
    score_msd = torch.log1p(ratio_agg)

    flow_median = float(torch.nanmedian(r_flow_agg).item())
    motion_median = float(torch.nanmedian(r_motion_agg).item())
    beta_est = float(motion_metrics.get("energy_removed_ratio", 0.0))
    zeta_est = max(0.0, 1.0 - flow_median)
    contrast_score = None
    score_mode = "msd"
    eta = 0.0

    if contrast_enabled and fd_motion.size > 0 and fd_flow.size > 0:
        contrast_score = torch.clamp(
            torch.log1p(r_flow_agg) - float(contrast_alpha) * torch.log1p(r_motion_agg),
            min=0.0,
        )
        if flow_median < 1e-3:
            contrast_score = None
        elif (
            beta_est > zeta_est
            and fd_flow.size >= min_flow_tones
            and flow_median > motion_median + 5e-2
        ):
            eta = 0.2
            score_mode = "msd_contrast_mix"
        else:
            contrast_score = None

    if contrast_score is not None:
        score_tile = (1.0 - eta) * score_msd + eta * contrast_score
    else:
        score_tile = score_msd

    # Winsorize at 99.9 percentile per tile to avoid extreme nulls
    flat = score_tile.flatten()
    if flat.numel() > 0:
        win = torch.quantile(flat, 0.999)
        if torch.isfinite(win):
            score_tile = torch.clamp(score_tile, max=float(win.item()))

    details = {
        "basis_mode": basis_mode,
        "zeta_est": float(zeta_est),
        "beta_est": float(beta_est),
        "fallback": bool(contrast_score is None),
        "score_mode": score_mode,
        "flow_fraction_median": float(flow_median),
        "motion_fraction_median": float(motion_median),
        "energy_kept_ratio": float(motion_metrics.get("energy_kept_ratio", 1.0)),
        "energy_removed_ratio": float(motion_metrics.get("energy_removed_ratio", 0.0)),
        "kc_flow": int(fd_flow.size),
        "kc_motion": int(fd_motion.size),
        "eta": float(eta),
    }
    details["flow_rank"] = motion_metrics.get("flow_rank")
    details["motion_rank_initial"] = motion_metrics.get("motion_rank_initial")
    details["motion_rank_eff"] = motion_metrics.get("motion_rank_eff")
    if ka_log:
        details["ka_last"] = ka_log[-1]
    if ka_details is not None and ka_log:
        ka_details.extend(ka_log)

    outputs: list = [score_tile, r_flow_agg, r_motion_agg]
    if return_details:
        outputs.append(details)
    if return_band_energy:
        outputs.append(Tb_flow)
    return tuple(outputs)


def _msd_contrast_core_batch(
    R_t_batch: torch.Tensor | np.ndarray,
    S_batch_Lt_N_hw: torch.Tensor | np.ndarray,
    *,
    prf_hz: float,
    fd_grid_hz: np.ndarray | List[float],
    motion_half_span_hz: float,
    lam_abs: float = 3e-2,
    ridge: float = 0.10,
    agg: Literal["mean", "median", "trim10"] = "trim10",
    contrast_alpha: float = 0.7,
    basis_mode: Literal["exp", "exp+deriv"] = "exp+deriv",
    ratio_rho: float = 0.0,
    R0_prior: Optional[np.ndarray | torch.Tensor] = None,
    ka_opts: Optional[Dict[str, float]] = None,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.complex64,
    eps: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, float]]]:
    """
    Batched wrapper around the MSD/contrast core (loops over batch).
    Returns score_tile (B,h,w), r_flow (B,h,w), r_motion (B,h,w), details list.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    Rb = to_tensor(R_t_batch, device=device, dtype=dtype)
    Sb = to_tensor(S_batch_Lt_N_hw, device=device, dtype=dtype)
    if Rb.dim() == 2:
        Rb = Rb.unsqueeze(0)
    if Sb.dim() == 3:
        Sb = Sb.unsqueeze(0)
    B = Rb.shape[0]
    scores = []
    r_flows = []
    r_motions = []
    details_list: List[Dict[str, float]] = []
    for b in range(B):
        out = _msd_contrast_core_single(
            Rb[b],
            Sb[b],
            prf_hz=prf_hz,
            fd_grid_hz=fd_grid_hz,
            motion_half_span_hz=motion_half_span_hz,
            lam_abs=lam_abs,
            ridge=ridge,
            agg=agg,
            contrast_alpha=contrast_alpha,
            basis_mode=basis_mode,
            ratio_rho=ratio_rho,
            R0_prior=R0_prior,
            ka_opts=ka_opts,
            ka_details=None,
            return_details=True,
            device=device,
            dtype=dtype,
            eps=eps,
            return_band_energy=False,
        )
        score_tile, r_flow_agg, r_motion_agg, det = out  # type: ignore[misc]
        scores.append(score_tile)
        r_flows.append(r_flow_agg)
        r_motions.append(r_motion_agg)
        details_list.append(det)
    score_stack = torch.stack(scores, dim=0)
    r_flow_stack = torch.stack(r_flows, dim=0)
    r_motion_stack = torch.stack(r_motions, dim=0)
    return score_stack, r_flow_stack, r_motion_stack, details_list


def build_temporal_hankels_and_cov(
    cube_T_hw: np.ndarray | torch.Tensor,
    Lt: int,
    *,
    center: bool = True,
    estimator: str = "huber",
    huber_c: float = 5.0,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.complex64,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Build per-pixel slow-time Hankel stacks and a pooled temporal covariance.

    Parameters
    ----------
    cube_T_hw : (T, h, w) complex ndarray/torch.Tensor
    Lt : int
        Slow-time aperture (rows of Hankel matrix).
    center : bool
        subtract per-row mean across slow-time columns.
    estimator : str
        robust_covariance estimator.
    huber_c : float
        Huber parameter when estimator == "huber".

    Returns
    -------
    S_Lt_N_hw : torch.Tensor
        Shape (Lt, N, h, w) complex Hankel stacks per pixel.
    R_t : torch.Tensor
        Shape (Lt, Lt) pooled temporal covariance.
    tel : dict
        Telemetry (Lt, N, trace, eff_rank, cond_est, plus estimator diagnostics).
    """

    if isinstance(cube_T_hw, np.ndarray):
        x = torch.as_tensor(cube_T_hw, dtype=dtype)
    else:
        x = cube_T_hw.to(dtype)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    x = x.to(device)

    T, h, w = x.shape
    if Lt < 2 or Lt >= T:
        raise ValueError(f"Need 2 <= Lt < T (got Lt={Lt}, T={T})")

    N = T - Lt + 1

    # reshape to (P, T)
    x_flat = x.permute(1, 2, 0).contiguous().view(h * w, T)  # (P, T)

    rows = []
    for k in range(Lt):
        rows.append(x_flat[:, k : k + N])
    S_P_Lt_N = torch.stack(rows, dim=1)  # (P, Lt, N)

    # Optional training snapshot subsampling to limit effective support.
    # Controlled via environment variables rather than plumbing new arguments:
    #   STAP_SNAPSHOT_STRIDE : integer stride along the N axis (>=1).
    #   STAP_MAX_SNAPSHOTS   : integer cap on N after striding.
    # This is intended for stress-testing regimes where we want fewer
    # Hankel columns per tile without changing the underlying slow-time cube.
    stride_env = os.getenv("STAP_SNAPSHOT_STRIDE", "").strip()
    max_env = os.getenv("STAP_MAX_SNAPSHOTS", "").strip()
    try:
        stride = int(stride_env) if stride_env else 1
    except ValueError:
        stride = 1
    if stride < 1:
        stride = 1
    if stride > 1 and N > 1:
        S_P_Lt_N = S_P_Lt_N[:, :, ::stride]
        N = S_P_Lt_N.shape[2]
    try:
        max_snaps = int(max_env) if max_env else None
    except ValueError:
        max_snaps = None
    if max_snaps is not None and max_snaps > 0 and N > max_snaps:
        # Evenly spaced subset of snapshots along N. Ensure the index tensor
        # lives on the same device as S_P_Lt_N to avoid device mismatch.
        idx = torch.linspace(0, N - 1, steps=max_snaps, device=S_P_Lt_N.device, dtype=torch.long)
        S_P_Lt_N = S_P_Lt_N.index_select(2, idx)
        N = S_P_Lt_N.shape[2]

    if center:
        S_P_Lt_N = S_P_Lt_N - S_P_Lt_N.mean(dim=2, keepdim=True)

    # reshape to (Lt, N, h, w)
    S_Lt_N_hw = S_P_Lt_N.permute(1, 2, 0).contiguous().view(Lt, N, h, w)

    # pooled covariance: concatenate across pixels
    X_pool = S_P_Lt_N.permute(1, 2, 0).contiguous().view(Lt, N * h * w)
    R_t, tel_cov = robust_covariance(
        X_pool,
        method=estimator,
        huber_c=huber_c,
        max_iter=100,
        tol=1e-4,
    )
    if not isinstance(R_t, torch.Tensor):
        R_t = torch.as_tensor(R_t, dtype=dtype, device=device)
    else:
        R_t = R_t.to(dtype=dtype, device=device)

    evals = torch.linalg.eigvalsh(R_t).real.clamp_min(0)
    trace = float(evals.sum().item())
    eff_rank = float((trace**2) / (torch.sum(evals**2).item() + 1e-30))
    cond_est = float((torch.max(evals).item() + 1e-12) / (torch.min(evals).item() + 1e-12))
    tel: Dict[str, float] = {
        "Lt": int(Lt),
        "N": int(N),
        "trace": trace,
        "eff_rank": eff_rank,
        "cond_est": cond_est,
    }
    tel_dict = asdict(tel_cov) if hasattr(tel_cov, "__dict__") else {}
    for k, v in tel_dict.items():
        if isinstance(v, (int, float)):
            tel[f"cov_{k}"] = float(v)
    return S_Lt_N_hw, R_t, tel


def bandpass_constraints_temporal(
    Lt: int,
    prf_hz: float,
    fd_grid_hz: list[float] | np.ndarray,
    *,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.complex64,
    mode: Literal["exp", "exp+deriv"] = "exp",
) -> torch.Tensor:
    """
    Build temporal band-pass constraint matrix of size (Lt, Kc).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    fd_arr = np.asarray(fd_grid_hz, dtype=np.float64)
    m = np.arange(Lt, dtype=np.float64)
    cols = []
    m_centered = m - m.mean()
    for fd in fd_arr:
        omega = 2.0 * np.pi * fd / float(prf_hz)
        s = np.exp(1j * omega * m)
        s = s / max(np.linalg.norm(s), 1e-12)
        cols.append(s[:, None])
        if mode == "exp+deriv":
            deriv = 1j * m_centered * s
            deriv = deriv / max(np.linalg.norm(deriv), 1e-12)
            cols.append(deriv[:, None])
    if not cols:
        return torch.zeros((Lt, 0), dtype=dtype, device=device)
    Ct = np.concatenate(cols, axis=1)
    Ct_torch = torch.as_tensor(Ct, dtype=torch.complex64, device=device)
    Ct_torch = Ct_torch.to(dtype=dtype)
    return Ct_torch


@torch.no_grad()
def msd_snapshot_energies_batched(
    R_t: torch.Tensor | np.ndarray,
    S_Lt_N_hw: torch.Tensor | np.ndarray,
    C_t: torch.Tensor | np.ndarray,
    *,
    lam_abs: float = 2e-2,
    kappa_target: float = 40.0,
    ridge: float = 0.10,
    ratio_rho: float = 0.0,
    R0_prior: Optional[np.ndarray | torch.Tensor] = None,
    Cf_flow: Optional[np.ndarray | torch.Tensor] = None,
    ka_opts: Optional[Dict[str, float]] = None,
    ka_details: Optional[List[Dict[str, float]]] = None,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.complex64,
    eps: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-snapshot band energy and total whitened power.

    Returns
    -------
    T_band : torch.Tensor (N, h, w)
        Energy captured by band projector in whitened space.
    sw_pow : torch.Tensor (N, h, w)
        Total whitened slow-time energy.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    Rt = to_tensor(R_t, device=device, dtype=dtype)
    S = to_tensor(S_Lt_N_hw, device=device, dtype=dtype)
    Ct = to_tensor(C_t, device=device, dtype=dtype)
    Cf = Cf_flow

    def _reduce_columns(mat: torch.Tensor, max_rank: int) -> torch.Tensor:
        if mat.shape[-1] <= max_rank:
            return mat
        try:
            U, _, _ = torch.linalg.svd(mat, full_matrices=False)
            return U[:, :max_rank]
        except RuntimeError:
            return mat[:, :max_rank]

    lam_eff = float(lam_abs)
    if R0_prior is not None:
        lam_override_req = None
        if ka_opts and "lambda_override" in ka_opts and ka_opts["lambda_override"] is not None:
            try:
                lam_override_req = float(ka_opts["lambda_override"])
            except (TypeError, ValueError):
                lam_override_req = None
        min_floor = lam_eff
        if lam_override_req is None and lam_eff < 0.02:
            min_floor = 0.02
        lam_candidates = [min_floor]
        if lam_override_req is not None:
            lam_candidates.append(lam_override_req)
        lam_eff = max(lam_candidates)

    Lt_dim = Rt.shape[0]
    if S.ndim < 2:
        raise ValueError(f"Expected S with at least 2 dims (Lt, ...), got shape {tuple(S.shape)}")
    if S.shape[0] != Lt_dim:
        # Handle mis-ordered snapshot tensors by moving the Lt axis up front
        # and selecting a reasonable slow-time axis based on size uniqueness.
        tail_axes = list(range(S.ndim))
        lt_axes = [ax for ax, sz in enumerate(S.shape) if sz == Lt_dim]
        if lt_axes:
            lt_ax = lt_axes[0]
            order = [lt_ax] + [ax for ax in tail_axes if ax != lt_ax]
            if len(order) >= 2:
                rem_axes = order[1:]
                rem_sizes = [S.shape[ax] for ax in rem_axes]
                size_counts: dict[int, int] = {}
                for sz in rem_sizes:
                    size_counts[int(sz)] = size_counts.get(int(sz), 0) + 1
                unique_axes = [
                    ax for ax in rem_axes if size_counts.get(int(S.shape[ax]), 0) == 1
                ]
                if unique_axes:
                    snap_ax = unique_axes[0]
                else:
                    snap_ax = rem_axes[int(np.argmax(rem_sizes))]
                # Put snapshots immediately after Lt.
                if snap_ax != rem_axes[0]:
                    snap_pos = order.index(snap_ax)
                    order[1], order[snap_pos] = order[snap_pos], order[1]
            S = S.permute(order).contiguous()
        if S.shape[0] != Lt_dim:
            raise ValueError(
                f"First dim of S ({S.shape[0]}) must match Lt={Lt_dim} from Rt; "
                f"got S shape {tuple(S.shape)} and Rt shape {tuple(Rt.shape)}"
            )
    elif S.ndim >= 3:
        # Lt is already leading; still normalize the snapshot axis so that it
        # sits directly after Lt.
        rem_axes = list(range(1, S.ndim))
        rem_sizes = [S.shape[ax] for ax in rem_axes]
        size_counts: dict[int, int] = {}
        for sz in rem_sizes:
            size_counts[int(sz)] = size_counts.get(int(sz), 0) + 1
        unique_axes = [ax for ax in rem_axes if size_counts.get(int(S.shape[ax]), 0) == 1]
        if unique_axes:
            snap_ax = unique_axes[0]
        else:
            snap_ax = rem_axes[int(np.argmax(rem_sizes))]
        if snap_ax != 1:
            order = [0, snap_ax] + [ax for ax in rem_axes if ax != snap_ax]
            S = S.permute(order).contiguous()
    # Snapshot / spatial shape (e.g., (N, h, w)); keep this generic so the same
    # code works for 3-D, 4-D, or higher-rank layouts.
    snap_shape = S.shape[1:]
    P = int(np.prod(snap_shape)) if snap_shape else 1

    if Ct.shape[-1] == 0:
        zeros = torch.zeros(snap_shape, dtype=torch.float32, device=device)
        return zeros, zeros

    guard_eps = 0.0
    guard_active = False
    ratio_rho_guard = 0.0
    if ratio_rho is not None:
        try:
            ratio_rho_guard = max(0.0, float(ratio_rho))
        except (TypeError, ValueError):
            ratio_rho_guard = 0.0
    if R0_prior is not None:
        guard_active = True
        if ka_opts:
            guard_eps = float(ka_opts.get("null_tail_guard_eps", 0.0))
            if guard_eps < 0.0:
                guard_active = False
            ratio_opt = ka_opts.get("ratio_rho", ka_opts.get("null_tail_ratio_rho"))
            if ratio_opt is not None:
                try:
                    ratio_rho_guard = max(ratio_rho_guard, float(ratio_opt))
                except (TypeError, ValueError):
                    pass
        opts: Dict[str, float] = {
            "kappa_target": float(kappa_target),
        }
        if ka_opts:
            opts.update(
                {
                    k: v
                    for k, v in ka_opts.items()
                    if v is not None
                    and k
                    not in {
                        "null_tail_guard_eps",
                        "null_tail_ratio_rho",
                        "flow_guard_ratio_thresh",
                        "flow_guard_drop",
                    }
                }
            )
        Cf_tensor = Cf
        if Cf_tensor is None:
            Cf_tensor = Ct.cpu().numpy()
        else:
            Cf_tensor_full = to_tensor(Cf_tensor, device="cpu", dtype=torch.complex128)
            if Cf_tensor_full.shape[-1] >= Lt_dim:
                try:
                    evals, evecs = torch.linalg.eigh(Rt.to(torch.complex128).to("cpu"))
                    v_flow = evecs[:, -1]
                    scores = torch.abs(Cf_tensor_full.conj().transpose(-2, -1) @ v_flow)
                    top_idx = torch.argsort(scores, descending=True)[: max(1, Lt_dim - 1)]
                    Cf_tensor_full = Cf_tensor_full[:, top_idx]
                except RuntimeError:
                    Cf_tensor_full = Cf_tensor_full[:, : max(1, Lt_dim - 1)]
            Cf_tensor = Cf_tensor_full.cpu().numpy()
        opts.setdefault("lambda_override", lam_eff)
        R_lam, ka_info = ka_blend_covariance_temporal(
            R_sample=Rt,
            R0_prior=R0_prior,
            Cf_flow=Cf_tensor,
            device=device,
            dtype=dtype,
            **opts,
        )
        if ka_details is not None:
            ka_details.append(ka_info)
        lam_used = float(ka_info.get("lambda_used", lam_eff))
        _lam_floor_used = float(ka_info.get("lambda_floor", lam_eff))
        # Numerical hygiene: enforce Hermitian symmetry before factorization.
        R_lam = 0.5 * (R_lam + R_lam.conj().transpose(-2, -1))
        try:
            L = torch.linalg.cholesky(R_lam)
        except RuntimeError:
            # Add adaptive jitter (scaled by mu) to avoid hard failures on
            # near-singular tiles in the fast path fallback.
            diag = torch.real(torch.diagonal(R_lam, dim1=-2, dim2=-1))
            scale_b = torch.mean(torch.abs(diag)) + 1e-12
            # Jitter ladder: last entries are intentionally large so we always
            # return a defined factorization rather than hard-failing.
            jitters = (1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0)
            eye = torch.eye(Lt_dim, dtype=dtype, device=device)
            for mult in jitters:
                try:
                    L = torch.linalg.cholesky(R_lam + float(mult) * scale_b * eye)
                    break
                except RuntimeError:
                    L = None  # type: ignore[assignment]
            if L is None:
                # Last resort: diagonal approximation (keeps the method defined).
                d = torch.clamp(torch.real(torch.diagonal(R_lam, dim1=-2, dim2=-1)), min=1e-8)
                L = torch.diag(torch.sqrt(d.to(dtype=torch.float32))).to(dtype=dtype)
    else:
        eye = torch.eye(Lt_dim, dtype=dtype, device=device)
        Rt_herm = 0.5 * (Rt + Rt.conj().transpose(-2, -1))
        lam_final, _, _ = conditioned_lambda(
            Rt_herm.to(torch.complex128) if Rt_herm.dtype.is_floating_point else Rt_herm,
            lam_eff,
            kappa_target=kappa_target,
        )
        R_loaded = Rt_herm + float(lam_final) * eye
        R_loaded = 0.5 * (R_loaded + R_loaded.conj().transpose(-2, -1))
        try:
            L = torch.linalg.cholesky(R_loaded)
        except RuntimeError:
            diag = torch.real(torch.diagonal(R_loaded, dim1=-2, dim2=-1))
            scale_b = torch.mean(torch.abs(diag)) + 1e-12
            jitters = (1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0)
            for mult in jitters:
                try:
                    L = torch.linalg.cholesky(R_loaded + float(mult) * scale_b * eye)
                    break
                except RuntimeError:
                    L = None  # type: ignore[assignment]
            if L is None:
                d = torch.clamp(torch.real(torch.diagonal(R_loaded, dim1=-2, dim2=-1)), min=1e-8)
                L = torch.diag(torch.sqrt(d.to(dtype=torch.float32))).to(dtype=dtype)

    # Flatten snapshots / spatial dimensions but keep Lt explicit.
    S_flat = S.reshape(Lt_dim, P)
    S_w = torch.linalg.solve_triangular(L, S_flat, upper=False, left=True)
    S_w = S_w.reshape(Lt_dim, *snap_shape)

    C_w = torch.linalg.solve_triangular(L, Ct, upper=False, left=True)
    Gram = C_w.conj().transpose(-2, -1) @ C_w
    if ridge > 0.0:
        Gram = Gram + float(ridge) * torch.eye(Gram.shape[-1], dtype=dtype, device=device)

    Sw_flat = S_w.reshape(Lt_dim, P)
    z = C_w.conj().transpose(-2, -1) @ Sw_flat  # (Kc, N*h*w)
    proj_coeffs, _ = cholesky_solve_hermitian(Gram, z, jitter_init=1e-10, max_tries=3)
    T_band = torch.sum(z.conj() * proj_coeffs, dim=0).real

    Sw_conj = Sw_flat.conj()
    sw_pow = torch.sum(Sw_conj * Sw_flat, dim=0).real

    T_band = torch.clamp(T_band, min=0.0)
    sw_pow = torch.clamp(sw_pow, min=eps)

    if guard_active:
        eye_base = torch.eye(Lt_dim, dtype=dtype, device=device)
        lam_guard_req = max(float(lam_abs), min(lam_used, 2e-2))
        Rt_herm = 0.5 * (Rt + Rt.conj().transpose(-2, -1))
        lam_guard_final, _, _ = conditioned_lambda(
            Rt_herm.to(torch.complex128) if Rt_herm.dtype.is_floating_point else Rt_herm,
            lam_guard_req,
            kappa_target=float(kappa_target),
        )
        R_loaded_base = Rt_herm + float(lam_guard_final) * eye_base
        R_loaded_base = 0.5 * (R_loaded_base + R_loaded_base.conj().transpose(-2, -1))
        try:
            L_base = torch.linalg.cholesky(R_loaded_base)
        except RuntimeError:
            diag = torch.real(torch.diagonal(R_loaded_base, dim1=-2, dim2=-1))
            scale_b = torch.mean(torch.abs(diag)) + 1e-12
            jitters = (1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0)
            for mult in jitters:
                try:
                    L_base = torch.linalg.cholesky(R_loaded_base + float(mult) * scale_b * eye_base)
                    break
                except RuntimeError:
                    L_base = None  # type: ignore[assignment]
            if L_base is None:
                d = torch.clamp(
                    torch.real(torch.diagonal(R_loaded_base, dim1=-2, dim2=-1)), min=1e-8
                )
                L_base = torch.diag(torch.sqrt(d.to(dtype=torch.float32))).to(dtype=dtype)
        Sw_flat_base = torch.linalg.solve_triangular(L_base, S_flat, upper=False, left=True)
        C_w_base = torch.linalg.solve_triangular(L_base, Ct, upper=False, left=True)
        Gram_base = C_w_base.conj().transpose(-2, -1) @ C_w_base
        if ridge > 0.0:
            Gram_base = Gram_base + float(ridge) * torch.eye(
                Gram_base.shape[-1], dtype=dtype, device=device
            )
        z_base = C_w_base.conj().transpose(-2, -1) @ Sw_flat_base
        proj_base, _ = cholesky_solve_hermitian(Gram_base, z_base, jitter_init=1e-10, max_tries=3)
        T_band_base = torch.sum(z_base.conj() * proj_base, dim=0).real
        sw_pow_base = torch.sum(Sw_flat_base.conj() * Sw_flat_base, dim=0).real
        T_band_base = torch.clamp(T_band_base, min=0.0).reshape(-1)
        sw_pow_base = torch.clamp(sw_pow_base, min=eps).reshape(-1)
        denom_base = torch.clamp(sw_pow_base - T_band_base, min=eps)
        T_band_flat = T_band.reshape(-1)
        sw_pow_flat = sw_pow.reshape(-1)
        denom_current = torch.clamp(sw_pow_flat - T_band_flat, min=eps)
        a = 1.0 + ratio_rho_guard
        ratio_base = torch.clamp(T_band_base / (denom_base * a), min=0.0)
        ratio_slack = max(0.0, guard_eps)
        ratio_limit = (1.0 + ratio_slack) * ratio_base
        ratio_current = torch.clamp(T_band_flat / (denom_current * a), min=0.0)
        ratio_limit = torch.where(torch.isfinite(ratio_limit), ratio_limit, ratio_current)
        ratio_clamped = torch.minimum(ratio_current, ratio_limit)

        flow_guard_thresh = 8.0
        flow_guard_drop = 0.02
        if ka_opts:
            raw_thresh = ka_opts.get("flow_guard_ratio_thresh")
            if raw_thresh is not None:
                try:
                    flow_guard_thresh = float(raw_thresh)
                except (TypeError, ValueError):
                    flow_guard_thresh = 8.0
            raw_drop = ka_opts.get("flow_guard_drop")
            if raw_drop is not None:
                try:
                    flow_guard_drop = float(raw_drop)
                except (TypeError, ValueError):
                    flow_guard_drop = 0.02
        flow_guard_drop = min(max(flow_guard_drop, 0.0), 0.5)
        high_mask = ratio_base >= flow_guard_thresh

        ratio_floor = ratio_base * (1.0 - flow_guard_drop)
        ratio_floor = torch.where(torch.isfinite(ratio_floor), ratio_floor, ratio_base)
        ratio_floor = torch.minimum(ratio_floor, ratio_limit)
        ratio_clamped = torch.where(
            high_mask, torch.maximum(ratio_clamped, ratio_floor), ratio_clamped
        )

        scale = ratio_clamped * a
        T_band_new = (scale * sw_pow_flat) / (1.0 + scale)

        T_band = T_band_new.reshape_as(T_band_flat)

    return T_band.reshape(*snap_shape), sw_pow.reshape(*snap_shape)


def _band_energy_whitened_batched(
    R_B_Lt_Lt: torch.Tensor,
    S_B_Lt_N_hw: torch.Tensor,
    C_t: torch.Tensor,
    lam_B: torch.Tensor,
    *,
    ridge: float = 0.10,
    ratio_rho: float = 0.0,
    kappa_target: float = 40.0,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.complex64,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batched band-energy computation on whitened snapshots.

    R_B_Lt_Lt: (B, Lt, Lt)
    S_B_Lt_N_hw: (B, Lt, N, h, w)
    C_t: (Lt, K)
    lam_B: (B,) absolute loading per tile
    Returns:
      T_band: (B, N, h, w)
      sw_pow: (B, N, h, w)
    """
    if device is None:
        device = R_B_Lt_Lt.device

    # Optional precision override for the band-energy path. By default we
    # operate in the same complex dtype as the caller (typically complex64).
    # When STAP_BAND_PRECISION=fp64, we up-cast to complex128 for numerical
    # experiments; lower precisions are not enabled here because PyTorch does
    # not yet provide full complex16 support across the required ops.
    prec = os.getenv("STAP_BAND_PRECISION", "").strip().lower()
    if prec == "fp64":
        work_dtype = torch.complex128
    else:
        work_dtype = dtype

    R = R_B_Lt_Lt.to(device=device, dtype=work_dtype)
    S = S_B_Lt_N_hw.to(device=device, dtype=work_dtype)
    Ct = C_t.to(device=device, dtype=work_dtype)
    lam_vec = lam_B.to(device=device).flatten()

    B_R, Lt_R, _ = R.shape
    # Normalize snapshot layout in case Lt is not at dim=1.
    if S.shape[1] != Lt_R:
        tail_axes = list(range(1, S.ndim))
        lt_axes = [ax for ax in tail_axes if S.shape[ax] == Lt_R]
        if lt_axes:
            lt_ax = lt_axes[0]
            order_tail = [lt_ax] + [ax for ax in tail_axes if ax != lt_ax]
            if len(order_tail) >= 2:
                rem_axes = order_tail[1:]
                rem_sizes = [S.shape[ax] for ax in rem_axes]
                size_counts: dict[int, int] = {}
                for sz in rem_sizes:
                    size_counts[int(sz)] = size_counts.get(int(sz), 0) + 1
                unique_axes = [
                    ax for ax in rem_axes if size_counts.get(int(S.shape[ax]), 0) == 1
                ]
                if unique_axes:
                    snap_ax = unique_axes[0]
                else:
                    snap_ax = rem_axes[int(np.argmax(rem_sizes))]
                if snap_ax != rem_axes[0]:
                    snap_pos = order_tail.index(snap_ax)
                    order_tail[1], order_tail[snap_pos] = order_tail[snap_pos], order_tail[1]
            perm = [0] + order_tail
            S = S.permute(perm).contiguous()
    else:
        # Lt already at dim=1; still normalize snapshot axis into dim=2.
        rem_axes = list(range(2, S.ndim))
        rem_sizes = [S.shape[ax] for ax in rem_axes]
        if rem_axes:
            size_counts: dict[int, int] = {}
            for sz in rem_sizes:
                size_counts[int(sz)] = size_counts.get(int(sz), 0) + 1
            unique_axes = [
                ax for ax in rem_axes if size_counts.get(int(S.shape[ax]), 0) == 1
            ]
            if unique_axes:
                snap_ax = unique_axes[0]
            else:
                snap_ax = rem_axes[int(np.argmax(rem_sizes))]
            if snap_ax != 2:
                order = [0, 1, snap_ax] + [ax for ax in rem_axes if ax != snap_ax]
                S = S.permute(order).contiguous()

    B_S, Lt_S = S.shape[:2]
    if S.ndim < 4:
        raise ValueError(f"Expected >=4 dims for S, got {tuple(S.shape)}")
    if S.ndim == 4:
        N, h = S.shape[2], S.shape[3]
        w = 1
    else:
        N, h, w = S.shape[2], S.shape[3], S.shape[4]
    if B_R != B_S or Lt_R != Lt_S:
        raise ValueError(
            f"R and S mismatch in _band_energy_whitened_batched: "
            f"R {R.shape}, S {S.shape}"
        )
    B, Lt = B_R, Lt_R
    if Ct.numel() == 0:
        # No band constraints: return zeros to avoid NaNs downstream.
        zeros = torch.zeros((B, N, h, w), dtype=torch.float32, device=device)
        return zeros, zeros

    # Numerical hygiene: enforce Hermitian symmetry and robustify the
    # factorization so a single bad tile doesn't abort the whole batch.
    R = 0.5 * (R + R.conj().transpose(-2, -1))
    eye = torch.eye(Lt, dtype=work_dtype, device=device).unsqueeze(0)  # (1,Lt,Lt)
    lam_mat = lam_vec.view(B, 1, 1) * eye  # (B,Lt,Lt)
    R_lam = 0.5 * (R + lam_mat + (R + lam_mat).conj().transpose(-2, -1))

    try:
        with _prof_ctx("stap:band_energy:chol_R"):
            L = torch.linalg.cholesky(R_lam)
    except RuntimeError:
        # Per-tile adaptive jitter (scaled by mu) fallback.
        eye_b = torch.eye(Lt, dtype=work_dtype, device=device)
        diag = torch.real(torch.diagonal(R_lam, dim1=-2, dim2=-1))
        diag_scale = torch.mean(torch.abs(diag), dim=1) + 1e-12
        jitter_mults = (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0)
        L_list = []
        for b in range(B):
            Rb = R_lam[b]
            Lb = None
            for mult in jitter_mults:
                try:
                    with _prof_ctx("stap:band_energy:chol_R"):
                        Lb = torch.linalg.cholesky(Rb + (float(mult) * diag_scale[b]) * eye_b)
                    break
                except RuntimeError:
                    Lb = None
            if Lb is None:
                # Last resort: diagonal approximation.
                d = torch.clamp(torch.real(torch.diagonal(Rb, dim1=-2, dim2=-1)), min=1e-8)
                Lb = torch.diag(torch.sqrt(d.to(dtype=torch.float32))).to(dtype=work_dtype)
            L_list.append(Lb)
        L = torch.stack(L_list, dim=0)

    # Whiten snapshots: reshape to (B, Lt, P) where P = N*h*w
    if S.ndim == 4:
        # (B, Lt, N, h) -> (B, Lt, N, h, 1) so downstream logic can stay uniform.
        S = S.unsqueeze(-1)
    Bsz, Lt_dim, N, h, w = S.shape
    # NOTE: preserve the legacy flattening order (h,w,N) used by the manuscript
    # baselines. This intentionally matches the historical (permute+view) layout
    # so downstream score maps remain bitwise-comparable across latency work.
    S_flat = S.permute(0, 1, 3, 4, 2).contiguous().view(Bsz, Lt_dim, -1)
    with _prof_ctx("stap:band_energy:whiten_snapshots"):
        # Whiten strategy for L^{-1} X (X is very wide when P=N*h*w is large).
        #
        # - direct: batched TRSM on the full RHS (L^{-1} X).
        # - inv_gemm: compute L^{-1} once using TRSM on I (small RHS), then GEMM.
        #
        # For large Lt and wide RHS on CUDA, inv+GEMM is often faster.
        solve_mode_env = os.getenv("STAP_BAND_SOLVE_MODE", "").strip().lower()
        if not solve_mode_env or solve_mode_env == "auto":
            use_inv_gemm = bool(S_flat.is_cuda) and int(Lt_dim) >= 32 and int(S_flat.shape[-1]) >= 2 * int(Lt_dim)
            solve_mode = "inv_gemm" if use_inv_gemm else "direct"
        elif solve_mode_env in {"direct", "trsm", "solve", "triangular"}:
            solve_mode = "direct"
        elif solve_mode_env in {"inv", "invgemm", "inv_gemm", "gemm", "matmul"}:
            solve_mode = "inv_gemm"
        else:
            raise ValueError(
                f"Unknown STAP_BAND_SOLVE_MODE='{solve_mode_env}'. Expected auto|direct|inv_gemm."
            )
        if solve_mode == "inv_gemm":
            eye_lt = torch.eye(Lt_dim, dtype=work_dtype, device=device)
            eye_expand = eye_lt.expand(Bsz, Lt_dim, Lt_dim)
            Linv = torch.linalg.solve_triangular(L, eye_expand, upper=False)
            Sw = torch.bmm(Linv, S_flat)
        else:
            Sw = torch.linalg.solve_triangular(L, S_flat, upper=False)

    # Whiten constraints: expand Ct across batch
    Ct_exp = Ct.unsqueeze(0).expand(Bsz, -1, -1)  # (B,Lt,K)
    with _prof_ctx("stap:band_energy:whiten_constraints"):
        Cw = torch.linalg.solve_triangular(L, Ct_exp, upper=False)

    # Total whitened power (used both for output and for the dual-form projector).
    sw_pow_flat = torch.sum(Sw.conj() * Sw, dim=1).real  # (B, P)

    # Project onto band subspace.
    #
    # IMPORTANT: use the proper projection energy z^H (Cw^H Cw + ridge I)^{-1} z
    # rather than ||z||^2. The latter assumes Cw columns are (approximately)
    # orthonormal; after whitening and conditioning this is often false, and the
    # resulting band-fraction score can become numerically degenerate.
    with _prof_ctx("stap:band_energy:project"):
        ridge_eff = float(ridge)
        try:
            project_mode = os.getenv("STAP_BAND_PROJECT_MODE", "").strip().lower()
            mode_pf = project_mode in {"pf", "proj", "projection"}
            mode_dual = project_mode in {"dual", "lt", "woodbury", "small"}
            mode_gram = project_mode in {"gram", "k", "reference", "chol"}

            Bm, Lt_m, K = Cw.shape
            CwH = Cw.conj().transpose(1, 2)  # (B, K, Lt)
            ridge_vec: torch.Tensor

            if not mode_pf and not mode_dual and not mode_gram:
                # Auto selection: when K is large relative to Lt, avoid the
                # K×K solve by using the mathematically equivalent dual form.
                mode_dual = bool(K > Lt_m)

            if mode_dual:
                # Dual/Woodbury form:
                #   y^H C (C^H C + λI)^{-1} C^H y
                #     = y^H y - λ y^H (C C^H + λI)^{-1} y
                # which replaces a (K×K) solve with an (Lt×Lt) solve when K > Lt.
                #
                # NOTE: the non-zero eigenvalues of (C^H C) and (C C^H) are
                # identical, so we can compute the ridge scaling using the much
                # smaller Lt×Lt matrix without changing the intended behavior.
                M = torch.bmm(Cw, CwH)  # (B, Lt, Lt)
                M = 0.5 * (M + M.conj().transpose(-2, -1))

                if ridge_eff > 0.0:
                    evals = torch.linalg.eigvalsh(M).real
                    evals = torch.clamp(evals, min=1e-12)
                    evals_max = evals.max(dim=1).values  # (B,)
                    ridge_vec = ridge_eff * torch.where(
                        evals_max < 1.0, evals_max, torch.ones_like(evals_max)
                    )
                    eye_lt = torch.eye(Lt_m, dtype=M.dtype, device=M.device).unsqueeze(0)
                    M = M + ridge_vec.view(-1, 1, 1) * eye_lt
                else:
                    ridge_vec = torch.zeros((Bm,), dtype=torch.float32, device=Cw.device)

                Lm = torch.linalg.cholesky(M)
                v = torch.cholesky_solve(Sw, Lm)  # (B, Lt, P) = (C C^H + λI)^{-1} y
                quad = torch.sum(Sw.conj() * v, dim=1).real  # (B, P)
                T_band_flat = sw_pow_flat - ridge_vec.view(-1, 1) * quad
            else:
                Gram = torch.bmm(CwH, Cw)  # (B, K, K)
                if ridge_eff > 0.0:
                    # Scale-aware ridge (mirrors head baseline behavior): when Gram is
                    # tiny in the whitened basis, an absolute ridge can dominate and
                    # collapse projected energy toward ~0.
                    evals = torch.linalg.eigvalsh(Gram).real
                    evals = torch.clamp(evals, min=1e-12)
                    evals_max = evals.max(dim=1).values  # (B,)
                    ridge_vec = ridge_eff * torch.where(
                        evals_max < 1.0, evals_max, torch.ones_like(evals_max)
                    )
                    eye_k = torch.eye(K, dtype=Gram.dtype, device=Gram.device).unsqueeze(0)
                    Gram = Gram + ridge_vec.view(-1, 1, 1) * eye_k
                else:
                    ridge_vec = torch.zeros((Bm,), dtype=torch.float32, device=Cw.device)

                if mode_pf:
                    # Fast exact form: precompute the projection matrix Pf and apply it
                    # directly to whitened snapshots. This avoids solving Gram against a
                    # large RHS (K×P) and replaces it with:
                    #   1) a small solve (K×Lt) to form Pf, then
                    #   2) a batched GEMM (Lt×Lt) on the large RHS (Lt×P).
                    #
                    # This is mathematically identical to z^H Gram^{-1} z because:
                    #   Pf = Cw Gram^{-1} Cw^H,  so  y^H Pf y = z^H Gram^{-1} z.
                    Lg = torch.linalg.cholesky(Gram)
                    X = torch.cholesky_solve(CwH, Lg)  # (B, K, Lt) = Gram^{-1} Cw^H
                    Pf = torch.bmm(Cw, X)  # (B, Lt, Lt)
                    Pf = 0.5 * (Pf + Pf.conj().transpose(-2, -1))
                    u = torch.bmm(Pf, Sw)  # (B, Lt, P)
                    T_band_flat = torch.sum(Sw.conj() * u, dim=1).real  # (B, P)
                else:
                    # Stable reference implementation: solve on the full RHS.
                    #
                    # We intentionally keep this form as the default because some
                    # profiles (e.g., very short N=T-Lt+1) can be numerically sensitive
                    # and historically exhibited heavy-tail artifacts under precomputed
                    # projection-matrix formulations. The env-controlled fast mode above
                    # lets latency experiments opt into Pf explicitly.
                    Lg = torch.linalg.cholesky(Gram)
                    # Projection energy: z^H Gram^{-1} z with Gram=Lg Lg^H.
                    #
                    # Algebra: ||Lg^{-1} z||^2 with z=(Cw^H)Sw.
                    # Prefer forming A = Lg^{-1} Cw^H once (small RHS K×Lt) then
                    # GEMM against the large RHS Sw (Lt×P) to avoid a TRSM on a
                    # very wide (K×P) matrix.
                    A = torch.linalg.solve_triangular(Lg, CwH, upper=False)  # (B, K, Lt)
                    tmp = torch.bmm(A, Sw)  # (B, K, P) = Lg^{-1} z
                    T_band_flat = torch.sum(tmp.conj() * tmp, dim=1).real  # (B, P)
        except RuntimeError:
            # Fallback: use robust Hermitian solve on the full RHS.
            CwH = Cw.conj().transpose(1, 2)  # (B, K, Lt)
            Gram = torch.bmm(CwH, Cw)  # (B, K, K)
            if ridge_eff > 0.0:
                evals = torch.linalg.eigvalsh(Gram).real
                evals = torch.clamp(evals, min=1e-12)
                evals_max = evals.max(dim=1).values  # (B,)
                ridge_vec = ridge_eff * torch.where(
                    evals_max < 1.0, evals_max, torch.ones_like(evals_max)
                )
                eye_k = torch.eye(Gram.shape[-1], dtype=Gram.dtype, device=Gram.device).unsqueeze(0)
                Gram = Gram + ridge_vec.view(-1, 1, 1) * eye_k
            z = torch.bmm(CwH, Sw)  # (B, K, P)
            proj, _ = cholesky_solve_hermitian(Gram, z, jitter_init=1e-8, max_tries=3)
            T_band_flat = torch.sum(z.conj() * proj, dim=1).real  # (B, P)

    # Reshape back to (B, N, h, w)
    T_band = T_band_flat.view(Bsz, N, h, w)
    sw_pow = sw_pow_flat.view(Bsz, N, h, w)
    return T_band, sw_pow


def _band_energy_whitened_covonly_batched(
    R_B_Lt_Lt: torch.Tensor,
    R_scm_B_Lt_Lt: torch.Tensor,
    C_t: torch.Tensor,
    lam_B: torch.Tensor,
    *,
    ridge: float = 0.10,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.complex64,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Covariance-only band-energy computation (tile statistic).

    This computes mean projected band energy and mean total whitened power per
    tile using only covariances, avoiding the (B, Lt, P) snapshot whitening path.

    Parameters
    ----------
    R_B_Lt_Lt : (B, Lt, Lt) covariance used for whitening (after shrinkage).
    R_scm_B_Lt_Lt : (B, Lt, Lt) pooled SCM covariance of the centered Hankel snapshots.
    C_t : (Lt, K) constraint matrix.
    lam_B : (B,) per-tile absolute loading used in whitening.

    Returns
    -------
    T_band_mean : (B,) float32
        Mean projected band energy.
    sw_pow_mean : (B,) float32
        Mean total whitened power (trace of whitened SCM covariance).
    """
    if device is None:
        device = R_B_Lt_Lt.device

    # Optional precision override for numerical experiments.
    prec = os.getenv("STAP_BAND_PRECISION", "").strip().lower()
    work_dtype = torch.complex128 if prec == "fp64" else dtype

    R = R_B_Lt_Lt.to(device=device, dtype=work_dtype)
    R_scm = R_scm_B_Lt_Lt.to(device=device, dtype=work_dtype)
    Ct = C_t.to(device=device, dtype=work_dtype)
    lam_vec = lam_B.to(device=device).flatten()
    if lam_vec.is_complex():
        lam_vec = lam_vec.real

    B, Lt, _ = R.shape
    if Ct.numel() == 0:
        zeros = torch.zeros((B,), dtype=torch.float32, device=device)
        return zeros, zeros

    # Numerical hygiene: enforce Hermitian symmetry and robustify the factorization.
    R = 0.5 * (R + R.conj().transpose(-2, -1))
    R_scm = 0.5 * (R_scm + R_scm.conj().transpose(-2, -1))
    eye = torch.eye(Lt, dtype=work_dtype, device=device).unsqueeze(0)  # (1,Lt,Lt)
    R_lam = R + lam_vec.view(B, 1, 1) * eye
    R_lam = 0.5 * (R_lam + R_lam.conj().transpose(-2, -1))

    try:
        with _prof_ctx("stap:band_energy_covonly:chol_R"):
            L = torch.linalg.cholesky(R_lam)
    except RuntimeError:
        # Per-tile adaptive jitter (scaled by mu) fallback.
        eye_b = torch.eye(Lt, dtype=work_dtype, device=device)
        diag = torch.real(torch.diagonal(R_lam, dim1=-2, dim2=-1))
        diag_scale = torch.mean(torch.abs(diag), dim=1) + 1e-12
        jitter_mults = (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0)
        L_list = []
        for b in range(B):
            Rb = R_lam[b]
            Lb = None
            for mult in jitter_mults:
                try:
                    with _prof_ctx("stap:band_energy_covonly:chol_R"):
                        Lb = torch.linalg.cholesky(Rb + (float(mult) * diag_scale[b]) * eye_b)
                    break
                except RuntimeError:
                    Lb = None
            if Lb is None:
                # Last resort: diagonal approximation.
                d = torch.clamp(torch.real(torch.diagonal(Rb, dim1=-2, dim2=-1)), min=1e-8)
                Lb = torch.diag(torch.sqrt(d.to(dtype=torch.float32))).to(dtype=work_dtype)
            L_list.append(Lb)
        L = torch.stack(L_list, dim=0)

    # Whiten constraints (batched).
    K = int(Ct.shape[-1])
    Ct_exp = Ct.unsqueeze(0).expand(B, Lt, K)
    with _prof_ctx("stap:band_energy_covonly:whiten_constraints"):
        Cw = torch.linalg.solve_triangular(L, Ct_exp, upper=False)  # (B, Lt, K)

    Gram = torch.bmm(Cw.conj().transpose(1, 2), Cw)  # (B, K, K)
    ridge_eff = float(ridge)
    if ridge_eff > 0.0:
        # Scale-aware ridge (mirrors `_band_energy_whitened_batched`).
        evals = torch.linalg.eigvalsh(Gram).real
        evals = torch.clamp(evals, min=1e-12)
        evals_max = evals.max(dim=1).values  # (B,)
        ridge_vec = ridge_eff * torch.where(evals_max < 1.0, evals_max, torch.ones_like(evals_max))
        eye_k = torch.eye(K, dtype=Gram.dtype, device=Gram.device).unsqueeze(0)
        Gram = Gram + ridge_vec.view(-1, 1, 1) * eye_k

    # Whiten SCM covariance: Rw = L^{-1} R_scm L^{-H}.
    with _prof_ctx("stap:band_energy_covonly:whiten_cov"):
        Y = torch.linalg.solve_triangular(L, R_scm, upper=False)
        Rw_h = torch.linalg.solve_triangular(L, Y.conj().transpose(-2, -1), upper=False)
    Rw = Rw_h.conj().transpose(-2, -1)
    Rw = 0.5 * (Rw + Rw.conj().transpose(-2, -1))

    sw_pow_mean = torch.real(torch.diagonal(Rw, dim1=-2, dim2=-1)).sum(dim=1).to(torch.float32)
    sw_pow_mean = torch.clamp(sw_pow_mean, min=float(eps))

    # Mean band energy: tr(G^{-1} Cw^H Rw Cw).
    with _prof_ctx("stap:band_energy_covonly:project"):
        Q = torch.bmm(Rw, Cw)  # (B, Lt, K)
        A = torch.bmm(Cw.conj().transpose(1, 2), Q)  # (B, K, K)
        A = 0.5 * (A + A.conj().transpose(-2, -1))
        try:
            Lg = torch.linalg.cholesky(Gram)
            X = torch.cholesky_solve(A, Lg)
        except RuntimeError:
            X, _ = cholesky_solve_hermitian(Gram, A, jitter_init=1e-8, max_tries=3)
        T_band_mean = torch.real(torch.diagonal(X, dim1=-2, dim2=-1)).sum(dim=1)
    T_band_mean = torch.clamp(T_band_mean, min=0.0).to(torch.float32)
    return T_band_mean, sw_pow_mean


def _band_energy_whitened_covonly_perpixel_batched(
    R_B_Lt_Lt: torch.Tensor,
    S_B_Lt_N_hw: torch.Tensor,
    C_t: torch.Tensor,
    lam_B: torch.Tensor,
    *,
    ridge: float = 0.10,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.complex64,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Covariance-only band-energy computation (per-pixel mean across snapshots).

    This produces spatially varying maps (B,h,w) using only covariances, avoiding
    the large (B, K, P) projection RHS in the snapshot path. It still requires
    the Hankel snapshot tensor `S_B_Lt_N_hw` (already constructed in the fast core)
    to form per-pixel SCM covariances.

    Returns
    -------
    T_band_mean_hw : (B,h,w) float32
        Mean projected band energy per pixel (averaged across the N snapshots).
    sw_pow_mean_hw : (B,h,w) float32
        Mean total whitened power per pixel (trace of whitened SCM covariance).
    """
    if device is None:
        device = R_B_Lt_Lt.device

    prec = os.getenv("STAP_BAND_PRECISION", "").strip().lower()
    work_dtype = torch.complex128 if prec == "fp64" else dtype

    R = R_B_Lt_Lt.to(device=device, dtype=work_dtype)
    S = S_B_Lt_N_hw.to(device=device, dtype=work_dtype)
    Ct = C_t.to(device=device, dtype=work_dtype)
    lam_vec = lam_B.to(device=device).flatten()
    if lam_vec.is_complex():
        lam_vec = lam_vec.real

    B, Lt, N, h, w = S.shape
    if Ct.numel() == 0:
        zeros = torch.zeros((B, h, w), dtype=torch.float32, device=device)
        return zeros, zeros

    # Hermitize and load.
    R = 0.5 * (R + R.conj().transpose(-2, -1))
    eye = torch.eye(Lt, dtype=work_dtype, device=device).unsqueeze(0)  # (1,Lt,Lt)
    R_lam = R + lam_vec.view(B, 1, 1) * eye
    R_lam = 0.5 * (R_lam + R_lam.conj().transpose(-2, -1))

    try:
        with _prof_ctx("stap:band_energy_covonly_pixel:chol_R"):
            L = torch.linalg.cholesky(R_lam)
    except RuntimeError:
        # Per-tile adaptive jitter (scaled by mu) fallback.
        eye_b = torch.eye(Lt, dtype=work_dtype, device=device)
        diag = torch.real(torch.diagonal(R_lam, dim1=-2, dim2=-1))
        diag_scale = torch.mean(torch.abs(diag), dim=1) + 1e-12
        jitter_mults = (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0)
        L_list = []
        for b in range(B):
            Rb = R_lam[b]
            Lb = None
            for mult in jitter_mults:
                try:
                    with _prof_ctx("stap:band_energy_covonly_pixel:chol_R"):
                        Lb = torch.linalg.cholesky(Rb + (float(mult) * diag_scale[b]) * eye_b)
                    break
                except RuntimeError:
                    Lb = None
            if Lb is None:
                d = torch.clamp(torch.real(torch.diagonal(Rb, dim1=-2, dim2=-1)), min=1e-8)
                Lb = torch.diag(torch.sqrt(d.to(dtype=torch.float32))).to(dtype=work_dtype)
            L_list.append(Lb)
        L = torch.stack(L_list, dim=0)

    # Whiten constraints (batched).
    K = int(Ct.shape[-1])
    Ct_exp = Ct.unsqueeze(0).expand(B, Lt, K)
    with _prof_ctx("stap:band_energy_covonly_pixel:whiten_constraints"):
        Cw = torch.linalg.solve_triangular(L, Ct_exp, upper=False)  # (B,Lt,K)
    CwH = Cw.conj().transpose(1, 2)  # (B,K,Lt)
    Gram = torch.bmm(CwH, Cw)  # (B,K,K)
    ridge_eff = float(ridge)
    if ridge_eff > 0.0:
        evals = torch.linalg.eigvalsh(Gram).real
        evals = torch.clamp(evals, min=1e-12)
        evals_max = evals.max(dim=1).values  # (B,)
        ridge_vec = ridge_eff * torch.where(evals_max < 1.0, evals_max, torch.ones_like(evals_max))
        eye_k = torch.eye(K, dtype=Gram.dtype, device=Gram.device).unsqueeze(0)
        Gram = Gram + ridge_vec.view(-1, 1, 1) * eye_k

    # Precompute Pf = Cw Gram^{-1} Cw^H (batched).
    with _prof_ctx("stap:band_energy_covonly_pixel:Pf"):
        try:
            Lg = torch.linalg.cholesky(Gram)
            X = torch.cholesky_solve(CwH, Lg)  # (B,K,Lt)
        except RuntimeError:
            X, _ = cholesky_solve_hermitian(Gram, CwH, jitter_init=1e-8, max_tries=3)
        Pf = torch.bmm(Cw, X)  # (B,Lt,Lt)
        Pf = 0.5 * (Pf + Pf.conj().transpose(-2, -1))

    # Per-pixel SCM covariances: (B,hw,Lt,Lt) with hw=h*w.
    hw = int(h * w)
    S_pix = S.reshape(B, Lt, N, hw).permute(0, 3, 1, 2).contiguous()  # (B,hw,Lt,N)
    with _prof_ctx("stap:band_energy_covonly_pixel:scm"):
        R_pix = torch.matmul(S_pix, S_pix.conj().transpose(-2, -1)) / float(max(int(N), 1))
        R_pix = 0.5 * (R_pix + R_pix.conj().transpose(-2, -1))

    # Whiten per-pixel covariances: Rw = L^{-1} R_pix L^{-H}.
    with _prof_ctx("stap:band_energy_covonly_pixel:whiten_cov"):
        L_exp = L.unsqueeze(1)  # (B,1,Lt,Lt) -> broadcast over hw
        Y = torch.linalg.solve_triangular(L_exp, R_pix, upper=False)
        Rw_h = torch.linalg.solve_triangular(L_exp, Y.conj().transpose(-2, -1), upper=False)
        Rw = Rw_h.conj().transpose(-2, -1)
    Rw = 0.5 * (Rw + Rw.conj().transpose(-2, -1))

    sw_pow = torch.real(torch.diagonal(Rw, dim1=-2, dim2=-1)).sum(dim=-1).to(torch.float32)
    sw_pow = torch.clamp(sw_pow, min=float(eps))  # (B,hw)

    with _prof_ctx("stap:band_energy_covonly_pixel:project"):
        T_band = torch.real(torch.sum(Pf.unsqueeze(1).conj() * Rw, dim=(-2, -1))).to(torch.float32)
    T_band = torch.clamp(T_band, min=0.0)  # (B,hw)

    return T_band.view(B, h, w), sw_pow.view(B, h, w)


def aggregate_over_snapshots(
    x_N_hw: torch.Tensor,
    *,
    mode: Literal["mean", "median", "trim10"] = "mean",
) -> torch.Tensor:
    """
    Aggregate per-snapshot statistic across the slow-time window.
    """
    if mode == "mean":
        return x_N_hw.mean(dim=0)
    if mode == "median":
        return x_N_hw.median(dim=0).values
    if mode == "trim10":
        N = x_N_hw.shape[0]
        k = max(1, int(0.1 * N))
        if 2 * k >= N:
            return x_N_hw.mean(dim=0)
        x_sorted, _ = torch.sort(x_N_hw, dim=0)
        x_trim = x_sorted[k:-k]
        return x_trim.mean(dim=0)
    raise ValueError(f"Unsupported aggregation mode: {mode}")


def aggregate_over_snapshots_batched(
    x_B_N_hw: torch.Tensor,
    *,
    mode: Literal["mean", "median", "trim10"] = "mean",
) -> torch.Tensor:
    """
    Batched variant of `aggregate_over_snapshots`.

    x_B_N_hw: (B, N, h, w) -> returns (B, h, w)
    """
    if x_B_N_hw.ndim != 4:
        raise ValueError(f"Expected (B,N,h,w) input, got shape {tuple(x_B_N_hw.shape)}")
    if mode == "mean":
        return x_B_N_hw.mean(dim=1)
    if mode == "median":
        return x_B_N_hw.median(dim=1).values
    if mode == "trim10":
        N = int(x_B_N_hw.shape[1])
        k = max(1, int(0.1 * N))
        if 2 * k >= N:
            return x_B_N_hw.mean(dim=1)
        x_sorted, _ = torch.sort(x_B_N_hw, dim=1)
        x_trim = x_sorted[:, k:-k]
        return x_trim.mean(dim=1)
    raise ValueError(f"Unsupported aggregation mode: {mode}")


def lcmv_temporal_apply_batched(
    R_t: torch.Tensor | np.ndarray,
    S_Lt_N_hw: torch.Tensor | np.ndarray,
    C_t: torch.Tensor | np.ndarray,
    *,
    load_mode: str = "auto",
    diag_load: float = 1e-2,
    auto_kappa_target: float = 50.0,
    auto_lambda_bounds: tuple[float, float] = (5e-4, 2e-1),
    constraint_ridge: float = 0.10,
    enforce_exact_if_possible: bool = True,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.complex64,
) -> Tuple[torch.Tensor, LCMVResult]:
    """
    Solve temporal LCMV once per tile and apply to every pixel.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    Rt = to_tensor(R_t, device=device, dtype=dtype)
    Ct = to_tensor(C_t, device=device, dtype=dtype)
    S = to_tensor(S_Lt_N_hw, device=device, dtype=dtype)

    Kc = Ct.shape[-1]
    target = (
        torch.ones((Kc,), dtype=dtype, device=device)
        if Kc > 0
        else torch.empty((0,), dtype=dtype, device=device)
    )

    res = lcmv_weights(
        Rt,
        Ct,
        target,
        load_mode=load_mode,
        diag_load=diag_load,
        auto_kappa_target=auto_kappa_target,
        auto_lambda_bounds=auto_lambda_bounds,
        constraint_ridge=constraint_ridge,
        enforce_exact_if_possible=enforce_exact_if_possible,
        device=device,
        dtype=dtype,
    )
    w = res.w.reshape(-1)
    y = torch.einsum("l,lnhw->nhw", torch.conj(w), S)
    return y, res


def capon_band_temporal_pd(
    R_t: torch.Tensor,
    S_Lt_N_hw: torch.Tensor,
    C_t: torch.Tensor,
    lam_abs: float,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Capon band integration in temporal subspace as fallback.

    Returns per-pixel PD map with shape (h, w) and spectrum values.
    """
    Lt = R_t.shape[0]
    eye = torch.eye(Lt, dtype=R_t.dtype, device=R_t.device)
    R_loaded = R_t + lam_abs * eye
    U, _ = cholesky_solve_hermitian(R_loaded, C_t, jitter_init=1e-8, max_tries=6)
    gram = torch.matmul(C_t.conj().transpose(-2, -1), U)
    denom = torch.real(torch.diagonal(gram))
    cap_spec = torch.reciprocal(torch.clamp(denom, min=1e-10))
    # Average temporal spectrum to get scalar PD for each pixel:
    # Equivalent to averaging outputs of matched filters.
    weights = torch.matmul(U, torch.ones((C_t.shape[1], 1), dtype=C_t.dtype, device=C_t.device))
    weights = weights.reshape(Lt)
    y = torch.einsum("l,lnhw->nhw", torch.conj(weights), S_Lt_N_hw)
    pd_hw = torch.mean(torch.real(y.conj() * y), dim=0)
    return pd_hw, cap_spec.detach().cpu().numpy()


@torch.no_grad()
def msd_band_energy_batched(
    R_t: torch.Tensor | np.ndarray,
    S_Lt_N_hw: torch.Tensor | np.ndarray,
    C_t: torch.Tensor | np.ndarray,
    *,
    lam_abs: float = 2e-2,
    ridge: float = 0.10,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.complex64,
    return_ratio: bool = True,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Matched-subspace (ACE/MSD) statistic in the temporal subspace.

    Returns a (h,w) map with the average statistic over slow-time snapshots.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    Rt = to_tensor(R_t, device=device, dtype=dtype)
    S = to_tensor(S_Lt_N_hw, device=device, dtype=dtype)
    Ct = to_tensor(C_t, device=device, dtype=dtype)

    Lt_dim = Rt.shape[0]
    Kc = Ct.shape[-1]
    if Kc == 0:
        h, w = S.shape[2], S.shape[3]
        return torch.zeros((h, w), dtype=torch.float32, device=device)

    N = S.shape[1]
    h, w = S.shape[2], S.shape[3]

    eye = torch.eye(Lt_dim, dtype=dtype, device=device)
    R_loaded = Rt + float(lam_abs) * eye
    L = torch.linalg.cholesky(R_loaded)

    S_flat = S.reshape(Lt_dim, N * h * w)
    Sw = _solve_lower_triangular(L, S_flat)
    Cw = _solve_lower_triangular(L, Ct)

    G = Cw.conj().transpose(-2, -1) @ Cw
    if ridge > 0.0:
        G = G + float(ridge) * torch.eye(Kc, dtype=dtype, device=device)

    z = Cw.conj().transpose(-2, -1) @ Sw  # (Kc, N*h*w)
    tmp, _ = cholesky_solve_hermitian(G, z, jitter_init=1e-10, max_tries=3)
    T_band = torch.sum(z.conj() * tmp, dim=0).real  # (N*h*w)

    if return_ratio:
        sw_pow = torch.sum(Sw.conj() * Sw, dim=0).real
        denom = torch.clamp(sw_pow - T_band, min=eps)
        stat = T_band / denom
    else:
        stat = T_band

    stat = stat.reshape(N, h, w).mean(dim=0)
    return stat


@torch.no_grad()
def capon_band_ratio_batched(
    R_t: torch.Tensor | np.ndarray,
    S_Lt_N_hw: torch.Tensor | np.ndarray,
    C_t: torch.Tensor | np.ndarray,
    *,
    lam_abs: float = 2e-2,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.complex64,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Capon-based band ratio (fallback when MSD fails)."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    Rt = to_tensor(R_t, device=device, dtype=dtype)
    S = to_tensor(S_Lt_N_hw, device=device, dtype=dtype)
    Ct = to_tensor(C_t, device=device, dtype=dtype)

    Lt_dim = Rt.shape[0]
    Kc = Ct.shape[-1]
    h, w = S.shape[2], S.shape[3]
    if Kc == 0:
        return torch.zeros((h, w), dtype=torch.float32, device=device)

    N = S.shape[1]
    eye = torch.eye(Lt_dim, dtype=dtype, device=device)
    R_loaded = Rt + float(lam_abs) * eye
    L = torch.linalg.cholesky(R_loaded)

    S_flat = S.reshape(Lt_dim, N * h * w)
    Sw = _solve_lower_triangular(L, S_flat)
    Cw = _solve_lower_triangular(L, Ct)

    denom = torch.real(torch.diagonal(Cw.conj().transpose(-2, -1) @ Cw))
    cap_spec = torch.reciprocal(torch.clamp(denom, min=1e-10))  # (Kc,)
    cap_mean = torch.mean(cap_spec)

    sw_pow = torch.sum(Sw.conj() * Sw, dim=0).real  # (N*h*w)
    ratio = cap_mean / torch.clamp(sw_pow.mean(dim=0), min=eps)
    return ratio.reshape(h, w)


# ---- Phase-3 fast-path helpers -------------------------------------------------


def _build_hankel_batch(
    cube_BT_hw: torch.Tensor,
    Lt: int,
    *,
    center: bool = True,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.complex64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build Hankel stacks and pooled SCM covariance for a batch of tiles.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    x = cube_BT_hw.to(device=device, dtype=dtype)
    B, T, h, w = x.shape
    if Lt < 2 or Lt >= T:
        raise ValueError(f"Need 2 <= Lt < T (got Lt={Lt}, T={T})")
    N = T - Lt + 1
    rows = [x[:, k : k + N] for k in range(Lt)]
    S_B_Lt_N_hw = torch.stack(rows, dim=1)  # (B, Lt, N, h, w)
    if center:
        S_B_Lt_N_hw = S_B_Lt_N_hw - S_B_Lt_N_hw.mean(dim=2, keepdim=True)
    # Preserve the manuscript baseline flattening order (h,w,N) before
    # collapsing to snapshots. This is mathematically irrelevant for covariance
    # estimation but keeps downstream fast-path mappings consistent when this
    # helper is used for profiling/experiments.
    S_flat = S_B_Lt_N_hw.permute(0, 1, 3, 4, 2).contiguous().view(B, Lt, -1)  # (B, Lt, P)
    P = S_flat.shape[-1]
    R = torch.matmul(S_flat, S_flat.conj().transpose(-2, -1)) / float(P)
    return S_B_Lt_N_hw, R


def _tyler_covariance_batched(
    S_flat: torch.Tensor,
    *,
    R_init: torch.Tensor | None = None,
    max_iter: int = 25,
    tol: float = 1e-4,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Lightweight Tyler iteration, batched over tiles.
    """
    # Optional runtime knobs for latency experiments (default preserves behavior).
    #
    # NOTE: Hard-capping Tyler iterations (e.g., STAP_TYLER_MAX_ITER=10) is
    # known to regress strict-tail performance on real-data audits (e.g., Shin
    # RatBrain / Twinkling-Gammex). Treat it as a latency-only knob and avoid
    # recording it in any "configs of record" tied to manuscript results.
    max_iter_env = os.getenv("STAP_TYLER_MAX_ITER", "").strip()
    if max_iter_env:
        try:
            max_iter = int(max_iter_env)
        except Exception:
            pass
    max_iter = max(1, int(max_iter))
    tol_env = os.getenv("STAP_TYLER_TOL", "").strip()
    if tol_env:
        try:
            tol = float(tol_env)
        except Exception:
            pass
    early_stop_env = os.getenv("STAP_TYLER_EARLY_STOP", "").strip().lower()
    early_stop = early_stop_env not in {"0", "false", "no", "off"}
    check_every_env = os.getenv("STAP_TYLER_CONV_CHECK_EVERY", "").strip()
    if check_every_env:
        try:
            check_every = int(check_every_env)
        except Exception:
            check_every = 1
    else:
        # Default: reduce per-iteration host sync + Frobenius-norm work on CUDA.
        check_every = 2 if S_flat.is_cuda else 1
    check_every = max(1, int(check_every))
    B, Lt, P = S_flat.shape
    eye = torch.eye(Lt, dtype=S_flat.dtype, device=S_flat.device)
    eye_B = eye.unsqueeze(0)
    # Solve strategy for the Tyler whitening step (L^{-1} X):
    #
    # - direct: use batched TRSM (`solve_triangular`) on the full RHS.
    # - inv_gemm: compute L^{-1} via TRSM on I (small RHS) then use GEMM (bmm).
    #
    # On CUDA, TRSM can be significantly slower than GEMM when the RHS has many
    # columns (P). The inverse+GEMM strategy is usually faster for large P, at
    # the cost of slightly different floating-point rounding (should not change
    # the fixed point in exact arithmetic).
    solve_mode_env = os.getenv("STAP_TYLER_SOLVE_MODE", "").strip().lower()
    if not solve_mode_env or solve_mode_env == "auto":
        # Heuristic:
        # - For small Lt (e.g., Lt=8 in Brain-*), direct TRSM is typically faster.
        # - For large Lt + wide RHS on CUDA, inverse+GEMM can be significantly faster.
        use_inv_gemm = bool(S_flat.is_cuda) and int(Lt) >= 32 and int(P) >= 2 * int(Lt)
        solve_mode = "inv_gemm" if use_inv_gemm else "direct"
    elif solve_mode_env in {"direct", "trsm", "solve", "triangular"}:
        solve_mode = "direct"
    elif solve_mode_env in {"inv", "invgemm", "inv_gemm", "gemm", "matmul"}:
        solve_mode = "inv_gemm"
    else:
        raise ValueError(
            f"Unknown STAP_TYLER_SOLVE_MODE='{solve_mode_env}'. Expected auto|direct|inv_gemm."
        )
    # Note: expanded identity is used as the RHS for the triangular solve; this
    # avoids allocating a (B,Lt,Lt) tensor in the common case.
    eye_expand = eye.expand(B, Lt, Lt)
    Xw = torch.empty_like(S_flat)
    Linv = torch.empty((B, Lt, Lt), device=S_flat.device, dtype=S_flat.dtype) if solve_mode == "inv_gemm" else None
    if R_init is None:
        R = eye_B.expand(B, Lt, Lt).clone()
    else:
        if R_init.shape != (B, Lt, Lt):
            raise ValueError(
                f"R_init must have shape (B,Lt,Lt)={B,Lt,Lt}, got {tuple(R_init.shape)}"
            )
        R = R_init.to(device=S_flat.device, dtype=S_flat.dtype)
        R = 0.5 * (R + R.conj().transpose(-2, -1))
        tr = torch.real(torch.diagonal(R, dim1=1, dim2=2).sum(dim=1)).clamp_min(eps)
        R = R * (float(Lt) / tr).view(B, 1, 1)
        # Keep the init strictly PD to avoid hard failures on the first Cholesky.
        mu = torch.real(torch.diagonal(R, dim1=1, dim2=2)).mean(dim=1).clamp_min(eps)
        R = R + (eps * mu).view(B, 1, 1) * eye_B
    # Reusable buffers to reduce per-iteration allocations.
    R_new = torch.empty_like(R)
    diff = torch.empty_like(R)
    # Optional Triton fusion for the Tyler weights + snapshot scaling stage.
    # Enabled by default on CUDA when Triton is available; can be disabled via:
    #   STAP_TYLER_TRITON_WEIGHTS=0
    triton_weights_env = os.getenv("STAP_TYLER_TRITON_WEIGHTS", "").strip().lower()
    if triton_weights_env in {"0", "false", "no", "off"}:
        use_triton_weights = False
    elif triton_weights_env in {"1", "true", "yes", "on"}:
        # Force-enable (for experiments).
        use_triton_weights = True
    else:
        # Auto: only enable for larger Lt regimes where weights is a clear hotspot.
        # For small Lt (e.g., Brain-* Lt=8), the Triton path can regress end-to-end
        # wall time due to launch/dispatch overhead dominating the relatively small
        # weights workload.
        use_triton_weights = int(Lt) >= 32
    use_triton_weights = bool(S_flat.is_cuda) and S_flat.dtype == torch.complex64 and use_triton_weights
    tyler_weights_triton = None
    tyler_weights_triton_cfg = None
    if use_triton_weights:
        try:
            from pipeline.stap.triton_ops import (
                TylerWeightsConfig,
                triton_available as _triton_available,
                tyler_weights_scale_triton,
            )

            if _triton_available():
                tyler_weights_triton = tyler_weights_scale_triton
                tyler_weights_triton_cfg = TylerWeightsConfig()
            else:
                use_triton_weights = False
        except Exception:
            use_triton_weights = False
    with _prof_ctx("stap:covariance:tyler"):
        tol2 = float(tol) * float(tol)
        for it in range(max_iter):
            # Numerical hygiene: enforce Hermitian symmetry before factorization.
            # On CUDA, small non-Hermitian drift can lead to `cholesky_ex` failures
            # and NaNs downstream (especially for short ensembles like T=17, Lt=16).
            with _prof_ctx("stap:covariance:tyler:herm"):
                torch.add(R, R.conj().transpose(-2, -1), out=R_new)
                R_new.mul_(0.5)
                R, R_new = R_new, R
            with _prof_ctx("stap:covariance:tyler:chol"):
                try:
                    # Fast path: avoid per-iteration host synchronization on the
                    # `info` tensor when everything is PD (the common case).
                    L = torch.linalg.cholesky(R)
                    info = None
                except RuntimeError:
                    # Rare path: fall back to `cholesky_ex` + adaptive jitter.
                    L, info = torch.linalg.cholesky_ex(R)
            if info is not None and torch.any(info != 0):
                # Adaptive jitter ladder (scaled by mu) so a few bad tiles do not
                # poison the whole batch with NaNs.
                mu = torch.real(torch.diagonal(R, dim1=1, dim2=2)).mean(dim=1).clamp_min(eps)
                # Start near the legacy epsilon and increase aggressively only if needed.
                for mult in (1.0, 1e2, 1e4, 1e6):
                    R = R + (float(mult) * eps * mu).view(B, 1, 1) * eye_B
                    with _prof_ctx("stap:covariance:tyler:chol"):
                        L, info = torch.linalg.cholesky_ex(R)
                    if not torch.any(info != 0):
                        break
            with _prof_ctx("stap:covariance:tyler:solve"):
                if solve_mode == "inv_gemm":
                    # L^{-1} X via explicit triangular inverse + GEMM.
                    torch.linalg.solve_triangular(L, eye_expand, upper=False, out=Linv)
                    torch.bmm(Linv, S_flat, out=Xw)
                else:
                    # L^{-1} X via batched TRSM.
                    torch.linalg.solve_triangular(L, S_flat, upper=False, out=Xw)
                Y = Xw
            with _prof_ctx("stap:covariance:tyler:weights"):
                # q_p = ||L^{-1} x_p||^2 (per snapshot), but compute it without
                # complex multiply to reduce kernel time.
                if use_triton_weights and tyler_weights_triton is not None and tyler_weights_triton_cfg is not None:
                    try:
                        tyler_weights_triton(
                            Y,
                            S_flat,
                            Xw,
                            Lt=int(Lt),
                            eps=float(eps),
                            cfg=tyler_weights_triton_cfg,
                        )
                    except Exception:
                        # Safety: fall back to the torch implementation if the
                        # Triton kernel fails for any reason.
                        use_triton_weights = False
                if not use_triton_weights:
                    # Torch fallback: compute q = ||Y||^2 without complex multiplies.
                    #
                    # NOTE: `Y` is not used after this stage, so it's safe (and faster)
                    # to square the float32 view in-place to avoid allocating the
                    # intermediate `Y_ri * Y_ri` tensor.
                    Y_ri = torch.view_as_real(Y)  # (B,Lt,P,2) float32 view
                    Y_ri.mul_(Y_ri)
                    q = torch.sum(Y_ri, dim=(1, 3))  # (B,P)
                    q.clamp_min_(eps)
                    # Reuse `q` storage for the weights to save an allocation.
                    q.reciprocal_().mul_(float(Lt)).clamp_max_(1e6)
                    # Single-pass weighting into the preallocated buffer (avoids an
                    # extra full-tensor copy each iteration).
                    torch.mul(S_flat, q.unsqueeze(1), out=Xw)
            with _prof_ctx("stap:covariance:tyler:update"):
                torch.bmm(Xw, Xw.conj().transpose(-2, -1), out=R_new)
                R_new.mul_(1.0 / float(P))
            # Trace normalization (Tyler is scale-invariant): enforce Tr(R)=Lt.
            #
            # IMPORTANT: compute the trace in float64 and avoid an overly-large
            # absolute clamp. Some regimes (notably Twinkling/Gammex after strong
            # baseline filtering) can produce extremely small intermediate traces
            # in float32, and clamping to `eps=1e-8` prevents proper normalization
            # (leading to near-zero covariances and downstream score regressions).
            with _prof_ctx("stap:covariance:tyler:trace"):
                tr64 = torch.real(torch.diagonal(R_new, dim1=1, dim2=2).sum(dim=1)).to(torch.float64)
                tr64 = torch.where(torch.isfinite(tr64), tr64, torch.zeros_like(tr64))
                tr64 = torch.clamp(tr64, min=1e-30)
                scale = (float(Lt) / tr64).to(dtype=torch.float32).view(B, 1, 1)
                R_new.mul_(scale)
            if early_stop:
                # Convergence check (relative Frobenius change), matching the
                # scalar criterion used by the reference per-tile Tyler solver.
                # NOTE: `float(rel.max())` forces a CUDA sync; avoid doing that
                # every iteration for latency. Checking every N iterations keeps
                # outputs essentially unchanged while reducing control-plane sync.
                do_check = ((it + 1) % check_every == 0) or (it == max_iter - 1)
                if do_check:
                    with _prof_ctx("stap:covariance:tyler:conv"):
                        torch.sub(R_new, R, out=diff)
                        diff_ri = torch.view_as_real(diff)  # (B,Lt,Lt,2)
                        num2 = torch.sum(diff_ri * diff_ri, dim=(1, 2, 3))
                        R_ri = torch.view_as_real(R)
                        den2 = torch.sum(R_ri * R_ri, dim=(1, 2, 3)).clamp_min(1e-24)
                        rel2 = num2 / den2
                        if float(rel2.max()) < tol2:
                            R = R_new
                            break
            R, R_new = R_new, R
    return R


def _huber_covariance_batched(
    S_flat: torch.Tensor,
    *,
    c: float = 5.0,
    max_iter: int = 50,
    tol: float = 1e-4,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Batched complex Huber covariance estimator via IRLS on scatter.

    This mirrors `pipeline.stap.robust_cov.huber_covariance`, but operates on
    an entire tile batch at once. The estimator preserves the SCM trace by
    re-scaling at each IRLS iteration.
    """
    B, Lt, P = S_flat.shape
    eye = torch.eye(Lt, dtype=S_flat.dtype, device=S_flat.device).unsqueeze(0)  # (1,Lt,Lt)

    # SCM initialization
    R = torch.matmul(S_flat, S_flat.conj().transpose(-2, -1)) / float(max(int(P), 1))
    R = 0.5 * (R + R.conj().transpose(-2, -1))

    tr0 = torch.real(torch.diagonal(R, dim1=1, dim2=2).sum(dim=1)).to(torch.float64)
    tr0 = torch.where(torch.isfinite(tr0), tr0, torch.zeros_like(tr0))
    mu0 = tr0 / float(max(int(Lt), 1))
    # Same tiny diagonal load as the reference implementation (scaled by mu).
    R = R + (1e-8 * mu0).to(dtype=torch.float32).view(B, 1, 1) * eye

    rel_change = torch.full((B,), float("inf"), device=S_flat.device, dtype=torch.float32)
    c_val = torch.as_tensor(float(c), device=S_flat.device, dtype=torch.float32)

    with _prof_ctx("stap:covariance:huber"):
        for _ in range(max(1, int(max_iter))):
            R = 0.5 * (R + R.conj().transpose(-2, -1))

            # Robust Cholesky per tile (jitter ladder scaled by mu).
            L, info = torch.linalg.cholesky_ex(R)
            if torch.any(info != 0):
                diag = torch.real(torch.diagonal(R, dim1=1, dim2=2)).to(torch.float32)
                mu = diag.mean(dim=1).clamp_min(1e-12)
                for mult in (1.0, 1e2, 1e4, 1e6):
                    R = R + (float(mult) * 1e-8 * mu).view(B, 1, 1) * eye
                    L, info = torch.linalg.cholesky_ex(R)
                    if not torch.any(info != 0):
                        break

            # Solve R^{-1} X via Cholesky solve.
            Y = torch.cholesky_solve(S_flat, L)  # (B,Lt,P)

            # d_i = Re{x_i^H R^{-1} x_i} / Lt
            d = torch.real(torch.sum(S_flat.conj() * Y, dim=1)) / float(Lt)  # (B,P)
            w = torch.minimum(torch.ones_like(d, dtype=torch.float32), c_val / torch.clamp(d, min=eps))

            R_new = torch.matmul(S_flat * w.unsqueeze(1), S_flat.conj().transpose(-2, -1)) / float(
                max(int(P), 1)
            )
            R_new = 0.5 * (R_new + R_new.conj().transpose(-2, -1))

            tr = torch.real(torch.diagonal(R_new, dim1=1, dim2=2).sum(dim=1)).to(torch.float64)
            tr = torch.where(torch.isfinite(tr), tr, torch.zeros_like(tr))
            scale = tr0 / torch.clamp(tr, min=1e-30)
            R_new = R_new * scale.to(dtype=torch.float32).view(B, 1, 1)

            num = torch.linalg.norm(R_new - R, ord="fro", dim=(-2, -1))
            den = torch.clamp(torch.linalg.norm(R, ord="fro", dim=(-2, -1)), min=1e-30)
            rel_change = (num / den).to(torch.float32)
            R = R_new
            if float(rel_change.max().item()) < float(tol):
                break

    return R


def _project_band_energy_batched(
    Sw_B_Lt_N_hw: torch.Tensor,
    C_t: torch.Tensor,
    *,
    ridge: float = 0.10,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Project whitened snapshots onto band constraints (batched).
    """
    if C_t.numel() == 0:
        B, _, N, h, w = Sw_B_Lt_N_hw.shape
        return torch.zeros((B, N, h, w), device=Sw_B_Lt_N_hw.device, dtype=torch.float32)
    B, Lt, N, h, w = Sw_B_Lt_N_hw.shape
    K = C_t.shape[-1]
    C_exp = C_t.unsqueeze(0).expand(B, Lt, K)  # (B, Lt, K)
    gram = torch.matmul(C_exp.conj().transpose(-2, -1), C_exp)
    if ridge > 0.0:
        gram = gram + float(ridge) * torch.eye(K, dtype=gram.dtype, device=gram.device).unsqueeze(
            0
        )
    Sw_flat = Sw_B_Lt_N_hw.reshape(B, Lt, N * h * w)
    z = torch.matmul(C_exp.conj().transpose(-2, -1), Sw_flat)  # (B, K, N*h*w)
    try:
        Lg = torch.linalg.cholesky(gram)
        tmp = torch.linalg.solve_triangular(Lg, z, upper=False)
        proj = torch.linalg.solve_triangular(Lg.conj().transpose(-2, -1), tmp, upper=True)
    except RuntimeError:
        proj, _ = cholesky_solve_hermitian(gram, z, jitter_init=1e-8, max_tries=3)
    T_band = torch.sum(z.conj() * proj, dim=1).real  # (B, N*h*w)
    return T_band.reshape(B, N, h, w)


def stap_temporal_core_batched(
    cube_batch_T_hw: torch.Tensor,
    *,
    prf_hz: float,
    Lt: int,
    diag_load: float,
    kappa_shrink: float = 200.0,
    kappa_msd: float = 200.0,
    cov_estimator: str,
    huber_c: float,
    grid_step_rel: float,
    fd_span_rel: Tuple[float, float],
    min_pts: int,
    max_pts: int,
    fd_min_abs_hz: float,
    motion_half_span_rel: Optional[float],
    msd_ridge: float,
    msd_agg_mode: str,
    msd_ratio_rho: float,
    msd_contrast_alpha: float,
    msd_lambda: Optional[float] = None,
    device: Optional[str] = None,
    use_ref_cov: bool = False,
    fd_span_mode: str = "psd",
    flow_band_hz: Optional[Tuple[float, float]] = None,
    return_torch: bool = False,
) -> Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor, List[dict]]:
    """
    Fast-path batched STAP core (KA/debug disabled).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    telemetry_env = os.getenv("STAP_FAST_TELEMETRY", "").strip().lower()
    telemetry_enabled = telemetry_env not in {"0", "false", "no", "off"}

    env_tile_stat = os.getenv("STAP_FAST_TILE_STATISTIC", "").strip().lower()
    tile_statistic = env_tile_stat in {"1", "true", "yes", "on"}
    if not tile_statistic:
        env_tile_stat = os.getenv("STAP_TILE_STATISTIC", "").strip().lower()
        tile_statistic = env_tile_stat in {"1", "true", "yes", "on"}
    if tile_statistic:
        _warn_tile_statistic_experimental()

    # Fine-grained runtime breakdown (seconds) for profiling.
    t_hankel = 0.0
    t_cov = 0.0
    t_shrink = 0.0
    t_fdgrid = 0.0
    t_msd = 0.0

    t0 = time.perf_counter()
    with _prof_ctx("stap:hankel"):
        S_B_Lt_N_hw, R_scm = build_temporal_hankels_batch(
            cube_batch_T_hw, Lt, center=True, device=device, dtype=cube_batch_T_hw.dtype
        )
    t_hankel = time.perf_counter() - t0

    B, _, N, h, w = S_B_Lt_N_hw.shape
    # Match the manuscript baseline snapshot flattening order (h,w,N) to avoid
    # silent stride/reshape-dependent behavior changes on non-contiguous Hankel
    # tensors (and to keep fast-path score maps comparable to the slow path).
    S_flat = S_B_Lt_N_hw.permute(0, 1, 3, 4, 2).contiguous().view(B, Lt, -1)
    cov_est = cov_estimator.lower()
    t1 = time.perf_counter()
    with _prof_ctx("stap:covariance"):
        if use_ref_cov:
            R_hat, _ = robust_temporal_cov_batch(
                S_flat, estimator=cov_estimator, huber_c=huber_c, max_iter=100, tol=1e-4
            )
        else:
            if cov_est == "scm":
                R_hat = R_scm
            elif cov_est in {"huber", "huber_s"}:
                R_hat = _huber_covariance_batched(
                    S_flat, c=float(huber_c), max_iter=50, tol=1e-4
                )
            else:
                # Very short ensembles (e.g., Twinkling/Gammex with T=17, Lt=16 -> N=2)
                # are numerically fragile for Tyler; match legacy behavior by using
                # a robust Huber IRLS covariance directly in this regime.
                if cov_est in {"tyler_pca", "tylerpca", "tyler-pca"} and int(N) <= 4:
                    R_hat = _huber_covariance_batched(
                        S_flat, c=float(huber_c), max_iter=50, tol=1e-4
                    )
                else:
                    try:
                        tyler_init_env = os.getenv("STAP_TYLER_INIT", "").strip().lower()
                        R_init = R_scm if tyler_init_env in {"scm", "r_scm", "cov_scm"} else None
                        R_hat = _tyler_covariance_batched(S_flat, R_init=R_init)
                    except Exception:
                        R_hat = _huber_covariance_batched(
                            S_flat, c=float(huber_c), max_iter=50, tol=1e-4
                        )
    t_cov = time.perf_counter() - t1

    t2 = time.perf_counter()
    # Per-tile shrinkage toward mu*I to match slow path.
    ev_min_herm: torch.Tensor | None = None
    ev_max_herm: torch.Tensor | None = None
    alphas_f: torch.Tensor | None = None
    mu_trace: torch.Tensor | None = None
    with _prof_ctx("stap:shrinkage"):
        kappa_shrink = float(max(kappa_shrink, 1.01))
        if shrinkage_alpha_for_kappa_batch is not None:
            alphas_f, ev_min_herm, ev_max_herm = shrinkage_alpha_for_kappa_batch(
                R_hat, kappa_target=kappa_shrink, return_eigs=True
            )
        else:
            alphas_f = torch.as_tensor(
                [float(_shrinkage_alpha_for_kappa(R_hat[b], kappa_shrink)) for b in range(B)],
                device=device,
                dtype=torch.float32,
            )
        mu_trace = torch.real(torch.diagonal(R_hat, dim1=1, dim2=2).sum(dim=1)).to(
            dtype=torch.float32
        ) / float(Lt)
        alpha = alphas_f.view(-1, 1, 1) if alphas_f is not None else torch.zeros((B, 1, 1), device=device)
        # Apply shrinkage in-place:
        #   R' = (1-a) R + a * mu * I
        # Off-diagonals scale by (1-a); the diagonal additionally gets +a*mu.
        R_hat.mul_(1.0 - alpha)
        diag_add = (alpha.view(B) * mu_trace).to(dtype=R_hat.dtype).view(B, 1)
        torch.diagonal(R_hat, dim1=1, dim2=2).add_(diag_add)
    t_shrink = time.perf_counter() - t2

    eye = torch.eye(Lt, device=device, dtype=R_hat.dtype)
    # Per-tile lambda for MSD via conditioned_lambda.
    with _prof_ctx("stap:lambda"):
        kappa_msd = float(max(kappa_msd, 1.01))
        lam_init = float(msd_lambda) if msd_lambda is not None else float(diag_load)
        if (
            ev_min_herm is not None
            and ev_max_herm is not None
            and alphas_f is not None
            and mu_trace is not None
        ):
            # Fast exact path: avoid a second batched eigendecomposition by reusing
            # eigen-extrema already computed in the shrinkage stage.
            #
            # After shrinkage: R' = (1-a) R + a * mu * I, where a=alpha per tile.
            # For Hermitian R, eigenvalues transform exactly as e' = (1-a) e + a * mu.
            a64 = alphas_f.to(device=device, dtype=torch.float64).flatten()
            mu64 = mu_trace.to(device=device, dtype=torch.float64).flatten()
            ev_min64 = ev_min_herm.to(device=device, dtype=torch.float64).flatten()
            ev_max64 = ev_max_herm.to(device=device, dtype=torch.float64).flatten()

            ev_min_post = (1.0 - a64) * ev_min64 + a64 * mu64
            ev_max_post = (1.0 - a64) * ev_max64 + a64 * mu64
            # Match conditioned_lambda_batch: clamp eigenvalues to non-negative.
            ev_min_post = torch.clamp(ev_min_post, min=0.0)
            ev_max_post = torch.clamp(ev_max_post, min=0.0)

            denom = torch.clamp(
                torch.as_tensor(kappa_msd - 1.0, device=device, dtype=torch.float64),
                min=1e-8,
            )
            lam_needed = torch.where(
                ev_min_post > 0.0,
                torch.clamp((ev_max_post - kappa_msd * ev_min_post) / denom, min=0.0),
                torch.clamp(ev_max_post / denom, min=0.0),
            )
            base = torch.full((B,), lam_init, device=device, dtype=torch.float64)
            lam_out = torch.maximum(base, lam_needed)
            lam_msd_tensor = lam_out.to(R_hat.dtype)
            lam_needed_tensor = lam_needed.to(R_hat.dtype)
        elif conditioned_lambda_batch_shared is not None:
            # Preserve manuscript baseline behavior: compute per-tile loading via the
            # shared conditioned-lambda helper (no eigenvalue-bound shortcut).
            lam_batch, _, lam_needed_batch = conditioned_lambda_batch_shared(R_hat, lam_init, kappa_msd)
            lam_msd_tensor = lam_batch.to(R_hat.dtype)
            lam_needed_tensor = lam_needed_batch.to(R_hat.dtype)
        else:
            lam_list = []
            lam_needed_list = []
            for b in range(B):
                lam_for_msd, _, _ = conditioned_lambda(
                    R_hat[b], lam_requested=lam_init, kappa_target=kappa_msd
                )
                lam_list.append(lam_for_msd)
                lam_needed_list.append(max(0.0, lam_for_msd - lam_init))
            lam_msd_tensor = torch.as_tensor(lam_list, device=device, dtype=R_hat.dtype)
            lam_needed_tensor = torch.as_tensor(lam_needed_list, device=device, dtype=R_hat.dtype)

    motion_half_span_hz = 0.0
    base = prf_hz / float(Lt)
    if motion_half_span_rel is not None:
        motion_half_span_hz = max(0.0, float(motion_half_span_rel) * base)
    else:
        motion_half_span_hz = 0.1 * base

    t3 = time.perf_counter()
    fd_mode = str(fd_span_mode or "psd").strip().lower()
    fd_grid_source = "span"
    flow_band_used: Tuple[float, float] | None = None
    with _prof_ctx("stap:fd_grid"):
        if fd_mode in {"flow_band", "band"} and flow_band_hz is not None:
            try:
                flow_band_used = (float(flow_band_hz[0]), float(flow_band_hz[1]))
            except Exception:
                flow_band_used = None
            if flow_band_used is not None:
                fd_grid, fd_meta = build_fd_grid_flow_band(
                    prf_hz,
                    Lt,
                    flow_band_used,
                    motion_half_span_hz=float(motion_half_span_hz),
                    min_pts=min_pts,
                    max_pts=max_pts,
                )
                fd_grid_source = "flow_band"
            else:
                fd_grid, fd_meta = build_fd_grid_span(
                    prf_hz,
                    Lt,
                    fd_span_rel=fd_span_rel,
                    grid_step_rel=grid_step_rel,
                    fd_min_abs_hz=fd_min_abs_hz,
                    min_pts=min_pts,
                    max_pts=max_pts,
                )
        else:
            fd_grid, fd_meta = build_fd_grid_span(
                prf_hz,
                Lt,
                fd_span_rel=fd_span_rel,
                grid_step_rel=grid_step_rel,
                fd_min_abs_hz=fd_min_abs_hz,
                min_pts=min_pts,
                max_pts=max_pts,
            )
    t_fdgrid = time.perf_counter() - t3

    with _prof_ctx("stap:constraints"):
        Ct = bandpass_constraints_temporal(
            Lt,
            prf_hz,
            fd_grid_hz=fd_grid.tolist(),
            device=device,
            dtype=R_hat.dtype,
            mode="exp+deriv",
        )

    # Batched band-energy computation (PD-focused fast path)
    t4 = time.perf_counter()
    if tile_statistic:
        covonly_pixel_env = os.getenv("STAP_FAST_TILE_STATISTIC_PIXEL", "").strip().lower()
        covonly_per_pixel = covonly_pixel_env in {"1", "true", "yes", "on"}
        try:
            with _prof_ctx("stap:band_energy:covonly"):
                if covonly_per_pixel:
                    T_band_mean_hw, sw_pow_mean_hw = _band_energy_whitened_covonly_perpixel_batched(
                        R_hat,
                        S_B_Lt_N_hw,
                        Ct,
                        lam_msd_tensor,
                        ridge=msd_ridge,
                        device=device,
                        dtype=R_hat.dtype,
                        eps=1e-10,
                    )
                    sw_pow_safe = torch.clamp(sw_pow_mean_hw, min=1e-10)
                    band_frac_stack = torch.clamp(T_band_mean_hw / sw_pow_safe, min=0.0, max=1.0)
                    denom = torch.clamp(
                        sw_pow_safe - T_band_mean_hw + float(msd_ratio_rho) * sw_pow_safe, min=1e-10
                    )
                    score_stack = torch.clamp(T_band_mean_hw / denom, min=0.0)
                else:
                    T_band_mean, sw_pow_mean = _band_energy_whitened_covonly_batched(
                        R_hat,
                        R_scm,
                        Ct,
                        lam_msd_tensor,
                        ridge=msd_ridge,
                        device=device,
                        dtype=R_hat.dtype,
                        eps=1e-10,
                    )
                    sw_pow_safe = torch.clamp(sw_pow_mean, min=1e-10)
                    band_frac_scalar = torch.clamp(T_band_mean / sw_pow_safe, min=0.0, max=1.0)
                    denom = torch.clamp(
                        sw_pow_safe - T_band_mean + float(msd_ratio_rho) * sw_pow_safe, min=1e-10
                    )
                    ratio_scalar = torch.clamp(T_band_mean / denom, min=0.0)
                    band_frac_stack = band_frac_scalar.view(B, 1, 1).expand(B, h, w)
                    score_stack = ratio_scalar.view(B, 1, 1).expand(B, h, w)
            det_list = [{} for _ in range(B)]
            t_msd = time.perf_counter() - t4
        except Exception:
            tile_statistic = False

    if not tile_statistic:
        try:
            with _prof_ctx("stap:band_energy:snapshots"):
                T_band, sw_pow = _band_energy_whitened_batched(
                    R_hat,
                    S_B_Lt_N_hw,
                    Ct,
                    lam_msd_tensor,
                    ridge=msd_ridge,
                    ratio_rho=msd_ratio_rho,
                    kappa_target=kappa_msd,
                    device=device,
                    dtype=R_hat.dtype,
                    eps=1e-10,
                )
            sw_pow_safe = torch.clamp(sw_pow, min=1e-10)
            r_flow = torch.clamp(T_band / sw_pow_safe, min=0.0, max=1.0)
            denom = torch.clamp(
                sw_pow_safe - T_band + float(msd_ratio_rho) * sw_pow_safe, min=1e-10
            )
            ratio = torch.clamp(T_band / denom, min=0.0)
            with _prof_ctx("stap:aggregate"):
                band_frac_stack = aggregate_over_snapshots_batched(r_flow, mode=msd_agg_mode)
                score_stack = aggregate_over_snapshots_batched(ratio, mode=msd_agg_mode)
            det_list = [{} for _ in range(B)]
            t_msd = time.perf_counter() - t4
        except Exception:
            # Fallback to per-tile band-energy computation if batched path fails.
            band_frac_list = []
            score_list = []
            det_list = []
            for b in range(B):
                T_band_b, sw_pow_b = msd_snapshot_energies_batched(
                    R_hat[b],
                    S_B_Lt_N_hw[b],
                    Ct,
                    lam_abs=float(lam_msd_tensor[b].real.item()),
                    kappa_target=kappa_msd,
                    ridge=msd_ridge,
                    ratio_rho=msd_ratio_rho,
                    R0_prior=None,
                    Cf_flow=None,
                    ka_opts=None,
                    ka_details=None,
                    device=device,
                    dtype=R_hat.dtype,
                )
                sw_pow_b_safe = torch.clamp(sw_pow_b, min=1e-10)
                r_flow_b = torch.clamp(T_band_b / sw_pow_b_safe, min=0.0, max=1.0)
                denom_b = torch.clamp(
                    sw_pow_b_safe - T_band_b + float(msd_ratio_rho) * sw_pow_b_safe,
                    min=1e-10,
                )
                ratio_b = torch.clamp(T_band_b / denom_b, min=0.0)
                band_frac_list.append(aggregate_over_snapshots(r_flow_b, mode=msd_agg_mode))
                score_list.append(aggregate_over_snapshots(ratio_b, mode=msd_agg_mode))
                det_list.append({})
            band_frac_stack = torch.stack(band_frac_list, dim=0)
            score_stack = torch.stack(score_list, dim=0)
            t_msd = time.perf_counter() - t4

    band_frac_out = band_frac_stack.to(dtype=torch.float32)
    score_out = score_stack.to(dtype=torch.float32)

    if not telemetry_enabled:
        infos = [{} for _ in range(int(B))]
        if return_torch:
            return band_frac_out.detach(), score_out.detach(), infos
        band_frac_np = band_frac_out.detach().cpu().numpy().astype(np.float32, copy=False)
        score_np = score_out.detach().cpu().numpy().astype(np.float32, copy=False)
        return band_frac_np, score_np, infos

    with _prof_ctx("stap:telemetry"):
        band_flat = band_frac_out.reshape(B, -1)
        vals_sorted, _ = torch.sort(band_flat, dim=1)
        K = int(vals_sorted.shape[1])
        if K <= 0:
            band_med_t = torch.zeros((B,), dtype=torch.float32, device=band_flat.device)
            band_p90_t = band_med_t
        else:
            if K % 2 == 1:
                band_med_t = vals_sorted[:, K // 2]
            else:
                band_med_t = 0.5 * (vals_sorted[:, K // 2 - 1] + vals_sorted[:, K // 2])
            pos = 0.90 * float(K - 1)
            i0 = int(math.floor(pos))
            i1 = min(K - 1, i0 + 1)
            frac = float(pos - i0)
            band_p90_t = (1.0 - frac) * vals_sorted[:, i0] + frac * vals_sorted[:, i1]

        flow_mean_t = band_flat.mean(dim=1)
        band_med = band_med_t.detach().cpu().numpy()
        band_p90 = band_p90_t.detach().cpu().numpy()
        flow_mean = flow_mean_t.detach().cpu().numpy()
        lam_cond = lam_msd_tensor.real.detach().cpu().numpy()
        lam_need = lam_needed_tensor.real.detach().cpu().numpy()

        if not return_torch:
            band_frac_np = band_frac_out.detach().cpu().numpy().astype(np.float32, copy=False)
            score_np = score_out.detach().cpu().numpy().astype(np.float32, copy=False)

        infos: List[dict] = []
        msd_lambda_base = float(msd_lambda) if msd_lambda is not None else float(diag_load)
        for b in range(B):
            det = det_list[b] if b < len(det_list) else {}
            flow = float(flow_mean[b])
            mot = 0.0
            alias = max(0.0, 1.0 - flow - mot)
            # Attach identical profiling timings to each tile; aggregators can
            # summarize these across tiles for stap_fallback_telemetry.
            infos.append(
                {
                    "Lt": Lt,
                    "fd_grid_len": int(len(fd_grid)),
                    "fd_grid_source": str(fd_grid_source),
                    "span_hz": float(fd_meta.get("span_hz", 0.0)),
                    "grid_step_hz": float(fd_meta.get("grid_step_hz", 0.0)),
                    "flow_band_hz": (
                        [float(flow_band_used[0]), float(flow_band_used[1])]
                        if flow_band_used is not None
                        else None
                    ),
                    "cov_estimator": cov_estimator,
                    "diag_load": float(diag_load),
                    "motion_half_span_hz": float(motion_half_span_hz),
                    "msd_lambda": float(msd_lambda_base),
                    "msd_lambda_conditioned": float(lam_cond[b]),
                    "msd_lambda_needed": float(lam_need[b]),
                    "kc_flow_cap": int(len(fd_grid)),
                    "band_fraction_median": float(band_med[b]),
                    "band_fraction_p90": float(band_p90[b]),
                    "flow_fraction_tile": flow,
                    "motion_fraction_tile": mot,
                    "alias_fraction_tile": alias,
                    "stap_tile_statistic_used": bool(tile_statistic),
                    "stap_hankel_ms": float(1000.0 * t_hankel),
                    "stap_cov_ms": float(1000.0 * t_cov),
                    "stap_shrink_ms": float(1000.0 * t_shrink),
                    "stap_fdgrid_ms": float(1000.0 * t_fdgrid),
                    "stap_msd_ms": float(1000.0 * t_msd),
                }
            )
            infos[-1].update(det)

    if return_torch:
        return band_frac_out.detach(), score_out.detach(), infos
    return band_frac_np, score_np, infos


def pd_temporal_core_batched(
    cube_batch_T_hw: torch.Tensor,
    *,
    prf_hz: float,
    Lt: int,
    diag_load: float,
    kappa_shrink: float = 200.0,
    kappa_msd: float = 200.0,
    cov_estimator: str,
    huber_c: float,
    grid_step_rel: float,
    fd_span_rel: Tuple[float, float],
    min_pts: int,
    max_pts: int,
    fd_min_abs_hz: float,
    motion_half_span_rel: Optional[float],
    msd_ridge: float,
    msd_agg_mode: str,
    msd_ratio_rho: float,
    msd_contrast_alpha: float,
    msd_lambda: Optional[float] = None,
    device: Optional[str] = None,
    use_ref_cov: bool = False,
    fd_span_mode: str = "psd",
    flow_band_hz: Optional[Tuple[float, float]] = None,
    return_torch: bool = False,
) -> Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor, List[dict]]:
    """
    PD-oriented fast-path core: compute band fraction maps only, skipping MSD contrast.

    This reuses the same covariance and loading logic as stap_temporal_core_batched
    but uses a lighter matched-subspace energy computation instead of the full MSD
    contrast kernel. The returned score map is a placeholder and should not be used
    as a detector score; PD should be derived from the band-fraction map.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    telemetry_env = os.getenv("STAP_FAST_TELEMETRY", "").strip().lower()
    telemetry_enabled = telemetry_env not in {"0", "false", "no", "off"}

    env_tile_stat = os.getenv("STAP_FAST_TILE_STATISTIC", "").strip().lower()
    tile_statistic = env_tile_stat in {"1", "true", "yes", "on"}
    if not tile_statistic:
        env_tile_stat = os.getenv("STAP_TILE_STATISTIC", "").strip().lower()
        tile_statistic = env_tile_stat in {"1", "true", "yes", "on"}
    if tile_statistic:
        _warn_tile_statistic_experimental()

    # Runtime breakdown (seconds) for profiling.
    t_hankel = 0.0
    t_cov = 0.0
    t_shrink = 0.0
    t_fdgrid = 0.0
    t_band = 0.0

    t0 = time.perf_counter()
    S_B_Lt_N_hw, R_scm = build_temporal_hankels_batch(
        cube_batch_T_hw, Lt, center=True, device=device, dtype=cube_batch_T_hw.dtype
    )
    t_hankel = time.perf_counter() - t0

    B, _, N, h, w = S_B_Lt_N_hw.shape
    # Match the manuscript baseline snapshot flattening order (h,w,N) to avoid
    # silent stride/reshape-dependent behavior changes on non-contiguous Hankel
    # tensors (and to keep fast-path score maps comparable to the slow path).
    S_flat = S_B_Lt_N_hw.permute(0, 1, 3, 4, 2).contiguous().view(B, Lt, -1)
    cov_est = cov_estimator.lower()

    t1 = time.perf_counter()
    if use_ref_cov:
        R_hat, _ = robust_temporal_cov_batch(
            S_flat, estimator=cov_estimator, huber_c=huber_c, max_iter=100, tol=1e-4
        )
    else:
        if cov_est == "scm":
            R_hat = R_scm
        elif cov_est in {"huber", "huber_s"}:
            R_hat = _huber_covariance_batched(
                S_flat, c=float(huber_c), max_iter=50, tol=1e-4
            )
        else:
            if cov_est in {"tyler_pca", "tylerpca", "tyler-pca"} and int(N) <= 4:
                R_hat = _huber_covariance_batched(
                    S_flat, c=float(huber_c), max_iter=50, tol=1e-4
                )
            else:
                try:
                    tyler_init_env = os.getenv("STAP_TYLER_INIT", "").strip().lower()
                    R_init = R_scm if tyler_init_env in {"scm", "r_scm", "cov_scm"} else None
                    R_hat = _tyler_covariance_batched(S_flat, R_init=R_init)
                except Exception:
                    R_hat = _huber_covariance_batched(
                        S_flat, c=float(huber_c), max_iter=50, tol=1e-4
                    )
    t_cov = time.perf_counter() - t1

    # Per-tile shrinkage toward mu*I to match slow path.
    t2 = time.perf_counter()
    ev_min_herm: torch.Tensor | None = None
    ev_max_herm: torch.Tensor | None = None
    alphas_f: torch.Tensor | None = None
    mu_trace: torch.Tensor | None = None
    with _prof_ctx("stap:shrinkage"):
        kappa_shrink = float(max(kappa_shrink, 1.01))
        eye = torch.eye(Lt, device=device, dtype=R_hat.dtype)
        if shrinkage_alpha_for_kappa_batch is not None:
            alphas_f, ev_min_herm, ev_max_herm = shrinkage_alpha_for_kappa_batch(
                R_hat, kappa_target=kappa_shrink, return_eigs=True
            )
        else:
            alphas_f = torch.as_tensor(
                [float(_shrinkage_alpha_for_kappa(R_hat[b], kappa_shrink)) for b in range(B)],
                device=device,
                dtype=torch.float32,
            )
        mu_trace = torch.real(torch.diagonal(R_hat, dim1=1, dim2=2).sum(dim=1)).to(
            dtype=torch.float32
        ) / float(Lt)
        alpha = alphas_f.view(-1, 1, 1) if alphas_f is not None else torch.zeros((B, 1, 1), device=device)
        R_hat = (1.0 - alpha) * R_hat + alpha * mu_trace.view(-1, 1, 1) * eye
    t_shrink = time.perf_counter() - t2

    # Per-tile lambda consistent with MSD loading.
    eye = torch.eye(Lt, device=device, dtype=R_hat.dtype)
    kappa_msd = float(max(kappa_msd, 1.01))
    lam_init = float(msd_lambda) if msd_lambda is not None else float(diag_load)
    with _prof_ctx("stap:lambda"):
        if (
            ev_min_herm is not None
            and ev_max_herm is not None
            and alphas_f is not None
            and mu_trace is not None
        ):
            a64 = alphas_f.to(device=device, dtype=torch.float64).flatten()
            mu64 = mu_trace.to(device=device, dtype=torch.float64).flatten()
            ev_min64 = ev_min_herm.to(device=device, dtype=torch.float64).flatten()
            ev_max64 = ev_max_herm.to(device=device, dtype=torch.float64).flatten()

            ev_min_post = (1.0 - a64) * ev_min64 + a64 * mu64
            ev_max_post = (1.0 - a64) * ev_max64 + a64 * mu64
            ev_min_post = torch.clamp(ev_min_post, min=0.0)
            ev_max_post = torch.clamp(ev_max_post, min=0.0)

            denom = torch.clamp(
                torch.as_tensor(kappa_msd - 1.0, device=device, dtype=torch.float64),
                min=1e-8,
            )
            lam_needed = torch.where(
                ev_min_post > 0.0,
                torch.clamp((ev_max_post - kappa_msd * ev_min_post) / denom, min=0.0),
                torch.clamp(ev_max_post / denom, min=0.0),
            )
            base = torch.full((B,), lam_init, device=device, dtype=torch.float64)
            lam_out = torch.maximum(base, lam_needed)
            lam_msd_tensor = lam_out.to(R_hat.dtype)
            lam_needed_tensor = lam_needed.to(R_hat.dtype)
        elif conditioned_lambda_batch_shared is not None:
            # Preserve manuscript baseline behavior: compute per-tile loading via the
            # shared conditioned-lambda helper (no eigenvalue-bound shortcut).
            lam_batch, _, lam_needed_batch = conditioned_lambda_batch_shared(R_hat, lam_init, kappa_msd)
            lam_msd_tensor = lam_batch.to(R_hat.dtype)
            lam_needed_tensor = lam_needed_batch.to(R_hat.dtype)
        else:
            lam_list = []
            lam_needed_list = []
            for b in range(B):
                lam_for_msd, _, _ = conditioned_lambda(
                    R_hat[b], lam_requested=lam_init, kappa_target=kappa_msd
                )
                lam_list.append(lam_for_msd)
                lam_needed_list.append(max(0.0, lam_for_msd - lam_init))
            lam_msd_tensor = torch.as_tensor(lam_list, device=device, dtype=R_hat.dtype)
            lam_needed_tensor = torch.as_tensor(lam_needed_list, device=device, dtype=R_hat.dtype)

    motion_half_span_hz = 0.0
    base = prf_hz / float(Lt)
    if motion_half_span_rel is not None:
        motion_half_span_hz = max(0.0, float(motion_half_span_rel) * base)
    else:
        motion_half_span_hz = 0.1 * base

    # Build a single band grid and constraint matrix for PD.
    t3 = time.perf_counter()
    fd_mode = str(fd_span_mode or "psd").strip().lower()
    fd_grid_source = "span"
    flow_band_used: Tuple[float, float] | None = None
    with _prof_ctx("stap:fd_grid"):
        if fd_mode in {"flow_band", "band"} and flow_band_hz is not None:
            try:
                flow_band_used = (float(flow_band_hz[0]), float(flow_band_hz[1]))
            except Exception:
                flow_band_used = None
            if flow_band_used is not None:
                fd_grid, fd_meta = build_fd_grid_flow_band(
                    prf_hz,
                    Lt,
                    flow_band_used,
                    motion_half_span_hz=float(motion_half_span_hz),
                    min_pts=min_pts,
                    max_pts=max_pts,
                )
                fd_grid_source = "flow_band"
            else:
                fd_grid, fd_meta = build_fd_grid_span(
                    prf_hz,
                    Lt,
                    fd_span_rel=fd_span_rel,
                    grid_step_rel=grid_step_rel,
                    fd_min_abs_hz=fd_min_abs_hz,
                    min_pts=min_pts,
                    max_pts=max_pts,
                )
        else:
            fd_grid, fd_meta = build_fd_grid_span(
                prf_hz,
                Lt,
                fd_span_rel=fd_span_rel,
                grid_step_rel=grid_step_rel,
                fd_min_abs_hz=fd_min_abs_hz,
                min_pts=min_pts,
                max_pts=max_pts,
            )
    t_fdgrid = time.perf_counter() - t3

    # Constraints for the entire band (no motion/alias split in this PD-only mode).
    with _prof_ctx("stap:constraints"):
        Ct = bandpass_constraints_temporal(
            Lt,
            prf_hz,
            fd_grid_hz=fd_grid.tolist(),
            device=device,
            dtype=R_hat.dtype,
            mode="exp+deriv",
        )

    # Compute per-snapshot band energy and total whitened power, then aggregate.
    t4 = time.perf_counter()
    with _prof_ctx("stap:band_energy"):
        if tile_statistic:
            covonly_pixel_env = os.getenv("STAP_FAST_TILE_STATISTIC_PIXEL", "").strip().lower()
            covonly_per_pixel = covonly_pixel_env in {"1", "true", "yes", "on"}
            if covonly_per_pixel:
                T_band_mean_hw, sw_pow_mean_hw = _band_energy_whitened_covonly_perpixel_batched(
                    R_hat,
                    S_B_Lt_N_hw,
                    Ct,
                    lam_msd_tensor,
                    ridge=msd_ridge,
                    device=device,
                    dtype=R_hat.dtype,
                    eps=1e-10,
                )
                sw_pow_safe = torch.clamp(sw_pow_mean_hw, min=1e-10)
                band_frac_stack = torch.clamp(T_band_mean_hw / sw_pow_safe, min=0.0, max=1.0)
                ratio_denom = torch.clamp(
                    sw_pow_safe - T_band_mean_hw + float(msd_ratio_rho) * sw_pow_safe, min=1e-10
                )
                score_stack = torch.clamp(T_band_mean_hw / ratio_denom, min=0.0)
            else:
                T_band_mean, sw_pow_mean = _band_energy_whitened_covonly_batched(
                    R_hat,
                    R_scm,
                    Ct,
                    lam_msd_tensor,
                    ridge=msd_ridge,
                    device=device,
                    dtype=R_hat.dtype,
                    eps=1e-10,
                )
                sw_pow_safe = torch.clamp(sw_pow_mean, min=1e-10)
                band_frac_scalar = torch.clamp(T_band_mean / sw_pow_safe, min=0.0, max=1.0)
                ratio_denom = torch.clamp(
                    sw_pow_safe - T_band_mean + float(msd_ratio_rho) * sw_pow_safe, min=1e-10
                )
                ratio_scalar = torch.clamp(T_band_mean / ratio_denom, min=0.0)
                band_frac_stack = band_frac_scalar.view(B, 1, 1).expand(B, h, w)
                score_stack = ratio_scalar.view(B, 1, 1).expand(B, h, w)
        else:
            # Use the same batched whitening + projection path as the full STAP fast core.
            T_band, sw_pow = _band_energy_whitened_batched(
                R_hat,
                S_B_Lt_N_hw,
                Ct,
                lam_msd_tensor,
                ridge=msd_ridge,
                ratio_rho=msd_ratio_rho,
                kappa_target=kappa_msd,
                device=device,
                dtype=R_hat.dtype,
                eps=1e-10,
            )
            sw_pow_safe = torch.clamp(sw_pow, min=1e-10)
            r_flow = torch.clamp(T_band / sw_pow_safe, min=0.0, max=1.0)
            ratio_denom = torch.clamp(
                sw_pow_safe - T_band + float(msd_ratio_rho) * sw_pow_safe, min=1e-10
            )
            ratio = torch.clamp(T_band / ratio_denom, min=0.0)
            with _prof_ctx("stap:aggregate"):
                band_frac_stack = aggregate_over_snapshots_batched(r_flow, mode=msd_agg_mode)
                score_stack = aggregate_over_snapshots_batched(ratio, mode=msd_agg_mode)
    t_band = time.perf_counter() - t4

    band_frac_out = band_frac_stack.to(dtype=torch.float32)
    score_out = score_stack.to(dtype=torch.float32)
    if not telemetry_enabled:
        infos = [{} for _ in range(int(B))]
        if return_torch:
            return band_frac_out.detach(), score_out.detach(), infos
        band_frac_np = band_frac_out.detach().cpu().numpy().astype(np.float32, copy=False)
        score_np = score_out.detach().cpu().numpy().astype(np.float32, copy=False)
        return band_frac_np, score_np, infos
    with _prof_ctx("stap:telemetry"):
        band_flat = band_frac_out.reshape(B, -1)
        vals_sorted, _ = torch.sort(band_flat, dim=1)
        K = int(vals_sorted.shape[1])
        if K <= 0:
            band_med_t = torch.zeros((B,), dtype=torch.float32, device=band_flat.device)
            band_p90_t = band_med_t
        else:
            if K % 2 == 1:
                band_med_t = vals_sorted[:, K // 2]
            else:
                band_med_t = 0.5 * (vals_sorted[:, K // 2 - 1] + vals_sorted[:, K // 2])
            pos = 0.90 * float(K - 1)
            i0 = int(math.floor(pos))
            i1 = min(K - 1, i0 + 1)
            frac = float(pos - i0)
            band_p90_t = (1.0 - frac) * vals_sorted[:, i0] + frac * vals_sorted[:, i1]

        flow_mean_t = band_flat.mean(dim=1)
        band_med = band_med_t.detach().cpu().numpy()
        band_p90 = band_p90_t.detach().cpu().numpy()
        flow_mean = flow_mean_t.detach().cpu().numpy()
        lam_cond = lam_msd_tensor.real.detach().cpu().numpy()
        lam_need = lam_needed_tensor.real.detach().cpu().numpy()

        if not return_torch:
            band_frac_np = band_frac_out.detach().cpu().numpy().astype(np.float32, copy=False)
            score_np = score_out.detach().cpu().numpy().astype(np.float32, copy=False)

        infos: List[dict] = []
        msd_lambda_base = float(msd_lambda) if msd_lambda is not None else float(diag_load)
        for b in range(B):
            flow = float(flow_mean[b])
            alias = max(0.0, 1.0 - flow)
            info = {
                "Lt": Lt,
                "fd_grid_len": int(len(fd_grid)),
                "fd_grid_source": str(fd_grid_source),
                "span_hz": float(fd_meta.get("span_hz", 0.0)),
                "grid_step_hz": float(fd_meta.get("grid_step_hz", 0.0)),
                "flow_band_hz": (
                    [float(flow_band_used[0]), float(flow_band_used[1])]
                    if flow_band_used is not None
                    else None
                ),
                "cov_estimator": cov_estimator,
                "diag_load": float(diag_load),
                "motion_half_span_hz": float(motion_half_span_hz),
                "msd_lambda": float(msd_lambda_base),
                "msd_lambda_conditioned": float(lam_cond[b]),
                "msd_lambda_needed": float(lam_need[b]),
                "kc_flow_cap": int(len(fd_grid)),
                "band_fraction_median": float(band_med[b]),
                "band_fraction_p90": float(band_p90[b]),
                "flow_fraction_tile": flow,
                "motion_fraction_tile": 0.0,
                "alias_fraction_tile": alias,
                "stap_tile_statistic_used": bool(tile_statistic),
                "stap_hankel_ms": float(1000.0 * t_hankel),
                "stap_cov_ms": float(1000.0 * t_cov),
                "stap_shrink_ms": float(1000.0 * t_shrink),
                "stap_fdgrid_ms": float(1000.0 * t_fdgrid),
                "stap_msd_ms": float(1000.0 * t_band),
            }
            infos.append(info)

    if return_torch:
        return band_frac_out.detach(), score_out.detach(), infos
    return band_frac_np, score_np, infos
