from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from ..gpu.linalg import cholesky_solve_hermitian, hermitianize, to_tensor
from .mvdr import steering_vector


@dataclass(frozen=True)
class LCMVResult:
    w: torch.Tensor
    A: torch.Tensor
    den: torch.Tensor
    jitter_used: float
    diag_load: float
    load_mode: str
    constraint_ridge: float
    constraint_residual: float
    auto_kappa_target: float | None = None


def _as_tensor(x, device: Optional[str], dtype: Optional[torch.dtype]) -> torch.Tensor:
    t = to_tensor(x, device=device)
    if dtype is not None:
        t = t.to(dtype)
    return t


def _cond_loaded(eigs: torch.Tensor, lam: float) -> float:
    lam = float(lam)
    lmin = float(torch.min(eigs).item())
    lmax = float(torch.max(eigs).item())
    return (lmax + lam) / max(lmin + lam, 1e-30)


def _auto_abs_load_from_eigs(
    eigs: torch.Tensor,
    kappa_target: float,
    lower: float,
    upper: float,
) -> float:
    lmin = float(torch.min(eigs).item())
    lmax = float(torch.max(eigs).item())
    if kappa_target <= 1.0:
        return lower
    lam_star = (lmax - kappa_target * lmin) / (kappa_target - 1.0)
    lam = max(lam_star, 0.0)
    if not np.isfinite(lam):
        lam = upper
    return float(np.clip(lam, lower, upper))


def lcmv_weights(
    R: torch.Tensor | np.ndarray,
    C: torch.Tensor | np.ndarray,
    f: torch.Tensor | np.ndarray,
    load_mode: str = "absolute",
    diag_load: float = 1e-2,
    auto_kappa_target: float = 1e5,
    auto_lambda_bounds: tuple[float, float] = (1e-4, 5e-1),
    constraint_ridge: float = 0.05,
    enforce_exact_if_possible: bool = True,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> LCMVResult:
    """
    Linear Constrained Minimum Variance weights with robust loading:
        w = R^{-1} C (C^H R^{-1} C)^{-1} f
    load_mode: "absolute" -> add diag_load * I
               "scaled"   -> add diag_load * mu * I , mu = trace(R)/M
    constraint_ridge: δ >= 0, applied to constraint Gram for soft equality.
    """
    Rt = _as_tensor(R, device=device, dtype=dtype)
    Rt = hermitianize(Rt)
    M = Rt.shape[-1]
    dev, dt = Rt.device, Rt.dtype

    Ct = _as_tensor(C, device=str(dev), dtype=dt)
    ft = _as_tensor(f, device=str(dev), dtype=dt).reshape(-1)

    mu = torch.real(torch.trace(Rt)) / float(M)
    mode = load_mode.lower()
    if mode == "scaled":
        lam = float(diag_load) * float(mu)
    elif mode == "auto":
        eigs = torch.linalg.eigvalsh(Rt).real
        lam = _auto_abs_load_from_eigs(
            eigs,
            kappa_target=float(auto_kappa_target),
            lower=float(auto_lambda_bounds[0]),
            upper=float(auto_lambda_bounds[1]),
        )
        mode = f"auto@{auto_kappa_target:.1e}"
    else:
        lam = float(diag_load)
        mode = "absolute"

    Rl = Rt + lam * torch.eye(M, dtype=dt, device=dev)

    U, jit = cholesky_solve_hermitian(Rl, Ct, jitter_init=1e-8, max_tries=6)
    A = torch.matmul(Ct.conj().transpose(-2, -1), U)
    if constraint_ridge > 0.0:
        A = A + (constraint_ridge * torch.eye(A.shape[-1], dtype=dt, device=dev))

    g, _ = cholesky_solve_hermitian(A, ft, jitter_init=1e-12, max_tries=4)
    w = torch.matmul(U, g)

    cf = torch.matmul(Ct.conj().transpose(-2, -1), w)
    if enforce_exact_if_possible and constraint_ridge == 0.0:
        delta, _ = cholesky_solve_hermitian(A, ft - cf, jitter_init=1e-12, max_tries=2)
        w = w + torch.matmul(U, delta)
        cf = torch.matmul(Ct.conj().transpose(-2, -1), w)

    resid = torch.linalg.norm(cf - ft).item()
    den = torch.real(torch.sum(w.conj() * torch.matmul(Rl, w)))
    return LCMVResult(
        w=w,
        A=A,
        den=den,
        jitter_used=jit,
        diag_load=float(lam),
        load_mode=mode,
        constraint_ridge=float(constraint_ridge),
        constraint_residual=resid,
        auto_kappa_target=float(auto_kappa_target) if "auto@" in mode else None,
    )


def bandpass_constraints(
    h: int,
    w: int,
    Lt: int,
    prf_hz: float,
    fd_grid_hz: Sequence[float],
    device: Optional[str] = None,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    cols = []
    for fd in fd_grid_hz:
        s = steering_vector(
            h=h,
            w=w,
            Lt=Lt,
            fd_hz=float(fd),
            prf_hz=prf_hz,
            device=device,
            dtype=dtype,
        )
        cols.append(s.reshape(-1, 1))
    return torch.cat(cols, dim=1)


def lcmv_bandpass_apply(
    R: torch.Tensor | np.ndarray,
    X: torch.Tensor | np.ndarray,
    C: torch.Tensor | np.ndarray,
    load_mode: str = "absolute",
    diag_load: float = 1e-2,
    auto_kappa_target: float = 1e5,
    auto_lambda_bounds: tuple[float, float] = (1e-4, 5e-1),
    constraint_ridge: float = 0.05,
    enforce_exact_if_possible: bool = True,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, LCMVResult]:
    """
    Compute LCMV band-pass weights and apply to snapshots X.
    Returns per-snapshot complex output y along with LCMVResult metadata.
    """
    Kc = C.shape[-1] if isinstance(C, torch.Tensor) else np.asarray(C).shape[-1]
    target = torch.ones(
        (Kc,),
        dtype=dtype or torch.complex64,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    res = lcmv_weights(
        R,
        C,
        f=target,
        load_mode=load_mode,
        diag_load=diag_load,
        auto_kappa_target=auto_kappa_target,
        auto_lambda_bounds=auto_lambda_bounds,
        constraint_ridge=constraint_ridge,
        enforce_exact_if_possible=enforce_exact_if_possible,
        device=device,
        dtype=dtype,
    )
    Xt = _as_tensor(X, device=device, dtype=res.w.dtype)
    y = torch.matmul(res.w.conj(), Xt)
    return y, res
