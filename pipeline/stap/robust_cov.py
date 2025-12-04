# pipeline/stap/robust_cov.py
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

try:
    import torch

    TORCH_OK = True
except Exception:  # pragma: no cover - torch optional
    TORCH_OK = False
    torch = None  # type: ignore

Array = np.ndarray
Tensor = "torch.Tensor"  # type: ignore
EPS = 1e-12


@dataclass(frozen=True)
class RobustCovTelemetry:
    method: str
    iters: int
    converged: bool
    rel_change: float
    diag_load: float
    trace_in: float
    trace_out: float
    notes: str = ""


def _is_torch(x) -> bool:
    return TORCH_OK and isinstance(x, torch.Tensor)


def _hermitianize(x):
    if _is_torch(x):
        return 0.5 * (x + x.conj().transpose(-2, -1))
    return 0.5 * (x + x.conj().T)


def _trace(x) -> float:
    if _is_torch(x):
        return float(torch.real(torch.trace(x)).item())
    return float(np.trace(x).real)


def _eye(M: int, like):
    if _is_torch(like):
        return torch.eye(M, dtype=like.dtype, device=like.device)
    return np.eye(M, dtype=like.dtype)


def _solve_RinvX(R, X):
    """Solve R Y = X for Hermitian positive-definite R."""
    if _is_torch(R):
        from pipeline.gpu.linalg import cholesky_robust

        L, _, _ = cholesky_robust(R, jitter_init=1e-8, max_tries=6)
        Y = torch.cholesky_solve(X, L)
        return Y

    try:
        L = np.linalg.cholesky(R)
        Y = np.linalg.solve(L.conj().T, np.linalg.solve(L, X))
        return Y
    except np.linalg.LinAlgError:
        mu = _trace(R) / R.shape[0]
        Rj = R + 1e-6 * mu * np.eye(R.shape[0], dtype=R.dtype)
        L = np.linalg.cholesky(Rj)
        Y = np.linalg.solve(L.conj().T, np.linalg.solve(L, X))
        return Y


def _fro_norm(a) -> float:
    if _is_torch(a):
        return float(torch.linalg.norm(a, ord="fro").item())
    return float(np.linalg.norm(a, ord="fro"))


# ------------------------------ Huber --------------------------------------- #


def huber_covariance(
    X_in,
    c: float = 5.0,
    max_iter: int = 50,
    tol: float = 1e-4,
    trace_target: Optional[float] = None,
) -> Tuple[object, RobustCovTelemetry]:
    """
    Complex Huber M-estimator via IRLS on scatter:
        d_i = Re{x_i^H R^{-1} x_i} / M
        w_i = min(1, c / max(d_i, eps))
        R = (1/N) sum_i w_i x_i x_i^H
    """
    X = X_in
    if _is_torch(X):
        M, N = X.shape[-2], X.shape[-1]
        R = (X @ X.conj().transpose(-2, -1)) / float(N)
    else:
        M, N = X.shape
        R = (X @ X.conj().T) / float(N)
    R = _hermitianize(R)

    tr0 = _trace(R) if trace_target is None else float(trace_target)
    mu = tr0 / max(M, 1)
    R = R + (1e-8 * mu) * _eye(M, R)

    rel_change = np.inf
    notes = ""

    last_iter = 0
    for it in range(1, max_iter + 1):
        last_iter = it
        Y = _solve_RinvX(R, X)
        if _is_torch(X):
            d = torch.real(torch.sum(X.conj() * Y, dim=-2)) / float(M)
            w = torch.minimum(
                torch.ones_like(d),
                torch.as_tensor(c, dtype=d.dtype, device=d.device) / torch.clamp(d, min=1e-12),
            )
            R_new = (X * w.unsqueeze(-2)) @ X.conj().transpose(-2, -1) / float(N)
        else:
            d = np.real(np.sum(X.conj() * Y, axis=0)) / float(M)
            w = np.minimum(1.0, c / np.maximum(d, 1e-12))
            R_new = (X * w[np.newaxis, :]) @ X.conj().T / float(N)
        R_new = _hermitianize(R_new)

        tr = _trace(R_new)
        if tr <= 0:
            notes = "nonpositive trace; added diag load"
            R_new = R_new + (1e-6 * mu) * _eye(M, R_new)
            tr = _trace(R_new)
        scale = tr0 / max(tr, EPS)
        R_new = R_new * scale

        rel_change = _fro_norm(R_new - R) / (max(_fro_norm(R), 1e-12))
        R = R_new
        if rel_change < tol:
            break

    tel = RobustCovTelemetry(
        method="huber",
        iters=last_iter,
        converged=rel_change < tol,
        rel_change=float(rel_change),
        diag_load=1e-8,
        trace_in=tr0,
        trace_out=_trace(R),
        notes=notes,
    )
    return R, tel


# ------------------------------ Tyler --------------------------------------- #


def tyler_covariance(
    X_in,
    max_iter: int = 200,
    tol: float = 1e-5,
    unit_trace: bool = True,
) -> Tuple[object, RobustCovTelemetry]:
    """
    Tyler's M-estimator (scatter for elliptical distributions).
    Requires N > M and data not contained in a subspace.
    """
    X = X_in
    if _is_torch(X):
        M, N = X.shape[-2], X.shape[-1]
        R = (X @ X.conj().transpose(-2, -1)) / float(N)
    else:
        M, N = X.shape
        if N <= M + 5:
            raise ValueError(f"Tyler requires N >> M (got N={N}, M={M})")
        R = (X @ X.conj().T) / float(N)
    R = _hermitianize(R)
    tr0 = _trace(R)
    mu = tr0 / max(M, 1)
    R = R + (1e-8 * mu) * _eye(M, R)

    if unit_trace:
        R = R * (M / max(_trace(R), EPS))

    rel_change = np.inf
    notes = ""

    last_iter = 0
    for it in range(1, max_iter + 1):
        last_iter = it
        Y = _solve_RinvX(R, X)
        if _is_torch(X):
            denom = torch.real(torch.sum(X.conj() * Y, dim=-2))
            inv_d = 1.0 / torch.clamp(denom, min=1e-12)
            R_new = (X * inv_d.unsqueeze(-2)) @ X.conj().transpose(-2, -1)
            R_new = R_new * (float(M) / float(N))
        else:
            denom = np.real(np.sum(X.conj() * Y, axis=0))
            inv_d = 1.0 / np.maximum(denom, 1e-12)
            R_new = (X * inv_d[np.newaxis, :]) @ X.conj().T
            R_new = R_new * (float(M) / float(N))
        R_new = _hermitianize(R_new)

        if unit_trace:
            R_new = R_new * (M / max(_trace(R_new), EPS))
        else:
            R_new = R_new * (tr0 / max(_trace(R_new), EPS))

        rel_change = _fro_norm(R_new - R) / (max(_fro_norm(R), 1e-12))
        R = R_new
        if rel_change < tol:
            break

    tel = RobustCovTelemetry(
        method="tyler",
        iters=last_iter,
        converged=rel_change < tol,
        rel_change=float(rel_change),
        diag_load=1e-8,
        trace_in=tr0,
        trace_out=_trace(R),
        notes=notes,
    )
    return R, tel


# ----------------------------- Unified entry -------------------------------- #


def tyler_pca_hybrid(
    X_in,
    energy: float = 0.90,
    r: Optional[int] = None,
    max_iter: int = 200,
    tol: float = 1e-5,
) -> Tuple[object, RobustCovTelemetry]:
    """
    Tyler-on-PCA hybrid for N < M or heavy contamination scenarios.

    Steps
    -----
    1) Compute SCM and eigendecompose -> principal subspace (rank r).
    2) Run Tyler in the r-dim subspace (unit-trace).
    3) Lift back and add isotropic residual on orthogonal complement to preserve trace.
    """
    X = X_in
    is_torch = _is_torch(X)

    if is_torch:
        M, N = X.shape[-2], X.shape[-1]
        R0 = (X @ X.conj().transpose(-2, -1)) / float(N)
        R0 = _hermitianize(R0)
        w, V = torch.linalg.eigh(R0)
        w = torch.real(w)
        order = torch.argsort(w, descending=True)
        w = w[order]
        V = V[:, order]
        cume = torch.cumsum(w, dim=0) / (torch.sum(w) + 1e-12)
        if r is None:
            idx = torch.searchsorted(cume, torch.as_tensor(energy, device=w.device))
            r = int(idx.item() + 1)
    else:
        M, N = X.shape
        R0 = (X @ X.conj().T) / float(N)
        R0 = _hermitianize(R0)
        w, V = np.linalg.eigh(R0)
        order = np.argsort(w)[::-1]
        w = w[order]
        V = V[:, order]
        cume = np.cumsum(w) / (np.sum(w) + 1e-12)
        if r is None:
            r = int(np.searchsorted(cume, energy) + 1)

    r = max(1, min(r, M - 1, N - 5))

    if is_torch:
        U = V[:, :r]
        Xr = U.conj().transpose(-2, -1) @ X
    else:
        U = V[:, :r]
        Xr = U.conj().T @ X

    Rr, tel_tyl = tyler_covariance(Xr, max_iter=max_iter, tol=tol, unit_trace=True)

    if is_torch:
        R_lift = U @ Rr @ U.conj().transpose(-2, -1)
        tr0 = _trace(R0)
        tr_lift = _trace(R_lift)
        Mr = M - r
        if Mr > 0:
            sigma2 = max((tr0 - tr_lift) / Mr, 1e-12)
            identity = torch.eye(M, dtype=R_lift.dtype, device=R_lift.device)
            Pres = identity - U @ U.conj().transpose(-2, -1)
            R = R_lift + sigma2 * Pres
        else:
            R = R_lift
    else:
        R_lift = U @ Rr @ U.conj().T
        tr0 = _trace(R0)
        tr_lift = _trace(R_lift)
        Mr = M - r
        if Mr > 0:
            sigma2 = max((tr0 - tr_lift) / Mr, 1e-12)
            identity = np.eye(M, dtype=R_lift.dtype)
            Pres = identity - U @ U.conj().T
            R = R_lift + sigma2 * Pres
        else:
            R = R_lift

    R = _hermitianize(R)
    tel = RobustCovTelemetry(
        method="tyler_pca",
        iters=tel_tyl.iters,
        converged=tel_tyl.converged,
        rel_change=tel_tyl.rel_change,
        diag_load=1e-8,
        trace_in=_trace(R0),
        trace_out=_trace(R),
        notes=f"r={r}, energy={energy:.2f}",
    )
    return R, tel


def robust_covariance(
    X_in,
    method: Literal["scm", "huber", "tyler", "tyler_pca"] = "huber",
    huber_c: float = 5.0,
    max_iter: int = 100,
    tol: float = 1e-4,
    trace_target: Optional[float] = None,
) -> Tuple[object, RobustCovTelemetry]:
    """
    Dispatch to SCM / Huber / Tyler M-estimators.
    """
    X = X_in
    if method == "scm":
        if _is_torch(X):
            M, N = X.shape[-2], X.shape[-1]
            R = (X @ X.conj().transpose(-2, -1)) / float(N)
        else:
            M, N = X.shape
            R = (X @ X.conj().T) / float(N)
        R = _hermitianize(R)
        tr = _trace(R)
        tel = RobustCovTelemetry("scm", 1, True, 0.0, 0.0, tr, tr)
        return R, tel

    if method == "tyler":
        try:
            return tyler_covariance(
                X, max_iter=max_iter, tol=tol, unit_trace=(trace_target is None)
            )
        except Exception as exc:  # pragma: no cover - fallback path
            warnings.warn(
                f"Tyler failed (or N<=M); falling back to Huber. Reason: {exc}",
                stacklevel=2,
            )
            method = "huber"

    if method == "tyler_pca":
        try:
            return tyler_pca_hybrid(X, energy=0.90, r=None, max_iter=max_iter, tol=tol)
        except Exception as exc:  # pragma: no cover - fallback path
            warnings.warn(
                f"Tyler-PCA failed; falling back to Huber. Reason: {exc}",
                stacklevel=2,
            )
            method = "huber"

    return huber_covariance(
        X,
        c=huber_c,
        max_iter=max_iter,
        tol=tol,
        trace_target=trace_target,
    )
