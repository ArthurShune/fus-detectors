# pipeline/stap/covariance.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .robust_cov import robust_covariance

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable in CI
    torch = None

Array = np.ndarray
EPS = 1e-12


def _is_torch_tensor(x) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


# ------------------------- Snapshots (Hankelization) ------------------------- #


def build_snapshots(tile_T_hw: Array, Lt: int, center: bool = True) -> Array:
    """
    Build space-time snapshots X in C^{M x Ntr} from a tile cube (T, h, w)
    using Lt temporal taps (Hankel stacking).

    M = h*w*Lt, Ntr = T - Lt + 1

    Parameters
    ----------
    tile_T_hw : (T, h, w) complex or real array
    Lt : int
        Number of temporal taps. Lt>=1
    center : bool
        If True, subtract per-dimension mean across snapshots (zero-mean columns).

    Returns
    -------
    X : (M, Ntr) complex64 array
    """
    assert tile_T_hw.ndim == 3, "tile_T_hw must be (T,h,w)"
    T, h, w = tile_T_hw.shape
    if Lt < 1 or Lt > T:
        raise ValueError(f"Lt must be in [1, T]. Got Lt={Lt}, T={T}")

    M = h * w * Lt
    Ntr = T - Lt + 1
    X = np.empty((M, Ntr), dtype=np.complex64 if np.iscomplexobj(tile_T_hw) else np.float32)
    for t in range(Lt - 1, T):
        cols = []
        for k in range(Lt):
            cols.append(tile_T_hw[t - k].ravel())
        X[:, t - (Lt - 1)] = np.concatenate(cols).astype(X.dtype, copy=False)

    if center:
        mu = X.mean(axis=1, keepdims=True)
        X -= mu
    return X


# --------------------------- Sample covariance ------------------------------- #


def sample_covariance(X: Array) -> Array:
    """
    Biased sample covariance: R = (1/N) X X^H

    Parameters
    ----------
    X : (M, N) array (complex or real), zero-mean across columns.

    Returns
    -------
    R : (M, M) Hermitian PSD (up to numerical eps).
    """
    if _is_torch_tensor(X):
        if X.ndim != 2:
            raise ValueError("X must be 2D (M, N)")
        M, N = X.shape
        if N < 2:
            raise ValueError("Need at least 2 snapshots.")
        dtype = torch.complex64 if torch.is_complex(X) else torch.float32
        Xh = X.to(dtype=dtype)
        R = (Xh @ Xh.conj().transpose(-2, -1)) / float(N)
        R = 0.5 * (R + R.conj().transpose(-2, -1))
        return make_psd(R, eps=1e-10)

    X_arr = np.asarray(X)
    if X_arr.ndim != 2:
        raise ValueError("X must be 2D (M, N)")
    M, N = X_arr.shape
    if N < 2:
        raise ValueError("Need at least 2 snapshots.")
    dtype = np.complex128 if np.iscomplexobj(X_arr) else np.float64
    Xh = np.asarray(X_arr, dtype=dtype, order="C")
    R_np = (Xh @ Xh.conj().T) / float(N)
    R_np = 0.5 * (R_np + R_np.conj().T)
    return make_psd(R_np, eps=1e-10)


# ---------------------------- Ledoit–Wolf (LW) -------------------------------- #


def _lw_pi_hat(X: Array, R: Array) -> float:
    """
    pi_hat = (1/N) * sum_i || x_i x_i^H - R ||_F^2  (complex-safe)
    X: (M,N), R: (M,M)
    """
    if _is_torch_tensor(X):
        M, N = X.shape
        acc = torch.zeros((), dtype=torch.float64, device=X.device)
        for i in range(N):
            xi = X[:, i : i + 1]
            outer = xi @ xi.conj().transpose(-2, -1)
            diff = outer - R
            acc = acc + torch.sum(diff.abs() ** 2)
        return float((acc / float(N)).item())

    X_arr = np.asarray(X)
    R_arr = np.asarray(R)
    M, N = X_arr.shape
    acc = 0.0
    for i in range(N):
        xi = X_arr[:, i : i + 1]  # (M,1)
        outer = xi @ xi.conj().T  # (M,M)
        diff = outer - R_arr
        acc += float(np.sum(np.abs(diff) ** 2))
    return acc / float(N)


def _fro_sq(A: Array) -> float:
    if _is_torch_tensor(A):
        return float(torch.sum(A.abs() ** 2).item())
    return float(np.sum(np.abs(A) ** 2))


def ledoit_wolf_shrinkage(R: Array, X: Array | None = None) -> tuple[Array, float, float, float]:
    """
    Ledoit–Wolf linear shrinkage towards mu I:
        R_alpha = (1 - alpha) R + alpha * mu I

    alpha* = pi_hat / gamma_hat, clamped to [0,1]
    where gamma_hat = ||R - mu I||_F^2

    If X is provided (M,N), we compute pi_hat exactly (O(N M^2)).
    Otherwise we default to a conservative alpha=0.1.

    Returns
    -------
    R_alpha, alpha, pi_hat, gamma_hat
    """
    if _is_torch_tensor(R):
        M = R.shape[0]
        mu = torch.real(torch.trace(R)) / M
        F = mu * torch.eye(M, dtype=R.dtype, device=R.device)
        gamma_hat = _fro_sq(R - F) + EPS

        if X is not None:
            pi_hat = _lw_pi_hat(X, R)
            alpha = float(np.clip(pi_hat / gamma_hat, 0.0, 1.0))
        else:
            pi_hat = float("nan")
            alpha = 0.1

        R_alpha = (1.0 - alpha) * R + alpha * F
        R_alpha = 0.5 * (R_alpha + R_alpha.conj().transpose(-2, -1))
        return R_alpha, alpha, pi_hat, gamma_hat

    R_arr = np.asarray(R)
    M = R_arr.shape[0]
    mu = (np.trace(R_arr).real) / M
    F = mu * np.eye(M, dtype=R_arr.dtype)
    gamma_hat = _fro_sq(R_arr - F) + EPS

    if X is not None:
        pi_hat = _lw_pi_hat(X, R_arr)
        alpha = float(np.clip(pi_hat / gamma_hat, 0.0, 1.0))
    else:
        pi_hat = float("nan")
        alpha = 0.1

    R_alpha = (1.0 - alpha) * R_arr + alpha * F
    R_alpha = 0.5 * (R_alpha + R_alpha.conj().T)
    return R_alpha, alpha, pi_hat, gamma_hat


# ------------------------------ KA prior ------------------------------------- #


def ka_prior(
    h: int,
    w: int,
    Lt: int,
    ell: float,
    rho_t: float,
    pix_spacing: float,
    trace_target: float,
    dtype=np.complex64,
) -> Array:
    """
    Knowledge-aided prior R0 = Rs kron Rt (Kronecker product).

    Rs_ij = exp(-||p_i - p_j||^2 / (2 ell^2)),  p_i 2D pixel coords (meters)
    Rt_ab = rho_t^{|a-b|}

    The result is scaled so tr(R0) == trace_target.
    """
    # Spatial kernel
    yy, xx = np.mgrid[0:h, 0:w]
    coords = np.stack([yy, xx], axis=-1).reshape(-1, 2).astype(np.float64)
    coords *= float(pix_spacing)
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=-1)  # (hw, hw)
    Rs = np.exp(-0.5 * d2 / max(ell**2, 1e-12)).astype(np.float64)

    # Temporal AR(1)
    t = np.arange(Lt, dtype=np.int64)
    Rt = (rho_t ** np.abs(t[:, None] - t[None, :])).astype(np.float64)

    R0 = np.kron(Rs, Rt).astype(np.float64)
    # Scale to match trace
    tr = float(np.trace(R0))
    if tr <= 0:
        raise ValueError("KA prior trace is non-positive, check parameters.")
    R0 *= trace_target / tr
    R0 = R0.astype(dtype, copy=False)
    # Ensure Hermitian
    R0 = 0.5 * (R0 + R0.conj().T)
    return R0


# -------------------------- PSD & Condition number --------------------------- #


def make_psd(R: Array, eps: float = 1e-8) -> Array:
    """Ensure PSD by shifting the diagonal if the smallest eigenvalue is slightly negative."""
    if _is_torch_tensor(R):
        herm = 0.5 * (R + R.conj().transpose(-2, -1))
        w = torch.linalg.eigvalsh(herm)
        lam_min = float(torch.min(torch.real(w)).item())
        if lam_min < eps:
            shift = float(eps - lam_min) * 1.01
            herm = herm + shift * torch.eye(R.shape[0], dtype=R.dtype, device=R.device)
        return 0.5 * (herm + herm.conj().transpose(-2, -1))

    herm_np = 0.5 * (R + R.conj().T)
    w = np.linalg.eigvalsh(herm_np)
    lam_min = float(w.min())
    if lam_min < eps:
        shift = float(eps - lam_min) * 1.01
        herm_np = herm_np + (shift * np.eye(R.shape[0], dtype=R.dtype))
    return 0.5 * (herm_np + herm_np.conj().T)


def hermitian_cond(R: Array, eps: float = 1e-12) -> float:
    if _is_torch_tensor(R):
        herm = 0.5 * (R + R.conj().transpose(-2, -1))
        w = torch.linalg.eigvalsh(herm)
        w = torch.clamp(torch.real(w), min=eps)
        numerator = float((torch.max(w) + eps).item())
        denominator = float((torch.min(w) + eps).item())
        return numerator / denominator

    herm_np = 0.5 * (R + R.conj().T)
    w = np.linalg.eigvalsh(herm_np)
    w = np.maximum(w, eps)
    return float((w.max() + eps) / (w.min() + eps))


# --------------------------- High-level assembly ----------------------------- #


@dataclass(frozen=True)
class CovarianceAssembly:
    R: Array  # final blended covariance (PSD Hermitian)
    R_hat: Array  # sample covariance
    R_alpha: Array  # LW-shrunk covariance
    R0: Array  # KA prior (scaled)
    alpha: float  # LW shrinkage coefficient
    beta: float  # KA blend coefficient
    cond_R: float  # condition number of R
    M: int  # dimension
    Ntr: int  # number of snapshots


def assemble_covariance(
    tile_T_hw: Array,
    Lt: int,
    lw_enable: bool = True,
    beta_ka: float = 0.2,
    ell: float = 0.002,
    rho_t: float = 0.95,
    pix_spacing: float = 0.0002,
    estimator: str = "scm",
    huber_c: float = 5.0,
    device: Optional[str] = None,
) -> CovarianceAssembly:
    """
    End-to-end covariance construction for one tile.

    Steps
    -----
    1) Build snapshots X (M x Ntr), centered.
    2) Sample covariance R_hat = (1/N) X X^H.
    3) LW shrinkage -> R_alpha.
    4) KA prior R0 (scaled to match tr(R_alpha)).
    5) Blend: R = (1 - beta) R_alpha + beta R0, then ensure PSD.
    """
    X = build_snapshots(tile_T_hw, Lt=Lt, center=True)
    if device is not None and torch is None:
        raise RuntimeError("PyTorch is required for GPU covariance assembly")

    est = (estimator or "scm").lower()
    if est not in {"scm", "huber", "tyler", "tyler_pca", "tyler-pca", "none"}:
        raise ValueError(f"Unknown covariance estimator '{estimator}'")
    if est in {"scm", "none"}:
        method = "scm"
    elif est in {"tyler_pca", "tyler-pca"}:
        method = "tyler_pca"
    else:
        method = est

    use_torch = device is not None and torch is not None
    dev = torch.device(device) if use_torch else None

    if use_torch:
        torch_dtype = torch.complex64 if np.iscomplexobj(X) else torch.float32
        X_used = torch.as_tensor(X, dtype=torch_dtype, device=dev)
        M, Ntr = X_used.shape
        R_hat, _ = robust_covariance(
            X_used,
            method=method,
            huber_c=huber_c,
            max_iter=100,
            tol=1e-4,
            trace_target=None,
        )
    else:
        M, Ntr = X.shape
        R_hat, _ = robust_covariance(
            X,
            method=method,
            huber_c=huber_c,
            max_iter=100,
            tol=1e-4,
            trace_target=None,
        )
        X_used = X

    if lw_enable:
        R_alpha, alpha, _, _ = ledoit_wolf_shrinkage(R_hat, X_used)
    else:
        R_alpha = R_hat.clone() if _is_torch_tensor(R_hat) else R_hat.copy()
        alpha = 0.0

    if _is_torch_tensor(R_alpha):
        trace_target = float(torch.real(torch.trace(R_alpha)).item())
    else:
        trace_target = float(np.trace(R_alpha).real)
    if _is_torch_tensor(R_alpha):
        dtype_np = np.complex64 if torch.is_complex(R_alpha) else np.float32
    else:
        dtype_np = R_alpha.dtype

    R0 = ka_prior(
        h=tile_T_hw.shape[1],
        w=tile_T_hw.shape[2],
        Lt=Lt,
        ell=ell,
        rho_t=rho_t,
        pix_spacing=pix_spacing,
        trace_target=trace_target,
        dtype=dtype_np,
    )

    if use_torch:
        R0 = torch.as_tensor(R0, dtype=R_alpha.dtype, device=dev)

    beta = float(np.clip(beta_ka, 0.0, 1.0))
    R = (1.0 - beta) * R_alpha + beta * R0
    R = make_psd(R, eps=1e-8)
    cond = hermitian_cond(R, eps=1e-9)

    return CovarianceAssembly(
        R=R, R_hat=R_hat, R_alpha=R_alpha, R0=R0, alpha=alpha, beta=beta, cond_R=cond, M=M, Ntr=Ntr
    )
