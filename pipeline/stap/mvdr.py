# pipeline/stap/mvdr.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ..gpu.linalg import (
    cholesky_solve_hermitian,
    get_device,
    hermitianize,
    quadratic_form,
    to_tensor,
)

# -------------------------- Steering construction ---------------------------- #


def steering_vector(
    h: int,
    w: int,
    Lt: int,
    fd_hz: float,
    prf_hz: float,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """
    Construct s = kron(s_t, s_s) consistent with snapshot stacking:
    X[:, t] = [vec(frame_t), vec(frame_{t-1}), ..., vec(frame_{t-Lt+1})]

    - s_s = ones(h*w)
    - s_t[m] = exp(j*2π*fd*(m)/PRF), m=0..Lt-1

    Returns
    -------
    s : (M=h*w*Lt,) complex tensor on device
    """
    M_s = h * w
    s_s = torch.ones((M_s,), dtype=dtype, device=get_device(device))
    m = torch.arange(Lt, device=get_device(device), dtype=torch.float32)
    omega = 2.0 * np.pi * float(fd_hz) / max(float(prf_hz), 1e-9)
    omega_tensor = torch.as_tensor(omega, dtype=torch.float32, device=get_device(device))
    s_t = torch.exp(1j * omega_tensor * m).to(dtype)
    s = torch.kron(s_t, s_s)  # (Lt*M_s,)
    return s


# ------------------------------ MVDR weights --------------------------------- #


@dataclass(frozen=True)
class MVDRResult:
    w: torch.Tensor  # (..., M)
    den: torch.Tensor  # (...,) sᴴ R⁻¹ s (complex ~ real)
    jitter_used: float
    diag_load: float  # applied as diag_load * mu * I per batch


def mvdr_weights(
    R: torch.Tensor | np.ndarray,
    s: torch.Tensor | np.ndarray,
    diag_load: float = 1e-3,
    jitter_init: float = 1e-8,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> MVDRResult:
    """
    Compute diagonal-loaded MVDR/Capon weights:
        w = R⁻¹ s / (sᴴ R⁻¹ s)

    Inputs can be numpy or torch. Supports batched R (..., M, M) and s (..., M).
    If s is 1D and R is batched, s is broadcast to each batch.

    Parameters
    ----------
    diag_load : float
        Diagonal loading coefficient (relative): add diag_load * mu * I, mu=tr(R)/M.

    Returns
    -------
    MVDRResult
    """
    # Convert inputs
    Rt = to_tensor(R, device=device)
    if dtype is not None:
        Rt = Rt.to(dtype)
    # Default dtype
    if dtype is None and Rt.dtype in (torch.float32, torch.float64):
        dtype = Rt.dtype
    elif dtype is None:
        dtype = Rt.dtype
    st = to_tensor(s, device=device, dtype=dtype)

    # Normalize shapes to batched
    if Rt.dim() == 2:
        Rt = Rt.unsqueeze(0)  # (1, M, M)
    if st.dim() == 1:
        st = st.unsqueeze(0)  # (1, M)
    if st.shape[0] != Rt.shape[0]:
        # broadcast s to all batches
        if st.shape[0] == 1:
            st = st.expand(Rt.shape[0], -1)
        else:
            raise ValueError("Batch size mismatch between R and s.")

    Rt = hermitianize(Rt)
    *batch, M, _ = Rt.shape
    device = Rt.device
    identity = torch.eye(M, dtype=Rt.dtype, device=device)

    # Diagonal loading scaled by mu per batch
    mu = torch.real(torch.diagonal(Rt, dim1=-2, dim2=-1).sum(-1)) / float(M)  # (B,)
    Rt_loaded = Rt + (diag_load * mu).reshape(*batch, 1, 1) * identity

    # Solve for u = R⁻¹ s
    u, jit = cholesky_solve_hermitian(Rt_loaded, st, jitter_init=jitter_init, max_tries=6)  # (B,M)
    den = torch.sum(st.conj() * u, dim=-1)  # (B,)

    # Normalize to enforce distortionless constraint
    w = u / den.unsqueeze(-1)

    # Final projection for numerical exactness: w <- w / (sᴴ w)
    c = torch.sum(st.conj() * w, dim=-1, keepdim=True)
    w = w / c

    return MVDRResult(w=w, den=den, jitter_used=jit, diag_load=diag_load)


# -------------------------- Apply weights / PD ------------------------------- #


def apply_weights_to_snapshots(
    w: torch.Tensor | np.ndarray,
    X: torch.Tensor | np.ndarray,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Apply wᴴ to snapshots X to form y_t = wᴴ x_t.

    Shapes
    ------
    w : (..., M)
    X : (..., M, N)   (same batch leading dims), N = # snapshots/time indices

    Returns
    -------
    y : (..., N)
    """
    wt = to_tensor(w, device=device, dtype=dtype)
    Xt = to_tensor(X, device=device, dtype=wt.dtype)
    # Align dims
    if wt.dim() == Xt.dim() - 1:
        wt = wt.unsqueeze(-2)  # (..., 1, M)
    elif wt.dim() != Xt.dim():
        raise ValueError("w and X must have compatible leading dims.")
    y = torch.matmul(wt.conj(), Xt).squeeze(-2)  # (..., N)
    return y


def output_variance(w: torch.Tensor | np.ndarray, R: torch.Tensor | np.ndarray) -> torch.Tensor:
    """
    Return wᴴ R w (real, >= 0) for diagnostic comparisons.
    """
    wt = to_tensor(w)
    Rt = to_tensor(R, dtype=wt.dtype, device=wt.device)
    return quadratic_form(wt, Rt)
