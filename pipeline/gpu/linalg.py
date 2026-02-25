# pipeline/gpu/linalg.py
from __future__ import annotations

from typing import Optional, Tuple

try:
    import torch
except Exception:  # pragma: no cover
    torch = None
    raise

# -------------------------- Device / dtype helpers --------------------------- #


def get_device(device: Optional[str] = None) -> "torch.device":
    """Pick CUDA if available unless device is explicitly given."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def to_tensor(
    a,
    device: Optional[str] = None,
    dtype: Optional["torch.dtype"] = None,
) -> "torch.Tensor":
    """Convert numpy/torch input to a torch Tensor on target device/dtype."""
    t = a if isinstance(a, torch.Tensor) else torch.as_tensor(a)
    if dtype is not None:
        t = t.to(dtype)
    return t.to(get_device(device))


def is_complex_dtype(dt: "torch.dtype") -> bool:
    return dt in (torch.complex64, torch.complex128)


def hermitianize(A: "torch.Tensor") -> "torch.Tensor":
    """Return (A + Aᴴ)/2 on the last two dims."""
    return 0.5 * (A + A.conj().transpose(-2, -1))


# ---------------------------- Robust Cholesky -------------------------------- #


def cholesky_robust(
    R: "torch.Tensor",
    jitter_init: float = 1e-8,
    max_tries: int = 6,
) -> Tuple["torch.Tensor", "torch.Tensor", float]:
    """
    Robust batched Cholesky for Hermitian PSD matrices.

    Adds diagonal jitter scaled by mu = tr(R)/M per batch until factorization succeeds.
    Returns the Cholesky factor L and the possibly jittered matrix Rj.

    Shapes
    ------
    R : (..., M, M)

    Returns
    -------
    L : (..., M, M) (lower-triangular)
    Rj: (..., M, M) the matrix actually factored
    jitter_used : float (scale coefficient multiplied by mu*I)
    """
    R = hermitianize(R)
    # Avoid NaN/Inf poisoning of the jitter scale and factorization.
    R = torch.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    *batch, M, _ = R.shape
    device = R.device
    dtype = R.dtype
    # Per-batch scale (real, positive): use mean(|diag|) rather than trace/M.
    # Using trace can be negative under numerical drift, which would apply a
    # *negative* "jitter" and worsen non-PD pathologies.
    diag = torch.real(torch.diagonal(R, dim1=-2, dim2=-1))  # (..., M)
    mu = torch.amax(torch.abs(diag), dim=-1) + 1e-12  # (...,)

    identity = torch.eye(M, dtype=dtype, device=device)
    jitter_dtype = mu.dtype if hasattr(mu, "dtype") else torch.float32
    jitter = torch.tensor(jitter_init, dtype=jitter_dtype, device=device)
    for attempt in range(max_tries):
        if attempt == 0 and jitter_init == 0.0:
            Rj = R
        else:
            load = (jitter * mu).to(dtype)
            if len(batch) == 0:
                Rj = R + load * identity
            else:
                Rj = R + load.reshape(*batch, 1, 1) * identity
        try:
            L = torch.linalg.cholesky(Rj)
            return L, Rj, float(jitter.item())
        except RuntimeError:  # numerical failure, increase jitter
            jitter = jitter * 10.0

    # Last-resort stronger load. If this still fails, fall back to a diagonal
    # proxy (keeps downstream code running deterministically).
    load = (1.0 * mu + 1e-6).to(dtype)
    if len(batch) == 0:
        Rj = R + load * identity
    else:
        Rj = R + load.reshape(*batch, 1, 1) * identity
    try:
        L = torch.linalg.cholesky(Rj)
        return L, Rj, 1.0
    except Exception:
        if len(batch) == 0:
            Rj = load * identity
            L = torch.sqrt(load).to(dtype) * identity
        else:
            Rj = load.reshape(*batch, 1, 1) * identity
            L = torch.sqrt(load).reshape(*batch, 1, 1).to(dtype) * identity
        return L, Rj, 1.0


def cholesky_solve_hermitian(
    R: "torch.Tensor",
    b: "torch.Tensor",
    jitter_init: float = 1e-8,
    max_tries: int = 6,
) -> Tuple["torch.Tensor", float]:
    """
    Solve R x = b for batched Hermitian PSD R via robust Cholesky.

    Shapes
    ------
    R : (..., M, M)
    b : (..., M) or (..., M, K)

    Returns
    -------
    x : (..., M) / (..., M, K)
    jitter_used : float
    """
    L, Rj, jit = cholesky_robust(R, jitter_init=jitter_init, max_tries=max_tries)
    x = torch.cholesky_solve(b.unsqueeze(-1) if b.dim() == R.dim() - 1 else b, L)
    if b.dim() == R.dim() - 1:
        x = x.squeeze(-1)
    return x, jit


# ------------------------------ Diagnostics ---------------------------------- #


def quadratic_form(w: "torch.Tensor", R: "torch.Tensor") -> "torch.Tensor":
    """
    Compute wᴴ R w for single or batched inputs.

    Shapes
    ------
    w : (..., M)
    R : (..., M, M)

    Returns
    -------
    q : (...) real scalar tensor
    """
    wH = w.conj().unsqueeze(-2)  # (..., 1, M)
    tmp = torch.matmul(wH, torch.matmul(R, w.unsqueeze(-1)))  # (..., 1, 1)
    q = torch.real(tmp.squeeze(-1).squeeze(-1))
    return q


def hermitian_cond(R: "torch.Tensor", eps: float = 1e-12) -> "torch.Tensor":
    """
    Rough condition number estimate via eigenvalues for Hermitian R (batched).

    Returns
    -------
    kappa : (...) tensor of condition numbers
    """
    R = hermitianize(R)
    w = torch.linalg.eigvalsh(R)
    w = torch.real(w)
    w = torch.clamp(w, min=eps)
    kappa = (w.max(dim=-1).values + eps) / (w.min(dim=-1).values + eps)
    return kappa
