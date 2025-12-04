from __future__ import annotations

import math

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore


def _to_numpy(array_like) -> np.ndarray:
    """Convert torch tensor or array-like to numpy complex array."""
    if torch is not None and isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def _orthonormal_basis(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return an orthonormal basis for column span of mat."""
    if mat.size == 0:
        return np.zeros((mat.shape[0], 0), dtype=mat.dtype)
    q, r = np.linalg.qr(mat, mode="reduced")
    diag = np.abs(np.diag(r))
    keep = diag > eps
    if not np.any(keep):
        return np.zeros((mat.shape[0], 0), dtype=mat.dtype)
    return q[:, keep]


def projected_flow_alignment(Cf, flow_vector, *, eps: float = 1e-12) -> float:
    """
    Compute ||P_Cf m|| / ||m|| where Cf is the design flow basis and m is an estimated flow vector.
    Returns cosine in [0,1].
    """
    Cf_np = _to_numpy(Cf)
    m_np = _to_numpy(flow_vector).reshape(-1)
    if m_np.size == 0:
        return 0.0
    basis = _orthonormal_basis(Cf_np)
    if basis.size == 0:
        return 0.0
    proj = basis @ (basis.conj().T @ m_np)
    num = np.linalg.norm(proj)
    den = np.linalg.norm(m_np) + eps
    if den <= eps:
        return 0.0
    cos_val = float(np.clip(num / den, 0.0, 1.0))
    return cos_val


def principal_angle(Cf, Cm, *, degrees: bool = True, eps: float = 1e-12) -> float:
    """
    Return the smallest principal angle between spans of Cf and Cm.
    Degenerate (empty) subspaces return 90 degrees (or pi/2 radians).
    """
    Cf_np = _to_numpy(Cf)
    Cm_np = _to_numpy(Cm)
    Qf = _orthonormal_basis(Cf_np, eps=eps)
    Qm = _orthonormal_basis(Cm_np, eps=eps)
    if Qf.shape[1] == 0 or Qm.shape[1] == 0:
        return 90.0 if degrees else math.pi / 2
    gram = Qf.conj().T @ Qm
    sv = np.linalg.svd(gram, compute_uv=False)
    if sv.size == 0:
        return 90.0 if degrees else math.pi / 2
    sigma_max = np.clip(np.max(np.abs(sv)), 0.0, 1.0)
    angle = math.acos(float(sigma_max))
    if degrees:
        angle = math.degrees(angle)
    return angle


__all__ = ["projected_flow_alignment", "principal_angle"]
