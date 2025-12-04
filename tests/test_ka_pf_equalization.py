import numpy as np
import torch

from pipeline.stap.temporal import (
    ka_blend_covariance_temporal,
)


def _rand_spd(Lt: int, scale: float = 1e-3, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((Lt, Lt)) + 1j * rng.standard_normal((Lt, Lt))
    A = A.astype(np.complex64)
    At = torch.as_tensor(A)
    R = At @ At.conj().transpose(-2, -1)
    R = 0.5 * (R + R.conj().transpose(-2, -1)) + scale * torch.eye(Lt, dtype=R.dtype)
    return R


def test_pf_trace_equalization_flag_preserves_flow_energy():
    torch.manual_seed(0)
    Lt = 6
    R = _rand_spd(Lt, scale=1e-3, seed=1)
    R0 = _rand_spd(Lt, scale=1e-3, seed=2)
    # Use a simple, well-conditioned projector onto the first two coordinates
    Cf = torch.zeros((Lt, 2), dtype=torch.complex64)
    Cf[0, 0] = 1.0 + 0.0j
    Cf[1, 1] = 1.0 + 0.0j

    # With flag disabled (default), do not force exact equality (allow small mismatch)
    R_loaded_noeq, _ = ka_blend_covariance_temporal(
        R_sample=R,
        R0_prior=R0,
        Cf_flow=Cf,
        beta_directional=True,
        ridge_split=True,
        device="cpu",
        dtype=torch.complex64,
    )

    # With flag enabled, enforce Pf-trace equality
    R_loaded_eq, _ = ka_blend_covariance_temporal(
        R_sample=R,
        R0_prior=R0,
        Cf_flow=Cf,
        beta_directional=True,
        ridge_split=True,
        equalize_pf_trace=True,
        device="cpu",
        dtype=torch.complex64,
    )

    # Construct Pf projector
    G = Cf.conj().transpose(-2, -1) @ Cf
    Pf = Cf @ torch.linalg.solve(
        G + 1e-6 * torch.eye(G.shape[-1], dtype=G.dtype), Cf.conj().transpose(-2, -1)
    )
    Pf = 0.5 * (Pf + Pf.conj().transpose(-2, -1))
    Rt = 0.5 * (R + R.conj().transpose(-2, -1))

    def tr_pf(M: torch.Tensor) -> float:
        return float(torch.real(torch.trace(Pf @ M @ Pf.conj().transpose(-2, -1))).item())

    tr_sample = tr_pf(Rt)
    tr_noeq = tr_pf(R_loaded_noeq)
    tr_eq = tr_pf(R_loaded_eq)

    # The equalized version should reduce the Pf-trace mismatch relative to no-equalization
    err_noeq = abs(tr_noeq - tr_sample) / max(abs(tr_sample), 1e-12)
    err_eq = abs(tr_eq - tr_sample) / max(abs(tr_sample), 1e-12)
    assert np.isfinite(err_noeq) and np.isfinite(err_eq)
    # Require at least 10x reduction or absolute <= 1e-3
    assert (err_eq <= err_noeq * 0.1) or (err_eq < 1e-3)
