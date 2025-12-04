import numpy as np
import torch

from pipeline.stap.temporal import equalize_pf_trace


def _rand_spd(Lt: int, scale: float = 1e-3, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((Lt, Lt)) + 1j * rng.standard_normal((Lt, Lt))
    A = A.astype(np.complex64)
    At = torch.as_tensor(A)
    R = At @ At.conj().transpose(-2, -1)
    R = 0.5 * (R + R.conj().transpose(-2, -1)) + scale * torch.eye(Lt, dtype=R.dtype)
    return R


def _make_pf_from_indices(Lt: int, idx: list[int]) -> torch.Tensor:
    C = torch.zeros((Lt, len(idx)), dtype=torch.complex64)
    for k, i in enumerate(idx):
        C[i, k] = 1.0 + 0.0j
    G = C.conj().transpose(-2, -1) @ C
    Pf = C @ torch.linalg.solve(
        G + 1e-6 * torch.eye(G.shape[-1], dtype=G.dtype), C.conj().transpose(-2, -1)
    )
    Pf = 0.5 * (Pf + Pf.conj().transpose(-2, -1))
    return Pf


def test_equalize_pf_trace_matches_pf_energy_and_preserves_complement():
    torch.manual_seed(0)
    Lt = 8
    R_loaded = _rand_spd(Lt, scale=1e-3, seed=1)
    Rt_sample = _rand_spd(Lt, scale=1e-3, seed=2)
    # Simple Pf onto first three coordinates
    Pf = _make_pf_from_indices(Lt, [0, 1, 2])

    R_eq_tuple = equalize_pf_trace(R_loaded, Rt_sample, Pf)
    if isinstance(R_eq_tuple, tuple):
        R_eq = R_eq_tuple[0]
    else:
        R_eq = R_eq_tuple

    def tr_pf(M: torch.Tensor) -> float:
        return float(torch.real(torch.trace(Pf @ M @ Pf)).item())

    tr_eq = tr_pf(R_eq)
    tr_sample = tr_pf(0.5 * (Rt_sample + Rt_sample.conj().transpose(-2, -1)))
    # Exact (within numerical tolerance) equality on Pf trace
    assert abs(tr_eq - tr_sample) / max(abs(tr_sample), 1e-12) < 1e-6

    # Complement block preserved: P_perp (R_eq - R_loaded) P_perp ~ 0
    eye = torch.eye(Lt, dtype=R_loaded.dtype)
    Pp = eye - Pf
    delta_perp = Pp @ (R_eq - R_loaded) @ Pp
    err = float(torch.linalg.norm(delta_perp).item())
    assert err < 1e-6

    # Hermitian and PSD-ish: smallest eigenvalue not << 0
    evals = torch.linalg.eigvalsh(0.5 * (R_eq + R_eq.conj().transpose(-2, -1))).real
    assert float(evals.min().item()) > -1e-7
