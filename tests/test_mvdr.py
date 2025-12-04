# tests/test_mvdr.py
import numpy as np
import torch

from pipeline.gpu.linalg import to_tensor
from pipeline.stap.mvdr import mvdr_weights


def _rand_spd_hermitian(M: int, scale: float = 1.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    A = (rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))).astype(np.complex64)
    R = A @ A.conj().T / M
    R += scale * np.eye(M, dtype=np.complex64)  # ensure PD
    return R


def test_mvdr_kkt_and_variance_single():
    M = 48
    R = _rand_spd_hermitian(M, scale=1e-3, seed=1)
    rng = np.random.default_rng(2)
    s = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex64)

    res = mvdr_weights(
        R,
        s,
        diag_load=1e-3,
        jitter_init=1e-10,
        device="cpu",
        dtype=torch.complex64,
    )
    w = res.w.squeeze(0).cpu().numpy()

    # KKT / distortionless: sᴴ w ≈ 1
    val = (s.conj().T @ w).item()
    assert np.allclose(val, 1.0 + 0j, atol=1e-5)

    # MVDR variance <= variance of any feasible baseline (e.g., normalized s)
    w0 = s / (s.conj().T @ s)
    var_mvdr = np.real((w.conj().T @ (R @ w)).item())
    var_w0 = np.real((w0.conj().T @ (R @ w0)).item())
    assert var_mvdr <= var_w0 * 1.0001


def test_mvdr_batched_shapes_and_constraints():
    B, M = 3, 32
    Rs = np.stack([_rand_spd_hermitian(M, scale=1e-4, seed=10 + b) for b in range(B)], axis=0)
    rng = np.random.default_rng(0)
    s = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex64)

    res = mvdr_weights(Rs, s, diag_load=1e-3, device="cpu", dtype=torch.complex64)
    w = res.w  # (B, M)
    assert w.shape == (B, M)

    # Distortionless per batch
    s_t = to_tensor(s, device="cpu", dtype=torch.complex64).expand(B, -1)
    vals = torch.sum(s_t.conj() * w, dim=-1)
    assert torch.allclose(vals, torch.ones_like(vals), atol=1e-5)


def test_mvdr_robust_on_ill_conditioned():
    M = 40
    # Ill-conditioned diagonal spectrum
    sv = 10.0 ** np.linspace(0, -8, M)  # spans 1 -> 1e-8
    R = (sv[:, None] * np.eye(M)).astype(np.complex64)
    rng = np.random.default_rng(5)
    s = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex64)

    # With diagonal loading, MVDR should still produce finite weights and satisfy constraint
    res = mvdr_weights(
        R,
        s,
        diag_load=1e-2,
        jitter_init=1e-10,
        device="cpu",
        dtype=torch.complex64,
    )
    w = res.w.squeeze(0)
    val = torch.sum(to_tensor(s, "cpu", torch.complex64).conj() * w).item()
    assert np.isfinite(val)
    assert abs(val - 1.0) < 5e-4

    # Variance must be non-negative
    R_t = to_tensor(R, "cpu", torch.complex64)
    var = torch.matmul(w.conj(), torch.matmul(R_t, w)).real.item()
    assert var >= -1e-6
