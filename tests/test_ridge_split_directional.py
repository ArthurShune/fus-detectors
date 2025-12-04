import numpy as np
import pytest

try:
    import torch
except ImportError:  # pragma: no cover - guard for environments without torch
    torch = None

if torch is None:  # pragma: no cover - skip entire module if torch missing
    pytest.skip("torch not available", allow_module_level=True)

from pipeline.stap.temporal import (
    bandpass_constraints_temporal,
    projector_from_tones,
    apply_ridge_split,
    choose_beta_directional,
)


def _rand_hermitian(Lt: int, scale: float = 1.0, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((Lt, Lt)) + 1j * rng.standard_normal((Lt, Lt))
    A = A.astype(np.complex64)
    H = (A + A.conj().T) * 0.5
    # add diagonal to keep PSD-ish
    H += (scale * Lt) * np.eye(Lt, dtype=np.complex64)
    return torch.as_tensor(H, dtype=torch.complex64)


def test_ridge_split_retains_passband():
    Lt = 6
    prf = 3000.0
    Cf = bandpass_constraints_temporal(
        Lt, prf, np.array([-600.0, 0.0, 600.0], dtype=np.float64), device="cpu"
    )
    Pf = projector_from_tones(Cf)
    Rhat = _rand_hermitian(Lt, scale=1e-3, seed=1)
    R0 = _rand_hermitian(Lt, scale=1e-3, seed=2)

    lam = 0.02
    Rb_global = (0.8 * Rhat + 0.2 * R0) + lam * torch.eye(Lt, dtype=torch.complex64)
    Rb_split = apply_ridge_split((0.8 * Rhat + 0.2 * R0), Pf, lam)

    # Passband retention: split keeps the original passband trace (within 1%)
    tr_f_hat = torch.real(torch.trace(Pf @ (0.8 * Rhat + 0.2 * R0)))
    tr_f_split = torch.real(torch.trace(Pf @ Rb_split))
    tr_f_global = torch.real(torch.trace(Pf @ Rb_global))
    assert torch.allclose(tr_f_split, tr_f_hat, rtol=1e-2, atol=1e-6)
    # Global ridge inflates the passband because it adds λI; split leaves it unchanged
    assert tr_f_global > tr_f_split


def test_choose_beta_directional_meets_targets():
    Lt = 6
    prf = 3000.0
    Cf = bandpass_constraints_temporal(
        Lt, prf, np.array([-450.0, 0.0, 450.0], dtype=np.float64), device="cpu"
    )
    Pf = projector_from_tones(Cf)
    Rhat = _rand_hermitian(Lt, scale=1e-3, seed=10)
    # Make prior slightly under the sample off-band, comparable in-band
    R0 = 0.9 * Rhat + 0.1 * torch.eye(Lt, dtype=torch.complex64)

    beta, info = choose_beta_directional(
        Rhat, R0, Pf, beta_max=0.25, target_retain_f=0.90, target_shrink_perp=0.95
    )
    assert 0.0 <= beta <= 0.25
    assert info["retain_f"] >= 0.90 - 1e-3
    assert info["shrink_perp"] <= 0.95 + 1e-3
