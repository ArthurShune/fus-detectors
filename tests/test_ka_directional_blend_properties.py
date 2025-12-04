import numpy as np
import pytest

try:
    import torch
except ImportError:  # pragma: no cover - guard for envs without torch
    torch = None

if torch is None:  # pragma: no cover
    pytest.skip("torch not available", allow_module_level=True)

from pipeline.stap.temporal import (
    bandpass_constraints_temporal,
    ka_blend_covariance_temporal,
    projector_from_tones,
)


def _rand_psd(Lt: int, scale: float, seed: int) -> torch.Tensor:
    """Return a well-conditioned Hermitian SPD matrix of size Lt x Lt."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((Lt, Lt)) + 1j * rng.standard_normal((Lt, Lt))
    H = A.conj().T @ A  # Hermitian PSD
    # Scale to keep magnitudes reasonable and add a ridge to ensure PD
    H = H / float(Lt) + (scale * Lt + 1e-2) * np.eye(Lt, dtype=np.complex64)
    return torch.as_tensor(H.astype(np.complex64))


def _trace_in_block(R: torch.Tensor, P: torch.Tensor) -> float:
    return float(torch.real(torch.trace(P @ R @ P.conj().transpose(-2, -1))).item())


def test_directional_blend_preserves_passband_and_shrinks_offband():
    Lt = 6
    prf = 3000.0
    # Use a simple flow band around +/- 450 Hz
    Cf = bandpass_constraints_temporal(
        Lt=Lt, prf_hz=prf, fd_grid_hz=np.array([-450.0, 0.0, 450.0]), device="cpu"
    )
    Pf = projector_from_tones(Cf.to(torch.complex128)).to(torch.complex64)
    eye = torch.eye(Lt, dtype=torch.complex64)
    Pperp = eye - Pf

    # Random sample/prior
    R_hat = _rand_psd(Lt, scale=1e-3, seed=2)
    # Make prior a bit lighter off-band on average
    R0 = 0.9 * R_hat + 0.1 * torch.eye(Lt, dtype=torch.complex64)

    # Blend with directional settings; use split ridge to avoid passband inflation
    R_loaded, info = ka_blend_covariance_temporal(
        R_sample=R_hat,  # (Lt,Lt)
        R0_prior=R0,
        Cf_flow=Cf.detach().cpu().numpy(),
        alpha=0.0,
        beta=None,
        beta_bounds=(0.0, 0.3),
        mismatch_tau=1.0,
        kappa_target=40.0,
        add_noise_floor_frac=0.0,
        lambda_override=None,
        beta_directional=True,
        target_retain_f=0.95,
        target_shrink_perp=0.90,
        beta_directional_strict=True,
        ridge_split=True,
        lambda_override_split=0.01,
        device="cpu",
        dtype=torch.complex64,
    )

    assert isinstance(R_loaded, torch.Tensor)
    R_loaded = 0.5 * (R_loaded + R_loaded.conj().transpose(-2, -1))

    # Compare against a baseline with the same split ridge applied to the sample
    lam_used = float(info.get("lambda_used", 0.01))
    R_base = 0.5 * (R_hat + lam_used * Pperp + (R_hat + lam_used * Pperp).conj().transpose(-2, -1))

    # Passband not inflated vs. baseline
    tr_f_base = _trace_in_block(R_base, Pf)
    tr_f_loaded = _trace_in_block(R_loaded, Pf)
    assert tr_f_loaded <= tr_f_base * (1.0 + 1e-3)

    # Off-band shrunk (within tolerance) vs. baseline
    tr_p_base = _trace_in_block(R_base, Pperp)
    tr_p_loaded = _trace_in_block(R_loaded, Pperp)
    assert tr_p_loaded <= tr_p_base * (1.0 + 1e-3)

    # Telemetry retains expected directional metrics
    assert info.get("beta_directional", False) is True
    assert info.get("directional_strict", False) is True
    if info.get("retain_f_total") is not None:
        assert info["retain_f_total"] <= 1.0 + 1e-3
        assert info["retain_f_total"] >= 0.90 - 1e-3
    # We enforce shrink at the loaded level vs. baseline; pre-load metric may differ


def test_directional_blend_improves_passband_quadratic_form():
    """
    For random passband vectors s, s^H (R_loaded)^{-1} s should not decrease
    compared to s^H (R_baseline)^{-1} s when using off-band shrink with split ridge.
    """
    Lt = 6
    prf = 3000.0
    Cf = bandpass_constraints_temporal(
        Lt=Lt, prf_hz=prf, fd_grid_hz=np.array([-600.0, 0.0, 600.0]), device="cpu"
    )
    Pf = projector_from_tones(Cf.to(torch.complex128)).to(torch.complex64)
    eye = torch.eye(Lt, dtype=torch.complex64)
    Pperp = eye - Pf

    R_hat = _rand_psd(Lt, scale=1e-3, seed=10)
    R0 = 0.9 * R_hat + 0.1 * torch.eye(Lt, dtype=torch.complex64)

    lam = 0.02
    R_loaded, info = ka_blend_covariance_temporal(
        R_sample=R_hat,
        R0_prior=R0,
        Cf_flow=Cf.detach().cpu().numpy(),
        alpha=0.0,
        beta=None,
        beta_bounds=(0.0, 0.3),
        mismatch_tau=1.0,
        kappa_target=40.0,
        add_noise_floor_frac=0.0,
        lambda_override=None,
        beta_directional=True,
        target_retain_f=0.95,
        target_shrink_perp=0.90,
        beta_directional_strict=True,
        ridge_split=True,
        lambda_override_split=lam,
        device="cpu",
        dtype=torch.complex64,
    )

    lam_used = float(info.get("lambda_used", lam))
    R_base = R_hat + lam_used * Pperp

    R_base = 0.5 * (R_base + R_base.conj().transpose(-2, -1))
    R_loaded = 0.5 * (R_loaded + R_loaded.conj().transpose(-2, -1))

    # Generate random passband vectors and compare quadratic forms
    rng = np.random.default_rng(0)
    for _ in range(16):
        # Random coefficients on Cf columns; project strictly into Pf subspace
        a = rng.standard_normal(Cf.shape[-1]) + 1j * rng.standard_normal(Cf.shape[-1])
        s_raw = (Cf @ torch.as_tensor(a, dtype=torch.complex64)).to(torch.complex64)
        s = Pf @ s_raw
        # Normalize to avoid scale issues
        s = s / (torch.linalg.norm(s) + 1e-12)
        # Evaluate via NumPy (complex128) for robust positive definiteness
        Rb_np = R_base.detach().cpu().to(torch.complex128).numpy()
        Rl_np = R_loaded.detach().cpu().to(torch.complex128).numpy()
        s_np = s.detach().cpu().to(torch.complex128).numpy()
        v_base = np.real(s_np.conj().T @ np.linalg.solve(Rb_np, s_np))
        v_loaded = np.real(s_np.conj().T @ np.linalg.solve(Rl_np, s_np))
        assert v_loaded >= v_base - 1e-8
