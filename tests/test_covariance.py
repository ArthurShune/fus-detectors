# tests/test_covariance.py
import numpy as np

from pipeline.stap.covariance import (
    assemble_covariance,
    build_snapshots,
    hermitian_cond,
    ledoit_wolf_shrinkage,
    sample_covariance,
)


def _random_complex(shape, rng):
    x = rng.standard_normal(shape)
    y = rng.standard_normal(shape)
    return (x + 1j * y).astype(np.complex64)


def test_snapshots_and_covariance_psd():
    rng = np.random.default_rng(0)
    T, h, w, Lt = 32, 8, 8, 4
    tile = _random_complex((T, h, w), rng)
    X = build_snapshots(tile, Lt=Lt, center=True)
    R = sample_covariance(X)
    # Hermitian check
    np.testing.assert_allclose(R, R.conj().T, rtol=1e-6, atol=1e-6)
    # PSD (allow tiny negative due to numeric)
    w = np.linalg.eigvalsh(R)
    assert w.min() > -1e-7


def test_ledoit_wolf_reduces_condition_on_ill_conditioned():
    rng = np.random.default_rng(1)
    M, N = 64, 256
    # Construct ill-conditioned X = A @ Z with diagonal A
    sv = 10.0 ** np.linspace(0, 4, M)  # cond ~ 1e4
    A = np.diag(sv).astype(np.complex64)
    Z = _random_complex((M, N), rng)
    X = A @ Z
    R = sample_covariance(X)
    R_alpha, alpha, _, _ = ledoit_wolf_shrinkage(R, X)
    cond_R = hermitian_cond(R)
    cond_A = hermitian_cond(R_alpha)
    assert alpha > 0.0
    assert cond_A < cond_R  # shrinkage should improve conditioning


def test_ka_prior_trace_and_assembly():
    rng = np.random.default_rng(2)
    T, h, w, Lt = 40, 8, 8, 3
    tile = _random_complex((T, h, w), rng)
    asm = assemble_covariance(
        tile, Lt=Lt, lw_enable=True, beta_ka=0.2, ell=0.0015, rho_t=0.9, pix_spacing=0.0002
    )
    # Hermitian PSD
    np.testing.assert_allclose(asm.R, asm.R.conj().T, rtol=1e-6, atol=1e-6)
    w = np.linalg.eigvalsh(asm.R)
    assert w.min() > -1e-7
    # Trace preservation after KA scaling/blend
    tr_alpha = np.trace(asm.R_alpha).real
    tr_R = np.trace(asm.R).real
    # Because of blending (convex combo), traces should be equal to tr_alpha
    np.testing.assert_allclose(tr_R, tr_alpha, rtol=1e-6, atol=1e-6)
    # Condition number sane
    assert asm.cond_R < 1e6
