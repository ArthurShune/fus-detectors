import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_absolute_load_and_ridge_preserve_band_energy():
    from pipeline.stap.lcmv import bandpass_constraints, lcmv_bandpass_apply

    torch.manual_seed(0)
    M, N = 64, 128
    h = w = 4
    Lt = 4
    prf = 3000.0

    # Ill-conditioned covariance
    U, _ = torch.linalg.qr(torch.randn(M, M, dtype=torch.complex64))
    evals = torch.logspace(0, -9, M, dtype=torch.float32)
    R = (U @ torch.diag(evals).to(torch.complex64) @ U.conj().T).to(torch.complex64)

    fd = np.linspace(-0.45 * prf / Lt, 0.45 * prf / Lt, 11).tolist()
    C = bandpass_constraints(h=h, w=w, Lt=Lt, prf_hz=prf, fd_grid_hz=fd, device="cpu")

    # snapshots with an in-band tone
    tone = C[:, [len(fd) // 2]]
    X = torch.randn(M, N, dtype=torch.complex64)
    X = X + 0.5 * tone @ torch.ones((1, N), dtype=torch.complex64)

    y_scaled, _ = lcmv_bandpass_apply(
        R,
        X,
        C,
        load_mode="scaled",
        diag_load=1e-6,
        constraint_ridge=0.0,
        device="cpu",
    )
    p_scaled = torch.mean(torch.real(y_scaled.conj() * y_scaled)).item()

    y_abs, res = lcmv_bandpass_apply(
        R,
        X,
        C,
        load_mode="absolute",
        diag_load=1e-2,
        constraint_ridge=0.05,
        device="cpu",
    )
    p_abs = torch.mean(torch.real(y_abs.conj() * y_abs)).item()

    assert p_abs >= 0.3 * max(p_scaled, 1e-12)
    assert res.constraint_residual > 0.0
    assert res.load_mode == "absolute"
    cond_raw = torch.linalg.cond(R).item()
    cond_loaded = torch.linalg.cond(R + res.diag_load * torch.eye(M, dtype=torch.complex64)).item()
    assert cond_loaded < cond_raw
    assert p_abs > 1e-6
