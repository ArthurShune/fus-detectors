import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pipeline.stap.temporal import (
    build_motion_basis_temporal,
    build_temporal_hankels_and_cov,
    project_out_motion_whitened,
)


def test_motion_null_reduces_dc_energy():
    """Whitening + motion projection should diminish low-frequency energy."""
    rng = np.random.default_rng(0)
    T, h, w, Lt = 64, 2, 2, 4
    prf_hz = 3000.0

    cube = (0.05 * (rng.standard_normal((T, h, w)) + 1j * rng.standard_normal((T, h, w)))).astype(
        np.complex64
    )
    # Inject strong DC component on a single pixel.
    cube[:, 0, 0] += 0.3 + 0j

    S, Rt, _ = build_temporal_hankels_and_cov(cube, Lt=Lt, device="cpu")
    motion_basis = build_motion_basis_temporal(Lt, prf_hz, device="cpu")

    Sw, L = project_out_motion_whitened(Rt, S, motion_basis, lam_abs=3e-2, device="cpu")
    S_flat = torch.as_tensor(S.reshape(Lt, -1))
    Sw_flat = torch.as_tensor(Sw.reshape(Lt, -1))
    motion_w = torch.linalg.solve_triangular(L, motion_basis, upper=False)
    U, _ = torch.linalg.qr(motion_w, mode="reduced")
    pre_coeff = U.conj().transpose(-2, -1) @ torch.linalg.solve_triangular(L, S_flat, upper=False)
    post_coeff = U.conj().transpose(-2, -1) @ Sw_flat

    assert torch.linalg.norm(post_coeff) < torch.linalg.norm(pre_coeff)
