import numpy as np
import pytest

from pipeline.stap.temporal import build_temporal_hankels_and_cov

torch = pytest.importorskip("torch")  # noqa: E305


def _make_cube(T: int, h: int, w: int, prf: float, tone_fd: float) -> np.ndarray:
    t = np.arange(T, dtype=np.float64)
    tone = np.exp(1j * 2.0 * np.pi * tone_fd * t / prf)
    cube = 0.05 * (np.random.randn(T, h, w) + 1j * np.random.randn(T, h, w))
    cube[:, h // 2, w // 2] += 0.5 * tone
    return cube.astype(np.complex64)


def test_temporal_covariance_phase_and_scale_invariance():
    T, h, w, Lt = 64, 4, 4, 4
    prf = 3000.0
    cube = _make_cube(T, h, w, prf, tone_fd=450.0)

    _, R1, _ = build_temporal_hankels_and_cov(
        cube, Lt=Lt, estimator="huber", huber_c=5.0, device="cpu"
    )

    cube_phase = cube * np.exp(1j * 0.37)
    _, R2, _ = build_temporal_hankels_and_cov(
        cube_phase, Lt=Lt, estimator="huber", huber_c=5.0, device="cpu"
    )
    assert np.allclose(R1.cpu(), R2.cpu(), rtol=1e-5, atol=1e-7)

    cube_scaled = cube * 3.0
    _, R3, _ = build_temporal_hankels_and_cov(
        cube_scaled, Lt=Lt, estimator="huber", huber_c=5.0, device="cpu"
    )
    # scaling by alpha should scale the covariance by |alpha|^2
    assert np.allclose(R3.cpu(), (3.0**2) * R1.cpu(), rtol=1e-5, atol=1e-7)

    # principal eigenvector invariant (up to global phase)
    v1 = torch.linalg.eigh(R1)[1][:, -1]
    v3 = torch.linalg.eigh(R3)[1][:, -1]
    corr = abs(torch.dot(torch.conj(v1), v3) / (torch.linalg.norm(v1) * torch.linalg.norm(v3)))
    assert float(corr) > 0.999
