import torch

from pipeline.stap.temporal import (
    _band_energy_whitened_batched,
    msd_snapshot_energies_batched,
)


def _constraint_matrix(Lt: int, k: int) -> torch.Tensor:
    """Simple orthonormal constraint matrix for tests."""
    eye = torch.eye(Lt, dtype=torch.complex64)
    return eye[:, :k]


def test_msd_snapshot_handles_misordered_snapshots():
    Lt = 8
    N = 5
    h = 6
    w = 6
    Rt = torch.eye(Lt, dtype=torch.complex64)
    # Deliberately misordered: (h, N, w, Lt) instead of (Lt, N, h, w).
    S = torch.randn(h, N, w, Lt, dtype=torch.complex64)
    Ct = _constraint_matrix(Lt, k=2)
    T_band, sw = msd_snapshot_energies_batched(
        Rt, S, Ct, lam_abs=0.01, kappa_target=40.0, device="cpu"
    )
    assert T_band.shape == (N, h, w)
    assert sw.shape == (N, h, w)
    assert torch.isfinite(T_band).all()
    assert torch.isfinite(sw).all()


def test_band_energy_batched_handles_misordered_snapshots():
    Lt = 8
    N = 5
    h = 6
    w = 6
    B = 3
    R = torch.eye(Lt, dtype=torch.complex64).expand(B, Lt, Lt).clone()
    # Misordered: (B, h, N, w, Lt) instead of (B, Lt, N, h, w).
    S = torch.randn(B, h, N, w, Lt, dtype=torch.complex64)
    lam = torch.full((B,), 0.01, dtype=torch.float32)
    Ct = _constraint_matrix(Lt, k=3)
    T_band, sw = _band_energy_whitened_batched(
        R, S, Ct, lam, device="cpu", dtype=torch.complex64
    )
    assert T_band.shape == (B, N, h, w)
    assert sw.shape == (B, N, h, w)
    assert torch.isfinite(T_band).all()
    assert torch.isfinite(sw).all()
