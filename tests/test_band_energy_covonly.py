import pytest

try:
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore


@pytest.mark.skipif(torch is None, reason="torch not available")
def test_band_energy_covonly_matches_snapshot_mean_cpu():
    """
    Sprint 3 guard:

    The covariance-only band-energy path should match the mean of the per-snapshot
    band-energy computed by `_band_energy_whitened_batched` (for the same inputs).
    """
    from pipeline.stap.temporal import (
        _band_energy_whitened_batched,
        _band_energy_whitened_covonly_batched,
    )
    from pipeline.stap.temporal_shared import build_temporal_hankels_batch

    torch.manual_seed(0)
    B, T, h, w = 3, 12, 4, 4
    Lt = 5
    K = 6

    cube = torch.randn(B, T, h, w, dtype=torch.float32) + 1j * torch.randn(
        B, T, h, w, dtype=torch.float32
    )
    cube = cube.to(dtype=torch.complex64)

    S, R_scm = build_temporal_hankels_batch(cube, Lt, center=True, device="cpu", dtype=torch.complex64)
    R_hat = R_scm.clone()
    Ct = torch.randn(Lt, K, dtype=torch.float32) + 1j * torch.randn(Lt, K, dtype=torch.float32)
    Ct = Ct.to(dtype=torch.complex64)
    lam = torch.full((B,), 0.05, dtype=torch.float32)

    T_band, sw_pow = _band_energy_whitened_batched(
        R_hat,
        S,
        Ct,
        lam,
        ridge=0.10,
        device="cpu",
        dtype=torch.complex64,
        eps=1e-10,
    )
    T_mean = T_band.mean(dim=(1, 2, 3))
    sw_mean = sw_pow.mean(dim=(1, 2, 3))

    T_cov, sw_cov = _band_energy_whitened_covonly_batched(
        R_hat,
        R_scm,
        Ct,
        lam,
        ridge=0.10,
        device="cpu",
        dtype=torch.complex64,
        eps=1e-10,
    )

    torch.testing.assert_close(T_cov, T_mean.to(dtype=T_cov.dtype), rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(sw_cov, sw_mean.to(dtype=sw_cov.dtype), rtol=3e-4, atol=3e-5)

