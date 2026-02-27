import torch


def _rand_complex(shape, *, device: str, dtype: torch.dtype) -> torch.Tensor:
    rr = torch.randn(shape, device=device, dtype=torch.float32)
    ri = torch.randn(shape, device=device, dtype=torch.float32)
    return (rr + 1j * ri).to(dtype=dtype)


def test_band_energy_project_pf_matches_reference(monkeypatch) -> None:
    from pipeline.stap.temporal import _band_energy_whitened_batched

    torch.manual_seed(0)
    device = "cpu"
    dtype = torch.complex64

    B, Lt, N, h, w, K = 3, 8, 7, 3, 2, 9

    A = _rand_complex((B, Lt, Lt), device=device, dtype=dtype)
    R = A @ A.conj().transpose(-2, -1)
    R = 0.5 * (R + R.conj().transpose(-2, -1))
    R = R + 0.5 * torch.eye(Lt, device=device, dtype=dtype).unsqueeze(0)

    S = _rand_complex((B, Lt, N, h, w), device=device, dtype=dtype)
    Ct = _rand_complex((Lt, K), device=device, dtype=dtype)
    lam_B = torch.full((B,), 0.07, device=device, dtype=torch.float32)

    monkeypatch.delenv("STAP_BAND_PROJECT_MODE", raising=False)
    T_ref, sw_ref = _band_energy_whitened_batched(
        R,
        S,
        Ct,
        lam_B,
        ridge=0.10,
        device=device,
        dtype=dtype,
    )

    monkeypatch.setenv("STAP_BAND_PROJECT_MODE", "pf")
    T_pf, sw_pf = _band_energy_whitened_batched(
        R,
        S,
        Ct,
        lam_B,
        ridge=0.10,
        device=device,
        dtype=dtype,
    )

    torch.testing.assert_close(sw_pf, sw_ref, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(T_pf, T_ref, rtol=1e-4, atol=1e-6)

