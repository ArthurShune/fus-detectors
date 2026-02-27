from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore

from sim.kwave.common import _baseline_pd_mcsvd, _fft_shift_apply, _register_stack_phasecorr_torch


@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="CUDA not available")
def test_phasecorr_torch_registration_recovers_static_scene() -> None:
    rng = np.random.default_rng(42)
    T, H, W = 16, 64, 64
    yy, xx = np.mgrid[0:H, 0:W]
    mag = (
        2.0 * np.exp(-((yy - 20) ** 2 + (xx - 32) ** 2) / 120.0)
        + 1.3 * np.exp(-((yy - 45) ** 2 + (xx - 18) ** 2) / 80.0)
    ).astype(np.float32)
    phase = (0.2 * rng.standard_normal((H, W))).astype(np.float32)
    base_frame = (mag * np.exp(1j * phase)).astype(np.complex64)
    clean_stack = np.tile(base_frame, (T, 1, 1))
    noisy = clean_stack + 0.01 * (
        rng.standard_normal((T, H, W)) + 1j * rng.standard_normal((T, H, W))
    ).astype(np.complex64)
    shifted = np.empty_like(noisy)
    for t in range(T):
        dy = 0.4 * np.sin(2 * np.pi * t / T)
        dx = -0.6 * np.cos(2 * np.pi * t / T)
        shifted[t] = _fft_shift_apply(noisy[t], dy, dx)

    cube_t = torch.as_tensor(shifted, dtype=torch.complex64, device="cuda")
    reg_t, tele = _register_stack_phasecorr_torch(
        cube_t, reg_enable=True, upsample=4, ref_strategy="median"
    )
    reg = reg_t.detach().cpu().numpy()
    rel_err = float(np.linalg.norm(reg - noisy) / (np.linalg.norm(noisy) + 1e-12))
    assert rel_err < 1.2e-1
    assert tele["reg_enable"] is True
    assert tele["reg_failed_fraction"] <= 1e-3
    assert tele.get("reg_backend") == "torch"


@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="CUDA not available")
def test_mcsvd_torch_matches_numpy_without_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(0)
    T, H, W = 24, 32, 32
    cube = (rng.standard_normal((T, H, W)) + 1j * rng.standard_normal((T, H, W))).astype(
        np.complex64
    )

    pd_cpu, _tele_cpu = _baseline_pd_mcsvd(
        cube,
        reg_enable=False,
        reg_reference="first",
        reg_subpixel=1,
        svd_rank=3,
        device="cpu",
    )

    monkeypatch.setenv("MC_SVD_TORCH", "1")
    pd_gpu, tele_gpu = _baseline_pd_mcsvd(
        cube,
        reg_enable=False,
        reg_reference="first",
        reg_subpixel=1,
        svd_rank=3,
        device="cuda",
    )
    np.testing.assert_allclose(pd_gpu, pd_cpu, rtol=1e-3, atol=1e-5)
    assert tele_gpu.get("svd_backend") == "torch"


@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="CUDA not available")
def test_mcsvd_torch_can_return_filtered_cube_on_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(1)
    T, H, W = 16, 24, 24
    cube = (rng.standard_normal((T, H, W)) + 1j * rng.standard_normal((T, H, W))).astype(
        np.complex64
    )

    monkeypatch.setenv("MC_SVD_TORCH", "1")
    monkeypatch.setenv("MC_SVD_TORCH_RETURN_CUBE", "1")
    pd, tele, cube_f = _baseline_pd_mcsvd(
        cube,
        reg_enable=False,
        reg_reference="first",
        reg_subpixel=1,
        svd_rank=2,
        device="cuda",
        return_filtered_cube=True,
    )

    assert pd.shape == (H, W)
    assert np.isfinite(pd).all()
    assert tele.get("baseline_type") == "mc_svd"
    assert isinstance(cube_f, torch.Tensor)
    assert cube_f.is_cuda
    assert cube_f.dtype == torch.complex64
    assert tuple(int(x) for x in cube_f.shape) == (T, H, W)

