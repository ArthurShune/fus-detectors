import numpy as np
import pytest
import torch

from pipeline.stap.temporal import (
    build_temporal_hankels_and_cov,
    ka_blend_covariance_temporal,
    ka_prior_temporal_from_psd,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ka_whitener_cpu_vs_gpu_parity() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    T, h, w, Lt = 32, 10, 10, 4
    prf = 3000.0
    t = torch.arange(T, dtype=torch.float32) / prf
    cube = 0.1 * (
        torch.randn(T, h, w, dtype=torch.float32) + 1j * torch.randn(T, h, w, dtype=torch.float32)
    )
    cube[:, 3, 4] += 0.1 * torch.exp(1j * 2 * np.pi * 600.0 * t)
    cube = cube.to(torch.complex64)

    S_cpu, Rt_cpu, _ = build_temporal_hankels_and_cov(
        cube,
        Lt=Lt,
        center=True,
        estimator="huber",
        huber_c=5.0,
        device="cpu",
        dtype=torch.complex64,
    )
    Rt_cpu = Rt_cpu.to(torch.complex64)

    R0_cpu = ka_prior_temporal_from_psd(
        Lt=Lt,
        prf_hz=prf,
        f_peaks_hz=(0.0,),
        width_bins=1,
        add_deriv=True,
        device="cpu",
    )
    Rlam_cpu, tel_cpu = ka_blend_covariance_temporal(
        R_sample=Rt_cpu,
        R0_prior=R0_cpu,
        Cf_flow=None,
        beta=None,
        kappa_target=40.0,
        device="cpu",
    )

    Rt_gpu = Rt_cpu.to("cuda")
    R0_gpu = R0_cpu.to("cuda")
    Rlam_gpu, tel_gpu = ka_blend_covariance_temporal(
        R_sample=Rt_gpu,
        R0_prior=R0_gpu,
        Cf_flow=None,
        beta=None,
        kappa_target=40.0,
        device="cuda",
    )

    herm_cpu = (Rlam_cpu + Rlam_cpu.conj().T) * 0.5
    herm_gpu = (Rlam_gpu + Rlam_gpu.conj().T) * 0.5
    ev_cpu = torch.linalg.eigvalsh(herm_cpu).cpu().numpy()
    ev_gpu = torch.linalg.eigvalsh(herm_gpu).cpu().numpy()

    cond_cpu = float(ev_cpu.max() / max(ev_cpu.min(), 1e-12))
    cond_gpu = float(ev_gpu.max() / max(ev_gpu.min(), 1e-12))
    assert abs(cond_cpu - cond_gpu) <= 1e-3
    assert abs(tel_cpu["lambda_used"] - tel_gpu["lambda_used"]) <= 1e-4
