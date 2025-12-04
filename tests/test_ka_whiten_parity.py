import numpy as np
import pytest
import torch

from pipeline.stap.temporal import (
    aggregate_over_snapshots,
    bandpass_constraints_temporal,
    build_temporal_hankels_and_cov,
    ka_prior_temporal_from_psd,
    msd_snapshot_energies_batched,
)


def _ratio_map(
    R_t,
    S,
    Cf,
    *,
    lam_abs,
    ridge,
    R0_prior=None,
    Cf_flow=None,
    ka_opts=None,
    device="cpu",
):
    opts = ka_opts
    if opts is None and R0_prior is not None:
        opts = {"kappa_target": 40.0}
    ratio_rho = 0.05
    T_band, sw_pow = msd_snapshot_energies_batched(
        R_t,
        S,
        Cf,
        lam_abs=lam_abs,
        ridge=ridge,
        ratio_rho=ratio_rho,
        R0_prior=R0_prior,
        Cf_flow=Cf_flow,
        ka_opts=opts,
        device=device,
    )
    ratio_snap = T_band / torch.clamp(sw_pow - T_band + ratio_rho * sw_pow, min=1e-10)
    ratio = aggregate_over_snapshots(ratio_snap, mode="mean")
    return ratio.cpu().numpy()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ka_whitened_cpu_gpu_parity() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    T, H, W, Lt, prf = 48, 18, 18, 4, 3000.0
    y0, x0 = 7, 7
    t = torch.arange(T, dtype=torch.float32) / prf
    cube = (
        0.10 * (torch.randn(T, H, W) + 1j * torch.randn(T, H, W))
        + 0.05 * torch.exp(1j * 2 * np.pi * 60.0 * t)[:, None, None]
    ).to(torch.complex64)
    cube[:, y0, x0] += 0.10 * torch.exp(1j * 2 * np.pi * 600.0 * t)

    S_cpu, R_cpu, _ = build_temporal_hankels_and_cov(
        cube.numpy(),
        Lt=Lt,
        estimator="huber",
        huber_c=5.0,
        device="cpu",
        dtype=torch.complex64,
    )
    Cf_cpu = bandpass_constraints_temporal(
        Lt, prf, np.array([-600.0, -300.0, 0.0, 300.0, 600.0], dtype=np.float64), device="cpu"
    )
    R0_cpu = ka_prior_temporal_from_psd(
        Lt=Lt, prf_hz=prf, f_peaks_hz=(0.0,), width_bins=1, add_deriv=True, device="cpu"
    )
    score_cpu = _ratio_map(
        torch.as_tensor(R_cpu, dtype=torch.complex64),
        torch.as_tensor(S_cpu, dtype=torch.complex64),
        Cf_cpu,
        lam_abs=0.0,
        ridge=0.20,
        R0_prior=R0_cpu,
        Cf_flow=Cf_cpu,
        device="cpu",
    )

    Cf_gpu = Cf_cpu.to("cuda")
    R0_gpu = ka_prior_temporal_from_psd(
        Lt=Lt, prf_hz=prf, f_peaks_hz=(0.0,), width_bins=1, add_deriv=True, device="cuda"
    )
    score_gpu = _ratio_map(
        torch.as_tensor(R_cpu, dtype=torch.complex64, device="cuda"),
        torch.as_tensor(S_cpu, dtype=torch.complex64, device="cuda"),
        Cf_gpu,
        lam_abs=0.0,
        ridge=0.20,
        R0_prior=R0_gpu,
        Cf_flow=Cf_gpu,
        device="cuda",
    )

    diff = np.max(np.abs(score_cpu - score_gpu))
    assert diff <= 5e-4, f"CPU/GPU mismatch too large ({diff:.2e})"
