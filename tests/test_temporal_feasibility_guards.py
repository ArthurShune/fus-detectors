import numpy as np
import torch

from pipeline.stap.temporal import (
    conditioned_lambda,
    msd_contrast_score_batched,
    build_motion_basis_temporal,
)


def _random_spd(dim: int, eps: float = 1e-4) -> torch.Tensor:
    torch.manual_seed(0)
    A = torch.randn(dim, dim, dtype=torch.complex64)
    R = (A @ A.conj().transpose(-2, -1)).real
    R += eps * torch.eye(dim, dtype=torch.float32)
    return R.to(torch.complex64)


def test_conditioned_lambda_meets_target():
    R = _random_spd(4)
    lam, _, _ = conditioned_lambda(R, kappa_target=25.0, safety_factor=1.1)
    R_loaded = R + lam * torch.eye(4, dtype=torch.complex64)
    kappa = torch.linalg.cond(R_loaded).item()
    assert kappa <= 25.0 * 1.05, f"conditioned lambda failed kappa bound: {kappa}"


def test_flow_grid_capped_by_motion_rank():
    Lt = 6
    prf = 3000.0
    R = torch.eye(Lt, dtype=torch.complex64) * 1.0
    N = 16
    hw = (2, 2)
    torch.manual_seed(1)
    S = torch.randn(Lt, N, hw[0], hw[1], dtype=torch.complex64)
    fd = np.linspace(-1200.0, 1200.0, 15)

    motion_basis = build_motion_basis_temporal(
        Lt=Lt,
        prf_hz=prf,
        width_bins=1,
        include_dc=True,
        device="cpu",
        dtype=torch.complex64,
    )
    motion_rank = motion_basis.shape[1]
    expected_max = max(1, Lt - motion_rank)

    _, _, _, details = msd_contrast_score_batched(
        R,
        S,
        prf_hz=prf,
        fd_grid_hz=fd,
        motion_half_span_hz=400.0,
        lam_abs=1e-2,
        ridge=0.1,
        ratio_rho=0.1,
        contrast_alpha=0.7,
        return_details=True,
        device="cpu",
    )

    assert (
        details["kc_flow"] <= expected_max
    ), f"flow grid size {details['kc_flow']} exceeds Lt - motion_rank = {expected_max}"


def test_msd_contrast_fallback_when_flow_absent():
    Lt = 4
    prf = 2000.0
    R = torch.eye(Lt, dtype=torch.complex64)
    N = 8
    S = torch.zeros(Lt, N, 1, 1, dtype=torch.complex64)
    fd = np.linspace(-800.0, 800.0, 7)

    _, _, _, details = msd_contrast_score_batched(
        R,
        S,
        prf_hz=prf,
        fd_grid_hz=fd,
        motion_half_span_hz=200.0,
        lam_abs=1e-2,
        ridge=0.1,
        ratio_rho=0.05,
        contrast_alpha=0.7,
        return_details=True,
        device="cpu",
    )

    assert details["fallback"] is True
    assert details["score_mode"] == "msd"
