import numpy as np
import pytest
import torch

from pipeline.stap.temporal import (
    _project_out_flow_from_prior,
    bandpass_constraints_temporal,
    ka_blend_covariance_temporal,
    ka_prior_temporal_from_psd,
)


def _condition_number(mat: torch.Tensor) -> float:
    herm = (mat + mat.conj().transpose(-2, -1)) * 0.5
    ev = torch.linalg.eigvalsh(herm).real
    ev = torch.clamp(ev, min=1e-12)
    return float((ev.max() / ev.min()).item())


@pytest.mark.parametrize(
    "device",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
def test_ka_conditioning_enforced(device: str) -> None:
    torch.manual_seed(0)
    Lt = 4
    vec = torch.randn(Lt, dtype=torch.float32, device=device)
    vec = (vec / (vec.norm() + 1e-9)).to(torch.complex64)
    Rt = 50.0 * (vec[:, None] @ vec[None, :].conj()) + 1e-6 * torch.eye(
        Lt, dtype=torch.complex64, device=device
    )
    R0 = ka_prior_temporal_from_psd(
        Lt=Lt,
        prf_hz=3000.0,
        f_peaks_hz=(0.0,),
        width_bins=1,
        add_deriv=True,
        device=device,
    )

    R_lam, tel = ka_blend_covariance_temporal(
        R_sample=Rt,
        R0_prior=R0,
        Cf_flow=None,
        alpha=None,
        beta=None,
        kappa_target=40.0,
        device=device,
    )

    assert _condition_number(R_lam) <= 40.0 * 1.05
    assert 0.0 <= tel["beta"] <= 0.5
    assert tel["kappa_target"] == 40.0
    assert tel["lambda_used"] >= 0.0
    assert tel["sigma_min_raw"] >= 0.0
    assert tel["sigma_max_raw"] >= tel["sigma_min_raw"]


def test_ka_projects_flow_band_from_prior() -> None:
    torch.manual_seed(0)
    device = "cpu"
    Lt = 4
    prf = 3000.0
    R0 = ka_prior_temporal_from_psd(
        Lt=Lt,
        prf_hz=prf,
        f_peaks_hz=(0.0,),
        width_bins=1,
        add_deriv=True,
        device=device,
    )
    fd = np.array([-600.0, 0.0, 600.0], dtype=np.float32)
    Cf = bandpass_constraints_temporal(
        Lt=Lt,
        prf_hz=prf,
        fd_grid_hz=fd,
        device=device,
        dtype=torch.complex64,
    )
    overlap_raw = torch.linalg.norm(Cf.conj().T @ R0 @ Cf, ord="fro").item()
    R0_proj = _project_out_flow_from_prior(R0, Cf)
    overlap_proj = torch.linalg.norm(Cf.conj().T @ R0_proj @ Cf, ord="fro").item()
    assert overlap_proj <= 0.2 * overlap_raw + 1e-9


def test_ka_auto_beta_tracks_mismatch() -> None:
    torch.manual_seed(0)
    device = "cpu"
    Lt = 4
    prf = 3000.0
    R0_match = ka_prior_temporal_from_psd(
        Lt=Lt,
        prf_hz=prf,
        f_peaks_hz=(0.0,),
        width_bins=1,
        add_deriv=True,
        device=device,
    )
    vec = torch.ones(Lt, dtype=torch.complex64, device=device) / np.sqrt(Lt)
    Rt = 10.0 * (vec[:, None] @ vec[None, :].conj()) + 1e-3 * torch.eye(
        Lt, dtype=torch.complex64, device=device
    )
    _, tel_match = ka_blend_covariance_temporal(
        R_sample=Rt,
        R0_prior=R0_match,
        Cf_flow=None,
        beta=None,
        device=device,
    )

    Q, _ = torch.linalg.qr(torch.randn(Lt, Lt, dtype=torch.float32))
    R0_mismatch = Q.to(torch.complex64) @ R0_match @ Q.to(torch.complex64).conj().T
    _, tel_mismatch = ka_blend_covariance_temporal(
        R_sample=Rt,
        R0_prior=R0_mismatch,
        Cf_flow=None,
        beta=None,
        device=device,
    )

    assert tel_match["mismatch"] < tel_mismatch["mismatch"]
    assert tel_match["beta"] >= tel_mismatch["beta"]


def test_ka_updated_mode_reports_snr_and_identity() -> None:
    torch.manual_seed(0)
    device = "cpu"
    Lt = 4
    Rt = torch.eye(Lt, dtype=torch.complex64, device=device) * 2.0
    R0 = ka_prior_temporal_from_psd(
        Lt=Lt,
        prf_hz=3000.0,
        f_peaks_hz=(0.0,),
        width_bins=1,
        add_deriv=True,
        device=device,
    )
    Cf = torch.eye(Lt, dtype=torch.complex64, device=device)

    _, tel = ka_blend_covariance_temporal(
        R_sample=Rt,
        R0_prior=R0,
        Cf_flow=Cf.cpu().numpy(),
        feasibility_mode="updated",
        device=device,
    )

    assert tel["lambda_strategy"] == "identity"
    assert tel["ridge_split"] is False
    assert tel.get("snr_flow_ratio") is not None
    assert tel.get("noise_perp_ratio") is not None
    assert tel.get("feasibility_mode") == "updated"


def test_repair_band_metrics_shrinks_alias() -> None:
    torch.manual_seed(0)
    Lt = 4
    Pf = torch.zeros((Lt, Lt), dtype=torch.complex64)
    Pf[0, 0] = 1.0
    Pf[1, 1] = 1.0
    R_sample = torch.eye(Lt, dtype=torch.complex64)
    R_beta = torch.eye(Lt, dtype=torch.complex64)
    R_beta[2:, 2:] = 3.0 * torch.eye(2, dtype=torch.complex64)

    from pipeline.stap.temporal import _repair_band_metrics

    R_fixed, beta_out, stats = _repair_band_metrics(
        R_sample.to(torch.complex128),
        R_beta.to(torch.complex128),
        Pf.to(torch.complex128),
        beta_init=0.2,
    )

    assert stats["perp_lambda_max"] <= 1.10
    assert beta_out <= 0.2


def test_ka_commuting_band_blend_reports_metrics() -> None:
    torch.manual_seed(0)
    device = "cpu"
    Lt = 8
    prf = 3000.0
    # Build a random sample covariance
    X = torch.randn(Lt, Lt, dtype=torch.complex64, device=device)
    Rt = X @ X.conj().transpose(-2, -1) + 0.2 * torch.eye(Lt, dtype=torch.complex64)
    R0 = ka_prior_temporal_from_psd(
        Lt=Lt,
        prf_hz=prf,
        f_peaks_hz=(0.0,),
        width_bins=1,
        add_deriv=True,
        device=device,
    )
    # Flow at bin 1 (~187.5 Hz), alias at bin 3 (~562.5 Hz)
    Cf_flow = (
        bandpass_constraints_temporal(
            Lt=Lt,
            prf_hz=prf,
            fd_grid_hz=[prf / Lt],
            device=device,
            dtype=torch.complex64,
        )
        .cpu()
        .numpy()
    )
    Cf_alias = (
        bandpass_constraints_temporal(
            Lt=Lt,
            prf_hz=prf,
            fd_grid_hz=[3 * (prf / Lt)],
            device=device,
            dtype=torch.complex64,
        )
        .cpu()
        .numpy()
    )

    _, tel = ka_blend_covariance_temporal(
        R_sample=Rt,
        R0_prior=R0,
        Cf_flow=Cf_flow,
        Cf_alias=Cf_alias,
        feasibility_mode="updated",
        device=device,
    )

    # Metrics should be populated and alias gain should exceed flow
    assert tel.get("pf_lambda_min") is not None
    assert tel.get("pf_lambda_max") is not None
    assert tel.get("ka_alias_lambda_mean") is not None
    assert tel.get("ka_noise_lambda_mean") is not None
    assert tel.get("mixing_epsilon") is not None
    assert tel["mixing_epsilon"] < 5e-2
    assert tel["ka_alias_lambda_mean"] > tel["pf_lambda_mean"]
