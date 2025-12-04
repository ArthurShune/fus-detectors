import torch

from pipeline.stap.temporal import ka_blend_covariance_temporal, ka_prior_temporal_from_psd


def test_ka_telemetry_fields_present_and_plausible() -> None:
    Lt = 4
    Rt = torch.eye(Lt, dtype=torch.complex64) * 2.0
    R0 = ka_prior_temporal_from_psd(
        Lt=Lt,
        prf_hz=3000.0,
        f_peaks_hz=(0.0,),
        width_bins=1,
        add_deriv=True,
        device="cpu",
    )
    _, tel = ka_blend_covariance_temporal(
        R_sample=Rt,
        R0_prior=R0,
        Cf_flow=None,
        beta=None,
        kappa_target=40.0,
        device="cpu",
    )

    required = [
        "ka_mode",
        "alpha",
        "beta",
        "mismatch",
        "lambda_used",
        "sigma_min_raw",
        "sigma_max_raw",
        "kappa_target",
    ]
    for key in required:
        assert key in tel, f"telemetry missing '{key}'"
    assert tel["ka_mode"] == "blend"
    assert 0.0 <= tel["beta"] <= 0.5
    assert tel["sigma_max_raw"] >= tel["sigma_min_raw"] >= 0.0


def test_ka_alias_noise_metrics_non_null_with_alias_projector() -> None:
    Lt = 4
    Rt = torch.diag(torch.tensor([2.0, 1.5, 0.6, 0.4], dtype=torch.float32)).to(torch.complex64)
    R0 = torch.eye(Lt, dtype=torch.complex64)
    # First two taps define flow, last two alias
    Cf_flow = torch.eye(Lt, dtype=torch.complex64)[:, :1]
    Cf_alias = torch.eye(Lt, dtype=torch.complex64)[:, 1:2]
    _, tel = ka_blend_covariance_temporal(
        R_sample=Rt,
        R0_prior=R0,
        Cf_flow=Cf_flow,
        Cf_alias=Cf_alias,
        beta=0.2,
        device="cpu",
    )
    for key in (
        "ka_alias_lambda_mean",
        "ka_noise_lambda_mean",
        "sample_alias_lambda_mean",
        "sample_noise_lambda_mean",
    ):
        assert tel.get(key) is not None, f"{key} missing"
