import torch

from pipeline.stap.temporal import ka_blend_covariance_temporal


def test_lambda_override_respected_and_conditioned_lambda_bounded():
    torch.manual_seed(1)
    Lt = 4
    A = torch.randn(Lt, Lt, dtype=torch.complex64)
    R = (A @ A.conj().T).real + 1e-3 * torch.eye(Lt)
    R0 = 0.6 * R + 0.4 * torch.eye(Lt, dtype=torch.complex64)

    override = 5e-2
    _, details = ka_blend_covariance_temporal(
        R_sample=R,
        R0_prior=R0,
        beta=0.25,
        lambda_override=override,
        kappa_target=20.0,
        device="cpu",
    )
    assert abs(details["lambda_used"] - override) < 1e-8

    # Without override, conditioned lambda must not exceed target-implied bound.
    R_blend, details_auto = ka_blend_covariance_temporal(
        R_sample=R,
        R0_prior=R0,
        beta=0.25,
        lambda_override=None,
        kappa_target=20.0,
        device="cpu",
    )
    lam_auto = details_auto["lambda_used"]
    assert lam_auto >= 0.0
    # Ensure resulting conditioning meets target.
    cond_val = torch.linalg.cond(R_blend).item()
    assert cond_val <= 20.0 * 1.05
