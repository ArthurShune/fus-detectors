import math

import numpy as np
import torch

from pipeline.stap.temporal import _generalized_band_metrics, _mixing_metric


def _make_projectors():
    """Build simple orthoprojectors for Lt=8 in the canonical basis."""
    Lt = 8
    eye = torch.eye(Lt, dtype=torch.complex128)
    # Flow: bins 2 and 6
    Pf = eye[:, [2, 6]] @ eye[:, [2, 6]].conj().transpose(-2, -1)
    # Alias: bins 3 and 5
    Pa = eye[:, [3, 5]] @ eye[:, [3, 5]].conj().transpose(-2, -1)
    Po = eye - Pf - Pa
    return Pf, Pa, Po


def _make_r_beta(
    R_hat: torch.Tensor, Pf: torch.Tensor, Pa: torch.Tensor, Po: torch.Tensor, sa: float
):
    """Construct R_beta = R_hat^{1/2} M R_hat^{1/2} with M band-block."""
    sf = 1.0
    so = 1.0
    inv_sf2 = 1.0 / (sf * sf)
    inv_sa2 = 1.0 / (sa * sa)
    inv_so2 = 1.0 / (so * so)
    M = inv_sf2 * Pf + inv_sa2 * Pa + inv_so2 * Po
    M = 0.5 * (M + M.conj().transpose(-2, -1))
    evals, evecs = torch.linalg.eigh(R_hat)
    evals_clamped = torch.clamp(evals.real, min=1e-6)
    Rhsqrt = (evecs * evals_clamped.sqrt()) @ evecs.conj().transpose(-2, -1)
    R_beta = Rhsqrt @ M @ Rhsqrt
    R_beta = 0.5 * (R_beta + R_beta.conj().transpose(-2, -1))
    return R_beta


def test_oc2_identity_hat():
    Lt = 8
    Pf, Pa, Po = _make_projectors()
    R_hat = torch.eye(Lt, dtype=torch.complex128)
    sa = 1.35

    R_beta = _make_r_beta(R_hat, Pf, Pa, Po, sa=sa)
    metrics = _generalized_band_metrics(R_hat, R_beta, Pf, Pa=Pa)
    mixing = _mixing_metric(R_hat, R_beta, Pf)

    assert metrics["pf_min"] >= 0.999
    assert abs(metrics["pf_mean"] - 1.0) < 1e-6
    expected_alias = sa * sa
    assert math.isclose(metrics["alias_mean"], expected_alias, rel_tol=1e-6, abs_tol=1e-6)
    assert metrics["noise_mean"] is not None
    assert abs(metrics["noise_mean"] - 1.0) < 1e-6
    assert mixing < 1e-6


def test_oc2_random_hat():
    torch.manual_seed(0)
    Lt = 8
    Pf, Pa, Po = _make_projectors()
    # Random SPD: A A^H + 0.1 I
    A = torch.randn(Lt, Lt, dtype=torch.complex128)
    R_hat = A @ A.conj().transpose(-2, -1) + 0.1 * torch.eye(Lt, dtype=torch.complex128)
    sa = 1.35

    R_beta = _make_r_beta(R_hat, Pf, Pa, Po, sa=sa)
    metrics = _generalized_band_metrics(R_hat, R_beta, Pf, Pa=Pa)
    mixing = _mixing_metric(R_hat, R_beta, Pf)

    expected_alias = sa * sa
    assert metrics["pf_min"] >= 0.98
    assert abs(metrics["pf_mean"] - 1.0) < 2e-2
    assert abs(metrics["alias_mean"] - expected_alias) < 2.5e-1
    assert metrics["noise_mean"] is not None
    assert abs(metrics["noise_mean"] - 1.0) < 5e-2
    assert mixing < 1e-1
