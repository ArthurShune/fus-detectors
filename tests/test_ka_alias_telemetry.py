import torch
import pytest
from pipeline.stap.temporal import _generalized_band_metrics, projector_from_tones


def test_generalized_band_metrics_with_alias() -> None:
    """
    Verify that _generalized_band_metrics correctly distinguishes between
    Alias and Noise bands when Pa is provided.
    """
    Lt = 10
    # Create orthogonal subspaces
    # Flow: indices 0, 1
    # Alias: indices 2, 3
    # Noise: indices 4..9
    eye = torch.eye(Lt, dtype=torch.complex128)
    Pf = torch.zeros_like(eye)
    Pf[0, 0] = 1.0
    Pf[1, 1] = 1.0

    Pa = torch.zeros_like(eye)
    Pa[2, 2] = 1.0
    Pa[3, 3] = 1.0

    # Create a covariance matrix R_beta where:
    # Flow energy = 1.0
    # Alias energy = 2.0 (Inflated)
    # Noise energy = 1.0
    R_beta = torch.zeros_like(eye)
    R_beta[0, 0] = 1.0
    R_beta[1, 1] = 1.0
    R_beta[2, 2] = 2.0
    R_beta[3, 3] = 2.0
    for i in range(4, Lt):
        R_beta[i, i] = 1.0

    # R_sample is identity (baseline)
    R_sample = eye.clone()

    # 1. Call WITHOUT Pa (Legacy behavior)
    # Perp should mix Alias (2.0) and Noise (1.0) in the covariance.
    # The metric computes eigenvalues of R_beta^{-1} R_sample.
    # Alias: 1.0 / 2.0 = 0.5
    # Noise: 1.0 / 1.0 = 1.0
    metrics_legacy = _generalized_band_metrics(R_sample, R_beta, Pf)

    assert metrics_legacy["pf_mean"] == pytest.approx(1.0)
    # Perp max should see the Noise (1.0)
    assert metrics_legacy["perp_max"] == pytest.approx(1.0)
    # Perp min should see the Alias suppression (0.5)
    assert metrics_legacy["perp_min"] == pytest.approx(0.5)

    # 2. Call WITH Pa (New behavior)
    metrics_new = _generalized_band_metrics(R_sample, R_beta, Pf, Pa=Pa)

    assert metrics_new["pf_mean"] == pytest.approx(1.0)

    # Alias metrics
    # R_beta is 2.0, R_sample is 1.0 -> metric is 0.5
    assert "alias_mean" in metrics_new
    assert metrics_new["alias_mean"] == pytest.approx(0.5)

    # Noise metrics
    # R_beta is 1.0, R_sample is 1.0 -> metric is 1.0
    assert "noise_mean" in metrics_new
    assert metrics_new["noise_mean"] == pytest.approx(1.0)

    # Legacy perp metrics should still be present
    assert metrics_new["perp_max"] == pytest.approx(1.0)
    assert metrics_new["perp_min"] == pytest.approx(0.5)
