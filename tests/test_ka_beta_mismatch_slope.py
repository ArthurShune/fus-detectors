import numpy as np
import torch

from pipeline.stap.temporal import ka_blend_covariance_temporal


def _make_rotated_prior(R: np.ndarray, angle: float) -> np.ndarray:
    """Rotate covariance in a 2-D subspace by `angle` radians."""
    Lt = R.shape[0]
    rot = np.eye(Lt, dtype=np.float64)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rot[0, 0] = cos_a
    rot[0, 1] = -sin_a
    rot[1, 0] = sin_a
    rot[1, 1] = cos_a
    return rot @ R @ rot.T


def test_auto_beta_monotonically_decreases_with_mismatch():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    Lt = 5
    eigs = np.geomspace(1.0, 0.2, Lt)
    Q, _ = np.linalg.qr(rng.standard_normal((Lt, Lt)))
    R_sample = (Q * eigs) @ Q.T

    angles = [0.0, 0.2, 0.5, 0.8]
    betas = []
    mismatches = []
    for ang in angles:
        R0 = _make_rotated_prior(R_sample, ang) + 1e-6 * np.eye(Lt)
        _, details = ka_blend_covariance_temporal(
            R_sample=torch.as_tensor(R_sample, dtype=torch.complex64),
            R0_prior=torch.as_tensor(R0, dtype=torch.complex64),
            Cf_flow=None,
            beta=None,
            alpha=None,
            device="cpu",
        )
        betas.append(details["beta"])
        mismatches.append(details["mismatch_beta_metric"])

    # Mismatch should increase with angle and beta should decrease.
    assert all(
        mismatches[i] <= mismatches[i + 1] + 1e-6 for i in range(len(mismatches) - 1)
    ), f"Mismatch should increase with angle: {mismatches}"
    assert all(
        betas[i] >= betas[i + 1] - 1e-6 for i in range(len(betas) - 1)
    ), f"Auto beta should decrease with prior mismatch: {betas}"
