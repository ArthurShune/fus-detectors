import numpy as np
import pytest

from pipeline.stap.temporal import bandpass_constraints_temporal

torch = pytest.importorskip("torch")  # noqa: E305


def test_bandpass_constraints_symmetry_and_center():
    Lt = 5
    prf = 3000.0
    grid = np.array([-600.0, -300.0, 0.0, 300.0, 600.0], dtype=np.float64)
    C = bandpass_constraints_temporal(Lt, prf, grid, device="cpu")

    assert C.shape == (Lt, grid.size)
    center = C[:, grid.size // 2].cpu().numpy()
    assert np.allclose(center, center[0])
    assert np.isclose(np.abs(center[0]), 1.0 / np.sqrt(Lt), atol=1e-6)

    for k in range(grid.size // 2):
        left = C[:, k].cpu().numpy()
        right = C[:, -(k + 1)].cpu().numpy()
        assert np.allclose(left, np.conj(right))


def test_bandpass_constraints_empty_grid_returns_zero_columns():
    Lt = 4
    prf = 2500.0
    C = bandpass_constraints_temporal(Lt, prf, [], device="cpu")
    assert C.shape == (Lt, 0)
