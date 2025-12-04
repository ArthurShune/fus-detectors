# tests/test_detector.py

import numpy as np

from pipeline.detect.cfar import (
    moving_average_pd,
    robust_null_stats_from_pd,
    score_cafar,
    score_from_null_mean,
)


def test_moving_average_shapes_and_values():
    T, H, W = 10, 4, 5
    x = np.arange(T * H * W, dtype=np.float32).reshape(T, H, W)
    y = moving_average_pd(x, W=3)  # shape (8, H, W)
    assert y.shape == (T - 3 + 1, H, W)
    # For a ramp, the moving average increases monotonically across time.
    assert np.all(y[1] > y[0])


def test_score_from_null_mean_exponential_null():
    rng = np.random.default_rng(0)
    T, H, W = 2000, 6, 7
    # Exponential PD under null with mean 1.0
    pd_null = rng.exponential(scale=1.0, size=(T, H, W)).astype(np.float32)
    stats = robust_null_stats_from_pd([pd_null], blocks=10)
    # New sequence from same distribution
    pd_new = rng.exponential(scale=1.0, size=(T, H, W)).astype(np.float32)
    S = score_from_null_mean(pd_new, stats, W=8)  # (T-7, H, W)
    m = float(S.mean())
    assert 0.9 < m < 1.1  # near 1 on average


def test_cafar_constant_field_yields_one():
    PD = np.ones((32, 32), dtype=np.float32)
    S = score_cafar(PD, guard=2, train=6)
    # Due to boundary effects with reflect mode, values should be ~1
    assert np.allclose(S, 1.0, atol=1e-5)
