import math

import numpy as np

from pipeline.stap.geometry import principal_angle, projected_flow_alignment


def test_projected_flow_alignment_matches_known_angle() -> None:
    Cf = np.array([[1.0], [0.0]], dtype=np.complex64)
    theta = math.radians(30.0)
    m = np.array([math.cos(theta), math.sin(theta)], dtype=np.complex64)
    cos_val = projected_flow_alignment(Cf, m)
    assert abs(cos_val - math.cos(theta)) < 1e-6


def test_projected_flow_alignment_zero_for_perpendicular() -> None:
    Cf = np.array([[1.0], [0.0]], dtype=np.complex64)
    m = np.array([0.0, 1.0], dtype=np.complex64)
    assert projected_flow_alignment(Cf, m) == 0.0


def test_principal_angle_simple_case() -> None:
    Cf = np.array([[1.0], [0.0]], dtype=np.complex64)
    Cm = np.array(
        [[math.cos(math.radians(45.0))], [math.sin(math.radians(45.0))]],
        dtype=np.complex64,
    )
    angle = principal_angle(Cf, Cm)
    assert abs(angle - 45.0) < 1e-6


def test_principal_angle_degenerate_returns_90() -> None:
    Cf = np.zeros((2, 0), dtype=np.complex64)
    Cm = np.array([[1.0], [0.0]], dtype=np.complex64)
    angle = principal_angle(Cf, Cm)
    assert angle == 90.0
