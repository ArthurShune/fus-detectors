"""Plane-wave delay computation utilities."""

from __future__ import annotations

import numpy as np


def delays(
    x: np.ndarray,
    z: np.ndarray,
    x_el: np.ndarray,
    theta_rad: float,
    c: float,
) -> np.ndarray:
    """Return round-trip delays for a plane-wave steered at ``theta_rad``.

    Args:
        x: lateral sample coordinates (1D).
        z: depth sample coordinates (1D).
        x_el: element x-positions (1D).
        theta_rad: steering angle in radians.
        c: speed of sound (m/s).
    """
    steering = (x * np.sin(theta_rad) + z * np.cos(theta_rad)) / c
    elem = np.sqrt((x_el[:, None] - x[None, :]) ** 2 + z[None, :] ** 2) / c
    return steering + elem
