"""Gaussian copula helpers."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def gaussian_copula(u: np.ndarray, v: np.ndarray, rho: float) -> np.ndarray:
    """Evaluate Gaussian copula density at (u, v)."""
    x = norm.ppf(u)
    y = norm.ppf(v)
    denom = np.sqrt(1 - rho**2)
    expo = -(x**2 - 2 * rho * x * y + y**2) / (2 * (1 - rho**2))
    return np.exp(expo) / denom
