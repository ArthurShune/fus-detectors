# pipeline/confirm2/rho_fit.py
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def pearson_rho_with_ci(
    z1: ArrayLike, z2: ArrayLike, alpha: float = 0.05
) -> tuple[float, float, float]:
    """
    Estimate Pearson correlation with Fisher-z CI.

    Parameters
    ----------
    z1, z2 : array-like
        Standardized look scores (z-scores) under null, same length.
    alpha : float
        CI level, default 0.05 => 95% CI.

    Returns
    -------
    rho_hat, lo, hi
    """
    x = np.asarray(z1, float).ravel()
    y = np.asarray(z2, float).ravel()
    if x.shape != y.shape or x.size < 20:
        raise ValueError("z1 and z2 must have the same length >= 20.")
    x = (x - x.mean()) / (x.std(ddof=1) + 1e-12)
    y = (y - y.mean()) / (y.std(ddof=1) + 1e-12)

    rho = float(np.corrcoef(x, y)[0, 1])
    rho = np.clip(rho, -0.999999, 0.999999)
    z = 0.5 * np.log((1 + rho) / (1 - rho))  # Fisher z
    se = 1.0 / np.sqrt(max(1, x.size - 3))
    zq = 1.959963984540054  # approx Phi^{-1}(0.975)
    lo_z, hi_z = z - zq * se, z + zq * se
    lo = np.tanh(lo_z)
    hi = np.tanh(hi_z)
    return rho, float(lo), float(hi)
