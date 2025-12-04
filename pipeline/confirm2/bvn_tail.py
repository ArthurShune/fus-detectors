# pipeline/confirm2/bvn_tail.py
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import multivariate_normal, norm


def joint_tail(z: float, rho: float) -> float:
    """
    P(Z1 > z, Z2 > z) for a standard bivariate normal with correlation rho.

    Uses inclusion-exclusion:
        P(Z1>z, Z2>z) = 1 - 2*Phi(z) + Phi2((z,z); rho)
    where Phi is univariate CDF and Phi2 is BVN CDF at (z,z).

    Parameters
    ----------
    z : float
        Per-look z threshold (standard normal units).
    rho : float
        Correlation in [-0.999, 0.999].

    Returns
    -------
    float
    """
    z = float(z)
    rho = float(np.clip(rho, -0.999, 0.999))
    cov = np.array([[1.0, rho], [rho, 1.0]], dtype=float)
    F2 = multivariate_normal(mean=[0.0, 0.0], cov=cov).cdf([z, z])
    return float(1.0 - 2.0 * norm.cdf(z) + F2)


def solve_z_for_pair_alpha(alpha2: float, rho: float) -> float:
    """
    Find z such that joint_tail(z, rho) = alpha2.

    alpha2 is extremely small in our use-cases (e.g., 1e-5, 1e-6).
    The function is monotone decreasing in z, so bracketing [0, 12] is safe.

    Returns
    -------
    z : float
    """
    alpha2 = float(alpha2)
    if not (0.0 < alpha2 < 0.25):
        raise ValueError("alpha2 must be in (0, 0.25).")

    def objective(z_val: float) -> float:
        return joint_tail(z_val, rho) - alpha2

    # Lower bound: z=0 => P ~ (1/4) + positive mass if rho>0; upper bound: z=12 => ~0
    return float(brentq(objective, a=0.0, b=12.0))


def per_look_alpha_from_pair(alpha2: float, rho: float) -> float:
    """
    Map pair false-alarm alpha2 to *per-look* miscoverage alpha1 via z.

        z* = solve_z_for_pair_alpha(alpha2, rho)
        alpha1 = 1 - Phi(z*)

    Returns
    -------
    alpha1 : float
    """
    z = solve_z_for_pair_alpha(alpha2, rho)
    return float(1.0 - norm.cdf(z))
