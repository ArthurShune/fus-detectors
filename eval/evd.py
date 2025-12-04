from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class WeibullPOT:
    u: float
    xi: float  # < 0 for bounded tails
    beta: float
    xF: float  # endpoint
    p_u: float
    n_exc: int
    n_total: int
    r2_mean_excess: float

    def as_dict(self) -> dict:
        return dict(
            u=self.u,
            xi=self.xi,
            beta=self.beta,
            xF=self.xF,
            p_u=self.p_u,
            n_exc=self.n_exc,
            n_total=self.n_total,
            r2=self.r2_mean_excess,
        )


def _linear_fit_r2(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m, b = np.polyfit(x, y, 1)
    yhat = m * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return float(m), float(b), float(r2)


def fit_weibull_pot(
    scores: np.ndarray,
    q0: float = 0.95,
    endpoint_hint: Optional[float] = None,
    min_exceed: int = 500,
) -> WeibullPOT:
    """
    Fit a bounded-tail (Weibull domain, xi<0) POT surrogate using a linear tail model.

    Assumes the score is bounded above by xF (use endpoint_hint if known; for MSD ratio xF≈1/rho).
    Uses the identity for conditional survival in the Weibull domain:
        log S_u(x) ≈ -(1/xi) [log(xF - x) - log(xF - u)]
    so slope = -1/xi on the transformed axis. We estimate slope by least squares on exceedances.
    Returns xi<0 and a simple R^2 diagnostic for monotonic mean-excess behaviour.
    """
    s = np.asarray(scores, float).ravel()
    n_total = int(s.size)
    if s.size < max(2000, min_exceed * 2):
        raise ValueError("Insufficient scores for stable Weibull POT fit.")
    if not np.all(np.isfinite(s)):
        raise ValueError("Scores contain non-finite values.")
    u = float(np.quantile(s, q0))
    tail = s[s >= u]
    n_exc = int(tail.size)
    if n_exc < min_exceed:
        raise ValueError(f"Too few exceedances above u={u:.6g}: {n_exc} < {min_exceed}")

    xF = float(endpoint_hint) if endpoint_hint is not None else float(np.max(s) + 1e-8)
    # Guard: if any exceedance equals/exceeds xF, nudge xF upward
    if np.max(tail) >= xF:
        xF = float(np.max(tail) * (1.0 + 1e-6))

    # Empirical conditional survival on exceedances
    xs = np.sort(tail)
    ranks = np.arange(xs.size - 1, -1, -1, dtype=np.float64)  # # of samples > x_i
    Su = (ranks + 1.0) / float(xs.size)
    # Transform axis
    denom = max(xF - u, 1e-12)
    z = (xF - xs) / denom
    z = np.clip(z, 1e-12, None)
    X = np.log(z)
    Y = np.log(Su)
    slope, intercept, r2 = _linear_fit_r2(X, Y)
    if slope >= 0.0:
        # degenerate/inconsistent fit; force negative xi via small negative slope
        slope = min(slope, -1e-3)
    xi_hat = float(-1.0 / slope)
    if xi_hat >= -1e-3:
        xi_hat = -1e-3
    beta = max((xF - u) * max(-xi_hat, 1e-6), 1e-12)
    p_u = n_exc / max(1, n_total)
    pot = WeibullPOT(
        u=u,
        xi=xi_hat,
        beta=beta,
        xF=xF,
        p_u=p_u,
        n_exc=n_exc,
        n_total=n_total,
        r2_mean_excess=float(r2),
    )
    return pot
