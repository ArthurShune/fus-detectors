import numpy as np
import pytest
from scipy.stats import norm

from pipeline.confirm2.bvn_tail import joint_tail, per_look_alpha_from_pair


def _empirical_pair_pfa(rho: float, threshold: float, n: int = 200000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, rho], [rho, 1.0]], dtype=np.float64)
    samples = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    hits = np.logical_and(samples[:, 0] > threshold, samples[:, 1] > threshold)
    return float(hits.mean())


@pytest.mark.parametrize("rho_pair", [0.0, 0.3, 0.6])
def test_joint_tail_monotone_vs_empirical(rho_pair: float):
    alpha2 = 1e-4
    alpha1 = per_look_alpha_from_pair(alpha2, rho_pair)
    z_thr = norm.isf(alpha1)
    analytic = joint_tail(z_thr, rho_pair)
    empirical = _empirical_pair_pfa(rho_pair, z_thr, n=120000, seed=42)
    assert empirical <= analytic * 1.5
    assert abs(analytic - alpha2) / alpha2 < 0.25


def test_joint_tail_increases_with_rho():
    z_thr = 4.0
    vals = [joint_tail(z_thr, rho) for rho in (-0.2, 0.0, 0.4, 0.8)]
    assert vals == sorted(vals)
