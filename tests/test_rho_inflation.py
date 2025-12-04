import numpy as np
from scipy.stats import norm

from pipeline.confirm2.validator import calibrate_confirm2, evaluate_confirm2


def _scores_from_bvn(n: int, rho: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((n, 2)) @ L.T
    p = 1.0 - norm.cdf(z)
    s = -np.log(p)
    return s[:, 0], s[:, 1]


def test_rho_inflation_conservatizes_pair_pfa():
    n = 200_000
    rho = 0.6
    alpha2 = 1e-4
    s1, s2 = _scores_from_bvn(n, rho, seed=123)
    s1_cal, s1_test = s1[: n // 2], s1[n // 2 :]
    s2_cal, s2_test = s2[: n // 2], s2[n // 2 :]

    calib0 = calibrate_confirm2(s1_cal, s2_cal, alpha2_target=alpha2, seed=0, rho_inflate=0.0)
    ev0 = evaluate_confirm2(calib0, s1_test, s2_test)

    calib1 = calibrate_confirm2(s1_cal, s2_cal, alpha2_target=alpha2, seed=0, rho_inflate=0.05)
    ev1 = evaluate_confirm2(calib1, s1_test, s2_test)

    assert ev1.empirical_pair_pfa <= ev0.empirical_pair_pfa * 1.05
    assert calib1.rho_eff > calib0.rho_hat
