# tests/test_confirm2_validate.py

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


def test_confirm2_empirical_matches_analytic_at_1e3():
    rho = 0.5
    n = 200_000  # yields ~100 joint exceedances at 1e-3
    s1, s2 = _scores_from_bvn(n, rho, seed=123)

    # Split (cal/test)
    s1_cal, s1_test = s1[: n // 2], s1[n // 2 :]
    s2_cal, s2_test = s2[: n // 2], s2[n // 2 :]

    alpha2 = 1e-3
    calib = calibrate_confirm2(s1_cal, s2_cal, alpha2_target=alpha2, seed=0)
    ev = evaluate_confirm2(calib, s1_test, s2_test)

    # Empirical should be close to analytic within ~30% relative error at this N
    rel_err = abs(ev.empirical_pair_pfa - ev.predicted_pair_pfa) / ev.predicted_pair_pfa
    assert rel_err < 0.3
    # Target should lie within binomial CI most of the time
    assert ev.pair_ci_lo <= alpha2 <= ev.pair_ci_hi
