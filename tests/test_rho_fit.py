import numpy as np

from pipeline.confirm2.rho_fit import pearson_rho_with_ci


def test_rho_estimation_ci():
    rng = np.random.default_rng(123)
    n = 20000
    rho_true = 0.5
    # BVN simulation
    cov = np.array([[1.0, rho_true], [rho_true, 1.0]])
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal(size=(n, 2)) @ L.T
    rho_hat, lo, hi = pearson_rho_with_ci(z[:, 0], z[:, 1])
    assert abs(rho_hat - rho_true) < 0.03
    assert lo < rho_true < hi
