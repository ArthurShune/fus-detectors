import numpy as np

from pipeline.stap.robust_cov import robust_covariance


def _cn_samples(Sigma: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw circular complex Gaussian samples CN(0, Sigma)."""
    M = Sigma.shape[0]
    L = np.linalg.cholesky(Sigma)
    Z = (rng.standard_normal((M, n)) + 1j * rng.standard_normal((M, n))) / np.sqrt(2.0)
    return (L @ Z).astype(np.complex64)


def test_huber_beats_scm_under_outliers():
    improvements = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        M, N = 12, 240
        U, _ = np.linalg.qr(rng.standard_normal((M, M)))
        ev = 10.0 ** np.linspace(0.0, -1.5, M)
        Sigma = U @ np.diag(ev) @ U.T
        Sigma += 1e-6 * np.eye(M)

        X_clean = _cn_samples(Sigma, int(N * 0.9), rng)
        X_out = _cn_samples(80.0 * Sigma, int(N * 0.1), rng)
        X = np.concatenate([X_clean, X_out], axis=1)
        X -= X.mean(axis=1, keepdims=True)

        R_scm, _ = robust_covariance(X, method="scm")
        R_hub, _ = robust_covariance(X, method="huber", huber_c=3.0, max_iter=100)

        err_scm = np.linalg.norm(R_scm - Sigma, ord="fro")
        err_hub = np.linalg.norm(R_hub - Sigma, ord="fro")
        improvements.append(err_scm - err_hub)

    assert np.mean(improvements) > 0.0
    assert any(diff > 0 for diff in improvements)


def test_tyler_converges_and_unit_trace():
    rng = np.random.default_rng(1)
    M, N = 12, 300
    Sigma = np.eye(M)
    X = _cn_samples(Sigma, N, rng)
    X -= X.mean(axis=1, keepdims=True)

    R_tyl, tel = robust_covariance(X, method="tyler", max_iter=200, tol=1e-5)
    assert abs(np.trace(R_tyl).real - M) < 1e-3
    assert tel.converged
