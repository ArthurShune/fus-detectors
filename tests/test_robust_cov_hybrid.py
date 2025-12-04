import numpy as np

from pipeline.stap.robust_cov import robust_covariance


def _cn_samples(sigma: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    m = sigma.shape[0]
    L = np.linalg.cholesky(sigma + 1e-8 * np.eye(m))
    z = (rng.standard_normal((m, n)) + 1j * rng.standard_normal((m, n))) / np.sqrt(2.0)
    return (L @ z).astype(np.complex64)


def test_tyler_pca_improves_when_n_less_m():
    rng = np.random.default_rng(0)
    M, N = 64, 40  # N < M triggers hybrid path
    Q, _ = np.linalg.qr(rng.standard_normal((M, M)))
    eig = 10.0 ** np.linspace(0, -3, M)
    sigma = Q @ np.diag(eig) @ Q.T
    X = _cn_samples(sigma, N, rng)
    # Inject heavy outliers in 15% of samples
    n_out = int(0.15 * N)
    X[:, :n_out] = _cn_samples(25.0 * sigma, n_out, rng)
    X -= X.mean(axis=1, keepdims=True)

    R_scm, _ = robust_covariance(X, method="scm")
    R_hybrid, _ = robust_covariance(X, method="tyler_pca", max_iter=150)

    err_scm = np.linalg.norm(R_scm - sigma, "fro")
    err_hybrid = np.linalg.norm(R_hybrid - sigma, "fro")
    assert err_hybrid <= err_scm * 0.95
