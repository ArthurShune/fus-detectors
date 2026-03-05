import numpy as np


def test_fwhm_1d_matches_gaussian_sigma():
    from scripts.psf_calib_point_target_kwave import fwhm_1d, sigma_from_fwhm_px

    rng = np.random.default_rng(0)
    sigma_true = 3.5
    x = np.arange(-64, 65, dtype=np.float64)
    prof = np.exp(-0.5 * (x / sigma_true) ** 2)
    # Add tiny noise to avoid any brittle ties in argmax.
    prof = prof + 1e-6 * rng.standard_normal(prof.shape)
    prof = prof.astype(np.float32)

    peak = int(np.argmax(prof))
    fwhm = fwhm_1d(prof, peak_idx=peak)
    sigma_est = sigma_from_fwhm_px(fwhm)

    # FWHM-based sigma should be very close for a clean Gaussian.
    assert np.isfinite(fwhm) and fwhm > 0.0
    assert abs(sigma_est - sigma_true) / sigma_true < 0.03


def test_fit_sigma_vs_depth_linear_recovers_parameters():
    from scripts.psf_calib_point_target_kwave import fit_sigma_vs_depth_linear

    sigma0 = 1.2
    alpha = 40.0  # px/m
    z = np.array([0.0, 0.005, 0.010, 0.015, 0.020], dtype=np.float64)
    sigma = sigma0 + alpha * z
    # Add tiny noise to make sure fit is stable but not exact.
    sigma = sigma + np.array([0.00, 0.01, -0.01, 0.00, 0.01], dtype=np.float64)

    sigma0_hat, alpha_hat, diag = fit_sigma_vs_depth_linear(z, sigma)
    assert diag["n"] == 5
    assert abs(sigma0_hat - sigma0) < 0.05
    assert abs(alpha_hat - alpha) < 1.0
    assert diag["r2"] > 0.99

