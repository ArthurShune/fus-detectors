import numpy as np

from eval.evd import fit_weibull_pot


def test_weibull_pot_fits_bounded_tail():
    rng = np.random.default_rng(0)
    # Simulate a bounded score in [0, 1] with mass near the endpoint
    x = rng.beta(5.0, 2.0, size=500_000)
    # Shift to make the top tail slightly heavier near 1
    scores = np.clip(0.85 + 0.15 * x, 0.0, 1.0)

    pot = fit_weibull_pot(scores, q0=0.98, endpoint_hint=1.0, min_exceed=2000)
    assert pot.n_exc >= 2000
    assert pot.n_total == scores.size
    assert pot.xi <= -1e-3  # bounded tail must yield sufficiently negative xi
    # Mean-excess linearity on the transformed axis should be decent
    assert pot.r2_mean_excess >= 0.9
    # Endpoint should remain close to the hint
    assert abs(pot.xF - 1.0) <= 1e-3
    assert pot.beta > 0.0
    assert 0.0 < pot.p_u < 1.0
