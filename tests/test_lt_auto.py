import numpy as np

from pipeline.stap.lt_auto import choose_lt_from_coherence


def test_choose_lt_tracks_ar1_correlation():
    rng = np.random.default_rng(123)
    T, H, W = 64, 8, 8
    lt_candidates = (3, 4, 5, 6, 7, 8)
    for rho, _expected_min in [(0.2, 3), (0.5, 5), (0.8, 6)]:
        noise = rng.normal(size=T)
        x = np.zeros(T)
        for t in range(1, T):
            x[t] = rho * x[t - 1] + noise[t]
        tile = np.tile(x[:, None, None], (1, H, W)).astype(np.float32)
        lt = choose_lt_from_coherence(tile, lt_candidates=lt_candidates, method="pd")
        if rho < 0.3:
            assert lt <= 4
        elif rho > 0.7:
            assert lt >= 6
