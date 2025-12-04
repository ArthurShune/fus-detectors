import numpy as np

from pipeline.stap.mvdr_bank import choose_fd_grid_auto


def test_fd_grid_auto_symmetry_and_scaling():
    prf_hz, Lt = 3000.0, 4
    grid_small = choose_fd_grid_auto(prf_hz, Lt, median_shift_px=0.05)
    grid_large = choose_fd_grid_auto(prf_hz, Lt, median_shift_px=0.8)

    assert len(grid_small) % 2 == 1
    assert len(grid_large) % 2 == 1
    np.testing.assert_allclose(grid_small[0], -grid_small[-1], atol=1e-8)
    np.testing.assert_allclose(grid_large[0], -grid_large[-1], atol=1e-8)
    assert len(grid_large) >= len(grid_small)
