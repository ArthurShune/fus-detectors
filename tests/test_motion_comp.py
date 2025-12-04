import numpy as np

from pipeline.stap.motion_comp import _fourier_shift_2d, register_tile_slowtime


def test_motion_comp_reduces_mse():
    rng = np.random.default_rng(0)
    T, H, W = 12, 32, 32
    base = rng.standard_normal((H, W)) + 1j * rng.standard_normal((H, W))
    dy, dx = 0.4, -0.7
    tile = np.empty((T, H, W), dtype=np.complex64)
    for t in range(T):
        noise = 0.01 * (rng.standard_normal((H, W)) + 1j * rng.standard_normal((H, W)))
        tile[t] = _fourier_shift_2d(base, dy * t, dx * t) + noise

    reg, shifts = register_tile_slowtime(tile, ref_strategy="median")
    assert reg.shape == tile.shape
    assert shifts.shape == (T, 2)
    mse_before = np.mean((np.abs(tile) - np.abs(tile[0])) ** 2)
    mse_after = np.mean((np.abs(reg) - np.abs(reg[0])) ** 2)
    assert mse_after < 0.75 * mse_before
