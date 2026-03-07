from __future__ import annotations

import numpy as np

from scripts.simus_motion_proxy_warp_check import _warp_icube


def test_warp_icube_identity_for_zero_displacement():
    icube = (np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4) + 1j).astype(np.complex64)
    dx = np.zeros((2, 4, 4), dtype=np.float32)
    dz = np.zeros((2, 4, 4), dtype=np.float32)
    warped = _warp_icube(icube, dx, dz)
    assert np.allclose(warped, icube)
