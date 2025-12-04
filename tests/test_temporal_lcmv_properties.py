import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pipeline.stap.temporal import (
    bandpass_constraints_temporal,
    build_temporal_hankels_and_cov,
    lcmv_temporal_apply_batched,
)


torch = pytest.importorskip("torch")  # noqa: E305


def _random_cube(T: int, h: int, w: int) -> np.ndarray:
    cube = 0.1 * (np.random.randn(T, h, w) + 1j * np.random.randn(T, h, w))
    cube += 0.5 * np.random.randn(T, 1, 1)
    return cube.astype(np.complex64)


def test_lcmv_cond_monotone_in_diag_load():
    T, h, w, Lt = 48, 3, 3, 4
    prf = 3000.0
    cube = _random_cube(T, h, w)
    S, R_t, _ = build_temporal_hankels_and_cov(
        cube, Lt=Lt, estimator="huber", huber_c=5.0, device="cpu"
    )
    grid = np.linspace(-900.0, 900.0, 9)
    C = bandpass_constraints_temporal(Lt, prf, grid, device="cpu")

    _, res_small = lcmv_temporal_apply_batched(
        R_t,
        S,
        C,
        load_mode="absolute",
        diag_load=1e-4,
        constraint_ridge=0.1,
        device="cpu",
    )
    _, res_large = lcmv_temporal_apply_batched(
        R_t,
        S,
        C,
        load_mode="absolute",
        diag_load=1e-2,
        constraint_ridge=0.1,
        device="cpu",
    )

    Rt_np = R_t.cpu().numpy()
    cond_small = np.linalg.cond(Rt_np + res_small.diag_load * np.eye(Lt, dtype=np.complex64))
    cond_large = np.linalg.cond(Rt_np + res_large.diag_load * np.eye(Lt, dtype=np.complex64))
    assert cond_large <= cond_small * 1.01  # allow small numerical wiggle


def test_lcmv_constraint_residual_grows_with_ridge():
    T, h, w, Lt = 40, 3, 3, 4
    prf = 3000.0
    cube = _random_cube(T, h, w)
    S, R_t, _ = build_temporal_hankels_and_cov(
        cube, Lt=Lt, estimator="huber", huber_c=5.0, device="cpu"
    )
    grid = np.linspace(-750.0, 750.0, 7)
    C = bandpass_constraints_temporal(Lt, prf, grid, device="cpu")

    _, res_lo = lcmv_temporal_apply_batched(
        R_t,
        S,
        C,
        load_mode="absolute",
        diag_load=5e-3,
        constraint_ridge=0.05,
        device="cpu",
    )
    _, res_hi = lcmv_temporal_apply_batched(
        R_t,
        S,
        C,
        load_mode="absolute",
        diag_load=5e-3,
        constraint_ridge=0.20,
        device="cpu",
    )
    assert res_hi.constraint_residual >= res_lo.constraint_residual - 1e-8
