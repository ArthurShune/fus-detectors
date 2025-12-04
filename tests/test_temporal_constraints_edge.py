import numpy as np
import torch

from pipeline.stap.temporal import bandpass_constraints_temporal, msd_snapshot_energies_batched


def test_bandpass_constraints_handles_degenerate_grids():
    Lt = 4
    prf = 3000.0

    empty = bandpass_constraints_temporal(Lt, prf, [], device="cpu")
    assert empty.shape == (Lt, 0)

    dc_only = bandpass_constraints_temporal(Lt, prf, [0.0], device="cpu")
    assert dc_only.shape == (Lt, 1)
    assert torch.allclose(dc_only, dc_only[0].reshape(-1, 1))

    large_grid = bandpass_constraints_temporal(Lt, prf, np.linspace(-1500, 1500, 12), device="cpu")
    # Should not exceed Lt*2 columns (basis adds derivatives), but finite.
    assert large_grid.shape[1] <= 2 * len(np.linspace(-1500, 1500, 12))
    assert torch.isfinite(large_grid).all()


def test_msd_snapshot_handles_zero_constraints_and_small_lt():
    Lt = 3
    R = torch.eye(Lt, dtype=torch.complex64)
    S = torch.zeros((Lt, 5, 1, 1), dtype=torch.complex64)
    Ct = bandpass_constraints_temporal(Lt, 3000.0, [], device="cpu")
    T_band, sw_pow = msd_snapshot_energies_batched(
        R,
        S,
        Ct,
        lam_abs=1e-2,
        ridge=0.1,
        device="cpu",
    )
    assert torch.allclose(T_band, torch.zeros_like(T_band))
    assert torch.allclose(sw_pow, torch.zeros_like(sw_pow))
