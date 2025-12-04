import numpy as np
import pytest

from pipeline.stap.temporal import (
    bandpass_constraints_temporal,
    capon_band_ratio_batched,
    msd_band_energy_batched,
    msd_snapshot_energies_batched,
    aggregate_over_snapshots,
)


torch = pytest.importorskip("torch")  # noqa: E305


def _make_constraint(Lt: int, prf_hz: float, fd_rel: float) -> torch.Tensor:
    grid = [fd_rel * prf_hz / Lt]
    return bandpass_constraints_temporal(
        Lt=Lt,
        prf_hz=prf_hz,
        fd_grid_hz=grid,
        device="cpu",
        dtype=torch.complex64,
    )


def _make_snapshots(
    Lt: int,
    N: int,
    h: int,
    w: int,
    signal_vec: torch.Tensor,
    noise_scale: float,
    seed: int,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    Y = torch.zeros((Lt, N, h, w), dtype=torch.complex64)
    sig = signal_vec.to(dtype=torch.complex64)
    for n in range(N):
        noise_real = torch.tensor(rng.standard_normal((Lt,)), dtype=torch.float32)
        noise_imag = torch.tensor(rng.standard_normal((Lt,)), dtype=torch.float32)
        noise = noise_scale * torch.complex(noise_real, noise_imag)
        Y[:, n, :, :] = (sig + noise).reshape(Lt, 1, 1)
    return Y


def test_msd_band_energy_highlights_inband_signal():
    Lt, N = 4, 8
    prf_hz = 4000.0
    fd_rel = 0.25
    C = _make_constraint(Lt, prf_hz, fd_rel)
    Rt = torch.eye(Lt, dtype=torch.complex64)

    s_vec = C[:, 0]
    inband = _make_snapshots(Lt, N, 1, 1, s_vec * 2.0, noise_scale=0.05, seed=0)
    outband = _make_snapshots(
        Lt,
        N,
        1,
        1,
        torch.zeros_like(s_vec),
        noise_scale=0.05,
        seed=1,
    )

    msd_in = (
        msd_band_energy_batched(
            Rt,
            inband,
            C,
            lam_abs=1e-4,
            ridge=1e-3,
            device="cpu",
            dtype=torch.complex64,
        )
        .detach()
        .cpu()
        .numpy()
    )
    msd_out = (
        msd_band_energy_batched(
            Rt,
            outband,
            C,
            lam_abs=1e-4,
            ridge=1e-3,
            device="cpu",
            dtype=torch.complex64,
        )
        .detach()
        .cpu()
        .numpy()
    )

    assert msd_in.shape == (1, 1)
    assert msd_out.shape == (1, 1)
    assert msd_in[0, 0] > 5.0
    assert msd_in[0, 0] > msd_out[0, 0] * 5.0


def test_msd_band_energy_zero_constraints_returns_zero():
    Lt, N = 4, 5
    Rt = torch.eye(Lt, dtype=torch.complex64)
    S = torch.zeros((Lt, N, 2, 2), dtype=torch.complex64)
    zeros = (
        msd_band_energy_batched(
            Rt,
            S,
            torch.zeros((Lt, 0), dtype=torch.complex64),
            device="cpu",
            dtype=torch.complex64,
        )
        .detach()
        .cpu()
        .numpy()
    )
    assert np.allclose(zeros, 0.0)


def test_capon_band_ratio_positive_definite():
    Lt, N = 4, 6
    prf_hz = 5000.0
    fd_rel = 0.2
    C = _make_constraint(Lt, prf_hz, fd_rel)
    Rt = torch.eye(Lt, dtype=torch.complex64)
    S = _make_snapshots(Lt, N, 1, 1, torch.zeros((Lt,), dtype=torch.complex64), 0.1, seed=3)

    ratio = (
        capon_band_ratio_batched(
            Rt,
            S,
            C,
            lam_abs=1e-3,
            device="cpu",
            dtype=torch.complex64,
        )
        .detach()
        .cpu()
        .numpy()
    )
    assert ratio.shape == (1, 1)
    assert ratio[0, 0] > 0.0


def test_msd_snapshot_energy_bounds():
    Lt, N, h, w = 4, 5, 2, 2
    Rt = torch.eye(Lt, dtype=torch.complex64)
    Ct = torch.ones((Lt, 1), dtype=torch.complex64)
    S = torch.randn(Lt, N, h, w, dtype=torch.complex64)
    T_band, sw_pow = msd_snapshot_energies_batched(
        Rt, S, Ct, lam_abs=1e-2, ridge=0.1, device="cpu"
    )
    assert torch.all(T_band >= 0)
    assert torch.all(sw_pow >= 0)
    assert torch.all(T_band <= sw_pow + 1e-6)


def test_trimmed_mean_reduces_outlier():
    x = torch.ones((6, 1, 1), dtype=torch.float32)
    x[0] = 10.0
    mean_val = aggregate_over_snapshots(x, mode="mean")
    trim_val = aggregate_over_snapshots(x, mode="trim10")
    assert trim_val.item() < mean_val.item()


def test_msd_ratio_orthogonal_component_zero():
    Lt, N = 4, 6
    prf_hz = 4000.0
    grid = np.array([-300.0, 0.0, 300.0], dtype=np.float64)
    C = bandpass_constraints_temporal(Lt=Lt, prf_hz=prf_hz, fd_grid_hz=grid, device="cpu")
    Rt = torch.eye(Lt, dtype=torch.complex64)

    # Construct vector in null(C^H)
    U, Svals, Vh = torch.linalg.svd(C, full_matrices=True)
    null_vec = U[:, -1]  # last column spans kernel of C^H
    ortho = torch.complex(null_vec.real, null_vec.imag)
    S = _make_snapshots(Lt, N, 1, 1, ortho, noise_scale=0.0, seed=5)

    stat = (
        msd_band_energy_batched(
            Rt,
            S,
            C,
            lam_abs=1e-2,
            ridge=0.0,
            device="cpu",
            dtype=torch.complex64,
        )
        .detach()
        .cpu()
        .numpy()
    )
    assert np.allclose(stat, 0.0, atol=1e-6)
