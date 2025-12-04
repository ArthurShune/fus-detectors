import pytest
import torch

from pipeline.stap.temporal import (
    _block_clip_covariance,
    _generalized_band_metrics,
    _mixing_metric,
)


def _pf_projector(rank: int, size: int, dtype=torch.complex64) -> torch.Tensor:
    Pf = torch.zeros((size, size), dtype=dtype)
    for i in range(min(rank, size)):
        Pf[i, i] = 1.0
    return Pf


def _random_spd(size: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    A = torch.randn((size, size), generator=gen, dtype=torch.complex64)
    return A @ A.conj().transpose(-2, -1) + 0.5 * torch.eye(size, dtype=torch.complex64)


def test_block_clip_removes_cross_terms() -> None:
    Lt = 5
    Pf = _pf_projector(2, Lt)
    R = _random_spd(Lt, seed=0)
    clipped = _block_clip_covariance(R, Pf)
    Pf_h = 0.5 * (Pf + Pf.conj().transpose(-2, -1))
    P_perp = torch.eye(Lt, dtype=Pf.dtype) - Pf_h
    cross = P_perp @ clipped @ Pf_h
    assert torch.linalg.norm(cross) < 1e-6


def test_generalized_band_metrics_match_explicit() -> None:
    Lt = 4
    Pf = _pf_projector(2, Lt)
    rf = 2
    Rp_dim = Lt - rf
    Rf_sample = _random_spd(rf, seed=1)
    Rp_sample = _random_spd(Rp_dim, seed=2)
    Rf_beta = _random_spd(rf, seed=3)
    Rp_beta = _random_spd(Rp_dim, seed=4)
    R_sample = torch.zeros((Lt, Lt), dtype=torch.complex64)
    R_beta = torch.zeros((Lt, Lt), dtype=torch.complex64)
    R_sample[:rf, :rf] = Rf_sample
    R_sample[rf:, rf:] = Rp_sample
    R_beta[:rf, :rf] = Rf_beta
    R_beta[rf:, rf:] = Rp_beta
    metrics = _generalized_band_metrics(
        R_sample.to(torch.complex128),
        R_beta.to(torch.complex128),
        Pf.to(torch.complex128),
    )
    Rf_sample = R_sample[:rf, :rf]
    Rf_beta = R_beta[:rf, :rf]
    Lf = torch.linalg.cholesky(Rf_beta)
    sol_f = torch.cholesky_solve(Rf_sample, Lf)
    evals_f = torch.linalg.eigvalsh(0.5 * (sol_f + sol_f.conj().transpose(-2, -1))).real
    assert pytest.approx(float(evals_f.min().item()), rel=1e-6) == metrics["pf_min"]
    assert pytest.approx(float(evals_f.max().item()), rel=1e-6) == metrics["pf_max"]

    Rp_sample = R_sample[rf:, rf:]
    Rp_beta = R_beta[rf:, rf:]
    Lp = torch.linalg.cholesky(Rp_beta)
    sol_p = torch.cholesky_solve(Rp_sample, Lp)
    evals_p = torch.linalg.eigvalsh(0.5 * (sol_p + sol_p.conj().transpose(-2, -1))).real
    assert pytest.approx(float(evals_p.max().item()), rel=1e-6) == metrics["perp_max"]


def test_mixing_metric_zero_for_block_diagonal() -> None:
    Lt = 4
    Pf = _pf_projector(2, Lt)
    R_sample = torch.zeros((Lt, Lt), dtype=torch.complex64)
    R_beta = torch.zeros((Lt, Lt), dtype=torch.complex64)
    R_sample[:2, :2] = _random_spd(2, seed=5)
    R_sample[2:, 2:] = _random_spd(2, seed=6)
    R_beta[:2, :2] = _random_spd(2, seed=7)
    R_beta[2:, 2:] = _random_spd(2, seed=8)
    mix_val = _mixing_metric(
        R_sample.to(torch.complex128),
        R_beta.to(torch.complex128),
        Pf.to(torch.complex128),
    )
    assert mix_val == pytest.approx(0.0, abs=1e-9)
