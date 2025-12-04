import numpy as np
import torch

from pipeline.stap.lcmv import bandpass_constraints, lcmv_bandpass_apply
from pipeline.stap.mvdr_bank import build_steering_bank, mvdr_glrt_bank


def _synthetic_snapshots(M: int, N: int, fd_true: float, prf: float) -> torch.Tensor:
    rng = np.random.default_rng(0)
    t = np.arange(N, dtype=np.float32)
    a = rng.standard_normal(M) + 1j * rng.standard_normal(M)
    a = a.astype(np.complex64)
    a /= np.linalg.norm(a)
    s = np.exp(1j * 2 * np.pi * (fd_true / prf) * t).astype(np.complex64)
    noise = 0.3 * (rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N)))
    noise = noise.astype(np.complex64)
    X = a[:, None] * s[None, :] + noise
    return torch.from_numpy(X)


def test_lcmv_band_resists_frequency_mismatch_vs_bank():
    h = w = 4
    Lt = 4
    M = h * w * Lt
    N = 128
    prf = 3000.0
    fd_true = 0.20 * prf / Lt

    R = torch.eye(M, dtype=torch.complex64)
    X = _synthetic_snapshots(M, N, fd_true, prf)

    fd_grid = [-0.20 * prf / Lt, 0.0, +0.20 * prf / Lt]
    bank = build_steering_bank(h, w, Lt, prf, fd_grid, device="cpu")
    scores, fused = mvdr_glrt_bank(R, X, bank, diag_load=1e-3, fuse="max", device="cpu")
    pd_bank = torch.real(fused).mean().item()

    C = bandpass_constraints(h, w, Lt, prf, fd_grid, device="cpu")
    y, _ = lcmv_bandpass_apply(R, X, C, diag_load=1e-3, device="cpu")
    pd_lcmv = torch.real(y.conj() * y).mean().item()

    assert pd_lcmv >= pd_bank * 1.05, (pd_lcmv, pd_bank)
