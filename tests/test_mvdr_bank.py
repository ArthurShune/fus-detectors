import numpy as np
import torch

from pipeline.stap.mvdr_bank import build_steering_bank, mvdr_glrt_bank


def test_bank_beats_single_when_fd_mismatch():
    rng = np.random.default_rng(0)
    h = w = 4
    Lt = 4
    M = h * w * Lt
    N = 64
    prf = 3000.0

    R = torch.eye(M, dtype=torch.complex64)
    fd_true = 0.12 * prf / Lt
    t = np.arange(N, dtype=np.float32)
    s = np.exp(1j * 2 * np.pi * (fd_true / prf) * t).astype(np.complex64)
    a = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex64)
    a /= np.linalg.norm(a)
    noise = 0.15 * (rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N))).astype(
        np.complex64
    )
    X = (a[:, None] * s[None, :] + noise).astype(np.complex64)

    fd_grid = [-0.12 * prf / Lt, 0.0, +0.12 * prf / Lt]
    bank = build_steering_bank(h, w, Lt, prf, fd_grid, device="cpu")
    S, _ = mvdr_glrt_bank(R, X, bank, diag_load=1e-3, fuse="max", device="cpu")

    zero_bank = build_steering_bank(h, w, Lt, prf, [0.0], device="cpu")
    S_zero, _ = mvdr_glrt_bank(R, X, zero_bank, diag_load=1e-3, fuse="max", device="cpu")

    frac_better = torch.mean((S[2] > S_zero[0]).float()).item()
    assert frac_better > 0.5
