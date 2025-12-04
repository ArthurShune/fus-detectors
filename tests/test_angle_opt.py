# tests/test_angle_opt.py
import itertools

import numpy as np
import torch

from pipeline.gpu.linalg import to_tensor
from pipeline.stap.angle_opt import _score_sH_invR_s, greedy_then_swap_optimize


def _synthetic_covariances(A: int = 12, M: int = 24, seed: int = 0):
    """
    Build angle-dependent Hermitian PSD covariances with smooth redundancy:
        R_k = sigma^2 I + alpha v_k v_k^H
    where v_k rotates smoothly in the 2D subspace span{u1,u2}.
    """
    rng = np.random.default_rng(seed)
    # Two orthonormal directions u1,u2
    U = rng.standard_normal((M, 2)) + 1j * rng.standard_normal((M, 2))
    # Orthonormalize (QR on real-augmented is fine because columns are only 2)
    # Simple Gram-Schmidt
    u1 = U[:, 0]
    u1 = u1 / np.linalg.norm(u1)
    u2 = U[:, 1] - (u1.conj().T @ U[:, 1]) * u1
    u2 = u2 / np.linalg.norm(u2)

    thetas = np.linspace(0.0, 2.0 * np.pi, A, endpoint=False)
    sigma2 = 1.0
    alpha = 5.0

    Rs = []
    for th in thetas:
        vk = np.cos(th) * u1 + np.sin(th) * u2
        vk = vk / np.linalg.norm(vk)
        Rk = sigma2 * np.eye(M, dtype=np.complex64) + alpha * np.outer(vk, vk.conj())
        Rs.append(0.5 * (Rk + Rk.conj().T))
    Rs = np.stack(Rs, axis=0)  # (A,M,M)
    return Rs


def _score_for_set(Rs: np.ndarray, s: np.ndarray, idx: tuple[int, ...], diag_load=1e-2) -> float:
    Rm = np.mean(Rs[list(idx)], axis=0)
    # torch path for stability
    Rt = to_tensor(Rm, device="cpu", dtype=torch.complex64)
    st = to_tensor(s, device="cpu", dtype=torch.complex64)
    return float(_score_sH_invR_s(Rt, st, diag_load=diag_load).item())


def test_angle_optimality_against_bruteforce():
    A, M, K = 12, 24, 4
    Rs = _synthetic_covariances(A=A, M=M, seed=42)
    rng = np.random.default_rng(123)
    # steering not aligned with u1/u2
    s = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex64)
    s = s / np.linalg.norm(s)

    # Brute-force global best over C(12,4)=495 sets
    all_sets = list(itertools.combinations(range(A), K))
    scores = [_score_for_set(Rs, s, idx, diag_load=1e-2) for idx in all_sets]
    best_idx = int(np.argmax(scores))
    best_score = scores[best_idx]

    # Run optimizer
    Rs_t = to_tensor(Rs, device="cpu", dtype=torch.complex64)
    s_t = to_tensor(s, device="cpu", dtype=torch.complex64)
    res = greedy_then_swap_optimize(
        Rs_t,
        s_t,
        K=K,
        diag_load=1e-2,
        max_swap_passes=3,
        rel_improve_tol=1e-6,
        device="cpu",
    )

    # The optimizer should recover the global optimum or be within 1%
    opt_score = _score_for_set(Rs, s, tuple(res.selected), diag_load=1e-2)
    assert opt_score >= 0.99 * best_score
    # And should be notably better than a consecutive-angle choice
    naive_set = set(range(K))
    naive_score = _score_for_set(Rs, s, tuple(sorted(naive_set)), diag_load=1e-2)
    # Synthetic covariances can yield modest separation; require clear improvement.
    assert opt_score >= naive_score * 1.02


def test_angle_optimality_with_spiky_angles():
    A, M, K = 12, 24, 4
    # Create covariances where only a few angles carry strong signal
    rng = np.random.default_rng(99)
    Rs = _synthetic_covariances(A=A, M=M, seed=17)
    hot_angles = [1, 4, 7, 9]
    spike = rng.standard_normal((M,)) + 1j * rng.standard_normal((M,))
    spike = spike / np.linalg.norm(spike)
    for idx in hot_angles:
        Rs[idx] += 20.0 * np.outer(spike, spike.conj()).astype(np.complex64)

    s = spike.astype(np.complex64)

    all_sets = list(itertools.combinations(range(A), K))
    scores = [_score_for_set(Rs, s, idx, diag_load=5e-3) for idx in all_sets]
    best_score = max(scores)

    Rs_t = to_tensor(Rs, device="cpu", dtype=torch.complex64)
    s_t = to_tensor(s, device="cpu", dtype=torch.complex64)
    res = greedy_then_swap_optimize(
        Rs_t,
        s_t,
        K=K,
        diag_load=5e-3,
        max_swap_passes=4,
        rel_improve_tol=5e-4,
        device="cpu",
    )

    opt_score = _score_for_set(Rs, s, tuple(res.selected), diag_load=5e-3)
    assert opt_score >= 0.99 * best_score
    naive_score = _score_for_set(Rs, s, tuple(range(K)), diag_load=5e-3)
    assert opt_score >= naive_score * 1.25  # substantially better than naive


def test_score_history_monotone_non_decreasing():
    A, M, K = 10, 16, 3
    Rs = _synthetic_covariances(A=A, M=M, seed=7)
    rng = np.random.default_rng(0)
    s = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex64)
    s = s / np.linalg.norm(s)

    Rs_t = to_tensor(Rs, device="cpu", dtype=torch.complex64)
    s_t = to_tensor(s, device="cpu", dtype=torch.complex64)
    res = greedy_then_swap_optimize(Rs_t, s_t, K=K, diag_load=1e-3, device="cpu")

    # Greedy history should be non-decreasing
    hist = res.score_history
    assert all(hist[i] >= hist[i - 1] - 1e-8 for i in range(1, len(hist)))
    # Final score should be at least as high as greedy
    assert res.final_score >= res.greedy_score - 1e-8
