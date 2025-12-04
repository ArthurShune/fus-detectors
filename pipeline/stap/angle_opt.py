# pipeline/stap/angle_opt.py
from __future__ import annotations

import itertools
from dataclasses import dataclass
from math import comb
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..gpu.linalg import cholesky_solve_hermitian, get_device, to_tensor
from .covariance import sample_covariance

# ------------------------------- Dataclasses -------------------------------- #


@dataclass(frozen=True)
class AngleOptResult:
    selected: List[int]  # sorted unique indices
    score_history: List[float]  # best score after each greedy step
    final_score: float  # score after local swaps
    greedy_score: float  # score after greedy, before swaps
    n_swaps: int  # number of successful swaps performed
    diag_load: float  # used diagonal loading
    device: str  # 'cpu' or 'cuda'


# ---------------------------- Utility functions ----------------------------- #


def _hermitianize_np(R: np.ndarray) -> np.ndarray:
    return 0.5 * (R + R.conj().T)


def _avg_covariance(Rs: torch.Tensor, idx: Sequence[int]) -> torch.Tensor:
    """Average covariance over selected indices. Rs: (A, M, M)."""
    if len(idx) == 0:
        raise ValueError("Empty index set.")
    R = torch.mean(Rs[torch.as_tensor(idx, device=Rs.device, dtype=torch.long)], dim=0)
    # enforce Hermitian numerically
    return 0.5 * (R + R.conj().transpose(-2, -1))


def _score_sH_invR_s(
    R: torch.Tensor,
    s: torch.Tensor,
    diag_load: float = 1e-3,
) -> torch.Tensor:
    """
    Compute sᴴ inv(R + dl * mu I) s, where mu = tr(R)/M, returned as real scalar tensor.
    """
    *_, M, _ = R.shape
    device, dtype = R.device, R.dtype
    mu = torch.real(torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)) / float(M)  # (...,)
    identity = torch.eye(M, dtype=dtype, device=device)
    Rl = R + (diag_load * mu).reshape(*mu.shape, 1, 1) * identity
    x, _ = cholesky_solve_hermitian(Rl, s)  # shape aligns (handle batch inside solver)
    # If s was 1D and R had no batch, solver returns (M,)
    if x.dim() == 1:
        return torch.real(torch.sum(s.conj() * x))
    # Else batched: (...,); for our use we ensure unbatched here
    return torch.real(torch.sum(s.conj() * x, dim=-1))


def _batched_scores_for_candidates(
    Rs_sum: torch.Tensor,  # (M,M) sum of covs in current set (can be zero if empty)
    count: int,  # |K| (can be 0)
    Rs_all: torch.Tensor,  # (A,M,M)
    candidates: Sequence[int],  # indices not yet selected
    s: torch.Tensor,  # (M,) complex
    diag_load: float,
) -> torch.Tensor:
    """
    Evaluate score for adding each candidate to the current set, **batched**.

    For candidate c: R_cand = (Rs_sum + Rs_all[c]) / (count+1)
    Returns a tensor of shape (len(candidates),) with scores.
    """
    device, dtype = Rs_all.device, Rs_all.dtype
    M = Rs_all.shape[-1]
    if count == 0:
        R_batch = Rs_all[torch.as_tensor(candidates, device=device)]
    else:
        # (len(cand), M, M)
        add = Rs_all[torch.as_tensor(candidates, device=device)]
        R_batch = (Rs_sum.unsqueeze(0) + add) / float(count + 1)
    # Hermitianize
    R_batch = 0.5 * (R_batch + R_batch.conj().transpose(-2, -1))

    # Diagonal load per candidate
    mu = torch.real(torch.diagonal(R_batch, dim1=-2, dim2=-1).sum(-1)) / float(M)  # (B,)
    identity = torch.eye(M, dtype=dtype, device=device)
    Rl = R_batch + (diag_load * mu).reshape(-1, 1, 1) * identity

    # Solve Rl x = s for all candidates (broadcast s)
    sB = s.expand(Rl.shape[0], -1)  # (B,M)
    x, _ = cholesky_solve_hermitian(Rl, sB)
    scores = torch.real(torch.sum(sB.conj() * x, dim=-1))  # (B,)
    return scores


# ---------------------------- Public API (snapshots) ------------------------- #


def covariances_from_snapshots(
    snapshots_per_angle: Sequence[np.ndarray],
    center: bool = True,
    dtype: np.dtype = np.complex64,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Convert a list of snapshot matrices X_k (M x N_k) into a torch tensor Rs (A,M,M).

    Each X_k should be the space-time snapshots for a given angle, column-centered
    if `center=True`. We use the *biased* covariance (1/N) X Xᴴ to match other modules.
    """
    Rs_np: List[np.ndarray] = []
    for X in snapshots_per_angle:
        X = np.asarray(X)
        assert X.ndim == 2, "Each snapshot matrix must be (M, N_k)."
        if center:
            X = X - X.mean(axis=1, keepdims=True)
        Rk = sample_covariance(X)
        Rs_np.append(_hermitianize_np(Rk.astype(dtype, copy=False)))
    Rs = np.stack(Rs_np, axis=0)
    dtype_out = torch.complex64 if np.iscomplexobj(Rs) else torch.float32
    return to_tensor(Rs, device=device, dtype=dtype_out)


# ----------------------------- Optimizer core -------------------------------- #


def greedy_then_swap_optimize(
    Rs: torch.Tensor,  # (A,M,M), complex Hermitian PSD (ish)
    s: torch.Tensor,  # (M,), complex steering
    K: int,
    diag_load: float = 1e-3,
    max_swap_passes: int = 2,
    rel_improve_tol: float = 1e-4,
    device: Optional[str] = None,
) -> AngleOptResult:
    """
    Greedy add K angles maximizing score sᴴ inv(AvgCov) s, then run local 1-swap passes.

    Returns selection (sorted unique) and score history.
    """
    if K < 1:
        raise ValueError("K must be >= 1")
    A, M, M2 = Rs.shape
    assert M == M2, "Covariance matrices must be square."
    if K > A:
        raise ValueError("K cannot exceed #angles A.")

    dev = get_device(device)
    Rs = Rs.to(dev)
    s = s.to(dev)

    # Greedy
    selected: List[int] = []
    score_hist: List[float] = []
    Rs_sum = torch.zeros((M, M), dtype=Rs.dtype, device=dev)
    remaining = set(range(A))

    for _ in range(K):
        cand_list = sorted(list(remaining))
        scores = _batched_scores_for_candidates(Rs_sum, len(selected), Rs, cand_list, s, diag_load)
        j = int(torch.argmax(scores).item())
        chosen = cand_list[j]
        selected.append(chosen)
        remaining.remove(chosen)
        # Update sum and history
        Rs_sum = Rs_sum + Rs[chosen]
        # Score of the selected set (not necessary but helpful telemetry)
        R_sel = Rs_sum / float(len(selected))
        sc = float(_score_sH_invR_s(R_sel, s, diag_load=diag_load).item())
        if score_hist:
            sc = max(sc, score_hist[-1])
        score_hist.append(sc)

    greedy_score = score_hist[-1]

    # Optional brute-force refinement when tractable
    n_swaps = 0
    base_score = greedy_score
    current_set = set(selected)

    def score_of_set(idx_set: Sequence[int]) -> float:
        Rm = _avg_covariance(Rs, idx_set)
        return float(_score_sH_invR_s(Rm, s, diag_load=diag_load).item())

    max_combinations = comb(A, K)
    if max_combinations <= 2000:
        best_score = base_score
        best_subset = tuple(sorted(current_set))
        for combo in itertools.combinations(range(A), K):
            combo_score = score_of_set(combo)
            if combo_score > best_score:
                best_score = combo_score
                best_subset = combo
        current_set = set(best_subset)
        base_score = best_score
    else:
        # Local 1-swap improvement
        for _pass in range(max_swap_passes):
            improved = False
            Rs_sum_curr = torch.stack([Rs[i] for i in current_set], dim=0).sum(dim=0)
            best_score = base_score
            best_swap: Optional[Tuple[int, int]] = None
            outside = [idx for idx in range(A) if idx not in current_set]

            for out in list(current_set):
                Rs_minus = Rs_sum_curr - Rs[out]
                for inp in outside:
                    R_try = (Rs_minus + Rs[inp]) / float(K)
                    sc_try = float(_score_sH_invR_s(R_try, s, diag_load=diag_load).item())
                    if sc_try > best_score * (1.0 + rel_improve_tol):
                        best_score = sc_try
                        best_swap = (out, inp)
            if best_swap is not None:
                out, inp = best_swap
                current_set.remove(out)
                current_set.add(inp)
                base_score = best_score
                n_swaps += 1
                improved = True
            if not improved:
                break

    final_sel = sorted(list(current_set))
    return AngleOptResult(
        selected=final_sel,
        score_history=score_hist,
        final_score=base_score,
        greedy_score=greedy_score,
        n_swaps=n_swaps,
        diag_load=diag_load,
        device=str(dev),
    )


# ---------------------------- Convenience wrapper ---------------------------- #


def optimize_angles_from_snapshots(
    snapshots_per_angle: Sequence[np.ndarray],
    s: np.ndarray | torch.Tensor,
    K: int,
    diag_load: float = 1e-3,
    center: bool = True,
    device: Optional[str] = None,
) -> AngleOptResult:
    """
    High-level path: supply per-angle snapshots (M x N_k), steering s, and K.
    """
    Rs = covariances_from_snapshots(snapshots_per_angle, center=center, device=device)
    steer_dtype = torch.complex64 if np.iscomplexobj(s) else torch.float32
    s_t = to_tensor(s, device=device, dtype=steer_dtype)
    return greedy_then_swap_optimize(Rs, s_t, K=K, diag_load=diag_load, device=device)
