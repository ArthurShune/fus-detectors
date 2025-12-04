from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HMMParams:
    p11: float = 0.98
    p00: float = 0.995
    sigma_flow: float = 0.8
    sigma_null: float = 1.0


def _log_norm(x: np.ndarray, sigma: float) -> np.ndarray:
    var = (sigma + 1e-12) ** 2
    return -0.5 * np.log(2.0 * np.pi * var) - 0.5 * (x**2) / var


def viterbi_binary(
    scores_t: np.ndarray, thr: float, params: HMMParams = HMMParams()
) -> np.ndarray:
    """
    Two-state HMM smoothing for thresholded scores.
    scores_t should already be standardized (e.g., CFAR output).
    Returns boolean array of length len(scores_t).
    """
    s = np.asarray(scores_t, dtype=np.float64).ravel()
    T = s.size
    if T == 0:
        return np.zeros(0, dtype=bool)

    A = np.array(
        [
            [params.p00, 1.0 - params.p11],
            [1.0 - params.p00, params.p11],
        ],
        dtype=np.float64,
    )
    logA = np.log(A + 1e-12)
    e0 = _log_norm(s, params.sigma_null)
    e1 = _log_norm(s, params.sigma_flow)

    dp = np.full((2, T), -1e18, dtype=np.float64)
    bp = np.zeros((2, T), dtype=np.int8)
    dp[0, 0], dp[1, 0] = e0[0], e1[0]

    for t in range(1, T):
        for j in (0, 1):
            prev = dp[:, t - 1] + logA[:, j]
            bp[j, t] = int(np.argmax(prev))
            dp[j, t] = np.max(prev) + (e1[t] if j == 1 else e0[t])

    path = np.zeros(T, dtype=np.int8)
    path[-1] = int(np.argmax(dp[:, -1]))
    for t in range(T - 2, -1, -1):
        path[t] = bp[path[t + 1], t + 1]

    decision = s > thr
    if T > 0:
        relax_thr = max(thr - 0.5, thr * 0.7)
        decision = np.logical_or(decision, np.logical_and(path == 1, s > relax_thr))
    return decision
