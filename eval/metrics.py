# eval/metrics.py
from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import beta

# =========================== PD-SNR ==========================================


def pd_snr_db(
    pd_map: ArrayLike,
    mask_flow: ArrayLike,
    mask_bg: ArrayLike,
    eps: float = 1e-12,
) -> float:
    """
    PD-SNR = 10 * log10( mu_flow^2 / var_bg ).
    """
    P = np.asarray(pd_map, dtype=np.float64)
    mf = np.asarray(mask_flow, dtype=bool)
    mb = np.asarray(mask_bg, dtype=bool)
    mu_flow = float(P[mf].mean()) if mf.any() else 0.0
    var_bg = float(P[mb].var()) if mb.any() else 0.0
    return 10.0 * np.log10((mu_flow * mu_flow + eps) / (var_bg + eps))


# =========================== ROC utilities ===================================


def roc_curve(
    scores_pos: ArrayLike,
    scores_neg: ArrayLike,
    num_thresh: int = 4096,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an approximate ROC curve via threshold sweep on the sample scores."""

    s_pos = np.asarray(scores_pos, dtype=np.float64).ravel()
    s_neg = np.asarray(scores_neg, dtype=np.float64).ravel()
    n_pos = s_pos.size
    n_neg = s_neg.size
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both positive and negative scores must be provided.")

    scores = np.concatenate([s_pos, s_neg])
    labels = np.concatenate(
        [np.ones_like(s_pos, dtype=np.int8), np.zeros_like(s_neg, dtype=np.int8)]
    )

    order = np.argsort(scores)[::-1]
    scores_sorted = scores[order]
    labels_sorted = labels[order]

    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)

    # Indices where the threshold (score) changes
    distinct = np.nonzero(np.diff(scores_sorted, prepend=np.inf))[0]
    tpr = tp[distinct] / float(n_pos)
    fpr = fp[distinct] / float(n_neg)
    thr = scores_sorted[distinct]

    # Prepend the operating point where threshold is +inf (everything rejected)
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    thr = np.concatenate([[np.inf], thr])

    # Append the point where threshold is -inf (everything accepted)
    tpr = np.concatenate([tpr, [1.0]])
    fpr = np.concatenate([fpr, [1.0]])
    thr = np.concatenate([thr, [-np.inf]])

    if num_thresh is not None and len(thr) > num_thresh:
        grid = np.linspace(0.0, 1.0, num_thresh)
        grid = grid**2  # concentrate samples near low FPR operating points
        idx = np.unique((grid * (len(thr) - 1)).astype(int))
        fpr = fpr[idx]
        tpr = tpr[idx]
        thr = thr[idx]

    return fpr, tpr, thr


def tpr_at_fpr_target(fpr: np.ndarray, tpr: np.ndarray, target_fpr: float) -> float:
    """
    Linear interpolation to estimate TPR at desired FPR (assumes fpr decreasing with threshold).
    """
    if target_fpr <= 0:
        return float(tpr[fpr.argmin()]) if fpr.size else 0.0
    if target_fpr >= 1:
        return float(tpr[fpr.argmax()]) if fpr.size else 1.0
    # Ensure fpr is ascending for interp
    order = np.argsort(fpr)
    fpr_sorted = fpr[order]
    tpr_sorted = tpr[order]
    return float(
        np.interp(target_fpr, fpr_sorted, tpr_sorted, left=tpr_sorted[0], right=tpr_sorted[-1])
    )


def partial_auc(fpr: np.ndarray, tpr: np.ndarray, fpr_max: float) -> float:
    """
    Trapezoidal partial AUC from FPR=0 to FPR=fpr_max.
    """
    order = np.argsort(fpr)
    f = fpr[order]
    t = tpr[order]
    # clip to [0, fpr_max]
    f = np.clip(f, 0.0, fpr_max)
    # ensure endpoints (0, t(0)) and (fpr_max, t(fpr_max))
    if f[0] > 0:
        f = np.concatenate([[0.0], f])
        t = np.concatenate([[t[0]], t])
    if f[-1] < fpr_max:
        t_end = float(np.interp(fpr_max, f, t))
        f = np.concatenate([f, [fpr_max]])
        t = np.concatenate([t, [t_end]])
    # trapezoid rule (compat: use trapz to support older NumPy)
    return float(np.trapz(f, t))


# =========================== Binomial CI (Clopper–Pearson) ====================


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Two-sided exact binomial CI.
    """
    if n <= 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2.0, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2.0, k + 1, n - k)
    return (float(lo), float(hi))
