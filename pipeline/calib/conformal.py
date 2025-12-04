# pipeline/calib/conformal.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from numpy.typing import ArrayLike

from .evt_pot import (
    MeanExcessDiagnostics,
    PotModel,
    fit_pot,
    invert_tail_pvalue,
    pick_u_by_mean_excess,
    tail_pvalue,
)
from eval.evd import WeibullPOT, fit_weibull_pot


@dataclass(frozen=True)
class ConformalThreshold:
    tau: float
    alpha1: float
    pot: PotModel
    gamma_orderstat: float  # gamma = kth smallest tail p-value on calib split
    k_index: int
    n_cal: int
    mean_excess_diag: MeanExcessDiagnostics
    evd_mode: Literal["gpd", "weibull"] = "gpd"
    weibull_pot: Optional[WeibullPOT] = None

    def as_dict(self) -> dict[str, float]:
        d = dict(
            tau=self.tau,
            alpha1=self.alpha1,
            gamma=self.gamma_orderstat,
            k=self.k_index,
            n_cal=self.n_cal,
            evd_mode=self.evd_mode,
        )
        d.update(
            mean_excess_status=self.mean_excess_diag.status,
            mean_excess_r2=(
                float(self.mean_excess_diag.r2) if self.mean_excess_diag.r2 is not None else None
            ),
            mean_excess_r2_threshold=self.mean_excess_diag.r2_threshold,
            mean_excess_n_exc=self.mean_excess_diag.n_exc,
            mean_excess_min_exc=self.mean_excess_diag.min_exceedances,
            mean_excess_selected_u=self.mean_excess_diag.selected_u,
        )
        d.update(self.pot.as_dict())
        if self.weibull_pot is not None:
            d.update(
                weibull_u=self.weibull_pot.u,
                weibull_xi=self.weibull_pot.xi,
                weibull_beta=self.weibull_pot.beta,
                weibull_xF=self.weibull_pot.xF,
                weibull_p_u=self.weibull_pot.p_u,
                weibull_n_exc=self.weibull_pot.n_exc,
                weibull_n_total=self.weibull_pot.n_total,
                weibull_r2=self.weibull_pot.r2_mean_excess,
            )
        return d


def split_fit_cal(
    scores: ArrayLike, ratio: float = 0.6, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=np.float64).ravel()
    rng = np.random.default_rng(seed)
    idx = np.arange(s.size)
    rng.shuffle(idx)
    cut = max(1, int(round(ratio * s.size)))
    fit_idx, cal_idx = idx[:cut], idx[cut:]
    return s[fit_idx], s[cal_idx]


def conformal_threshold_from_scores(
    scores: ArrayLike,
    alpha1: float,
    q0: float = 0.98,
    grid_size: int = 8,
    min_exceedances: int = 200,
    r2_threshold: float = 0.98,
    split_ratio: float = 0.6,
    seed: int = 0,
    evd_mode: Literal["gpd", "weibull"] = "weibull",
    endpoint_hint: float | None = None,
    min_exceedances_weibull: int = 500,
) -> ConformalThreshold:
    """
    Compute a split-conformal threshold for a single look.

    Steps
    -----
    1) Split calibration scores into FIT and CAL parts.
    2) On FIT: pick u via mean-excess; fit POT to exceedances.
    3) On CAL: compute tail p-values with the fitted POT.
    4) Set gamma as the k-th order statistic with k = ceil(alpha1*(n_cal+1)).
    5) Invert gamma via the POT model to obtain tau (overall miscoverage alpha1 in finite sample).

    Returns
    -------
    ConformalThreshold
    """
    s_fit, s_cal = split_fit_cal(scores, ratio=split_ratio, seed=seed)
    u, mean_diag = pick_u_by_mean_excess(
        s_fit,
        q0=q0,
        grid_size=grid_size,
        min_exceedances=min_exceedances,
        r2_threshold=r2_threshold,
    )
    weibull_model: Optional[WeibullPOT] = None
    if evd_mode == "weibull":
        wb = fit_weibull_pot(
            s_fit,
            q0=q0,
            endpoint_hint=endpoint_hint,
            min_exceed=max(min_exceedances_weibull, min_exceedances),
        )
        beta = max((wb.xF - wb.u) * max(-wb.xi, 1e-6), 1e-12)
        pm = PotModel(
            u=wb.u,
            xi=wb.xi,
            beta=beta,
            p_u=wb.p_u,
            n_exc=wb.n_exc,
            n_total=wb.n_total,
        )
        weibull_model = wb
    else:
        pm = fit_pot(s_fit, u=u)

    # Tail p-values on calibration split
    pvals = np.array([tail_pvalue(pm, float(x)) for x in s_cal], dtype=np.float64)
    n_cal = int(pvals.size)
    if n_cal < 10:
        raise ValueError("Too few calibration points in conformal split.")

    k = int(np.ceil(alpha1 * (n_cal + 1)))
    k = max(1, min(k, n_cal))  # clamp to [1, n_cal]
    gamma = float(np.partition(pvals, k - 1)[k - 1])  # k-th smallest

    # If gamma lies within the POT tail support, use the parametric inversion.
    # Otherwise fall back to the empirical quantile on the calibration split.
    if gamma <= pm.p_u + 1e-12:
        tau = float(invert_tail_pvalue(pm, gamma))
    else:
        # gamma corresponds to the k-th smallest tail p-value, i.e. the k-th largest score.
        tau = float(np.sort(s_cal)[-k])
    return ConformalThreshold(
        tau=tau,
        alpha1=float(alpha1),
        pot=pm,
        gamma_orderstat=gamma,
        k_index=k,
        n_cal=n_cal,
        mean_excess_diag=mean_diag,
        evd_mode=evd_mode,
        weibull_pot=weibull_model,
    )


def apply_threshold(scores: ArrayLike, thr: ConformalThreshold) -> np.ndarray:
    """Return boolean alarms: 1 if score > tau else 0."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    return (s > thr.tau).astype(np.uint8)


def empirical_pfa(scores_null: ArrayLike, thr: ConformalThreshold) -> float:
    s = np.asarray(scores_null, dtype=np.float64).ravel()
    if s.size == 0:
        return np.nan
    return float((s > thr.tau).mean())
