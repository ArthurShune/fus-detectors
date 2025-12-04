# pipeline/confirm2/validator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.stats import norm, t as student_t

try:
    import torch
except Exception:  # torch optional
    torch = None

from ..calib.conformal import conformal_threshold_from_scores
from ..calib.evt_pot import (
    MeanExcessDiagnostics,
    PotModel,
    fit_pot,
    pick_u_by_mean_excess,
    tail_pvalue,
)
from .bvn_tail import joint_tail, per_look_alpha_from_pair
from .rho_fit import pearson_rho_with_ci


@dataclass(frozen=True)
class Confirm2Calib:
    """All artifacts from Confirm-2 calibration on calibration-null pools."""

    alpha2: float
    alpha1: float  # per-look miscoverage implied by rho
    rho_hat: float
    rho_lo: float
    rho_hi: float
    pm1: PotModel  # POT model for look 1 (fit split)
    pm2: PotModel  # POT model for look 2 (fit split)
    tau1: float  # conformalized threshold for look 1
    tau2: float  # conformalized threshold for look 2
    gamma1: float  # calibration gamma (order-stat pvalue) for look 1
    gamma2: float  # calibration gamma for look 2
    k1: int
    k2: int
    n_cal1: int
    n_cal2: int
    rho_eff: float
    rho_inflate: float
    copula_mode: str
    lambda_u_emp: float
    lambda_u_gauss: float
    df_t: float | None
    mean_excess_diag1: MeanExcessDiagnostics
    mean_excess_diag2: MeanExcessDiagnostics


@dataclass(frozen=True)
class Confirm2Eval:
    """Evaluation on a (preferably disjoint) null holdout."""

    empirical_pair_pfa: float
    pair_ci_lo: float
    pair_ci_hi: float
    predicted_pair_pfa: float  # analytic BVN tail prediction at z*(rho_hat)
    alpha2_target: float
    alpha1_per_look: float
    n_null_pairs: int
    k_joint: int


# -------------------------- Helper: z-scores from scores --------------------- #


def _torch_available() -> bool:
    return torch is not None


def _torch_tail_pvalues(x: "torch.Tensor", pm: PotModel, clip: float) -> "torch.Tensor":
    xi = torch.tensor(pm.xi, dtype=x.dtype, device=x.device)
    beta = torch.tensor(pm.beta, dtype=x.dtype, device=x.device)
    u = torch.tensor(pm.u, dtype=x.dtype, device=x.device)
    p_u = torch.tensor(pm.p_u, dtype=x.dtype, device=x.device)

    y = torch.clamp((x - u) / beta, min=0.0)
    if abs(pm.xi) < 1e-12:
        p = p_u * torch.exp(-y)
    else:
        base = 1.0 + xi * y
        base = torch.clamp(base, min=clip)
        p = p_u * torch.pow(base, -1.0 / xi)
    return torch.clamp(p, clip, 1.0 - clip)


def zscores_from_scores(
    scores: ArrayLike,
    pm: PotModel,
    clip: float = 1e-12,
    device: Optional[str] = None,
) -> np.ndarray:
    """Map scores to approximately N(0,1) via a hybrid POT + empirical PIT."""

    if device is not None and not _torch_available():
        raise RuntimeError("torch not available but device specified")

    if device is not None and _torch_available():
        dev = torch.device(device)
        s_t = torch.as_tensor(scores, dtype=torch.float64, device=dev).flatten()
        n = s_t.numel()
        order = torch.argsort(s_t)
        ranks = torch.empty_like(order, dtype=torch.float64)
        ranks[order] = torch.arange(1, n + 1, dtype=torch.float64, device=dev)
        cdf_emp = (ranks + 0.5) / (n + 1.0)
        tail_mask = s_t > pm.u
        if torch.any(tail_mask):
            p_tail = _torch_tail_pvalues(s_t[tail_mask], pm, clip)
            cdf_emp = cdf_emp.clone()
            cdf_emp[tail_mask] = 1.0 - p_tail
        cdf_emp = torch.clamp(cdf_emp, clip, 1.0 - clip)
        normal = torch.distributions.Normal(
            torch.tensor(0.0, device=dev), torch.tensor(1.0, device=dev)
        )
        z = normal.icdf(cdf_emp)
        return z.cpu().numpy()

    s = np.asarray(scores, dtype=np.float64).ravel()
    from scipy.stats import rankdata

    cdf_emp = rankdata(s, method="average") / (s.size + 1.0)
    tail_mask = s > pm.u
    if tail_mask.any():
        p_tail = np.array([tail_pvalue(pm, float(x)) for x in s[tail_mask]], dtype=np.float64)
        p_tail = np.clip(p_tail, clip, 1.0 - clip)
        cdf_emp[tail_mask] = 1.0 - p_tail

    cdf_emp = np.clip(cdf_emp, clip, 1.0 - clip)
    z = norm.ppf(cdf_emp)
    return z


TAIL_PROB_DEFAULT = 0.98
LAMBDA_EPS = 1e-8
LAMBDA_THRESHOLD = 0.02
LAMBDA_TOL = 0.01


def _empirical_tail_dependence(
    z1: np.ndarray,
    z2: np.ndarray,
    u: float = TAIL_PROB_DEFAULT,
) -> tuple[float, int, int]:
    u1 = norm.cdf(z1)
    u2 = norm.cdf(z2)
    mask = (u1 > u) & (u2 > u)
    joint = float(mask.mean())
    lambda_emp = joint / max(1.0 - u, LAMBDA_EPS)
    return float(lambda_emp), int(mask.sum()), int(z1.size)


def _gaussian_tail_dependence(rho: float, u: float = TAIL_PROB_DEFAULT) -> float:
    z = norm.ppf(u)
    jt = joint_tail(z, rho)
    return float(jt / max(1.0 - u, LAMBDA_EPS))


def _t_copula_lambda(rho: float, df: float) -> float:
    if df <= 2.0:
        raise ValueError("df must exceed 2 for finite covariance.")
    arg = np.sqrt((df + 1.0) * max(1e-12, 1.0 - rho) / max(1e-12, 1.0 + rho))
    return float(2.0 * student_t.cdf(-arg, df=df + 1.0))


def _solve_t_df_for_lambda(rho: float, lambda_target: float) -> float:
    if lambda_target <= 0.0 or lambda_target >= 1.0:
        raise ValueError("Tail dependence target must lie in (0,1).")
    lam_max = _t_copula_lambda(rho, df=2.1)
    if lambda_target > lam_max:
        raise ValueError("Target tail dependence exceeds t-copula capability.")

    if lambda_target < 1e-6:
        return float("inf")

    def objective(df: float) -> float:
        return _t_copula_lambda(rho, df) - lambda_target

    # As df -> inf, lambda -> 0. Use a generous upper bound.
    hi = 2000.0
    lo = 2.1
    return float(brentq(objective, lo, hi, maxiter=256))


# -------------------------- Main: calibration & evaluation ------------------- #


def calibrate_confirm2(
    scores1_cal: ArrayLike,
    scores2_cal: ArrayLike,
    alpha2_target: float,
    q0: float = 0.98,
    min_exceedances: int = 300,
    split_ratio: float = 0.6,
    seed: int = 0,
    rho_inflate: float = 0.0,
    device: Optional[str] = None,
    evd_mode: Literal["gpd", "weibull"] = "gpd",
    endpoint_hint: float | None = None,
    min_exceedances_weibull: int = 500,
) -> Confirm2Calib:
    """
    Calibrate Confirm-2 from two *calibration-null* score sequences (same length recommended).

    Steps
    -----
    1) Fit POT models per-look (threshold u via mean-excess on the FIT split).
    2) Transform calibration scores to z-scores via POT tail p-values; estimate rho with CI.
    3) Map alpha2 -> per-look alpha1 using BVN tail and rho_hat.
    4) Compute conformalized thresholds tau1,tau2 at alpha1 on each look.

    Returns
    -------
    Confirm2Calib
    """
    s1 = np.asarray(scores1_cal, dtype=np.float64).ravel()
    s2 = np.asarray(scores2_cal, dtype=np.float64).ravel()
    if s1.size < 1000 or s2.size < 1000:
        raise ValueError(
            "Need >=1000 calibration scores per look for stable Confirm-2 calibration."
        )

    # --- Fit POT on random FIT split (conformal handles split internally too)
    rng = np.random.default_rng(seed)
    idx1 = rng.permutation(s1.size)
    idx2 = rng.permutation(s2.size)
    fit1 = s1[idx1[: int(split_ratio * s1.size)]]
    fit2 = s2[idx2[: int(split_ratio * s2.size)]]

    u1, _diag_fit1 = pick_u_by_mean_excess(fit1, q0=q0, min_exceedances=min_exceedances)
    u2, _diag_fit2 = pick_u_by_mean_excess(fit2, q0=q0, min_exceedances=min_exceedances)
    pm1 = fit_pot(fit1, u1)
    pm2 = fit_pot(fit2, u2)

    # --- Estimate rho in z-space (Gaussian copula proxy)
    z1 = zscores_from_scores(s1, pm1, device=device)
    z2 = zscores_from_scores(s2, pm2, device=device)
    rho_hat, rho_lo, rho_hi = pearson_rho_with_ci(z1, z2)

    rho_eff = float(np.clip(rho_hat + rho_inflate, -0.999, 0.999))
    alpha1_gauss = per_look_alpha_from_pair(alpha2_target, rho_eff)

    lambda_emp, _tail_hits, _tail_total = _empirical_tail_dependence(z1, z2)
    lambda_gauss = _gaussian_tail_dependence(rho_eff)
    copula_mode = "gaussian"
    df_t: float | None = None
    alpha1 = float(alpha1_gauss)

    if lambda_emp > lambda_gauss + LAMBDA_TOL and lambda_emp > LAMBDA_THRESHOLD:
        lam_ref = min(1.0, lambda_emp)
        try:
            df_candidate = _solve_t_df_for_lambda(rho_eff, lam_ref)
            if np.isfinite(df_candidate):
                df_t = float(df_candidate)
                copula_mode = "t"
            else:
                copula_mode = "gaussian"
                df_t = None
        except Exception:
            copula_mode = "empirical"
            df_t = None
        alt_alpha = alpha2_target / max(lam_ref, 1e-6)
        alt_alpha = min(0.2, max(alpha1_gauss, alt_alpha))
        alpha1 = float(alt_alpha)
    # If the tail diagnostics force a more conservative mode, alpha1 is widened above.

    # --- Conformal thresholds (finite-sample)
    # Use full calibration pools for robust conformalization
    thr1 = conformal_threshold_from_scores(
        s1,
        alpha1=alpha1,
        split_ratio=split_ratio,
        seed=seed,
        evd_mode=evd_mode,
        endpoint_hint=endpoint_hint,
        min_exceedances_weibull=min_exceedances_weibull,
    )
    thr2 = conformal_threshold_from_scores(
        s2,
        alpha1=alpha1,
        split_ratio=split_ratio + 1e-9,
        seed=seed + 7,
        evd_mode=evd_mode,
        endpoint_hint=endpoint_hint,
        min_exceedances_weibull=min_exceedances_weibull,
    )

    return Confirm2Calib(
        alpha2=alpha2_target,
        alpha1=float(alpha1),
        rho_hat=float(rho_hat),
        rho_lo=float(rho_lo),
        rho_hi=float(rho_hi),
        pm1=pm1,
        pm2=pm2,
        tau1=thr1.tau,
        tau2=thr2.tau,
        gamma1=thr1.gamma_orderstat,
        gamma2=thr2.gamma_orderstat,
        k1=thr1.k_index,
        k2=thr2.k_index,
        n_cal1=thr1.n_cal,
        n_cal2=thr2.n_cal,
        rho_eff=rho_eff,
        rho_inflate=float(rho_inflate),
        copula_mode=copula_mode,
        lambda_u_emp=float(lambda_emp),
        lambda_u_gauss=float(lambda_gauss),
        df_t=df_t,
        mean_excess_diag1=thr1.mean_excess_diag,
        mean_excess_diag2=thr2.mean_excess_diag,
    )


def _predicted_pair_pfa(calib: Confirm2Calib) -> float:
    z_star = norm.isf(calib.alpha1)
    gauss_pred = joint_tail(z_star, calib.rho_hat)
    tail_pred = calib.lambda_u_emp * calib.alpha1
    mode = calib.copula_mode.lower()
    if mode == "gaussian":
        return float(gauss_pred)
    if not np.isfinite(tail_pred) or tail_pred <= 0.0:
        tail_pred = 0.0
    # For heavy tails (t or empirical) ensure the prediction is not optimistic.
    return float(max(gauss_pred, tail_pred))


def evaluate_confirm2(
    calib: Confirm2Calib,
    scores1_test: ArrayLike,
    scores2_test: ArrayLike,
    alpha: float = 0.05,
    device: Optional[str] = None,
) -> Confirm2Eval:
    """
    Evaluate empirical pair-Pfa on a (disjoint) null holdout and compare to analytic prediction.

    Analytic predicted pair-Pfa is computed as BVN tail at z* where alpha1 = 1 - Φ(z*).
    """
    s1 = np.asarray(scores1_test, dtype=np.float64).ravel()
    s2 = np.asarray(scores2_test, dtype=np.float64).ravel()
    if s1.size != s2.size:
        N = min(s1.size, s2.size)
        s1, s2 = s1[:N], s2[:N]

    # Empirical pair false alarms
    k_joint = int(np.sum((s1 > calib.tau1) & (s2 > calib.tau2)))
    n = int(s1.size)
    p_emp = k_joint / max(1, n)

    # Binomial two-sided CI
    from eval.metrics import clopper_pearson_ci

    lo, hi = clopper_pearson_ci(k_joint, n, alpha=alpha)

    # Analytic prediction: z from per-look alpha
    p_pred = _predicted_pair_pfa(calib)

    return Confirm2Eval(
        empirical_pair_pfa=float(p_emp),
        pair_ci_lo=float(lo),
        pair_ci_hi=float(hi),
        predicted_pair_pfa=float(p_pred),
        alpha2_target=float(calib.alpha2),
        alpha1_per_look=float(calib.alpha1),
        n_null_pairs=n,
        k_joint=k_joint,
    )


# -------------------------- Convenience sweep -------------------------------- #


def sweep_alpha2(
    scores1_cal: ArrayLike,
    scores2_cal: ArrayLike,
    scores1_test: ArrayLike,
    scores2_test: ArrayLike,
    alpha2_list: List[float],
    seed: int = 0,
) -> List[Dict[str, float]]:
    """
    Calibrate/evaluate Confirm-2 over several target pair-Pfa values.

    Returns
    -------
    list of dicts with keys:
      ['alpha2', 'alpha1', 'rho_hat', 'emp_pair', 'emp_lo', 'emp_hi', 'pred_pair', 'k_joint', 'n']
    """
    out = []
    for a2 in alpha2_list:
        calib = calibrate_confirm2(scores1_cal, scores2_cal, alpha2_target=a2, seed=seed)
        ev = evaluate_confirm2(calib, scores1_test, scores2_test)
        out.append(
            dict(
                alpha2=a2,
                alpha1=calib.alpha1,
                rho_hat=calib.rho_hat,
                rho_eff=calib.rho_eff,
                rho_inflate=calib.rho_inflate,
                emp_pair=ev.empirical_pair_pfa,
                emp_lo=ev.pair_ci_lo,
                emp_hi=ev.pair_ci_hi,
                pred_pair=ev.predicted_pair_pfa,
                k_joint=ev.k_joint,
                n=ev.n_null_pairs,
            )
        )
    return out
