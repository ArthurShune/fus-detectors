# pipeline/calib/evt_pot.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import genpareto

EPS = 1e-12


@dataclass(frozen=True)
class PotModel:
    """
    Peaks-Over-Threshold (POT) model for right-tail scores S.

    Parameters
    ----------
    u : float
        Threshold.
    xi : float
        GPD shape parameter (ξ).
    beta : float
        GPD scale parameter (β) > 0.
    p_u : float
        Empirical exceedance rate p_u = P(S > u) estimated from the fit sample.
    n_exc : int
        Number of exceedances used in the fit.
    n_total : int
        Total number of calibration samples used to estimate p_u.
    """

    u: float
    xi: float
    beta: float
    p_u: float
    n_exc: int
    n_total: int

    def as_dict(self) -> dict[str, float]:
        return dict(
            u=self.u,
            xi=self.xi,
            beta=self.beta,
            p_u=self.p_u,
            n_exc=self.n_exc,
            n_total=self.n_total,
        )


@dataclass(frozen=True)
class MeanExcessDiagnostics:
    status: Literal["ok", "fallback_r2", "fallback_count", "fallback_default"]
    selected_u: float
    selected_index: int
    r2: float | None
    r2_threshold: float
    n_exc: int
    min_exceedances: int
    grid: np.ndarray
    mean_excess: np.ndarray


# ----------------------------- Utilities ------------------------------------ #


def fit_gpd(exceedances: ArrayLike) -> tuple[float, float]:
    """
    Convenience wrapper returning the (shape, scale) parameters of a GPD fit.

    Parameters
    ----------
    exceedances : array-like
        Samples above a threshold (already shifted).

    Returns
    -------
    (shape, scale) : tuple of floats
    """
    y = np.asarray(exceedances, dtype=np.float64).ravel()
    if y.size < 10:
        raise ValueError("Need at least 10 exceedances to fit GPD.")
    if np.any(y < 0):
        raise ValueError("Exceedances must be non-negative.")
    xi, _, beta = genpareto.fit(y, floc=0.0)
    return float(xi), float(beta)


def _check_scores(scores: ArrayLike) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64).ravel()
    if s.size < 100:
        raise ValueError("Need at least 100 scores for a stable POT fit.")
    if not np.all(np.isfinite(s)):
        raise ValueError("Scores contain non-finite values.")
    return s


def mean_excess(scores: ArrayLike, u_values: ArrayLike) -> np.ndarray:
    """
    Mean-excess function e(u) = E[S - u | S > u] estimated empirically.

    Returns
    -------
    e : np.ndarray
        Mean-excess values for each u in u_values; NaN if no exceedances.
    """
    s = _check_scores(scores)
    u = np.asarray(u_values, dtype=np.float64)
    e = np.full_like(u, np.nan, dtype=np.float64)
    s_sort = np.sort(s)
    for i, ui in enumerate(u):
        idx = np.searchsorted(s_sort, ui, side="right")
        exc = s_sort[idx:] - ui
        if exc.size > 0:
            e[i] = exc.mean()
    return e


def _linear_r2(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m, b = np.polyfit(x, y, 1)
    yhat = m * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + EPS
    return 1.0 - ss_res / ss_tot


def pick_u_by_mean_excess(
    scores: ArrayLike,
    q0: float = 0.98,
    grid_size: int = 8,
    min_exceedances: int = 200,
    r2_threshold: float = 0.98,
) -> tuple[float, MeanExcessDiagnostics]:
    """
    Pick a high threshold u by mean-excess linearity.

    Strategy
    --------
    Build a grid of candidate thresholds between the q0-quantile and the 0.999-quantile.
    Compute mean-excess e(u) on the tail of the grid and select the *highest* u
    such that the last K points (K = min(grid_size, 8)) yield R^2 >= r2_threshold
    and the number of exceedances >= min_exceedances. If none qualifies, fall back
    to the max candidate satisfying the min_exceedances constraint; if still none,
    use the q0-quantile.

    Returns
    -------
    u : float
        Selected threshold.
    """
    s = _check_scores(scores)
    lo = float(np.quantile(s, q0))
    hi = float(np.quantile(s, min(0.999, max(q0 + 0.01, q0 + 0.005))))
    grid = np.linspace(lo, hi, grid_size, dtype=np.float64)
    e = mean_excess(s, grid)

    # Use last K points to assess linearity
    K = min(grid_size, 8)
    best_u = None
    best_idx = -1
    best_r2 = None
    best_nexc = 0
    for i in range(grid_size - 1, -1, -1):
        u_i = grid[i]
        # count exceedances
        n_exc = int((s > u_i).sum())
        if n_exc < min_exceedances:
            continue
        # R^2 on the last K points up to i
        j0 = max(0, i - (K - 1))
        xu = grid[j0 : i + 1]
        yu = e[j0 : i + 1]
        if np.any(~np.isfinite(yu)) or yu.size < 3:
            continue
        r2 = _linear_r2(xu, yu)
        if r2 >= r2_threshold:
            best_u = float(u_i)
            best_idx = i
            best_r2 = float(r2)
            best_nexc = n_exc
            break

    if best_u is not None:
        diag = MeanExcessDiagnostics(
            status="ok",
            selected_u=best_u,
            selected_index=best_idx,
            r2=best_r2,
            r2_threshold=float(r2_threshold),
            n_exc=int(best_nexc),
            min_exceedances=int(min_exceedances),
            grid=grid.copy(),
            mean_excess=e.copy(),
        )
        return best_u, diag

    fallback_status: Literal["fallback_r2", "fallback_count", "fallback_default"] = "fallback_r2"

    # Fallback: highest u with enough exceedances
    for i in range(grid_size - 1, -1, -1):
        u_i = grid[i]
        if (s > u_i).sum() >= min_exceedances:
            fallback_status = "fallback_r2"
            diag = MeanExcessDiagnostics(
                status=fallback_status,
                selected_u=float(u_i),
                selected_index=i,
                r2=None,
                r2_threshold=float(r2_threshold),
                n_exc=int((s > u_i).sum()),
                min_exceedances=int(min_exceedances),
                grid=grid.copy(),
                mean_excess=e.copy(),
            )
            return float(u_i), diag

    fallback_status = "fallback_default"
    diag = MeanExcessDiagnostics(
        status=fallback_status,
        selected_u=float(lo),
        selected_index=0,
        r2=None,
        r2_threshold=float(r2_threshold),
        n_exc=int((s > lo).sum()),
        min_exceedances=int(min_exceedances),
        grid=grid.copy(),
        mean_excess=e.copy(),
    )
    return float(lo), diag  # last resort


# ----------------------------- Fitting -------------------------------------- #


def _fit_gpd_mle(exceedances: np.ndarray) -> tuple[float, float]:
    """
    Fit GPD(y; xi, beta) to exceedances y >= 0 via MLE (SciPy).

    Returns
    -------
    xi, beta
    """
    c, loc, scale = genpareto.fit(exceedances, floc=0.0)
    if not np.isfinite(c) or not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("GPD MLE returned invalid params.")
    return float(c), float(scale)


def _fit_gpd_moments(exceedances: np.ndarray) -> tuple[float, float]:
    """
    Method-of-moments fallback (requires xi < 0.5 to have finite variance).

    Using
        m = E[Y] = beta / (1 - xi)
        v = Var[Y] = beta^2 / ((1 - xi)^2 (1 - 2xi))

    So r = v / m^2 = 1 / (1 - 2xi)  =>  xi = (1 - 1/r) / 2
       beta = m (1 - xi)

    If the sample violates r <= 1 (numerical issues) or gives xi >= 0.49, we clip xi to 0.49.
    """
    y = exceedances
    m = float(np.mean(y))
    v = float(np.var(y, ddof=1) + EPS)
    r = v / (m * m + EPS)
    if r <= 1.0 + 1e-6:  # extremely light tail or numerical issue
        xi = 0.0
        beta = max(m, EPS)
    else:
        xi = (1.0 - 1.0 / r) / 2.0
        xi = float(np.clip(xi, -0.4, 0.49))
        beta = max(m * (1.0 - xi), EPS)
    return xi, beta


def fit_pot(
    scores: ArrayLike,
    u: float,
    prefer_mle: bool = True,
) -> PotModel:
    """
    Fit a POT/GPD model on exceedances above u.

    Parameters
    ----------
    scores : array-like
        Calibration scores (1D).
    u : float
        Threshold.
    prefer_mle : bool
        If True, attempt GPD MLE first and fallback to moments; otherwise use moments.

    Returns
    -------
    PotModel
    """
    s = _check_scores(scores)
    y = s[s > u] - u
    n_exc = int(y.size)
    n_total = int(s.size)
    if n_exc < 50:
        raise ValueError(f"Too few exceedances above u={u:.4g}: n_exc={n_exc} (<50).")
    p_u = n_exc / max(1, n_total)

    if prefer_mle:
        try:
            xi, beta = _fit_gpd_mle(y)
        except Exception:
            xi, beta = _fit_gpd_moments(y)
    else:
        xi, beta = _fit_gpd_moments(y)

    return PotModel(
        u=float(u), xi=float(xi), beta=float(beta), p_u=float(p_u), n_exc=n_exc, n_total=n_total
    )


# ------------------------- Tail functions ----------------------------------- #


def tail_pvalue(pm: PotModel, s: float) -> float:
    """
    Overall right-tail p-value: P(S >= s) under the POT model.
    For s <= u, returns 1.0 (no exceedance).

    Uses
        p(S > s) = p_u * (1 + xi*(s-u)/beta)^(-1/xi),  for xi != 0
                 = p_u * exp(-(s-u)/beta),              for xi -> 0
    """
    if s <= pm.u:
        return 1.0
    y = (s - pm.u) / max(pm.beta, EPS)
    if abs(pm.xi) < 1e-9:
        return float(pm.p_u * np.exp(-y))
    base = 1.0 + pm.xi * y
    if base <= 0.0:
        # For xi < 0, support is bounded: s cannot exceed u - beta/xi.
        return 0.0
    return float(pm.p_u * base ** (-1.0 / pm.xi))


def tail_quantile(pm: PotModel, alpha: float) -> float:
    """
    Quantile Q(alpha) such that P(S > Q(alpha)) = alpha with alpha <= p_u.

    Q(alpha) = u + (beta/xi)*[(alpha/p_u)^(-xi) - 1],  xi != 0
             = u + beta * log(p_u/alpha),             xi -> 0
    """
    alpha = float(alpha)
    if alpha <= 0.0:
        return np.inf
    if alpha > pm.p_u + 1e-15:
        raise ValueError(
            f"alpha={alpha:.3g} > p_u={pm.p_u:.3g}; quantile lies below u, "
            "use empirical quantile instead."
        )
    if abs(pm.xi) < 1e-9:
        return float(pm.u + pm.beta * np.log(pm.p_u / alpha))
    return float(pm.u + (pm.beta / pm.xi) * ((alpha / pm.p_u) ** (-pm.xi) - 1.0))


def invert_tail_pvalue(pm: PotModel, gamma: float) -> float:
    """
    Invert tail p-value gamma -> threshold tau so that P(S > tau) = gamma.
    """
    return tail_quantile(pm, gamma)


# ------------------------- Bootstrap quantiles ------------------------------- #


def bootstrap_quantile_ci(
    scores: ArrayLike,
    u: float,
    alpha: float,
    B: int = 200,
    prefer_mle: bool = True,
    random_state: np.random.Generator | None = None,
    ci: tuple[float, float] = (0.025, 0.975),
) -> tuple[float, float, float]:
    """
    Parametric bootstrap CI for the POT quantile Q(alpha).

    Returns
    -------
    q_hat, q_lo, q_hi
    """
    rng = np.random.default_rng() if random_state is None else random_state
    s = _check_scores(scores)
    pm = fit_pot(s, u=u, prefer_mle=prefer_mle)
    q_hat = tail_quantile(pm, alpha)

    # Resample exceedances with replacement; re-fit and re-evaluate Q
    exc = s[s > u] - u
    if exc.size < 50:
        raise ValueError("Too few exceedances for bootstrap.")
    qs = np.empty(B, dtype=np.float64)
    for b in range(B):
        yb = rng.choice(exc, size=exc.size, replace=True)
        try:
            xib, betab = _fit_gpd_mle(yb)
        except Exception:
            xib, betab = _fit_gpd_moments(yb)
        pmb = PotModel(u=u, xi=xib, beta=betab, p_u=pm.p_u, n_exc=pm.n_exc, n_total=pm.n_total)
        qs[b] = tail_quantile(pmb, alpha)
    q_lo, q_hi = np.quantile(qs, ci)
    return float(q_hat), float(q_lo), float(q_hi)
