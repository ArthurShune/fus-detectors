from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class KaContractV2Config:
    """Configuration for the KA Contract v2 state machine.

    Phase 1: this config drives state/telemetry only (no score modification).
    """

    # Coverage proxies
    c_bg: float = 0.05
    c_flow: float = 0.20

    # Candidate/protection quantiles on the baseline score
    q_lo_candidate: float = 0.30
    # Upper score quantile for the candidate set. Set to 1.0 to allow KA to act
    # on extreme-score background tails (needed for ultra-low-FPR reporting).
    q_hi_candidate: float = 1.0
    # Optional extra protection: exclude the top score quantile from gating. When
    # enabled, this should be set strictly above the smallest evaluated FPR, i.e.
    # protect < alpha_min fraction of the score mass, otherwise KA cannot shift
    # the negative tail quantiles at that FPR.
    protect_hi_by_score: bool = True
    q_hi_protect: float = 0.99999

    # Risk threshold (quantile of m_alias on background-proxy tiles)
    q_risk: float = 0.90

    # R1 telemetry integrity checks
    alias_iqr_min: float = 0.25
    # Guard fraction is treated as an uplift veto (strict) and, optionally, a
    # safety-only lane (looser) when guard is actionable in background tails.
    guard_q90_max_uplift: float = 0.25
    guard_q90_max_safety: float = 0.70
    guard_iqr_bg_min: float = 0.03
    delta_tail_guard_min: float = 0.01

    # R2 differential evidence (label-free) for uplift eligibility (C2)
    delta_bf_min: float = 0.30
    delta_tail_min: float = 0.20
    # Tail association is evaluated by comparing the risk metric on an extreme
    # right-tail subset of background-proxy tiles versus a mid-quantile subset.
    # For small sample sizes, we enforce a minimum tail size via `n_tail_min`
    # by expanding the tail fraction as needed (see implementation).
    tail_frac_bg: float = 0.01
    mid_q_lo_bg: float = 0.40
    mid_q_hi_bg: float = 0.60

    # C2 guardrail: require minimum Pf realization (label-free proxy). This is
    # intentionally modest; it is meant to prevent "uplift-eligible" states
    # from triggering when the declared Pf band is effectively empty.
    pf_peak_min_c2: float = 0.05
    n_flow_min_c2: int = 50

    # Coverage sentinels (fraction of candidate tiles gated)
    pmin_safety: float = 0.02
    pmax_safety: float = 0.25
    pmin_safety_guard: float = 0.02
    pmax_safety_guard: float = 0.15
    pmin_uplift: float = 0.05
    pmax_uplift: float = 0.40

    # Shrink mapping parameters (used only to estimate invariance in Phase 1)
    wmin_safety: float = 0.75
    k_safety: float = 0.7
    wmin_guard: float = 0.75
    k_guard: float = 0.5
    wmin_uplift: float = 0.50
    k_uplift: float = 1.0
    iqr_logw_min: float = 0.02

    # Sample support
    n_bg_min: int = 200
    n_min: int = 500
    n_cand_min: int = 200
    n_tail_min: int = 20
    n_mid_min: int = 20

    eps: float = 1e-12


def _as_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _quantile(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    q = float(np.clip(q, 0.0, 1.0))
    return float(np.quantile(values, q))


def evaluate_ka_contract_v2(
    *,
    s_base: np.ndarray | None,
    m_alias: np.ndarray | None,
    r_guard: np.ndarray | None,
    pf_peak: np.ndarray | None = None,
    c_flow: np.ndarray | None,
    valid_mask: np.ndarray | None = None,
    config: KaContractV2Config | None = None,
) -> dict[str, Any]:
    """Evaluate KA Contract v2 and return a JSON-serializable report.

    Parameters are per-tile arrays (shape: (N_tiles,)).

    Notes
    -----
    - Phase 1 is logging-only: the resulting state does not alter scores.
    - m_alias is expected to increase with alias dominance (e.g. log(Ea/Ef)).
    - s_base is expected to increase with flow evidence (e.g. STAP PD).
    """

    cfg = config or KaContractV2Config()
    report: dict[str, Any] = {
        "state": "C0_OFF",
        "reason": "uninitialized",
        "config": asdict(cfg),
        "metrics": {},
    }

    if s_base is None or m_alias is None or r_guard is None or c_flow is None:
        report["reason"] = "missing_inputs"
        return report

    s_base = np.asarray(s_base, dtype=np.float64).ravel()
    m_alias = np.asarray(m_alias, dtype=np.float64).ravel()
    r_guard = np.asarray(r_guard, dtype=np.float64).ravel()
    pf_peak = None if pf_peak is None else np.asarray(pf_peak, dtype=bool).ravel()
    c_flow = np.asarray(c_flow, dtype=np.float64).ravel()
    if not (s_base.size == m_alias.size == r_guard.size == c_flow.size):
        report["reason"] = "shape_mismatch"
        report["metrics"] = {
            "n_s_base": int(s_base.size),
            "n_m_alias": int(m_alias.size),
            "n_r_guard": int(r_guard.size),
            "n_c_flow": int(c_flow.size),
        }
        return report
    if pf_peak is not None and pf_peak.size != s_base.size:
        report["reason"] = "pf_peak_shape_mismatch"
        report["metrics"] = {
            "n_tiles": int(s_base.size),
            "n_pf_peak": int(pf_peak.size),
        }
        return report

    if valid_mask is None:
        valid = np.isfinite(s_base) & np.isfinite(m_alias) & np.isfinite(r_guard) & np.isfinite(c_flow)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool).ravel()
        if valid_mask.size != s_base.size:
            report["reason"] = "valid_mask_shape_mismatch"
            report["metrics"] = {
                "n_tiles": int(s_base.size),
                "n_valid_mask": int(valid_mask.size),
            }
            return report
        valid = valid_mask & np.isfinite(s_base) & np.isfinite(m_alias) & np.isfinite(r_guard) & np.isfinite(c_flow)

    n_total = int(s_base.size)
    n_valid = int(np.sum(valid))
    report["metrics"].update(
        {
            "n_tiles": n_total,
            "n_valid": n_valid,
        }
    )
    if n_valid < int(cfg.n_min):
        report["reason"] = "insufficient_samples"
        return report

    c_bg = float(cfg.c_bg)
    c_flow_thr = float(cfg.c_flow)
    T_bg = valid & (c_flow <= c_bg)
    T_flow = valid & (c_flow >= c_flow_thr)
    n_bg = int(np.sum(T_bg))
    n_flow = int(np.sum(T_flow))
    report["metrics"].update(
        {
            "n_bg_proxy": n_bg,
            "n_flow_proxy": n_flow,
            "c_bg": float(cfg.c_bg),
            "c_flow": float(cfg.c_flow),
        }
    )
    # Log Pf-peak telemetry as early as possible so it is present even when the
    # contract exits early (e.g. guard-dominance reasons). This is used for
    # regime anchoring comparisons across sim vs real datasets.
    pf_peak_nonbg: float | None = None
    pf_peak_flow: float | None = None
    n_nonbg = 0
    if pf_peak is not None:
        # Non-background proxy tiles: anything not in the bg proxy.
        T_nonbg = valid & (c_flow > c_bg)
        n_nonbg = int(np.sum(T_nonbg))
        if n_nonbg > 0:
            pf_peak_nonbg = float(np.mean(pf_peak[T_nonbg]))
        if n_flow > 0:
            pf_peak_flow = float(np.mean(pf_peak[T_flow]))
        report["metrics"].update(
            {
                "n_nonbg_proxy": n_nonbg,
                "pf_peak_nonbg": pf_peak_nonbg,
                "pf_peak_flow": pf_peak_flow,
            }
        )
    if n_bg < int(cfg.n_bg_min):
        report["reason"] = "insufficient_bg_samples"
        return report

    # --- R1: telemetry integrity ---
    m_alias_bg = m_alias[T_bg]
    q25 = _quantile(m_alias_bg, 0.25)
    q75 = _quantile(m_alias_bg, 0.75)
    iqr_alias_bg = None if q25 is None or q75 is None else float(q75 - q25)
    guard_q90 = _quantile(r_guard[valid], 0.90)
    report["metrics"].update(
        {
            "iqr_alias_bg": _as_float(iqr_alias_bg),
            "guard_q90": _as_float(guard_q90),
        }
    )
    if iqr_alias_bg is None or iqr_alias_bg < float(cfg.alias_iqr_min):
        report["reason"] = "alias_metric_flat"
        return report
    if guard_q90 is None:
        report["reason"] = "guard_metric_missing"
        return report

    # --- R2: differential evidence (label-free) on m_alias (always logged) ---
    med_bg = float(np.median(m_alias[T_bg])) if n_bg else float(np.median(m_alias[valid]))
    if n_flow:
        med_flow = float(np.median(m_alias[T_flow]))
    else:
        med_flow = float(np.median(m_alias[valid]))
    delta_bf = float(med_bg - med_flow)

    s_bg = s_base[T_bg]
    if s_bg.size >= 5:
        # Size-aware tail definition: for small n_bg, a fixed 1% tail can be
        # smaller than `n_tail_min` and make delta_tail undefined. Enforce a
        # minimum tail count by expanding the tail fraction as needed.
        tail_frac = max(float(cfg.tail_frac_bg), float(cfg.n_tail_min) / float(max(n_bg, 1)))
        tail_frac = float(np.clip(tail_frac, 0.0, 1.0))
        tail_q = float(np.clip(1.0 - tail_frac, 0.0, 1.0))
        mid_q_lo = float(np.clip(cfg.mid_q_lo_bg, 0.0, 1.0))
        mid_q_hi = float(np.clip(cfg.mid_q_hi_bg, 0.0, 1.0))
        if mid_q_hi < mid_q_lo:
            mid_q_lo, mid_q_hi = mid_q_hi, mid_q_lo

        tail_thr = float(np.quantile(s_bg, tail_q))
        mid_thr_lo = float(np.quantile(s_bg, mid_q_lo))
        mid_thr_hi = float(np.quantile(s_bg, mid_q_hi))
    else:
        tail_frac = float(cfg.tail_frac_bg)
        tail_q = 0.99
        mid_q_lo = float(cfg.mid_q_lo_bg)
        mid_q_hi = float(cfg.mid_q_hi_bg)
        tail_thr = float(np.max(s_bg)) if s_bg.size else float(np.max(s_base[valid]))
        mid_thr_lo, mid_thr_hi = float("-inf"), float("inf")

    T_tail = T_bg & (s_base >= tail_thr)
    T_mid = T_bg & (s_base >= mid_thr_lo) & (s_base <= mid_thr_hi)
    n_tail = int(np.sum(T_tail))
    n_mid = int(np.sum(T_mid))
    if n_tail >= int(cfg.n_tail_min) and n_mid >= int(cfg.n_mid_min):
        delta_tail = float(np.median(m_alias[T_tail]) - np.median(m_alias[T_mid]))
    else:
        delta_tail = float("-inf")
    uplift_eligible_raw = bool(
        delta_bf >= float(cfg.delta_bf_min) and delta_tail >= float(cfg.delta_tail_min)
    )

    risk_mode = "alias"
    uplift_vetoed_by_guard = False
    guard_q90_max_uplift = float(cfg.guard_q90_max_uplift)
    guard_q90_max_safety = float(cfg.guard_q90_max_safety)
    report["metrics"].update(
        {
            "delta_bg_flow_median": float(delta_bf),
            "delta_tail": float(delta_tail) if np.isfinite(delta_tail) else None,
            "uplift_eligible_raw": uplift_eligible_raw,
            "tail_frac_bg": float(tail_frac),
            "tail_q_bg": float(tail_q),
            "tail_thr_bg": float(tail_thr),
            "mid_q_lo_bg": float(mid_q_lo),
            "mid_q_hi_bg": float(mid_q_hi),
            "mid_thr_lo_bg": float(mid_thr_lo),
            "mid_thr_hi_bg": float(mid_thr_hi),
            "n_tail": n_tail,
            "n_mid": n_mid,
            "guard_q90_max_uplift": guard_q90_max_uplift,
            "guard_q90_max_safety": guard_q90_max_safety,
        }
    )

    # Guard dominates: veto uplift always, but allow a safety-only lane when the
    # guard metric is actionable in background tail tiles.
    if float(guard_q90) > guard_q90_max_uplift:
        uplift_vetoed_by_guard = True
        if float(guard_q90) > guard_q90_max_safety:
            report["metrics"].update(
                {"risk_mode": risk_mode, "uplift_vetoed_by_guard": uplift_vetoed_by_guard}
            )
            report["reason"] = "guard_extreme_out_of_regime"
            return report

        r_guard_bg = r_guard[T_bg]
        g25 = _quantile(r_guard_bg, 0.25)
        g75 = _quantile(r_guard_bg, 0.75)
        guard_iqr_bg = None if g25 is None or g75 is None else float(g75 - g25)
        if n_tail >= int(cfg.n_tail_min) and n_mid >= int(cfg.n_mid_min):
            delta_tail_guard = float(np.median(r_guard[T_tail]) - np.median(r_guard[T_mid]))
        else:
            delta_tail_guard = float("-inf")
        report["metrics"].update(
            {
                "guard_iqr_bg": _as_float(guard_iqr_bg),
                "delta_tail_guard": float(delta_tail_guard)
                if np.isfinite(delta_tail_guard)
                else None,
            }
        )
        if guard_iqr_bg is None or guard_iqr_bg < float(cfg.guard_iqr_bg_min):
            report["metrics"].update(
                {"risk_mode": risk_mode, "uplift_vetoed_by_guard": uplift_vetoed_by_guard}
            )
            report["reason"] = "guard_dominates_not_actionable"
            return report
        if delta_tail_guard < float(cfg.delta_tail_guard_min):
            report["metrics"].update(
                {"risk_mode": risk_mode, "uplift_vetoed_by_guard": uplift_vetoed_by_guard}
            )
            report["reason"] = "guard_dominates_not_actionable"
            return report

        risk_mode = "guard"

    uplift_vetoed_by_pf_peak = False

    uplift_eligible = bool(uplift_eligible_raw and not uplift_vetoed_by_guard and risk_mode == "alias")
    pf_peak_min_c2 = float(cfg.pf_peak_min_c2)
    n_flow_min_c2 = int(cfg.n_flow_min_c2)
    uplift_veto_pf_peak_reason: str | None = None
    if uplift_eligible and pf_peak_min_c2 > 0.0:
        # If Pf peak telemetry is missing or indicates Pf is not realized on the
        # *flow proxy* set, downgrade to C1. This keeps C2 aligned with "uplift
        # eligible" meaning without relying on non-bg tiles that may be dominated
        # by cluttery or ambiguous regions.
        if pf_peak is None:
            uplift_vetoed_by_pf_peak = True
            uplift_veto_pf_peak_reason = "pf_peak_missing"
        elif n_flow < n_flow_min_c2:
            uplift_vetoed_by_pf_peak = True
            uplift_veto_pf_peak_reason = "pf_peak_flow_insufficient_samples"
        elif pf_peak_flow is None:
            uplift_vetoed_by_pf_peak = True
            uplift_veto_pf_peak_reason = "pf_peak_flow_missing"
        elif float(pf_peak_flow) < pf_peak_min_c2:
            uplift_vetoed_by_pf_peak = True
            uplift_veto_pf_peak_reason = "pf_peak_flow_below_min"
        if uplift_vetoed_by_pf_peak:
            uplift_eligible = False

    report["metrics"].update(
        {
            "uplift_vetoed_by_guard": uplift_vetoed_by_guard,
            "uplift_vetoed_by_pf_peak": uplift_vetoed_by_pf_peak,
            "uplift_veto_pf_peak_reason": uplift_veto_pf_peak_reason,
            "pf_peak_min_c2": pf_peak_min_c2,
            "n_flow_min_c2": n_flow_min_c2,
            "uplift_eligible": uplift_eligible,
            "risk_mode": risk_mode,
        }
    )

    # --- Candidate/protected sets (coverage sentinels) ---
    s_valid = s_base[valid]
    score_q_lo = float(np.quantile(s_valid, float(cfg.q_lo_candidate)))
    score_q_hi_candidate = float(np.quantile(s_valid, float(cfg.q_hi_candidate)))
    if score_q_hi_candidate < score_q_lo:
        score_q_hi_candidate = score_q_lo
    score_q_hi_protect = float(np.quantile(s_valid, float(cfg.q_hi_protect)))

    T_prot = valid & (c_flow >= c_flow_thr)
    if bool(cfg.protect_hi_by_score):
        T_prot |= valid & (s_base >= score_q_hi_protect)

    T_cand = (
        valid
        & (~T_prot)
        & (s_base >= score_q_lo)
        & (s_base <= score_q_hi_candidate)
    )
    n_prot = int(np.sum(T_prot))
    n_cand = int(np.sum(T_cand))
    report["metrics"].update(
        {
            "q_lo_candidate": float(cfg.q_lo_candidate),
            "q_hi_candidate": float(cfg.q_hi_candidate),
            "protect_hi_by_score": bool(cfg.protect_hi_by_score),
            "q_hi_protect": float(cfg.q_hi_protect),
            "score_q_lo": float(score_q_lo),
            "score_q_hi_candidate": float(score_q_hi_candidate),
            "score_q_hi_protect": float(score_q_hi_protect),
            "n_protected": n_prot,
            "n_candidates": n_cand,
        }
    )
    if n_cand < int(cfg.n_cand_min):
        report["reason"] = "too_few_candidates"
        return report

    risk_bg = m_alias_bg if risk_mode == "alias" else r_guard[T_bg]
    risk_all = m_alias if risk_mode == "alias" else r_guard
    tau_alias = float(np.quantile(risk_bg, float(cfg.q_risk)))
    med_alias_bg = float(np.median(risk_bg))
    mad_alias = float(np.median(np.abs(risk_bg - med_alias_bg))) + float(cfg.eps)
    report["metrics"].update(
        {
            "q_risk": float(cfg.q_risk),
            "tau_alias": float(tau_alias),
            "mad_alias": float(mad_alias),
        }
    )

    gated = T_cand & (risk_all >= tau_alias)
    p_shrink = float(np.sum(gated) / max(1, n_cand))
    report["metrics"].update(
        {
            "n_gated": int(np.sum(gated)),
            "p_shrink": float(p_shrink),
        }
    )

    if risk_mode == "guard":
        pmin, pmax = float(cfg.pmin_safety_guard), float(cfg.pmax_safety_guard)
    elif uplift_eligible:
        pmin, pmax = float(cfg.pmin_uplift), float(cfg.pmax_uplift)
    else:
        pmin, pmax = float(cfg.pmin_safety), float(cfg.pmax_safety)
    report["metrics"].update({"pmin": pmin, "pmax": pmax})
    if p_shrink < pmin or p_shrink > pmax:
        report["reason"] = "coverage_out_of_bounds"
        return report

    # Estimate whether the shrink map would be effectively invariant.
    if risk_mode == "guard":
        w_min = float(cfg.wmin_guard)
        k = float(cfg.k_guard)
    else:
        w_min = float(cfg.wmin_uplift if uplift_eligible else cfg.wmin_safety)
        k = float(cfg.k_uplift if uplift_eligible else cfg.k_safety)
    # Normalized excess risk in units of MAD.
    u = np.maximum(0.0, (risk_all - tau_alias) / mad_alias)
    w = np.ones_like(s_base, dtype=np.float64)
    w[gated] = np.maximum(w_min, np.exp(-k * u[gated]))
    if np.sum(gated) >= 50:
        logw = np.log(w[gated] + float(cfg.eps))
        iqr_logw = float(np.quantile(logw, 0.75) - np.quantile(logw, 0.25))
    else:
        iqr_logw = 0.0
    report["metrics"]["iqr_logw_gated"] = float(iqr_logw)
    if iqr_logw < float(cfg.iqr_logw_min):
        report["reason"] = "invariance_no_effect"
        return report

    report["state"] = "C2_UPLIFT" if uplift_eligible else "C1_SAFETY"
    report["reason"] = "ok"
    return report


def derive_score_shrink_v2_tile_scales(
    *,
    report: dict[str, Any],
    s_base: np.ndarray,
    m_alias: np.ndarray,
    c_flow: np.ndarray,
    valid_mask: np.ndarray | None = None,
    mode: str = "safety",
) -> dict[str, Any]:
    """Derive per-tile shrink-only scale factors from a v2 contract report.

    This helper computes scale factors >= 1 that can be applied as a *division*
    on PD score maps so that the PD score is shrunk on risky tiles:
      S_post = S_pre / scale,  scale >= 1.

    Parameters
    ----------
    report:
        Output of evaluate_ka_contract_v2.
    s_base:
        Baseline per-tile detector score used by the contract state machine
        (higher = more flow evidence). For PD scoring, this should be PD
        (i.e., the right-tail PD score map).
    m_alias:
        Per-tile alias metric (larger = more alias-like), e.g. log(Ea/Ef).
    c_flow:
        Per-tile flow coverage fraction in [0,1].
    mode:
        "safety" (Phase 2) or "uplift" (Phase 3+). Phase 2 uses "safety".
    """

    out: dict[str, Any] = {
        "apply": False,
        "reason": "uninitialized",
        "scale_tiles": None,
        "gated_tiles": None,
        "protected_tiles": None,
        "candidate_tiles": None,
        "stats": {},
    }

    if not isinstance(report, dict):
        out["reason"] = "invalid_report"
        return out
    if report.get("reason") != "ok":
        out["reason"] = f"contract_{report.get('reason')}"
        return out

    cfg = report.get("config") or {}
    metrics = report.get("metrics") or {}
    risk_mode = str(metrics.get("risk_mode") or "alias").strip().lower()
    tau_alias = metrics.get("tau_alias")
    mad_alias = metrics.get("mad_alias")
    score_q_lo = metrics.get("score_q_lo")
    score_q_hi_candidate = metrics.get("score_q_hi_candidate", metrics.get("score_q_hi"))
    score_q_hi_protect = metrics.get("score_q_hi_protect", metrics.get("score_q_hi"))
    protect_hi_by_score = bool(cfg.get("protect_hi_by_score", True))
    if tau_alias is None or mad_alias is None or score_q_lo is None or score_q_hi_candidate is None:
        out["reason"] = "missing_thresholds"
        return out
    if protect_hi_by_score and score_q_hi_protect is None:
        out["reason"] = "missing_protect_threshold"
        return out

    try:
        c_bg = float(cfg.get("c_bg", 0.05))
        c_flow_thr = float(cfg.get("c_flow", 0.20))
    except Exception:
        out["reason"] = "invalid_config"
        return out

    mode_norm = str(mode or "safety").strip().lower()
    if mode_norm not in {"safety", "uplift"}:
        out["reason"] = "invalid_mode"
        return out

    if mode_norm == "uplift":
        w_min = float(cfg.get("wmin_uplift", 0.50))
        k = float(cfg.get("k_uplift", 1.0))
    else:
        if risk_mode == "guard":
            w_min = float(cfg.get("wmin_guard", 0.75))
            k = float(cfg.get("k_guard", 0.5))
        else:
            w_min = float(cfg.get("wmin_safety", 0.75))
            k = float(cfg.get("k_safety", 0.7))

    s_base = np.asarray(s_base, dtype=np.float64).ravel()
    m_alias = np.asarray(m_alias, dtype=np.float64).ravel()
    c_flow = np.asarray(c_flow, dtype=np.float64).ravel()
    if not (s_base.size == m_alias.size == c_flow.size):
        out["reason"] = "shape_mismatch"
        return out

    if valid_mask is None:
        valid = np.isfinite(s_base) & np.isfinite(m_alias) & np.isfinite(c_flow)
    else:
        vm = np.asarray(valid_mask, dtype=bool).ravel()
        if vm.size != s_base.size:
            out["reason"] = "valid_mask_shape_mismatch"
            return out
        valid = vm & np.isfinite(s_base) & np.isfinite(m_alias) & np.isfinite(c_flow)

    # Reconstruct candidate/protected and gate conditions using the same
    # thresholds recorded in the report.
    T_bg = valid & (c_flow <= c_bg)
    T_prot = valid & (c_flow >= c_flow_thr)
    if protect_hi_by_score:
        T_prot |= valid & (s_base >= float(score_q_hi_protect))
    T_cand = (
        valid
        & (~T_prot)
        & (s_base >= float(score_q_lo))
        & (s_base <= float(score_q_hi_candidate))
    )
    gated = T_cand & (m_alias >= float(tau_alias))

    eps = float(cfg.get("eps", 1e-12))
    denom = float(mad_alias) if float(mad_alias) > 0.0 else eps
    u = np.maximum(0.0, (m_alias - float(tau_alias)) / denom)
    w = np.ones_like(s_base, dtype=np.float64)
    if np.any(gated):
        w[gated] = np.maximum(w_min, np.exp(-k * u[gated]))
    # Protected set identity.
    w[T_prot] = 1.0
    scale = 1.0 / np.maximum(w, eps)

    out["apply"] = True
    out["reason"] = "ok"
    out["scale_tiles"] = scale.astype(np.float32, copy=False)
    out["gated_tiles"] = gated.astype(bool)
    out["protected_tiles"] = T_prot.astype(bool)
    out["candidate_tiles"] = T_cand.astype(bool)
    out["stats"] = {
        "n_valid": int(np.sum(valid)),
        "n_bg_proxy": int(np.sum(T_bg)),
        "n_protected": int(np.sum(T_prot)),
        "n_candidates": int(np.sum(T_cand)),
        "n_gated": int(np.sum(gated)),
        "scale_min": float(np.min(scale[valid])) if np.any(valid) else 1.0,
        "scale_max": float(np.max(scale[valid])) if np.any(valid) else 1.0,
        "scale_median": float(np.median(scale[gated])) if np.any(gated) else 1.0,
    }
    return out


def derive_score_shrink_v2_tile_scales_forced(
    *,
    report: dict[str, Any] | None,
    s_base: np.ndarray,
    m_alias: np.ndarray,
    c_flow: np.ndarray,
    valid_mask: np.ndarray | None = None,
    mode: str = "safety",
    risk_mode: str | None = None,
) -> dict[str, Any]:
    """Forced variant of `derive_score_shrink_v2_tile_scales` for ablations.

    Unlike the contract-governed helper, this function does not require the
    contract report to be `reason=ok` and does not enforce coverage/invariance
    sentinels. It recomputes candidate/protected sets and the risk threshold
    from the provided arrays and the (frozen) config.

    Intended use: quantify worst-case harm when KA is applied even when the
    contract would disable it.
    """

    out: dict[str, Any] = {
        "apply": False,
        "reason": "uninitialized",
        "scale_tiles": None,
        "gated_tiles": None,
        "protected_tiles": None,
        "candidate_tiles": None,
        "stats": {},
    }

    cfg_obj: KaContractV2Config
    cfg_dict = None
    if isinstance(report, dict):
        cfg_dict = report.get("config")
    if isinstance(cfg_dict, dict):
        try:
            cfg_obj = KaContractV2Config(**cfg_dict)
        except Exception:
            cfg_obj = KaContractV2Config()
    else:
        cfg_obj = KaContractV2Config()

    rm = (risk_mode or "").strip().lower()
    if not rm and isinstance(report, dict):
        try:
            metrics = report.get("metrics") or {}
            rm = str(metrics.get("risk_mode") or "").strip().lower()
        except Exception:
            rm = ""
    if rm not in {"alias", "guard"}:
        rm = "alias"

    mode_norm = str(mode or "safety").strip().lower()
    if mode_norm not in {"safety", "uplift"}:
        out["reason"] = "invalid_mode"
        return out

    if mode_norm == "uplift":
        w_min = float(cfg_obj.wmin_uplift)
        k = float(cfg_obj.k_uplift)
    else:
        if rm == "guard":
            w_min = float(cfg_obj.wmin_guard)
            k = float(cfg_obj.k_guard)
        else:
            w_min = float(cfg_obj.wmin_safety)
            k = float(cfg_obj.k_safety)

    s_base = np.asarray(s_base, dtype=np.float64).ravel()
    m_alias = np.asarray(m_alias, dtype=np.float64).ravel()
    c_flow = np.asarray(c_flow, dtype=np.float64).ravel()
    if not (s_base.size == m_alias.size == c_flow.size):
        out["reason"] = "shape_mismatch"
        return out

    if valid_mask is None:
        valid = np.isfinite(s_base) & np.isfinite(m_alias) & np.isfinite(c_flow)
    else:
        vm = np.asarray(valid_mask, dtype=bool).ravel()
        if vm.size != s_base.size:
            out["reason"] = "valid_mask_shape_mismatch"
            return out
        valid = vm & np.isfinite(s_base) & np.isfinite(m_alias) & np.isfinite(c_flow)

    if not np.any(valid):
        out["reason"] = "no_valid_tiles"
        return out

    c_bg = float(cfg_obj.c_bg)
    c_flow_thr = float(cfg_obj.c_flow)
    eps = float(cfg_obj.eps)

    s_valid = s_base[valid]
    score_q_lo = float(np.quantile(s_valid, float(cfg_obj.q_lo_candidate)))
    score_q_hi_candidate = float(np.quantile(s_valid, float(cfg_obj.q_hi_candidate)))
    if score_q_hi_candidate < score_q_lo:
        score_q_hi_candidate = score_q_lo
    score_q_hi_protect = float(np.quantile(s_valid, float(cfg_obj.q_hi_protect)))

    T_prot = valid & (c_flow >= c_flow_thr)
    if bool(cfg_obj.protect_hi_by_score):
        T_prot |= valid & (s_base >= score_q_hi_protect)

    T_cand = (
        valid
        & (~T_prot)
        & (s_base >= score_q_lo)
        & (s_base <= score_q_hi_candidate)
    )

    T_bg = valid & (c_flow <= c_bg)
    risk_bg = m_alias[T_bg]
    if risk_bg.size == 0:
        risk_bg = m_alias[valid]

    tau_alias = float(np.quantile(risk_bg, float(cfg_obj.q_risk)))
    med_alias_bg = float(np.median(risk_bg))
    mad_alias = float(np.median(np.abs(risk_bg - med_alias_bg))) + eps
    denom = mad_alias if mad_alias > 0.0 else eps

    gated = T_cand & (m_alias >= tau_alias)

    u = np.maximum(0.0, (m_alias - tau_alias) / denom)
    w = np.ones_like(s_base, dtype=np.float64)
    if np.any(gated):
        w[gated] = np.maximum(w_min, np.exp(-k * u[gated]))
    w[T_prot] = 1.0
    scale = 1.0 / np.maximum(w, eps)

    out["apply"] = True
    out["reason"] = "ok_forced"
    out["scale_tiles"] = scale.astype(np.float32, copy=False)
    out["gated_tiles"] = gated.astype(bool)
    out["protected_tiles"] = T_prot.astype(bool)
    out["candidate_tiles"] = T_cand.astype(bool)
    out["stats"] = {
        "risk_mode": rm,
        "mode": mode_norm,
        "q_risk": float(cfg_obj.q_risk),
        "tau_alias": float(tau_alias),
        "mad_alias": float(mad_alias),
        "n_valid": int(np.sum(valid)),
        "n_bg_proxy": int(np.sum(T_bg)),
        "n_protected": int(np.sum(T_prot)),
        "n_candidates": int(np.sum(T_cand)),
        "n_gated": int(np.sum(gated)),
        "scale_min": float(np.min(scale[valid])) if np.any(valid) else 1.0,
        "scale_max": float(np.max(scale[valid])) if np.any(valid) else 1.0,
        "scale_median": float(np.median(scale[gated])) if np.any(gated) else 1.0,
    }
    return out
