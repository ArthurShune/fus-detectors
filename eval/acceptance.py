# eval/acceptance.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from pipeline.calib.conformal import (
    ConformalThreshold,
    conformal_threshold_from_scores,
    empirical_pfa,
)

from .metrics import clopper_pearson_ci, partial_auc, pd_snr_db, roc_curve, tpr_at_fpr_target

# ======================== Data containers ====================================


@dataclass(frozen=True)
class DetectorDataset:
    """
    Container for detection scores and PD map for one method (baseline or STAP).
    """

    scores_pos: np.ndarray  # 1D array of scores on flow pixels
    scores_null: np.ndarray  # 1D array of scores on null pixels (held-out if possible)
    pd_map: Optional[np.ndarray]  # (H, W) PD map for PD-SNR computation (optional)
    pd_stats: Optional[Dict[str, float]] = None  # Optional expected PD stats for validation


@dataclass(frozen=True)
class Masks:
    mask_flow: Optional[np.ndarray]  # (H,W) bool
    mask_bg: Optional[np.ndarray]  # (H,W) bool


@dataclass(frozen=True)
class AcceptanceTargets:
    delta_pdsnrdB_min: float = 3.0
    delta_tpr_at_fpr_min: float = 0.05
    fpr_target: float = 1e-5
    alpha_for_calibration: float = 1e-5  # single-look calibration target


DEFAULT_TARGETS = AcceptanceTargets()


def _pd_stats_from_map(pd_map: Optional[np.ndarray], masks: Optional[Masks]) -> Dict[str, float]:
    stats: Dict[str, float] = {"flow_mean": None, "bg_var": None}
    if pd_map is None or masks is None:
        return stats
    if masks.mask_flow is not None:
        flow_mask = masks.mask_flow.astype(bool, copy=False)
        if flow_mask.any():
            stats["flow_mean"] = float(pd_map[flow_mask].mean())
    if masks.mask_bg is not None:
        bg_mask = masks.mask_bg.astype(bool, copy=False)
        if bg_mask.any():
            stats["bg_var"] = float(pd_map[bg_mask].var())
    return stats


def _validate_pd_stats(dataset: DetectorDataset, masks: Optional[Masks], label: str) -> None:
    expected = dataset.pd_stats
    if expected is None or dataset.pd_map is None or masks is None:
        return
    actual = _pd_stats_from_map(dataset.pd_map, masks)
    for key in ("flow_mean", "bg_var"):
        exp_val = expected.get(key) if isinstance(expected, dict) else None
        act_val = actual.get(key)
        if exp_val is None or act_val is None:
            continue
        if not np.isfinite(exp_val) or not np.isfinite(act_val):
            continue
        if not np.isclose(act_val, exp_val, rtol=1e-4, atol=1e-8):
            raise ValueError(
                f"PD map stats mismatch for {label} ({key}): expected {exp_val}, got {act_val}"
            )


# ======================== Evaluation / acceptance ============================


def evaluate_performance(
    base: DetectorDataset,
    stap: DetectorDataset,
    masks: Optional[Masks] = None,
    targets: Optional[AcceptanceTargets] = None,
) -> Dict[str, float]:
    """
    Compute key performance metrics used in acceptance gates.
    Returns a dict with PD-SNR (baseline, stap, delta), TPR@FPR (both, delta), and partial AUCs.
    """
    targets = targets or DEFAULT_TARGETS

    # PD-SNR (requires PD maps and masks)
    pdsnrb = np.nan
    pdsnrs = np.nan
    pdsnrb_band = np.nan
    pdsnrs_band = np.nan
    if base.pd_map is not None and stap.pd_map is not None and masks is not None:
        _validate_pd_stats(base, masks, "baseline")
        _validate_pd_stats(stap, masks, "stap")
        pdsnrb = pd_snr_db(base.pd_map, masks.mask_flow, masks.mask_bg)
        pdsnrs = pd_snr_db(stap.pd_map, masks.mask_flow, masks.mask_bg)
        eps = 1e-12
        band_frac = np.clip(stap.pd_map / (base.pd_map + eps), 0.0, 1.0)
        pd_band_base = base.pd_map * band_frac
        pd_band_stap = stap.pd_map
        pdsnrb_band = pd_snr_db(pd_band_base, masks.mask_flow, masks.mask_bg)
        pdsnrs_band = pd_snr_db(pd_band_stap, masks.mask_flow, masks.mask_bg)

    # ROC near low FPR
    fpr_b, tpr_b, _ = roc_curve(base.scores_pos, base.scores_null, num_thresh=4096)
    fpr_s, tpr_s, _ = roc_curve(stap.scores_pos, stap.scores_null, num_thresh=4096)

    tprb = tpr_at_fpr_target(fpr_b, tpr_b, target_fpr=targets.fpr_target)
    tprs = tpr_at_fpr_target(fpr_s, tpr_s, target_fpr=targets.fpr_target)

    aucb = partial_auc(fpr_b, tpr_b, fpr_max=targets.fpr_target)
    aucs = partial_auc(fpr_s, tpr_s, fpr_max=targets.fpr_target)

    return {
        "pd_snr_baseline_db": float(pdsnrb),
        "pd_snr_stap_db": float(pdsnrs),
        "pd_snr_delta_db": (
            float(pdsnrs - pdsnrb) if np.isfinite(pdsnrb) and np.isfinite(pdsnrs) else np.nan
        ),
        "pd_snr_band_baseline_db": float(pdsnrb_band),
        "pd_snr_band_stap_db": float(pdsnrs_band),
        "pd_snr_band_delta_db": (
            float(pdsnrs_band - pdsnrb_band)
            if np.isfinite(pdsnrb_band) and np.isfinite(pdsnrs_band)
            else np.nan
        ),
        "tpr_at_fpr_baseline": float(tprb),
        "tpr_at_fpr_stap": float(tprs),
        "tpr_at_fpr_delta": float(tprs - tprb),
        "pauc_baseline": float(aucb),
        "pauc_stap": float(aucs),
        "pauc_delta": float(aucs - aucb),
    }


def evaluate_calibration(
    scores_null: np.ndarray,
    alpha_target: float = 1e-5,
    seed: int = 0,
    *,
    evd_mode: str = "weibull",
    evd_endpoint: float | None = None,
    min_exceedances_weibull: int = 500,
) -> tuple[Dict[str, float], Dict[str, float], ConformalThreshold]:
    """
    Run split-conformal (EVT+conformal) on null scores and report empirical Pfa.
    Uses the same pool for calibration split and empirical estimate if no holdout is provided;
    for strict validation, pass a disjoint holdout pool to empirical_pfa.
    """
    thr = conformal_threshold_from_scores(
        scores_null,
        alpha1=alpha_target,
        split_ratio=0.6,
        seed=seed,
        evd_mode="weibull" if evd_mode.lower() == "weibull" else "gpd",
        endpoint_hint=evd_endpoint,
        min_exceedances_weibull=min_exceedances_weibull,
    )
    # If you have a holdout pool, evaluate empirical_pfa on that split instead.
    pfa_emp = float(empirical_pfa(scores_null, thr))
    # Binomial CI
    k = int((scores_null > thr.tau).sum())
    n = int(scores_null.size)
    lo, hi = clopper_pearson_ci(k, n, alpha=0.05)
    calib_dict = {
        "alpha_target": float(alpha_target),
        "tau": float(thr.tau),
        "empirical_pfa": pfa_emp,
        "pfa_ci_lo": float(lo),
        "pfa_ci_hi": float(hi),
        "n_null": n,
        "k_false_alarms": k,
    }
    evt_diag = {
        "status": thr.mean_excess_diag.status,
        "r2": thr.mean_excess_diag.r2,
        "r2_threshold": thr.mean_excess_diag.r2_threshold,
        "n_exc": thr.mean_excess_diag.n_exc,
        "min_exceedances": thr.mean_excess_diag.min_exceedances,
        "selected_u": thr.mean_excess_diag.selected_u,
        "mode": thr.evd_mode,
    }
    if thr.weibull_pot is not None:
        evt_diag.update(
            weibull_r2=thr.weibull_pot.r2_mean_excess,
            weibull_n_exc=thr.weibull_pot.n_exc,
            weibull_endpoint=thr.weibull_pot.xF,
            weibull_p_u=thr.weibull_pot.p_u,
        )
    return calib_dict, evt_diag, thr


def acceptance_report(
    base: DetectorDataset,
    stap: DetectorDataset,
    masks: Optional[Masks] = None,
    targets: Optional[AcceptanceTargets] = None,
    seed: int = 0,
    evd_mode: str = "weibull",
    evd_endpoint: float | None = None,
    min_exceedances_weibull: int = 500,
) -> Dict[str, object]:
    """
    Compute acceptance metrics and PASS/FAIL gates.
    """
    targets = targets or DEFAULT_TARGETS

    perf = evaluate_performance(base, stap, masks, targets)
    calib_stap, evt_diag, thr = evaluate_calibration(
        stap.scores_null,
        alpha_target=targets.alpha_for_calibration,
        seed=seed,
        evd_mode=evd_mode,
        evd_endpoint=evd_endpoint,
        min_exceedances_weibull=min_exceedances_weibull,
    )
    evt_ok = bool(evt_diag["status"] == "ok" and evt_diag["n_exc"] >= evt_diag["min_exceedances"])
    if evd_mode.lower() == "weibull" and thr.weibull_pot is not None:
        evt_ok = evt_ok and thr.weibull_pot.n_exc >= min_exceedances_weibull
        evt_ok = evt_ok and thr.weibull_pot.r2_mean_excess >= 0.9

    gates = {
        "gate_delta_pd_snr": bool(
            np.isfinite(perf["pd_snr_delta_db"])
            and perf["pd_snr_delta_db"] >= targets.delta_pdsnrdB_min
        ),
        "gate_delta_pd_snr_band": bool(
            np.isfinite(perf["pd_snr_band_delta_db"])
            and perf["pd_snr_band_delta_db"] >= targets.delta_pdsnrdB_min
        ),
        "gate_delta_tpr_at_fpr": bool(perf["tpr_at_fpr_delta"] >= targets.delta_tpr_at_fpr_min),
        "gate_calibration_ci": bool(
            calib_stap["pfa_ci_lo"] <= targets.alpha_for_calibration <= calib_stap["pfa_ci_hi"]
        ),
        "gate_evt_diagnostics": evt_ok,
    }
    overall = bool(
        gates["gate_delta_pd_snr_band"]
        and gates["gate_delta_tpr_at_fpr"]
        and gates["gate_calibration_ci"]
        and gates["gate_evt_diagnostics"]
    )

    report = {
        "performance": perf,
        "calibration": calib_stap,
        "evt_diagnostics": evt_diag,
        "gates": gates,
        "overall_pass": overall,
        "targets": targets.__dict__,
        "conformal": thr.as_dict(),
    }
    return report
