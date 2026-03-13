#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.brain_whitening_policy_validation import _eval_at_tau, _summarize, _tau_for_fpr
from scripts.fair_filter_comparison import _bundle_map_by_window
from sim.kwave.common import _apply_hybrid_rescue_score_map, _normalize_hybrid_rescue_rule


def _load_pair_scores(
    whitened_dir: Path,
    unwhitened_dir: Path,
    rule_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    score_w = np.load(whitened_dir / "score_stap_preka.npy").astype(np.float64, copy=False)
    score_u = np.load(unwhitened_dir / "score_stap_preka.npy").astype(np.float64, copy=False)
    mask_flow = np.load(whitened_dir / "mask_flow.npy").astype(bool, copy=False)
    mask_bg = np.load(whitened_dir / "mask_bg.npy").astype(bool, copy=False)
    rule = _normalize_hybrid_rescue_rule(rule_name)
    feature_name = str(rule["feature"])
    feat = np.load(whitened_dir / f"{feature_name}.npy").astype(np.float64, copy=False)
    hybrid, _mask, _stats = _apply_hybrid_rescue_score_map(
        score_w,
        score_u,
        feature_name=feature_name,
        feature_map=feat,
        direction=str(rule["direction"]),
        threshold=float(rule["threshold"]),
    )
    return hybrid[mask_flow].ravel(), hybrid[mask_bg].ravel()


def _load_fixed_scores(bundle_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    score = np.load(bundle_dir / "score_stap_preka.npy").astype(np.float64, copy=False)
    mf = np.load(bundle_dir / "mask_flow.npy").astype(bool, copy=False)
    mb = np.load(bundle_dir / "mask_bg.npy").astype(bool, copy=False)
    return score[mf].ravel(), score[mb].ravel()


def _summarize_split(posneg_by_win: list[tuple[np.ndarray, np.ndarray]], alpha: float) -> dict[str, Any]:
    within_tpr: list[float] = []
    within_fpr: list[float] = []
    cross_tpr: list[float] = []
    cross_fpr: list[float] = []
    fixed_tpr: list[float] = []
    fixed_fpr: list[float] = []

    for pos, neg in posneg_by_win:
        tau = _tau_for_fpr(neg, alpha)
        tpr, fpr = _eval_at_tau(pos, neg, tau)
        within_tpr.append(tpr)
        within_fpr.append(fpr)

    for i, (pos_i, neg_i) in enumerate(posneg_by_win):
        tau = _tau_for_fpr(neg_i, alpha)
        for j, (pos_j, neg_j) in enumerate(posneg_by_win):
            if i == j:
                continue
            tpr, fpr = _eval_at_tau(pos_j, neg_j, tau)
            cross_tpr.append(tpr)
            cross_fpr.append(fpr)

    for i, (pos_i, neg_i) in enumerate(posneg_by_win):
        neg_pool = np.concatenate([posneg_by_win[j][1] for j in range(len(posneg_by_win)) if j != i])
        tau = _tau_for_fpr(neg_pool, alpha)
        tpr, fpr = _eval_at_tau(pos_i, neg_i, tau)
        fixed_tpr.append(tpr)
        fixed_fpr.append(fpr)

    return {
        "within": {"tpr": _summarize(within_tpr), "fpr": _summarize(within_fpr)},
        "cross": {"tpr": _summarize(cross_tpr), "fpr": _summarize(cross_fpr)},
        "fixed_cal": {"tpr": _summarize(fixed_tpr), "fpr": _summarize(fixed_fpr)},
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate fixed hybrid-rescue rules on the labeled-brain stress-test bundles.")
    ap.add_argument("--window-length", type=int, default=64)
    ap.add_argument("--alphas", type=str, default="1e-4,3e-4,1e-3")
    ap.add_argument(
        "--rules",
        type=str,
        default="guard_frac_v1,alias_rescue_v1,band_ratio_v1",
        help="Comma-separated hybrid rescue rule names.",
    )
    ap.add_argument(
        "--open-whitened-root",
        type=Path,
        default=Path("runs/pilot/brain_whitening_policy_validation/open_seed1_huber_trim8"),
    )
    ap.add_argument(
        "--open-unwhitened-root",
        type=Path,
        default=Path("runs/pilot/brain_whitening_policy_validation/open_seed1_unwhitened_ref"),
    )
    ap.add_argument(
        "--skullor-whitened-root",
        type=Path,
        default=Path("runs/pilot/brain_whitening_policy_validation/skullor_seed2_huber_trim8"),
    )
    ap.add_argument(
        "--skullor-unwhitened-root",
        type=Path,
        default=Path("runs/pilot/stap_whitening_regime_sweep/skullor_seed2_gamma0p00"),
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/brain_hybrid_rule_eval.json"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    alphas = [float(x) for x in str(args.alphas).split(",") if x.strip()]
    rules = [x.strip() for x in str(args.rules).split(",") if x.strip()]
    regime_roots = {
        "open": (args.open_whitened_root, args.open_unwhitened_root),
        "skullor": (args.skullor_whitened_root, args.skullor_unwhitened_root),
    }
    payload: dict[str, Any] = {"alphas": alphas, "rules": rules, "regimes": {}}
    for regime, (whitened_root, unwhitened_root) in regime_roots.items():
        w_bundles = _bundle_map_by_window(whitened_root, int(args.window_length))
        u_bundles = _bundle_map_by_window(unwhitened_root, int(args.window_length))
        offsets = sorted(set(w_bundles) & set(u_bundles))
        if not offsets:
            raise SystemExit(f"No paired bundles for regime={regime}: {whitened_root} vs {unwhitened_root}")
        regime_out: dict[str, Any] = {}

        fixed = {
            "whitened": [_load_fixed_scores(w_bundles[o]) for o in offsets],
            "unwhitened": [_load_fixed_scores(u_bundles[o]) for o in offsets],
        }
        regime_out["fixed"] = {
            name: {str(alpha): _summarize_split(posneg, alpha) for alpha in alphas}
            for name, posneg in fixed.items()
        }

        regime_out["rules"] = {}
        for rule in rules:
            posneg_by_win = [_load_pair_scores(w_bundles[o], u_bundles[o], rule) for o in offsets]
            regime_out["rules"][rule] = {
                str(alpha): _summarize_split(posneg_by_win, alpha) for alpha in alphas
            }
        payload["regimes"][regime] = regime_out

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out_json)


if __name__ == "__main__":
    main()
