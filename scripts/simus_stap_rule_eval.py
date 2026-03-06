#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _safe_float(x: Any) -> float:
    if x in (None, ""):
        return float("nan")
    return float(x)


def _utility(row: dict[str, Any]) -> float:
    return float(row["delta_auc_main_vs_bg"]) + float(row["delta_auc_main_vs_nuisance"]) - float(
        row["delta_fpr_nuisance_match@0p5"]
    )


@dataclass
class Case:
    case_key: str
    seed: int
    simus_profile: str
    motion_scale: float
    features: dict[str, float]
    baseline: dict[str, Any]
    candidates: dict[str, dict[str, Any]]


def _load_cases(path: Path) -> list[Case]:
    rows = list(csv.DictReader(path.open()))
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_key"]), []).append(row)
    cases: list[Case] = []
    for case_key, items in by_case.items():
        baseline = next(r for r in items if r["role"] == "baseline")
        candidates = {str(r["stap_profile"]): r for r in items if r["role"] == "stap"}
        features = {
            "reg_shift_rms": _safe_float(baseline.get("reg_shift_rms")),
            "reg_shift_p90": _safe_float(baseline.get("reg_shift_p90")),
            "reg_psr_median": _safe_float(baseline.get("reg_psr_median")),
            "motion_disp_rms_px": _safe_float(baseline.get("motion_disp_rms_px")),
        }
        cases.append(
            Case(
                case_key=case_key,
                seed=int(baseline["seed"]),
                simus_profile=str(baseline["simus_profile"]),
                motion_scale=float(baseline["motion_scale"]),
                features=features,
                baseline=baseline,
                candidates=candidates,
            )
        )
    return cases


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows]
    return float(sum(vals) / max(len(vals), 1))


def _midpoints(values: list[float]) -> list[float]:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return [0.0]
    out = [vals[0] - 1e-6]
    out.extend((a + b) * 0.5 for a, b in zip(vals[:-1], vals[1:], strict=False))
    out.append(vals[-1] + 1e-6)
    return out


def _evaluate_selection(cases: list[Case], chooser: Callable[[Case], str]) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    oracle: list[dict[str, Any]] = []
    for case in cases:
        chosen_profile = chooser(case)
        row = case.candidates[chosen_profile]
        selected.append(row)
        oracle.append(max(case.candidates.values(), key=_utility))
    mean_utility = _mean_metric(selected, "delta_auc_main_vs_bg") + _mean_metric(
        selected, "delta_auc_main_vs_nuisance"
    ) - _mean_metric(selected, "delta_fpr_nuisance_match@0p5")
    oracle_utility = _mean_metric(oracle, "delta_auc_main_vs_bg") + _mean_metric(
        oracle, "delta_auc_main_vs_nuisance"
    ) - _mean_metric(oracle, "delta_fpr_nuisance_match@0p5")
    recovery = None
    if oracle_utility > 0:
        recovery = float(mean_utility / oracle_utility)
    counts: dict[str, int] = {}
    for row in selected:
        counts[str(row["stap_profile"])] = counts.get(str(row["stap_profile"]), 0) + 1
    return {
        "selected_rows": selected,
        "mean_delta_auc_main_vs_bg": _mean_metric(selected, "delta_auc_main_vs_bg"),
        "mean_delta_auc_main_vs_nuisance": _mean_metric(selected, "delta_auc_main_vs_nuisance"),
        "mean_delta_fpr_nuisance_match@0p5": _mean_metric(selected, "delta_fpr_nuisance_match@0p5"),
        "mean_auc_main_vs_bg": _mean_metric(selected, "auc_main_vs_bg"),
        "mean_auc_main_vs_nuisance": _mean_metric(selected, "auc_main_vs_nuisance"),
        "mean_fpr_nuisance_match@0p5": _mean_metric(selected, "fpr_nuisance_match@0p5"),
        "mean_utility": float(mean_utility),
        "oracle_mean_utility": float(oracle_utility),
        "oracle_recovery_frac": recovery,
        "profile_counts": counts,
    }


def _best_fixed_profile(train_cases: list[Case], profiles: list[str]) -> tuple[str, dict[str, Any]]:
    best_profile = profiles[0]
    best_eval = _evaluate_selection(train_cases, lambda case: best_profile)
    for profile in profiles[1:]:
        cand_eval = _evaluate_selection(train_cases, lambda case, p=profile: p)
        if cand_eval["mean_utility"] > best_eval["mean_utility"]:
            best_profile = profile
            best_eval = cand_eval
    return best_profile, best_eval


def _best_threshold_rule(
    train_cases: list[Case],
    *,
    feature: str,
    low_profile: str,
    high_profile: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    vals = [case.features.get(feature, float("nan")) for case in train_cases]
    thresholds = _midpoints([float(v) for v in vals if math.isfinite(v)])
    best_rule = {"feature": feature, "type": "2way", "low_profile": low_profile, "high_profile": high_profile, "threshold": thresholds[0]}
    best_eval = _evaluate_selection(
        train_cases,
        lambda case, t=thresholds[0]: low_profile if case.features[feature] <= t else high_profile,
    )
    for thr in thresholds[1:]:
        cand_eval = _evaluate_selection(
            train_cases,
            lambda case, t=thr: low_profile if case.features[feature] <= t else high_profile,
        )
        if cand_eval["mean_utility"] > best_eval["mean_utility"]:
            best_rule = {
                "feature": feature,
                "type": "2way",
                "low_profile": low_profile,
                "high_profile": high_profile,
                "threshold": float(thr),
            }
            best_eval = cand_eval
    return best_rule, best_eval


def _best_three_way_rule(
    train_cases: list[Case],
    *,
    feature: str,
    low_profile: str,
    mid_profile: str,
    high_profile: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    vals = [case.features.get(feature, float("nan")) for case in train_cases]
    thresholds = _midpoints([float(v) for v in vals if math.isfinite(v)])
    best_rule = {
        "feature": feature,
        "type": "3way",
        "low_profile": low_profile,
        "mid_profile": mid_profile,
        "high_profile": high_profile,
        "threshold_lo": thresholds[0],
        "threshold_hi": thresholds[-1],
    }
    best_eval = _evaluate_selection(
        train_cases,
        lambda case, t0=thresholds[0], t1=thresholds[-1]: (
            low_profile
            if case.features[feature] <= t0
            else mid_profile
            if case.features[feature] <= t1
            else high_profile
        ),
    )
    for i, t0 in enumerate(thresholds):
        for t1 in thresholds[i + 1 :]:
            cand_eval = _evaluate_selection(
                train_cases,
                lambda case, a=t0, b=t1: (
                    low_profile
                    if case.features[feature] <= a
                    else mid_profile
                    if case.features[feature] <= b
                    else high_profile
                ),
            )
            if cand_eval["mean_utility"] > best_eval["mean_utility"]:
                best_rule = {
                    "feature": feature,
                    "type": "3way",
                    "low_profile": low_profile,
                    "mid_profile": mid_profile,
                    "high_profile": high_profile,
                    "threshold_lo": float(t0),
                    "threshold_hi": float(t1),
                }
                best_eval = cand_eval
    return best_rule, best_eval


def _apply_rule(rule: dict[str, Any], case: Case) -> str:
    if rule["type"] == "fixed":
        return str(rule["profile"])
    feat = float(case.features[str(rule["feature"])])
    if rule["type"] == "2way":
        return str(rule["low_profile"]) if feat <= float(rule["threshold"]) else str(rule["high_profile"])
    if rule["type"] == "3way":
        if feat <= float(rule["threshold_lo"]):
            return str(rule["low_profile"])
        if feat <= float(rule["threshold_hi"]):
            return str(rule["mid_profile"])
        return str(rule["high_profile"])
    raise ValueError(f"unknown rule type {rule['type']!r}")


def _profiles_available(profiles: list[str], *needed: str) -> bool:
    profile_set = set(profiles)
    return all(p in profile_set for p in needed)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate frozen telemetry-based STAP profile rules with leave-one-seed-out validation.")
    ap.add_argument(
        "--search-csv",
        type=Path,
        default=Path("reports/simus_motion/simus_stap_compromise_search.csv"),
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/simus_motion/simus_stap_rule_eval.csv"),
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/simus_motion/simus_stap_rule_eval.json"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cases = _load_cases(Path(args.search_csv))
    seeds = sorted({case.seed for case in cases})
    profiles = sorted({p for case in cases for p in case.candidates.keys()})
    folds: list[dict[str, Any]] = []
    row_out: list[dict[str, Any]] = []

    for heldout_seed in seeds:
        train_cases = [case for case in cases if case.seed != heldout_seed]
        test_cases = [case for case in cases if case.seed == heldout_seed]

        fixed_profile, fixed_train = _best_fixed_profile(train_cases, profiles)
        candidate_rules: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
            ("fixed_best", {"type": "fixed", "profile": fixed_profile}, fixed_train),
        ]
        if _profiles_available(profiles, "Brain-SIMUS-Clin-MotionShort-v0", "Brain-SIMUS-Clin-MotionLong-v0"):
            candidate_rules.append(
                (
                    "reg_shift_rms_short_long",
                    *_best_threshold_rule(
                        train_cases,
                        feature="reg_shift_rms",
                        low_profile="Brain-SIMUS-Clin-MotionShort-v0",
                        high_profile="Brain-SIMUS-Clin-MotionLong-v0",
                    ),
                )
            )
        if _profiles_available(
            profiles,
            "Brain-SIMUS-Clin-MotionShort-v0",
            "Brain-SIMUS-Clin-MotionMid-v0",
            "Brain-SIMUS-Clin-MotionLong-v0",
        ):
            candidate_rules.append(
                (
                    "reg_shift_rms_short_mid_long",
                    *_best_three_way_rule(
                        train_cases,
                        feature="reg_shift_rms",
                        low_profile="Brain-SIMUS-Clin-MotionShort-v0",
                        mid_profile="Brain-SIMUS-Clin-MotionMid-v0",
                        high_profile="Brain-SIMUS-Clin-MotionLong-v0",
                    ),
                )
            )
        if _profiles_available(
            profiles,
            "Brain-SIMUS-Clin-MotionShort-v0",
            "Brain-SIMUS-Clin-MotionRobust-v0",
            "Brain-SIMUS-Clin-MotionLong-v0",
        ):
            candidate_rules.append(
                (
                    "reg_shift_rms_short_robust_long",
                    *_best_three_way_rule(
                        train_cases,
                        feature="reg_shift_rms",
                        low_profile="Brain-SIMUS-Clin-MotionShort-v0",
                        mid_profile="Brain-SIMUS-Clin-MotionRobust-v0",
                        high_profile="Brain-SIMUS-Clin-MotionLong-v0",
                    ),
                )
            )
        if _profiles_available(
            profiles,
            "Brain-SIMUS-Clin-MotionShort-v0",
            "Brain-SIMUS-Clin-MotionMid-v0",
            "Brain-SIMUS-Clin-MotionLong-v0",
        ):
            candidate_rules.append(
                (
                    "reg_shift_p90_short_mid_long",
                    *_best_three_way_rule(
                        train_cases,
                        feature="reg_shift_p90",
                        low_profile="Brain-SIMUS-Clin-MotionShort-v0",
                        mid_profile="Brain-SIMUS-Clin-MotionMid-v0",
                        high_profile="Brain-SIMUS-Clin-MotionLong-v0",
                    ),
                )
            )
        if _profiles_available(
            profiles,
            "Brain-SIMUS-Clin-MotionShort-v0",
            "Brain-SIMUS-Clin-MotionMidRobust-v0",
        ):
            candidate_rules.append(
                (
                    "motion_disp_rms_short_midrobust",
                    *_best_threshold_rule(
                        train_cases,
                        feature="motion_disp_rms_px",
                        low_profile="Brain-SIMUS-Clin-MotionShort-v0",
                        high_profile="Brain-SIMUS-Clin-MotionMidRobust-v0",
                    ),
                )
            )
        if _profiles_available(
            profiles,
            "Brain-SIMUS-Clin-MotionRobust-v0",
            "Brain-SIMUS-Clin-MotionMidRobust-v0",
        ):
            candidate_rules.append(
                (
                    "motion_disp_rms_robust_midrobust",
                    *_best_threshold_rule(
                        train_cases,
                        feature="motion_disp_rms_px",
                        low_profile="Brain-SIMUS-Clin-MotionRobust-v0",
                        high_profile="Brain-SIMUS-Clin-MotionMidRobust-v0",
                    ),
                )
            )
            candidate_rules.append(
                (
                    "reg_shift_rms_robust_midrobust",
                    *_best_threshold_rule(
                        train_cases,
                        feature="reg_shift_rms",
                        low_profile="Brain-SIMUS-Clin-MotionRobust-v0",
                        high_profile="Brain-SIMUS-Clin-MotionMidRobust-v0",
                    ),
                )
            )
            candidate_rules.append(
                (
                    "reg_shift_p90_robust_midrobust",
                    *_best_threshold_rule(
                        train_cases,
                        feature="reg_shift_p90",
                        low_profile="Brain-SIMUS-Clin-MotionRobust-v0",
                        high_profile="Brain-SIMUS-Clin-MotionMidRobust-v0",
                    ),
                )
            )
            candidate_rules.append(
                (
                    "reg_psr_median_midrobust_robust",
                    *_best_threshold_rule(
                        train_cases,
                        feature="reg_psr_median",
                        low_profile="Brain-SIMUS-Clin-MotionMidRobust-v0",
                        high_profile="Brain-SIMUS-Clin-MotionRobust-v0",
                    ),
                )
            )

        for rule_name, rule, train_eval in candidate_rules:
            test_eval = _evaluate_selection(test_cases, lambda case, rr=rule: _apply_rule(rr, case))
            row = {
                "heldout_seed": int(heldout_seed),
                "rule_name": rule_name,
                "train_mean_utility": float(train_eval["mean_utility"]),
                "test_mean_utility": float(test_eval["mean_utility"]),
                "test_oracle_mean_utility": float(test_eval["oracle_mean_utility"]),
                "test_oracle_recovery_frac": test_eval["oracle_recovery_frac"],
                "test_mean_delta_auc_main_vs_bg": float(test_eval["mean_delta_auc_main_vs_bg"]),
                "test_mean_delta_auc_main_vs_nuisance": float(test_eval["mean_delta_auc_main_vs_nuisance"]),
                "test_mean_delta_fpr_nuisance_match@0p5": float(test_eval["mean_delta_fpr_nuisance_match@0p5"]),
                "test_mean_auc_main_vs_bg": float(test_eval["mean_auc_main_vs_bg"]),
                "test_mean_auc_main_vs_nuisance": float(test_eval["mean_auc_main_vs_nuisance"]),
                "test_mean_fpr_nuisance_match@0p5": float(test_eval["mean_fpr_nuisance_match@0p5"]),
                "selected_profile_counts": json.dumps(test_eval["profile_counts"], sort_keys=True),
                "rule": json.dumps(rule, sort_keys=True),
            }
            row_out.append(row)
            folds.append(
                {
                    "heldout_seed": int(heldout_seed),
                    "rule_name": rule_name,
                    "rule": rule,
                    "train_eval": train_eval,
                    "test_eval": test_eval,
                }
            )

    by_rule: dict[str, list[dict[str, Any]]] = {}
    for row in row_out:
        by_rule.setdefault(str(row["rule_name"]), []).append(row)
    summary: dict[str, Any] = {}
    for rule_name, items in by_rule.items():
        summary[rule_name] = {
            "folds": len(items),
            "mean_test_utility": float(sum(float(r["test_mean_utility"]) for r in items) / len(items)),
            "mean_test_oracle_recovery_frac": float(
                sum(float(r["test_oracle_recovery_frac"] or 0.0) for r in items) / len(items)
            ),
            "mean_test_delta_auc_main_vs_bg": float(
                sum(float(r["test_mean_delta_auc_main_vs_bg"]) for r in items) / len(items)
            ),
            "mean_test_delta_auc_main_vs_nuisance": float(
                sum(float(r["test_mean_delta_auc_main_vs_nuisance"]) for r in items) / len(items)
            ),
            "mean_test_delta_fpr_nuisance_match@0p5": float(
                sum(float(r["test_mean_delta_fpr_nuisance_match@0p5"]) for r in items) / len(items)
            ),
        }

    _write_csv(Path(args.out_csv), row_out)
    _write_json(
        Path(args.out_json),
        {
            "schema_version": "simus_stap_rule_eval.v1",
            "folds": folds,
            "summary_by_rule": summary,
        },
    )
    print(f"[simus-stap-rule-eval] wrote {args.out_csv}")
    print(f"[simus-stap-rule-eval] wrote {args.out_json}")


if __name__ == "__main__":
    main()
