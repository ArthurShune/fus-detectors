#!/usr/bin/env python3
"""
Analyze whether simple telemetry-driven routing between the frozen Shin
advanced-whitened head and the unwhitened head can improve the tradeoff
between proxy-AUC, flow retention, and map hygiene.

The script consumes the paired clip-level outputs from:
  - scripts/shin_whitening_diagnostics.py
  - scripts/shin_score_hygiene_compare.py

It searches single-feature decision stumps of the form:
    choose whitened if feature <= threshold else unwhitened
or the reverse direction.

Two summaries are reported:
  1. A "Pareto-like" shortlist: rules that stay within a small AUC drop budget
     relative to always-whitened while improving both strict-tail flow hit and
     background-cluster burden.
  2. A utility-maximizing shortlist: rules that optimize a simple composite
     utility over AUC, strict-tail flow hit, and hygiene endpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FEATURES = [
    "br_peak_p50_hz",
    "br_peak_p90_hz",
    "br_pf_peak_nonbg",
    "br_pa_peak_bg",
    "median_condR",
    "median_cond_loaded",
    "median_cov_rank_proxy",
    "median_score_q50",
    "median_score_q90",
    "median_band_fraction",
    "median_flow_mu_ratio",
    "reg_shift_p90",
]


@dataclass
class ClipPair:
    iq_file: str
    features: dict[str, float]
    whitened: dict[str, float]
    unwhitened: dict[str, float]


def _to_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        val = float(value)
    except Exception:
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _zscore_map(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(var) if var > 0 else 1.0
    return mean, std


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = sum((x - mx) ** 2 for x in xs)
    deny = sum((y - my) ** 2 for y in ys)
    den = math.sqrt(denx * deny)
    if den <= 0:
        return None
    return num / den


def _load_pairs(diag_csv: Path, hygiene_csv: Path) -> list[ClipPair]:
    diag_rows = _load_csv(diag_csv)
    hyg_rows = _load_csv(hygiene_csv)
    by_clip: dict[str, dict[str, dict[str, str]]] = {}
    for row in diag_rows:
        iq = str(row["iq_file"])
        by_clip.setdefault(iq, {})[f"diag::{row['detector_variant']}"] = row
    for row in hyg_rows:
        iq = str(row["iq_file"])
        by_clip.setdefault(iq, {})[f"hyg::{row['detector_variant']}"] = row

    clips: list[ClipPair] = []
    for iq_file, rows in sorted(by_clip.items()):
        if not {
            "diag::msd_ratio",
            "diag::unwhitened_ratio",
            "hyg::msd_ratio",
            "hyg::unwhitened_ratio",
        } <= set(rows):
            continue
        dw = rows["diag::msd_ratio"]
        du = rows["diag::unwhitened_ratio"]
        hw = rows["hyg::msd_ratio"]
        hu = rows["hyg::unwhitened_ratio"]
        features = {
            feat: _to_float(dw.get(feat))
            for feat in FEATURES
            if _to_float(dw.get(feat)) is not None
        }
        whitened = {
            "delta_auc": float(dw["delta_auc_stap_minus_pd"]),
            "strict_hit": float(dw["flow_hit_fpr1e3_stap"]),
            "strict_margin_q50": float(dw["flow_margin_q50_fpr1e3_stap"]),
            "hyg_hit": float(hw["stap_hit_flow"]),
            "hyg_clusters": float(hw["stap_bg_clusters"]),
            "hyg_excess_mass": float(hw["stap_bg_excess_mass"]),
        }
        unwhitened = {
            "delta_auc": float(du["delta_auc_stap_minus_pd"]),
            "strict_hit": float(du["flow_hit_fpr1e3_stap"]),
            "strict_margin_q50": float(du["flow_margin_q50_fpr1e3_stap"]),
            "hyg_hit": float(hu["stap_hit_flow"]),
            "hyg_clusters": float(hu["stap_bg_clusters"]),
            "hyg_excess_mass": float(hu["stap_bg_excess_mass"]),
        }
        clips.append(
            ClipPair(
                iq_file=iq_file,
                features=features,
                whitened=whitened,
                unwhitened=unwhitened,
            )
        )
    return clips


def _fixed_summary(clips: list[ClipPair], head: str) -> dict[str, float]:
    key = "whitened" if head == "whitened" else "unwhitened"
    vals = [getattr(c, key) for c in clips]
    return {
        "n_clips": float(len(vals)),
        "mean_delta_auc": sum(v["delta_auc"] for v in vals) / len(vals),
        "mean_strict_hit": sum(v["strict_hit"] for v in vals) / len(vals),
        "mean_strict_margin_q50": sum(v["strict_margin_q50"] for v in vals) / len(vals),
        "mean_hyg_hit": sum(v["hyg_hit"] for v in vals) / len(vals),
        "mean_hyg_clusters": sum(v["hyg_clusters"] for v in vals) / len(vals),
        "mean_hyg_excess_mass": sum(v["hyg_excess_mass"] for v in vals) / len(vals),
    }


def _build_utilities(clips: list[ClipPair]) -> None:
    metrics = [
        ("delta_auc", +1.0),
        ("strict_hit", +0.40),
        ("hyg_hit", +0.40),
        ("hyg_clusters", -0.30),
        ("hyg_excess_mass", -0.10),
    ]
    stats: dict[str, tuple[float, float]] = {}
    for metric, _ in metrics:
        all_vals: list[float] = []
        for clip in clips:
            all_vals.append(clip.whitened[metric])
            all_vals.append(clip.unwhitened[metric])
        stats[metric] = _zscore_map(all_vals)
    for clip in clips:
        for label in ("whitened", "unwhitened"):
            src = getattr(clip, label)
            util = 0.0
            for metric, weight in metrics:
                mean, std = stats[metric]
                util += weight * ((src[metric] - mean) / std)
            src["utility"] = util


def _evaluate_rule(
    clips: list[ClipPair],
    feature: str,
    direction: str,
    threshold: float,
) -> dict[str, Any]:
    choose_whitened = 0
    chosen: list[dict[str, float]] = []
    per_clip: list[dict[str, Any]] = []
    for clip in clips:
        feat_val = clip.features.get(feature)
        if feat_val is None:
            use_whitened = True
        elif direction == "<=":
            use_whitened = feat_val <= threshold
        else:
            use_whitened = feat_val >= threshold
        head = clip.whitened if use_whitened else clip.unwhitened
        if use_whitened:
            choose_whitened += 1
        chosen.append(head)
        per_clip.append(
            {
                "iq_file": clip.iq_file,
                "feature_value": feat_val,
                "choose_whitened": bool(use_whitened),
                "delta_auc": head["delta_auc"],
                "strict_hit": head["strict_hit"],
                "hyg_hit": head["hyg_hit"],
                "hyg_clusters": head["hyg_clusters"],
                "utility": head["utility"],
            }
        )
    n = len(chosen)
    return {
        "feature": feature,
        "direction": direction,
        "threshold": float(threshold),
        "choose_whitened_count": int(choose_whitened),
        "choose_unwhitened_count": int(n - choose_whitened),
        "mean_delta_auc": sum(v["delta_auc"] for v in chosen) / n,
        "mean_strict_hit": sum(v["strict_hit"] for v in chosen) / n,
        "mean_hyg_hit": sum(v["hyg_hit"] for v in chosen) / n,
        "mean_hyg_clusters": sum(v["hyg_clusters"] for v in chosen) / n,
        "mean_hyg_excess_mass": sum(v["hyg_excess_mass"] for v in chosen) / n,
        "mean_utility": sum(v["utility"] for v in chosen) / n,
        "per_clip": per_clip,
    }


def _feature_correlations(clips: list[ClipPair]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feat in FEATURES:
        xs: list[float] = []
        d_auc: list[float] = []
        d_strict_hit: list[float] = []
        d_hyg_hit: list[float] = []
        d_clusters: list[float] = []
        for clip in clips:
            x = clip.features.get(feat)
            if x is None:
                continue
            xs.append(x)
            d_auc.append(clip.whitened["delta_auc"] - clip.unwhitened["delta_auc"])
            d_strict_hit.append(clip.whitened["strict_hit"] - clip.unwhitened["strict_hit"])
            d_hyg_hit.append(clip.whitened["hyg_hit"] - clip.unwhitened["hyg_hit"])
            d_clusters.append(clip.unwhitened["hyg_clusters"] - clip.whitened["hyg_clusters"])
        rows.append(
            {
                "feature": feat,
                "n": len(xs),
                "corr_delta_auc": _pearson(xs, d_auc),
                "corr_delta_strict_hit": _pearson(xs, d_strict_hit),
                "corr_delta_hyg_hit": _pearson(xs, d_hyg_hit),
                "corr_unwhitened_cluster_advantage": _pearson(xs, d_clusters),
            }
        )
    rows.sort(
        key=lambda r: abs(r["corr_delta_hyg_hit"] or 0.0) + abs(r["corr_unwhitened_cluster_advantage"] or 0.0),
        reverse=True,
    )
    return rows


def _search_rules(
    clips: list[ClipPair],
    auc_drop_budget: float,
    max_rules: int,
) -> dict[str, Any]:
    base_w = _fixed_summary(clips, "whitened")
    base_u = _fixed_summary(clips, "unwhitened")
    candidates: list[dict[str, Any]] = []
    for feature in FEATURES:
        vals = sorted({clip.features.get(feature) for clip in clips if clip.features.get(feature) is not None})
        if len(vals) < 2:
            continue
        thresholds = [(a + b) / 2.0 for a, b in zip(vals[:-1], vals[1:])]
        for direction in ("<=", ">="):
            for threshold in thresholds:
                rule = _evaluate_rule(clips, feature, direction, threshold)
                rule["delta_vs_whitened_auc"] = rule["mean_delta_auc"] - base_w["mean_delta_auc"]
                rule["delta_vs_whitened_strict_hit"] = rule["mean_strict_hit"] - base_w["mean_strict_hit"]
                rule["delta_vs_whitened_hyg_hit"] = rule["mean_hyg_hit"] - base_w["mean_hyg_hit"]
                rule["delta_vs_whitened_hyg_clusters"] = rule["mean_hyg_clusters"] - base_w["mean_hyg_clusters"]
                rule["delta_vs_unwhitened_auc"] = rule["mean_delta_auc"] - base_u["mean_delta_auc"]
                rule["delta_vs_unwhitened_hyg_hit"] = rule["mean_hyg_hit"] - base_u["mean_hyg_hit"]
                rule["delta_vs_unwhitened_hyg_clusters"] = rule["mean_hyg_clusters"] - base_u["mean_hyg_clusters"]
                rule["auc_within_budget"] = (
                    rule["mean_delta_auc"] >= base_w["mean_delta_auc"] - float(auc_drop_budget)
                )
                rule["improves_vs_whitened"] = (
                    rule["delta_vs_whitened_hyg_hit"] > 0
                    and rule["delta_vs_whitened_hyg_clusters"] < 0
                    and rule["auc_within_budget"]
                )
                candidates.append(rule)

    pareto_like = [
        {
            k: v
            for k, v in rule.items()
            if k != "per_clip"
        }
        for rule in candidates
        if rule["improves_vs_whitened"]
        and 0 < rule["choose_whitened_count"] < len(clips)
    ]
    pareto_like.sort(
        key=lambda r: (
            r["delta_vs_whitened_hyg_hit"],
            -r["delta_vs_whitened_hyg_clusters"],
            r["mean_delta_auc"],
        ),
        reverse=True,
    )
    utility_best = sorted(
        (
            {
                k: v
                for k, v in rule.items()
                if k != "per_clip"
            }
            for rule in candidates
            if 0 < rule["choose_whitened_count"] < len(clips)
        ),
        key=lambda r: r["mean_utility"],
        reverse=True,
    )

    recommended = pareto_like[0] if pareto_like else (utility_best[0] if utility_best else None)
    return {
        "baseline_whitened": base_w,
        "baseline_unwhitened": base_u,
        "recommended_rule": recommended,
        "pareto_like_rules": pareto_like[: max_rules],
        "utility_best_rules": utility_best[: max_rules],
        "all_rule_count": len(candidates),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--diag-csv",
        type=Path,
        default=Path("reports/shin_whitening_allclips_S_Lt64_cuda.csv"),
        help="Clip-level diagnostics CSV from shin_whitening_diagnostics.py.",
    )
    ap.add_argument(
        "--hygiene-csv",
        type=Path,
        default=Path("reports/shin_score_hygiene_allclips_S_Lt64_cuda.csv"),
        help="Clip-level hygiene CSV from shin_score_hygiene_compare.py.",
    )
    ap.add_argument(
        "--auc-drop-budget",
        type=float,
        default=0.01,
        help="Allowable drop in mean delta-AUC relative to always-whitened when selecting shortlist rules.",
    )
    ap.add_argument(
        "--max-rules",
        type=int,
        default=12,
        help="Number of top routing rules to keep in each summary.",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/shin_dual_head_routing_analysis.csv"),
        help="CSV for the Pareto-like shortlist.",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/shin_dual_head_routing_analysis.json"),
        help="JSON summary payload.",
    )
    args = ap.parse_args()

    clips = _load_pairs(Path(args.diag_csv), Path(args.hygiene_csv))
    if not clips:
        raise SystemExit("No paired clips found.")
    _build_utilities(clips)
    correlations = _feature_correlations(clips)
    search = _search_rules(clips, auc_drop_budget=float(args.auc_drop_budget), max_rules=int(args.max_rules))

    csv_rows = search["pareto_like_rules"] if search["pareto_like_rules"] else search["utility_best_rules"]
    _write_csv(Path(args.out_csv), csv_rows)
    payload = {
        "config": {
            "diag_csv": str(args.diag_csv),
            "hygiene_csv": str(args.hygiene_csv),
            "auc_drop_budget": float(args.auc_drop_budget),
            "max_rules": int(args.max_rules),
            "n_clips": len(clips),
        },
        "summary": search,
        "feature_correlations": correlations,
    }
    _write_json(Path(args.out_json), payload)

    rec = search["recommended_rule"]
    print(f"Loaded {len(clips)} paired clips.")
    print("Baseline whitened:", search["baseline_whitened"])
    print("Baseline unwhitened:", search["baseline_unwhitened"])
    if rec is not None:
        print(
            "Recommended rule:",
            f"{rec['feature']} {rec['direction']} {rec['threshold']:.6g}",
            f"(choose whitened on {rec['choose_whitened_count']}/{len(clips)} clips)",
        )
        print(
            "Means:",
            f"delta_auc={rec['mean_delta_auc']:.6f}",
            f"hyg_hit={rec['mean_hyg_hit']:.6f}",
            f"hyg_clusters={rec['mean_hyg_clusters']:.6f}",
            f"utility={rec['mean_utility']:.6f}",
        )


if __name__ == "__main__":
    main()
