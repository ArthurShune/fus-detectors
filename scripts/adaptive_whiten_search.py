#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.brain_whitening_policy_validation import _eval_at_tau, _tau_for_fpr
from scripts.fair_filter_comparison import _bundle_map_by_window
from scripts.shin_map_routing_analysis import _score_metrics


HIGHER_BETTER = {
    "brain_open_tpr",
    "brain_skullor_tpr",
    "shin_auc",
    "shin_hit",
    "along_auc",
    "along_hit",
    "across_auc",
    "across_hit",
}
LOWER_BETTER = {
    "shin_clusters",
    "along_clusters",
    "across_clusters",
}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite(arr: np.ndarray) -> np.ndarray:
    vals = np.asarray(arr, dtype=np.float64).ravel()
    return vals[np.isfinite(vals)]


def _robust_norm(score: np.ndarray, mask_bg: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    bg = np.asarray(mask_bg, dtype=bool)
    vals = score[bg]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return score
    q50 = float(np.quantile(vals, 0.50))
    q999 = float(np.quantile(vals, 0.999))
    scale = max(q999 - q50, 1e-9)
    return (score - q50) / scale


def _safe_log_map(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64)
    out = np.full_like(x, np.nan, dtype=np.float64)
    finite = np.isfinite(x)
    pos = finite & (x > 0.0)
    out[pos] = np.log(np.maximum(x[pos], 1e-12))
    return out


def _load_pair_maps(whitened_dir: Path, unwhitened_dir: Path) -> dict[str, np.ndarray] | None:
    req = [
        whitened_dir / "score_stap_preka.npy",
        whitened_dir / "score_base.npy",
        whitened_dir / "mask_flow.npy",
        whitened_dir / "mask_bg.npy",
        whitened_dir / "base_guard_frac_map.npy",
        whitened_dir / "base_m_alias_map.npy",
        whitened_dir / "stap_bg_var_inflation_map.npy",
        whitened_dir / "stap_cond_loaded_map.npy",
        whitened_dir / "stap_flow_mu_ratio_map.npy",
        unwhitened_dir / "score_stap_preka.npy",
    ]
    if not all(p.exists() for p in req):
        return None
    score_w = np.load(whitened_dir / "score_stap_preka.npy", allow_pickle=False).astype(
        np.float64, copy=False
    )
    score_u = np.load(unwhitened_dir / "score_stap_preka.npy", allow_pickle=False).astype(
        np.float64, copy=False
    )
    flow = np.load(whitened_dir / "mask_flow.npy", allow_pickle=False).astype(bool, copy=False)
    bg = np.load(whitened_dir / "mask_bg.npy", allow_pickle=False).astype(bool, copy=False)
    z_w = _robust_norm(score_w, bg)
    z_u = _robust_norm(score_u, bg)
    return {
        "score_w": score_w,
        "score_u": score_u,
        "score_base": np.load(whitened_dir / "score_base.npy", allow_pickle=False).astype(
            np.float64, copy=False
        ),
        "z_w": z_w,
        "z_u": z_u,
        "flow": flow,
        "bg": bg,
        "guard": np.load(whitened_dir / "base_guard_frac_map.npy", allow_pickle=False).astype(
            np.float64, copy=False
        ),
        "alias": np.load(whitened_dir / "base_m_alias_map.npy", allow_pickle=False).astype(
            np.float64, copy=False
        ),
        "bg_inflation": np.load(
            whitened_dir / "stap_bg_var_inflation_map.npy", allow_pickle=False
        ).astype(np.float64, copy=False),
        "cond_loaded": np.load(
            whitened_dir / "stap_cond_loaded_map.npy", allow_pickle=False
        ).astype(np.float64, copy=False),
        "flow_mu_ratio": np.load(
            whitened_dir / "stap_flow_mu_ratio_map.npy", allow_pickle=False
        ).astype(np.float64, copy=False),
    }


def _load_brain_pairs(whitened_root: Path, unwhitened_root: Path) -> list[dict[str, np.ndarray]]:
    w_bundles = _bundle_map_by_window(whitened_root, 64)
    u_bundles = _bundle_map_by_window(unwhitened_root, 64)
    rows: list[dict[str, np.ndarray]] = []
    for offset in sorted(set(w_bundles) & set(u_bundles)):
        pair = _load_pair_maps(w_bundles[offset], u_bundles[offset])
        if pair is None:
            continue
        rows.append(pair)
    return rows


def _load_stem_pairs(
    whitened_root: Path,
    unwhitened_root: Path,
    suffix_w: str,
    suffix_u: str,
) -> list[dict[str, np.ndarray]]:
    def _stem(name: str, suffix: str) -> str:
        if suffix:
            return name[: -len(suffix)] if name.endswith(suffix) else name
        return name

    w_dirs = {
        _stem(p.name, suffix_w): p
        for p in whitened_root.iterdir()
        if p.is_dir() and p.name.endswith(suffix_w)
    }
    u_dirs = {
        _stem(p.name, suffix_u): p
        for p in unwhitened_root.iterdir()
        if p.is_dir() and p.name.endswith(suffix_u)
    }
    out: list[dict[str, np.ndarray]] = []
    for stem in sorted(set(w_dirs) & set(u_dirs)):
        pair = _load_pair_maps(w_dirs[stem], u_dirs[stem])
        if pair is not None:
            out.append(pair)
    return out


def _collect_thresholds(
    pairs: list[dict[str, np.ndarray]],
    field: str,
    quantiles: list[float],
    *,
    transform: str = "identity",
) -> list[float]:
    vals_all: list[np.ndarray] = []
    for pair in pairs:
        arr = np.asarray(pair[field], dtype=np.float64)
        if transform == "log":
            arr = _safe_log_map(arr)
        vals = _finite(arr)
        if vals.size:
            vals_all.append(vals)
    if not vals_all:
        return []
    vals_cat = np.concatenate(vals_all)
    return sorted({float(np.quantile(vals_cat, q)) for q in quantiles})


def _mean_metric(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr))


def _score_summary(score: np.ndarray, pair: dict[str, np.ndarray], alpha: float) -> dict[str, float]:
    met = _score_metrics(score, pair["flow"], pair["bg"], alpha=alpha, connectivity=4)
    base = _score_metrics(pair["score_base"], pair["flow"], pair["bg"], alpha=alpha, connectivity=4)
    auc = None
    if met.get("auc_flow_bg") is not None and base.get("auc_flow_bg") is not None:
        auc = float(met["auc_flow_bg"]) - float(base["auc_flow_bg"])
    return {
        "auc": float("nan") if auc is None else float(auc),
        "hit": float("nan") if met.get("hit_flow") is None else float(met["hit_flow"]),
        "clusters": float("nan") if met.get("bg_clusters") is None else float(met["bg_clusters"]),
    }


def _fixed_metrics(
    pairs: list[dict[str, np.ndarray]],
    *,
    alpha_brain: float,
    alpha_other: float,
    head: str,
) -> dict[str, float]:
    out: dict[str, float] = {}
    if not pairs:
        return out
    if "flow" in pairs[0] and "bg" in pairs[0]:
        if head == "whitened":
            score_key = "score_w"
        elif head == "unwhitened":
            score_key = "score_u"
        else:
            raise ValueError(head)
        # Brain-style strict TPR
        vals: list[float] = []
        for pair in pairs:
            tau = _tau_for_fpr(pair[score_key][pair["bg"]], alpha_brain)
            tpr, _ = _eval_at_tau(pair[score_key][pair["flow"]], pair[score_key][pair["bg"]], tau)
            vals.append(float(tpr))
        out["brain_tpr"] = float(np.median(np.asarray(vals, dtype=np.float64)))
        # Other-style score metrics
        summaries = [_score_summary(pair[score_key], pair, alpha_other) for pair in pairs]
        out["auc"] = _mean_metric([s["auc"] for s in summaries])
        out["hit"] = _mean_metric([s["hit"] for s in summaries])
        out["clusters"] = _mean_metric([s["clusters"] for s in summaries])
    return out


def _apply_rule(pair: dict[str, np.ndarray], rule: dict[str, Any]) -> np.ndarray:
    guard = pair["guard"]
    alias = pair["alias"]
    bg_inflation = pair["bg_inflation"]
    cond_loaded = pair["cond_loaded"]
    flow_mu_ratio = pair["flow_mu_ratio"]

    cond_guard_hi = np.isfinite(guard) & (guard >= float(rule["guard_hi"]))
    cond_alias_hi = np.isfinite(alias) & (alias >= float(rule["alias_hi"]))
    cond_bg_hi = np.isfinite(bg_inflation) & (_safe_log_map(bg_inflation) >= float(rule["bg_hi_log"]))
    cond_cond_ok = np.isfinite(cond_loaded) & (cond_loaded <= float(rule["cond_ok_hi"]))
    cond_flow_ok = np.isfinite(flow_mu_ratio) & (flow_mu_ratio >= float(rule["flow_ok_lo"]))
    cond_cov_ok = cond_cond_ok & (cond_flow_ok | ~np.isfinite(flow_mu_ratio))

    family = str(rule["family"])
    if family == "promote_w_guard_alias":
        use_w = (cond_guard_hi | cond_alias_hi) & cond_cov_ok
    elif family == "promote_w_guard_alias_bg":
        use_w = (cond_guard_hi | cond_alias_hi | cond_bg_hi) & cond_cov_ok
    elif family == "promote_w_guard_only":
        use_w = cond_guard_hi & cond_cov_ok
    else:
        raise ValueError(f"Unknown family {family!r}")
    return np.where(use_w, pair["score_w"], pair["score_u"])


def _brain_median_tpr_alpha(pairs: list[dict[str, np.ndarray]], alpha: float, rule: dict[str, Any]) -> float:
    vals: list[float] = []
    for pair in pairs:
        score = _apply_rule(pair, rule)
        tau = _tau_for_fpr(score[pair["bg"]], alpha)
        tpr, _ = _eval_at_tau(score[pair["flow"]], score[pair["bg"]], tau)
        vals.append(float(tpr))
    return float(np.median(np.asarray(vals, dtype=np.float64))) if vals else float("nan")


def _mean_score_metrics(
    pairs: list[dict[str, np.ndarray]],
    alpha: float,
    rule: dict[str, Any],
) -> dict[str, float]:
    summaries = [_score_summary(_apply_rule(pair, rule), pair, alpha) for pair in pairs]
    return {
        "mean_delta_auc_vs_pd": _mean_metric([s["auc"] for s in summaries]),
        "mean_hit": _mean_metric([s["hit"] for s in summaries]),
        "mean_bg_clusters": _mean_metric([s["clusters"] for s in summaries]),
    }


def _baseline_best_metric(
    fixed_w: dict[str, float],
    fixed_u: dict[str, float],
    key: str,
) -> float:
    if key in HIGHER_BETTER:
        return max(float(fixed_w[key]), float(fixed_u[key]))
    if key in LOWER_BETTER:
        return min(float(fixed_w[key]), float(fixed_u[key]))
    raise KeyError(key)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Search telemetry-aware adaptive whitening policies on paired "
            "whitened/unwhitened bundles."
        )
    )
    ap.add_argument("--brain-open-whitened-root", type=Path, required=True)
    ap.add_argument("--brain-open-unwhitened-root", type=Path, required=True)
    ap.add_argument("--brain-skullor-whitened-root", type=Path, required=True)
    ap.add_argument("--brain-skullor-unwhitened-root", type=Path, required=True)
    ap.add_argument("--shin-root", type=Path, required=True)
    ap.add_argument("--twinkling-along-whitened-root", type=Path, required=True)
    ap.add_argument("--twinkling-along-unwhitened-root", type=Path, required=True)
    ap.add_argument("--twinkling-across-whitened-root", type=Path, required=True)
    ap.add_argument("--twinkling-across-unwhitened-root", type=Path, required=True)
    ap.add_argument("--alpha-brain", type=float, default=1e-4)
    ap.add_argument("--alpha-other", type=float, default=1e-3)
    ap.add_argument("--quantiles", type=str, default="0.2,0.4,0.6,0.8")
    ap.add_argument("--out-json", type=Path, default=Path("reports/adaptive_whiten_search.json"))
    ap.add_argument("--out-csv", type=Path, default=Path("reports/adaptive_whiten_search.csv"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    quantiles = [float(x) for x in str(args.quantiles).split(",") if x.strip()]

    brain_open = _load_brain_pairs(args.brain_open_whitened_root, args.brain_open_unwhitened_root)
    brain_skullor = _load_brain_pairs(
        args.brain_skullor_whitened_root, args.brain_skullor_unwhitened_root
    )
    shin_pairs = _load_stem_pairs(args.shin_root, args.shin_root, "_msd_ratio", "_unwhitened_ratio")
    tw_along = _load_stem_pairs(
        args.twinkling_along_whitened_root, args.twinkling_along_unwhitened_root, "", ""
    )
    tw_across = _load_stem_pairs(
        args.twinkling_across_whitened_root, args.twinkling_across_unwhitened_root, "", ""
    )

    all_pairs = brain_open + brain_skullor + shin_pairs + tw_along + tw_across
    if not all_pairs:
        raise SystemExit("No paired bundles with telemetry maps were found.")

    guard_thrs = _collect_thresholds(all_pairs, "guard", quantiles)
    alias_thrs = _collect_thresholds(all_pairs, "alias", quantiles)
    bg_log_thrs = _collect_thresholds(all_pairs, "bg_inflation", quantiles, transform="log")
    cond_thrs = _collect_thresholds(all_pairs, "cond_loaded", quantiles)
    flow_thrs = _collect_thresholds(all_pairs, "flow_mu_ratio", quantiles)

    baseline = {
        "brain_open": {
            "whitened": _fixed_metrics(brain_open, alpha_brain=args.alpha_brain, alpha_other=args.alpha_other, head="whitened"),
            "unwhitened": _fixed_metrics(brain_open, alpha_brain=args.alpha_brain, alpha_other=args.alpha_other, head="unwhitened"),
        },
        "brain_skullor": {
            "whitened": _fixed_metrics(brain_skullor, alpha_brain=args.alpha_brain, alpha_other=args.alpha_other, head="whitened"),
            "unwhitened": _fixed_metrics(brain_skullor, alpha_brain=args.alpha_brain, alpha_other=args.alpha_other, head="unwhitened"),
        },
        "shin": {
            "whitened": _fixed_metrics(shin_pairs, alpha_brain=args.alpha_brain, alpha_other=args.alpha_other, head="whitened"),
            "unwhitened": _fixed_metrics(shin_pairs, alpha_brain=args.alpha_brain, alpha_other=args.alpha_other, head="unwhitened"),
        },
        "along": {
            "whitened": _fixed_metrics(tw_along, alpha_brain=args.alpha_brain, alpha_other=args.alpha_other, head="whitened"),
            "unwhitened": _fixed_metrics(tw_along, alpha_brain=args.alpha_brain, alpha_other=args.alpha_other, head="unwhitened"),
        },
        "across": {
            "whitened": _fixed_metrics(tw_across, alpha_brain=args.alpha_brain, alpha_other=args.alpha_other, head="whitened"),
            "unwhitened": _fixed_metrics(tw_across, alpha_brain=args.alpha_brain, alpha_other=args.alpha_other, head="unwhitened"),
        },
    }

    rows: list[dict[str, Any]] = []
    families = (
        "promote_w_guard_alias",
        "promote_w_guard_alias_bg",
        "promote_w_guard_only",
    )
    for family in families:
        for guard_hi in guard_thrs:
            for alias_hi in alias_thrs:
                for bg_hi_log in (bg_log_thrs if family == "promote_w_guard_alias_bg" else [float("nan")]):
                    for cond_ok_hi in cond_thrs:
                        for flow_ok_lo in flow_thrs:
                            rule = {
                                "family": family,
                                "guard_hi": float(guard_hi),
                                "alias_hi": float(alias_hi),
                                "bg_hi_log": float(bg_hi_log),
                                "cond_ok_hi": float(cond_ok_hi),
                                "flow_ok_lo": float(flow_ok_lo),
                            }
                            brain_open_tpr = _brain_median_tpr_alpha(
                                brain_open, float(args.alpha_brain), rule
                            )
                            brain_skullor_tpr = _brain_median_tpr_alpha(
                                brain_skullor, float(args.alpha_brain), rule
                            )
                            shin = _mean_score_metrics(shin_pairs, float(args.alpha_other), rule)
                            along = _mean_score_metrics(tw_along, float(args.alpha_other), rule)
                            across = _mean_score_metrics(tw_across, float(args.alpha_other), rule)
                            row = {
                                **rule,
                                "brain_open_tpr": brain_open_tpr,
                                "brain_skullor_tpr": brain_skullor_tpr,
                                "shin_auc": shin["mean_delta_auc_vs_pd"],
                                "shin_hit": shin["mean_hit"],
                                "shin_clusters": shin["mean_bg_clusters"],
                                "along_auc": along["mean_delta_auc_vs_pd"],
                                "along_hit": along["mean_hit"],
                                "along_clusters": along["mean_bg_clusters"],
                                "across_auc": across["mean_delta_auc_vs_pd"],
                                "across_hit": across["mean_hit"],
                                "across_clusters": across["mean_bg_clusters"],
                            }

                            bests = {
                                "brain_open_tpr": _baseline_best_metric(
                                    {"brain_open_tpr": baseline["brain_open"]["whitened"]["brain_tpr"]},
                                    {"brain_open_tpr": baseline["brain_open"]["unwhitened"]["brain_tpr"]},
                                    "brain_open_tpr",
                                ),
                                "brain_skullor_tpr": _baseline_best_metric(
                                    {"brain_skullor_tpr": baseline["brain_skullor"]["whitened"]["brain_tpr"]},
                                    {"brain_skullor_tpr": baseline["brain_skullor"]["unwhitened"]["brain_tpr"]},
                                    "brain_skullor_tpr",
                                ),
                                "shin_auc": _baseline_best_metric(
                                    {"shin_auc": baseline["shin"]["whitened"]["auc"]},
                                    {"shin_auc": baseline["shin"]["unwhitened"]["auc"]},
                                    "shin_auc",
                                ),
                                "shin_hit": _baseline_best_metric(
                                    {"shin_hit": baseline["shin"]["whitened"]["hit"]},
                                    {"shin_hit": baseline["shin"]["unwhitened"]["hit"]},
                                    "shin_hit",
                                ),
                                "shin_clusters": _baseline_best_metric(
                                    {"shin_clusters": baseline["shin"]["whitened"]["clusters"]},
                                    {"shin_clusters": baseline["shin"]["unwhitened"]["clusters"]},
                                    "shin_clusters",
                                ),
                                "along_auc": _baseline_best_metric(
                                    {"along_auc": baseline["along"]["whitened"]["auc"]},
                                    {"along_auc": baseline["along"]["unwhitened"]["auc"]},
                                    "along_auc",
                                ),
                                "along_hit": _baseline_best_metric(
                                    {"along_hit": baseline["along"]["whitened"]["hit"]},
                                    {"along_hit": baseline["along"]["unwhitened"]["hit"]},
                                    "along_hit",
                                ),
                                "along_clusters": _baseline_best_metric(
                                    {"along_clusters": baseline["along"]["whitened"]["clusters"]},
                                    {"along_clusters": baseline["along"]["unwhitened"]["clusters"]},
                                    "along_clusters",
                                ),
                                "across_auc": _baseline_best_metric(
                                    {"across_auc": baseline["across"]["whitened"]["auc"]},
                                    {"across_auc": baseline["across"]["unwhitened"]["auc"]},
                                    "across_auc",
                                ),
                                "across_hit": _baseline_best_metric(
                                    {"across_hit": baseline["across"]["whitened"]["hit"]},
                                    {"across_hit": baseline["across"]["unwhitened"]["hit"]},
                                    "across_hit",
                                ),
                                "across_clusters": _baseline_best_metric(
                                    {"across_clusters": baseline["across"]["whitened"]["clusters"]},
                                    {"across_clusters": baseline["across"]["unwhitened"]["clusters"]},
                                    "across_clusters",
                                ),
                            }

                            beat_count = 0
                            margin_sum = 0.0
                            for key, best_val in bests.items():
                                cur = float(row[key])
                                if key in HIGHER_BETTER:
                                    margin = cur - float(best_val)
                                    if margin >= 0.0:
                                        beat_count += 1
                                else:
                                    margin = float(best_val) - cur
                                    if cur <= float(best_val):
                                        beat_count += 1
                                margin_sum += margin
                            row["beats_best_count"] = int(beat_count)
                            row["margin_sum"] = float(margin_sum)
                            row["dominates_fixed"] = bool(beat_count == len(bests))
                            rows.append(row)

    rows_sorted = sorted(
        rows,
        key=lambda r: (int(r["beats_best_count"]), float(r["margin_sum"])),
        reverse=True,
    )
    payload = {
        "counts": {
            "brain_open": len(brain_open),
            "brain_skullor": len(brain_skullor),
            "shin": len(shin_pairs),
            "twinkling_along": len(tw_along),
            "twinkling_across": len(tw_across),
        },
        "baseline": baseline,
        "top_rows": rows_sorted[:50],
    }
    _write_json(args.out_json, payload)
    _write_csv(args.out_csv, rows_sorted)
    print(args.out_json)


if __name__ == "__main__":
    main()
