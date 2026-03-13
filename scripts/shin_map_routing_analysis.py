#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_MAPS = [
    "base_peak_freq_map",
    "base_band_ratio_map",
    "base_m_alias_map",
    "base_guard_frac_map",
]


def _finite(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def _auc_pos_vs_neg(pos: np.ndarray, neg: np.ndarray) -> float | None:
    pos = _finite(pos)
    neg = _finite(neg)
    if pos.size == 0 or neg.size == 0:
        return None
    neg_sorted = np.sort(neg)
    less = np.searchsorted(neg_sorted, pos, side="left")
    right = np.searchsorted(neg_sorted, pos, side="right")
    equal = right - less
    return float((float(np.sum(less)) + 0.5 * float(np.sum(equal))) / float(pos.size * neg.size))


def _connected_components(binary: np.ndarray, connectivity: int = 4) -> int:
    binary = np.asarray(binary, dtype=bool)
    if binary.size == 0:
        return 0
    try:
        import scipy.ndimage as ndi  # type: ignore

        structure = np.ones((3, 3), dtype=int) if connectivity == 8 else np.array(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int
        )
        _, n = ndi.label(binary, structure=structure)
        return int(n)
    except Exception:
        H, W = binary.shape
        visited = np.zeros_like(binary, dtype=bool)
        n = 0
        neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if connectivity == 8:
            neigh += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for y in range(H):
            for x in range(W):
                if not binary[y, x] or visited[y, x]:
                    continue
                n += 1
                stack = [(y, x)]
                visited[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    for dy, dx in neigh:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W and binary[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
        return int(n)


def _score_metrics(
    score: np.ndarray,
    flow: np.ndarray,
    bg: np.ndarray,
    *,
    alpha: float,
    connectivity: int,
) -> dict[str, float | int | None]:
    score = np.asarray(score, dtype=np.float64)
    flow = np.asarray(flow, dtype=bool)
    bg = np.asarray(bg, dtype=bool)
    flow_vals = _finite(score[flow])
    bg_vals = _finite(score[bg])
    out: dict[str, float | int | None] = {
        "auc_flow_bg": _auc_pos_vs_neg(flow_vals, bg_vals),
        "flow_q50": float(np.quantile(flow_vals, 0.50)) if flow_vals.size else None,
        "flow_q90": float(np.quantile(flow_vals, 0.90)) if flow_vals.size else None,
        "bg_q999": float(np.quantile(bg_vals, 0.999)) if bg_vals.size else None,
    }
    if bg_vals.size == 0:
        out.update(
            {
                "thr": None,
                "hit_flow": None,
                "bg_clusters": None,
                "bg_area": None,
                "bg_excess_mass": None,
            }
        )
        return out
    thr = float(np.quantile(bg_vals, 1.0 - float(alpha)))
    hit_bg = bg & (score >= thr)
    hit_flow = flow & (score >= thr)
    out["thr"] = thr
    out["hit_flow"] = float(np.mean(hit_flow[flow])) if np.any(flow) else None
    out["bg_clusters"] = int(_connected_components(hit_bg, connectivity=connectivity))
    out["bg_area"] = int(np.sum(hit_bg))
    excess = np.clip(score[bg] - thr, a_min=0.0, a_max=None)
    out["bg_excess_mass"] = float(np.sum(excess[np.isfinite(excess)]))
    return out


def _iter_bundle_pairs(root: Path) -> list[tuple[str, Path, Path]]:
    msd: dict[str, Path] = {}
    unw: dict[str, Path] = {}
    for meta_path in root.rglob("meta.json"):
        if not meta_path.is_file():
            continue
        bundle_dir = meta_path.parent
        name = bundle_dir.name
        if name.endswith("_msd_ratio"):
            stem = name[: -len("_msd_ratio")]
            msd[stem] = bundle_dir
        elif name.endswith("_unwhitened_ratio"):
            stem = name[: -len("_unwhitened_ratio")]
            unw[stem] = bundle_dir
    stems = sorted(set(msd) & set(unw))
    return [(stem, msd[stem], unw[stem]) for stem in stems]


def _load_map(bundle_dir: Path, name: str) -> np.ndarray:
    return np.load(bundle_dir / f"{name}.npy", allow_pickle=False)


def _robust_normalize(score: np.ndarray, bg: np.ndarray) -> np.ndarray:
    vals = _finite(np.asarray(score, dtype=np.float64)[np.asarray(bg, dtype=bool)])
    if vals.size == 0:
        return np.asarray(score, dtype=np.float64)
    q50 = float(np.quantile(vals, 0.50))
    q999 = float(np.quantile(vals, 0.999))
    scale = max(q999 - q50, 1e-9)
    return (np.asarray(score, dtype=np.float64) - q50) / scale


def _collect_feature_thresholds(pairs: list[tuple[str, Path, Path]], quantiles: list[float]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for feat in FEATURE_MAPS:
        vals_all: list[np.ndarray] = []
        for _stem, msd_dir, _unw_dir in pairs:
            arr = _load_map(msd_dir, feat)
            vals = _finite(arr)
            if vals.size:
                vals_all.append(vals)
        if not vals_all:
            out[feat] = []
            continue
        vals_cat = np.concatenate(vals_all)
        thresholds = sorted({float(np.quantile(vals_cat, q)) for q in quantiles})
        out[feat] = thresholds
    return out


def _summarize_rows(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [r.get(key) for r in rows]
    arr = np.asarray([v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def _evaluate_fixed(
    pairs: list[tuple[str, Path, Path]],
    *,
    head: str,
    alpha: float,
    connectivity: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stem, msd_dir, unw_dir in pairs:
        bundle_dir = msd_dir if head == "whitened" else unw_dir
        score = _load_map(bundle_dir, "score_stap_preka")
        base = _load_map(bundle_dir, "score_base")
        flow = _load_map(bundle_dir, "mask_flow")
        bg = _load_map(bundle_dir, "mask_bg")
        met = _score_metrics(score, flow, bg, alpha=alpha, connectivity=connectivity)
        base_auc = _score_metrics(base, flow, bg, alpha=alpha, connectivity=connectivity).get("auc_flow_bg")
        rows.append(
            {
                "bundle": stem,
                "mode": head,
                "mean_feature_fraction_whitened": 1.0 if head == "whitened" else 0.0,
                "auc_flow_bg": met.get("auc_flow_bg"),
                "delta_auc_vs_pd": (
                    None
                    if met.get("auc_flow_bg") is None or base_auc is None
                    else float(met["auc_flow_bg"]) - float(base_auc)
                ),
                "strict_hit": met.get("hit_flow"),
                "hyg_hit": met.get("hit_flow"),
                "hyg_clusters": met.get("bg_clusters"),
                "hyg_area": met.get("bg_area"),
                "hyg_excess_mass": met.get("bg_excess_mass"),
            }
        )
    summary = {
        "n_clips": len(rows),
        "mean_delta_auc_vs_pd": _summarize_rows(rows, "delta_auc_vs_pd"),
        "mean_strict_hit": _summarize_rows(rows, "strict_hit"),
        "mean_hyg_hit": _summarize_rows(rows, "hyg_hit"),
        "mean_hyg_clusters": _summarize_rows(rows, "hyg_clusters"),
        "mean_hyg_area": _summarize_rows(rows, "hyg_area"),
        "mean_hyg_excess_mass": _summarize_rows(rows, "hyg_excess_mass"),
    }
    return rows, summary


def _evaluate_rule(
    pairs: list[tuple[str, Path, Path]],
    *,
    feature_name: str,
    direction: str,
    threshold: float,
    alpha: float,
    connectivity: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for stem, msd_dir, unw_dir in pairs:
        score_w = _load_map(msd_dir, "score_stap_preka")
        score_u = _load_map(unw_dir, "score_stap_preka")
        base = _load_map(msd_dir, "score_base")
        flow = _load_map(msd_dir, "mask_flow")
        bg = _load_map(msd_dir, "mask_bg")
        feat = _load_map(msd_dir, feature_name).astype(np.float64, copy=False)
        valid = np.isfinite(feat)
        cond = feat <= threshold if direction == "<=" else feat >= threshold
        cond &= valid
        score = np.where(cond, score_w, score_u)
        met = _score_metrics(score, flow, bg, alpha=alpha, connectivity=connectivity)
        base_auc = _score_metrics(base, flow, bg, alpha=alpha, connectivity=connectivity).get("auc_flow_bg")
        rows.append(
            {
                "bundle": stem,
                "mode": f"{feature_name}{direction}{threshold:.6g}",
                "mean_feature_fraction_whitened": float(np.mean(cond)),
                "auc_flow_bg": met.get("auc_flow_bg"),
                "delta_auc_vs_pd": (
                    None
                    if met.get("auc_flow_bg") is None or base_auc is None
                    else float(met["auc_flow_bg"]) - float(base_auc)
                ),
                "strict_hit": met.get("hit_flow"),
                "hyg_hit": met.get("hit_flow"),
                "hyg_clusters": met.get("bg_clusters"),
                "hyg_area": met.get("bg_area"),
                "hyg_excess_mass": met.get("bg_excess_mass"),
            }
        )
    return {
        "feature": feature_name,
        "direction": direction,
        "threshold": float(threshold),
        "mean_choose_whitened_fraction": _summarize_rows(rows, "mean_feature_fraction_whitened"),
        "mean_delta_auc_vs_pd": _summarize_rows(rows, "delta_auc_vs_pd"),
        "mean_strict_hit": _summarize_rows(rows, "strict_hit"),
        "mean_hyg_hit": _summarize_rows(rows, "hyg_hit"),
        "mean_hyg_clusters": _summarize_rows(rows, "hyg_clusters"),
        "mean_hyg_area": _summarize_rows(rows, "hyg_area"),
        "mean_hyg_excess_mass": _summarize_rows(rows, "hyg_excess_mass"),
        "rows": rows,
    }


def _evaluate_blend(
    pairs: list[tuple[str, Path, Path]],
    *,
    alpha_mix: float,
    alpha: float,
    connectivity: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for stem, msd_dir, unw_dir in pairs:
        score_w = _load_map(msd_dir, "score_stap_preka")
        score_u = _load_map(unw_dir, "score_stap_preka")
        base = _load_map(msd_dir, "score_base")
        flow = _load_map(msd_dir, "mask_flow")
        bg = _load_map(msd_dir, "mask_bg")
        score_w_n = _robust_normalize(score_w, bg)
        score_u_n = _robust_normalize(score_u, bg)
        hybrid = float(alpha_mix) * score_w_n + (1.0 - float(alpha_mix)) * score_u_n
        met = _score_metrics(hybrid, flow, bg, alpha=alpha, connectivity=connectivity)
        base_auc = _score_metrics(base, flow, bg, alpha=alpha, connectivity=connectivity).get("auc_flow_bg")
        rows.append(
            {
                "bundle": stem,
                "mode": f"blend_{alpha_mix:.2f}",
                "mean_feature_fraction_whitened": float(alpha_mix),
                "auc_flow_bg": met.get("auc_flow_bg"),
                "delta_auc_vs_pd": (
                    None
                    if met.get("auc_flow_bg") is None or base_auc is None
                    else float(met["auc_flow_bg"]) - float(base_auc)
                ),
                "strict_hit": met.get("hit_flow"),
                "hyg_hit": met.get("hit_flow"),
                "hyg_clusters": met.get("bg_clusters"),
                "hyg_area": met.get("bg_area"),
                "hyg_excess_mass": met.get("bg_excess_mass"),
            }
        )
    return {
        "alpha_mix": float(alpha_mix),
        "mean_delta_auc_vs_pd": _summarize_rows(rows, "delta_auc_vs_pd"),
        "mean_strict_hit": _summarize_rows(rows, "strict_hit"),
        "mean_hyg_hit": _summarize_rows(rows, "hyg_hit"),
        "mean_hyg_clusters": _summarize_rows(rows, "hyg_clusters"),
        "mean_hyg_area": _summarize_rows(rows, "hyg_area"),
        "mean_hyg_excess_mass": _summarize_rows(rows, "hyg_excess_mass"),
        "rows": rows,
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
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("runs/shin_whitening_allclips_S_Lt64_cuda"))
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--connectivity", type=int, default=4, choices=[4, 8])
    ap.add_argument(
        "--feature-quantiles",
        type=str,
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
    )
    ap.add_argument("--auc-drop-budget", type=float, default=0.01)
    ap.add_argument(
        "--blend-alphas",
        type=str,
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    )
    ap.add_argument("--out-csv", type=Path, default=Path("reports/shin_map_routing_analysis.csv"))
    ap.add_argument("--out-json", type=Path, default=Path("reports/shin_map_routing_analysis.json"))
    args = ap.parse_args()

    quantiles = [float(x) for x in args.feature_quantiles.split(",") if x.strip()]
    blend_alphas = [float(x) for x in args.blend_alphas.split(",") if x.strip()]
    pairs = _iter_bundle_pairs(Path(args.root))
    if not pairs:
        raise SystemExit(f"No paired msd/unwhitened bundles found under {args.root}")

    _fixed_w_rows, fixed_w = _evaluate_fixed(pairs, head="whitened", alpha=float(args.alpha), connectivity=int(args.connectivity))
    _fixed_u_rows, fixed_u = _evaluate_fixed(pairs, head="unwhitened", alpha=float(args.alpha), connectivity=int(args.connectivity))

    thresholds = _collect_feature_thresholds(pairs, quantiles)
    rule_results: list[dict[str, Any]] = []
    for feat, vals in thresholds.items():
        for thr in vals:
            for direction in ("<=", ">="):
                rule_results.append(
                    _evaluate_rule(
                        pairs,
                        feature_name=feat,
                        direction=direction,
                        threshold=float(thr),
                        alpha=float(args.alpha),
                        connectivity=int(args.connectivity),
                    )
                )

    blend_results = [
        _evaluate_blend(
            pairs,
            alpha_mix=float(a),
            alpha=float(args.alpha),
            connectivity=int(args.connectivity),
        )
        for a in blend_alphas
    ]

    for item in rule_results:
        item["delta_vs_whitened_auc"] = (
            None if item["mean_delta_auc_vs_pd"] is None or fixed_w["mean_delta_auc_vs_pd"] is None
            else float(item["mean_delta_auc_vs_pd"]) - float(fixed_w["mean_delta_auc_vs_pd"])
        )
        item["delta_vs_whitened_hyg_hit"] = (
            None if item["mean_hyg_hit"] is None or fixed_w["mean_hyg_hit"] is None
            else float(item["mean_hyg_hit"]) - float(fixed_w["mean_hyg_hit"])
        )
        item["delta_vs_whitened_hyg_clusters"] = (
            None if item["mean_hyg_clusters"] is None or fixed_w["mean_hyg_clusters"] is None
            else float(item["mean_hyg_clusters"]) - float(fixed_w["mean_hyg_clusters"])
        )
        item["auc_within_budget"] = bool(
            item["mean_delta_auc_vs_pd"] is not None
            and fixed_w["mean_delta_auc_vs_pd"] is not None
            and float(item["mean_delta_auc_vs_pd"]) >= float(fixed_w["mean_delta_auc_vs_pd"]) - float(args.auc_drop_budget)
        )
        item["improves_vs_whitened"] = bool(
            item["auc_within_budget"]
            and item["delta_vs_whitened_hyg_hit"] is not None
            and item["delta_vs_whitened_hyg_clusters"] is not None
            and float(item["delta_vs_whitened_hyg_hit"]) > 0.0
            and float(item["delta_vs_whitened_hyg_clusters"]) < 0.0
        )
        item["dominates_both_fixed"] = bool(
            item["mean_delta_auc_vs_pd"] is not None
            and item["mean_hyg_hit"] is not None
            and item["mean_hyg_clusters"] is not None
            and fixed_w["mean_delta_auc_vs_pd"] is not None
            and fixed_u["mean_delta_auc_vs_pd"] is not None
            and fixed_w["mean_hyg_hit"] is not None
            and fixed_u["mean_hyg_hit"] is not None
            and fixed_w["mean_hyg_clusters"] is not None
            and fixed_u["mean_hyg_clusters"] is not None
            and float(item["mean_delta_auc_vs_pd"])
            >= max(float(fixed_w["mean_delta_auc_vs_pd"]), float(fixed_u["mean_delta_auc_vs_pd"]))
            and float(item["mean_hyg_hit"])
            >= max(float(fixed_w["mean_hyg_hit"]), float(fixed_u["mean_hyg_hit"]))
            and float(item["mean_hyg_clusters"])
            <= min(float(fixed_w["mean_hyg_clusters"]), float(fixed_u["mean_hyg_clusters"]))
        )

    pareto_rules = [
        {k: v for k, v in item.items() if k != "rows"}
        for item in rule_results
        if item["improves_vs_whitened"]
    ]
    pareto_rules.sort(
        key=lambda r: (
            float(r["delta_vs_whitened_hyg_hit"]),
            -float(r["delta_vs_whitened_hyg_clusters"]),
            float(r["mean_delta_auc_vs_pd"]),
        ),
        reverse=True,
    )

    dominating_rules = [
        {k: v for k, v in item.items() if k != "rows"}
        for item in rule_results
        if item["dominates_both_fixed"]
    ]
    dominating_rules.sort(
        key=lambda r: (
            float(r["mean_delta_auc_vs_pd"]),
            -float(r["mean_hyg_clusters"]),
            float(r["mean_hyg_hit"]),
        ),
        reverse=True,
    )

    blend_results_clean = [{k: v for k, v in item.items() if k != "rows"} for item in blend_results]
    blend_results_clean.sort(
        key=lambda r: (
            float(r["mean_delta_auc_vs_pd"]) if r["mean_delta_auc_vs_pd"] is not None else -1e9,
            float(r["mean_hyg_hit"]) if r["mean_hyg_hit"] is not None else -1e9,
            -float(r["mean_hyg_clusters"]) if r["mean_hyg_clusters"] is not None else -1e9,
        ),
        reverse=True,
    )

    recommended_rule = (
        dominating_rules[0]
        if dominating_rules
        else (pareto_rules[0] if pareto_rules else None)
    )
    best_blend = blend_results_clean[0] if blend_results_clean else None

    csv_rows = pareto_rules[:12] if pareto_rules else blend_results_clean[:12]
    _write_csv(Path(args.out_csv), csv_rows)
    payload = {
        "config": {
            "root": str(args.root),
            "alpha": float(args.alpha),
            "connectivity": int(args.connectivity),
            "feature_quantiles": quantiles,
            "blend_alphas": blend_alphas,
            "auc_drop_budget": float(args.auc_drop_budget),
            "n_pairs": len(pairs),
        },
        "baseline_whitened": fixed_w,
        "baseline_unwhitened": fixed_u,
        "recommended_rule": recommended_rule,
        "dominating_rules": dominating_rules[:20],
        "pareto_rules": pareto_rules[:20],
        "best_blend": best_blend,
        "blend_results": blend_results_clean,
    }
    _write_json(Path(args.out_json), payload)

    print(f"Loaded {len(pairs)} paired clips.")
    print("Baseline whitened:", fixed_w)
    print("Baseline unwhitened:", fixed_u)
    if recommended_rule is not None:
        print("Recommended map rule:", recommended_rule)
    if best_blend is not None:
        print("Best blend:", best_blend)


if __name__ == "__main__":
    main()
