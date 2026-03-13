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


def _robust_center_scale(score: np.ndarray, mask_bg: np.ndarray) -> tuple[float, float]:
    score = np.asarray(score, dtype=np.float64)
    bg = np.asarray(mask_bg, dtype=bool)
    vals = score[bg]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    q50 = float(np.quantile(vals, 0.50))
    q999 = float(np.quantile(vals, 0.999))
    return q50, max(q999 - q50, 1e-9)


def _load_pair_maps(whitened_dir: Path, unwhitened_dir: Path) -> dict[str, np.ndarray] | None:
    req = [
        whitened_dir / "score_stap_preka.npy",
        unwhitened_dir / "score_stap_preka.npy",
        whitened_dir / "score_base.npy",
        whitened_dir / "mask_flow.npy",
        whitened_dir / "mask_bg.npy",
        whitened_dir / "base_guard_frac_map.npy",
        whitened_dir / "base_m_alias_map.npy",
    ]
    if not all(p.exists() for p in req):
        return None
    score_w = np.load(whitened_dir / "score_stap_preka.npy", allow_pickle=False)
    score_u = np.load(unwhitened_dir / "score_stap_preka.npy", allow_pickle=False)
    flow = np.load(whitened_dir / "mask_flow.npy", allow_pickle=False).astype(bool, copy=False)
    bg = np.load(whitened_dir / "mask_bg.npy", allow_pickle=False).astype(bool, copy=False)
    return {
        "score_w": score_w.astype(np.float64, copy=False),
        "score_u": score_u.astype(np.float64, copy=False),
        "score_base": np.load(whitened_dir / "score_base.npy", allow_pickle=False).astype(np.float64, copy=False),
        "flow": flow,
        "bg": bg,
        "guard": np.load(whitened_dir / "base_guard_frac_map.npy", allow_pickle=False).astype(np.float64, copy=False),
        "alias": np.load(whitened_dir / "base_m_alias_map.npy", allow_pickle=False).astype(np.float64, copy=False),
    }


def _load_brain_pairs(whitened_root: Path, unwhitened_root: Path) -> list[dict[str, np.ndarray]]:
    w_bundles = _bundle_map_by_window(whitened_root, 64)
    u_bundles = _bundle_map_by_window(unwhitened_root, 64)
    rows: list[dict[str, np.ndarray]] = []
    for offset in sorted(set(w_bundles) & set(u_bundles)):
        pair = _load_pair_maps(w_bundles[offset], u_bundles[offset])
        if pair is None:
            continue
        pair["id"] = np.array([offset], dtype=np.int32)
        rows.append(pair)
    return rows


def _load_stem_pairs(whitened_root: Path, unwhitened_root: Path, suffix_w: str, suffix_u: str) -> list[dict[str, np.ndarray]]:
    w_dirs = {
        p.name[: -len(suffix_w)]: p
        for p in whitened_root.iterdir()
        if p.is_dir() and p.name.endswith(suffix_w)
    }
    u_dirs = {
        p.name[: -len(suffix_u)]: p
        for p in unwhitened_root.iterdir()
        if p.is_dir() and p.name.endswith(suffix_u)
    }
    rows: list[dict[str, np.ndarray]] = []
    for stem in sorted(set(w_dirs) & set(u_dirs)):
        pair = _load_pair_maps(w_dirs[stem], u_dirs[stem])
        if pair is None:
            continue
        pair["id"] = np.array([hash(stem) & 0x7FFFFFFF], dtype=np.int32)
        rows.append(pair)
    return rows


def _load_frame_pairs(whitened_root: Path, unwhitened_root: Path) -> list[dict[str, np.ndarray]]:
    w_dirs = {p.name: p for p in whitened_root.iterdir() if p.is_dir()}
    u_dirs = {p.name: p for p in unwhitened_root.iterdir() if p.is_dir()}
    rows: list[dict[str, np.ndarray]] = []
    for stem in sorted(set(w_dirs) & set(u_dirs)):
        pair = _load_pair_maps(w_dirs[stem], u_dirs[stem])
        if pair is None:
            continue
        pair["id"] = np.array([hash(stem) & 0x7FFFFFFF], dtype=np.int32)
        rows.append(pair)
    return rows


def _gate_map(
    *,
    z_w: np.ndarray,
    z_u: np.ndarray,
    guard: np.ndarray,
    alias: np.ndarray,
    guard_thr: float,
    alias_thr: float,
    delta_thr: float,
    z_u_thr: float,
    logic: str,
    mode: str,
    center_w: float,
    scale_w: float,
) -> np.ndarray:
    cond_guard = guard <= float(guard_thr)
    cond_alias = alias <= float(alias_thr)
    if logic == "and":
        rescue = cond_guard & cond_alias
    else:
        rescue = cond_guard | cond_alias
    rescue &= (z_u - z_w) >= float(delta_thr)
    rescue &= z_u >= float(z_u_thr)
    if mode == "choose":
        return np.where(rescue, z_u, z_w)
    if mode == "boost":
        return z_w + rescue * np.maximum(0.0, z_u - z_w)
    if mode == "calibrated_max":
        rescue_on_w_scale = float(center_w) + float(scale_w) * z_u
        return np.where(rescue, np.maximum(z_w, rescue_on_w_scale), z_w)
    raise ValueError(f"unsupported mode {mode!r}")


def _brain_median_tpr_alpha(pairs: list[dict[str, np.ndarray]], alpha: float, rule: dict[str, Any]) -> float:
    vals: list[float] = []
    for pair in pairs:
        z = _gate_map(
            z_w=pair["z_w"],
            z_u=pair["z_u"],
            guard=pair["guard"],
            alias=pair["alias"],
            center_w=float(pair["center_w"]),
            scale_w=float(pair["scale_w"]),
            **rule,
        )
        tau = _tau_for_fpr(z[pair["bg"]], alpha)
        tpr, _ = _eval_at_tau(z[pair["flow"]], z[pair["bg"]], tau)
        vals.append(float(tpr))
    return float(np.median(np.asarray(vals, dtype=np.float64))) if vals else float("nan")


def _mean_score_metrics(pairs: list[dict[str, np.ndarray]], alpha: float, rule: dict[str, Any]) -> dict[str, float]:
    aucs: list[float] = []
    hits: list[float] = []
    clusters: list[float] = []
    for pair in pairs:
        z = _gate_map(
            z_w=pair["z_w"],
            z_u=pair["z_u"],
            guard=pair["guard"],
            alias=pair["alias"],
            center_w=float(pair["center_w"]),
            scale_w=float(pair["scale_w"]),
            **rule,
        )
        met = _score_metrics(z, pair["flow"], pair["bg"], alpha=alpha, connectivity=4)
        base = _score_metrics(pair["score_base"], pair["flow"], pair["bg"], alpha=alpha, connectivity=4)
        auc = None
        if met.get("auc_flow_bg") is not None and base.get("auc_flow_bg") is not None:
            auc = float(met["auc_flow_bg"]) - float(base["auc_flow_bg"])
        aucs.append(float("nan") if auc is None else float(auc))
        hits.append(float("nan") if met.get("hit_flow") is None else float(met["hit_flow"]))
        clusters.append(float("nan") if met.get("bg_clusters") is None else float(met["bg_clusters"]))
    arr_auc = np.asarray(aucs, dtype=np.float64)
    arr_hit = np.asarray(hits, dtype=np.float64)
    arr_clusters = np.asarray(clusters, dtype=np.float64)
    return {
        "mean_delta_auc_vs_pd": float(np.nanmean(arr_auc)),
        "mean_hit": float(np.nanmean(arr_hit)),
        "mean_bg_clusters": float(np.nanmean(arr_clusters)),
    }


def _prepare_pairs(raw_pairs: list[dict[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
    out: list[dict[str, np.ndarray]] = []
    for pair in raw_pairs:
        rec = dict(pair)
        rec["z_w"] = _robust_norm(pair["score_w"], pair["bg"])
        rec["z_u"] = _robust_norm(pair["score_u"], pair["bg"])
        rec["center_w"], rec["scale_w"] = _robust_center_scale(pair["score_w"], pair["bg"])
        out.append(rec)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Search score-aware hybrid routing rules across brain, Shin, and Gammex scenarios.")
    ap.add_argument("--brain-open-whitened-root", type=Path, default=Path("runs/pilot/brain_whitening_policy_validation/open_seed1_huber_trim8"))
    ap.add_argument("--brain-open-unwhitened-root", type=Path, default=Path("runs/pilot/brain_whitening_policy_validation/open_seed1_unwhitened_ref"))
    ap.add_argument("--brain-skullor-whitened-root", type=Path, default=Path("runs/pilot/brain_whitening_policy_validation/skullor_seed2_huber_trim8"))
    ap.add_argument("--brain-skullor-unwhitened-root", type=Path, default=Path("runs/pilot/stap_whitening_regime_sweep/skullor_seed2_gamma0p00"))
    ap.add_argument("--shin-root", type=Path, default=Path("runs/shin_whitening_hybrid_allclips"))
    ap.add_argument("--twinkling-along-whitened-root", type=Path, default=Path("runs/real/twinkling_gammex_alonglinear17_prf2500_str6_msd_ratio_fast/data_twinkling_artifact_Flow_in_Gammex_phantom_Flow_in_Gammex_phantom__along_-_linear_probe___RawBCFCine"))
    ap.add_argument("--twinkling-along-unwhitened-root", type=Path, default=Path("runs/real/twinkling_gammex_alonglinear17_prf2500_str6_unwhitened_ratio/data_twinkling_artifact_Flow_in_Gammex_phantom_Flow_in_Gammex_phantom__along_-_linear_probe___RawBCFCine"))
    ap.add_argument("--twinkling-across-whitened-root", type=Path, default=Path("runs/real/twinkling_gammex_across17_prf2500_str4_msd_ratio_fast/data_twinkling_artifact_Flow_in_Gammex_phantom_Flow_in_Gammex_phantom__across_-_linear_probe___RawBCFCine_08062017_145434_17"))
    ap.add_argument("--twinkling-across-unwhitened-root", type=Path, default=Path("runs/real/twinkling_gammex_across17_prf2500_str4_unwhitened_ratio/data_twinkling_artifact_Flow_in_Gammex_phantom_Flow_in_Gammex_phantom__across_-_linear_probe___RawBCFCine_08062017_145434_17"))
    ap.add_argument("--alpha-brain", type=float, default=1e-4)
    ap.add_argument("--alpha-other", type=float, default=1e-3)
    ap.add_argument("--logic", type=str, default="or,and", help="Comma-separated routing logics.")
    ap.add_argument("--mode", type=str, default="boost,choose,calibrated_max", help="Comma-separated fusion modes.")
    ap.add_argument(
        "--guard-quantiles",
        type=str,
        default="0.02,0.05,0.10,0.15,0.20,0.25",
        help="Comma-separated quantiles for guard-threshold search.",
    )
    ap.add_argument(
        "--alias-quantiles",
        type=str,
        default="0.02,0.05,0.10,0.15,0.20,0.25",
        help="Comma-separated quantiles for alias-threshold search.",
    )
    ap.add_argument(
        "--delta-thresholds",
        type=str,
        default="0,0.1,0.25,0.5,1.0",
        help="Comma-separated explicit normalized score-delta thresholds.",
    )
    ap.add_argument(
        "--z-u-thresholds",
        type=str,
        default="0,median",
        help="Comma-separated explicit z_u thresholds; use 'median' for the pooled median.",
    )
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--out-json", type=Path, default=Path("reports/hybrid_router_search.json"))
    ap.add_argument("--out-csv", type=Path, default=Path("reports/hybrid_router_search.csv"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    brain_pairs = {
        "open": _prepare_pairs(_load_brain_pairs(args.brain_open_whitened_root, args.brain_open_unwhitened_root)),
        "skullor": _prepare_pairs(_load_brain_pairs(args.brain_skullor_whitened_root, args.brain_skullor_unwhitened_root)),
    }
    shin_pairs = _prepare_pairs(_load_stem_pairs(args.shin_root, args.shin_root, "_msd_ratio", "_unwhitened_ratio"))
    twink_pairs = {
        "along": _prepare_pairs(_load_frame_pairs(args.twinkling_along_whitened_root, args.twinkling_along_unwhitened_root)),
        "across": _prepare_pairs(_load_frame_pairs(args.twinkling_across_whitened_root, args.twinkling_across_unwhitened_root)),
    }

    if not shin_pairs:
        raise SystemExit("No Shin paired bundles found")

    all_guard = np.concatenate(
        [pair["guard"].ravel() for items in brain_pairs.values() for pair in items]
        + [pair["guard"].ravel() for pair in shin_pairs]
        + [pair["guard"].ravel() for items in twink_pairs.values() for pair in items]
    )
    all_alias = np.concatenate(
        [pair["alias"].ravel() for items in brain_pairs.values() for pair in items]
        + [pair["alias"].ravel() for pair in shin_pairs]
        + [pair["alias"].ravel() for items in twink_pairs.values() for pair in items]
    )
    all_delta = np.concatenate(
        [(pair["z_u"] - pair["z_w"]).ravel() for items in brain_pairs.values() for pair in items]
        + [(pair["z_u"] - pair["z_w"]).ravel() for pair in shin_pairs]
        + [(pair["z_u"] - pair["z_w"]).ravel() for items in twink_pairs.values() for pair in items]
    )
    all_z_u = np.concatenate(
        [pair["z_u"].ravel() for items in brain_pairs.values() for pair in items]
        + [pair["z_u"].ravel() for pair in shin_pairs]
        + [pair["z_u"].ravel() for items in twink_pairs.values() for pair in items]
    )

    guard_thresholds = sorted(
        {
            float(np.quantile(all_guard, float(q)))
            for q in [x.strip() for x in str(args.guard_quantiles).split(",") if x.strip()]
        }
    )
    alias_thresholds = sorted(
        {
            float(np.quantile(all_alias, float(q)))
            for q in [x.strip() for x in str(args.alias_quantiles).split(",") if x.strip()]
        }
    )
    delta_thresholds = sorted(
        set(
            [float(x) for x in [s.strip() for s in str(args.delta_thresholds).split(",") if s.strip()]]
            + [float(np.quantile(all_delta, q)) for q in (0.50, 0.70, 0.80, 0.90)]
        )
    )
    z_u_thresholds: list[float] = []
    for raw in [x.strip().lower() for x in str(args.z_u_thresholds).split(",") if x.strip()]:
        if raw == "median":
            z_u_thresholds.append(float(np.quantile(all_z_u, 0.50)))
        else:
            z_u_thresholds.append(float(raw))
    z_u_thresholds = sorted(set(z_u_thresholds))

    baselines = {
        "brain_open_tpr": max(0.8523644752018454, 0.8688965782391388),
        "brain_skullor_tpr": max(0.9165705497885429, 0.9104190695886197),
        "shin_auc": -0.17702760188044953,
        "shin_hit": 0.050118964026903814,
        "shin_clusters": 8.6875,
        "across_auc": 0.9372882493322326,
        "across_hit": 0.24996115427302995,
        "across_clusters": 5.79,
        "along_auc": 0.9186531705286443,
        "along_hit": 0.00030722595444863184,
        "along_clusters": 18.235294117647058,
    }

    rows: list[dict[str, Any]] = []
    logic_list = [x.strip() for x in str(args.logic).split(",") if x.strip()]
    mode_list = [x.strip() for x in str(args.mode).split(",") if x.strip()]

    for logic in logic_list:
        for mode in mode_list:
            for guard_thr in guard_thresholds:
                for alias_thr in alias_thresholds:
                    for delta_thr in delta_thresholds:
                        for z_u_thr in z_u_thresholds:
                            rule = {
                                "guard_thr": guard_thr,
                                "alias_thr": alias_thr,
                                "delta_thr": delta_thr,
                                "z_u_thr": z_u_thr,
                                "logic": logic,
                                "mode": mode,
                            }
                            rec: dict[str, Any] = dict(rule)
                            rec["brain_open_tpr"] = _brain_median_tpr_alpha(
                                brain_pairs["open"], float(args.alpha_brain), rule
                            )
                            rec["brain_skullor_tpr"] = _brain_median_tpr_alpha(
                                brain_pairs["skullor"], float(args.alpha_brain), rule
                            )
                            for name, pairs in (("shin", shin_pairs), ("across", twink_pairs["across"]), ("along", twink_pairs["along"])):
                                met = _mean_score_metrics(pairs, float(args.alpha_other), rule)
                                rec[f"{name}_auc"] = met["mean_delta_auc_vs_pd"]
                                rec[f"{name}_hit"] = met["mean_hit"]
                                rec[f"{name}_clusters"] = met["mean_bg_clusters"]

                            rec["passes_brain_floor"] = bool(
                                rec["brain_open_tpr"] >= 0.845 and rec["brain_skullor_tpr"] >= 0.912
                            )
                            rec["passes_auc_floor"] = bool(
                                rec["shin_auc"] >= baselines["shin_auc"] - 0.02
                                and rec["across_auc"] >= baselines["across_auc"] - 0.02
                                and rec["along_auc"] >= baselines["along_auc"] - 0.02
                            )
                            rec["composite_score"] = (
                                rec["brain_open_tpr"]
                                + rec["brain_skullor_tpr"]
                                + 2.0 * (rec["shin_auc"] - baselines["shin_auc"])
                                + 2.0 * (rec["across_auc"] - baselines["across_auc"])
                                + 2.0 * (rec["along_auc"] - baselines["along_auc"])
                                + 2.0 * (rec["shin_hit"] - baselines["shin_hit"])
                                + 2.0 * (rec["across_hit"] - baselines["across_hit"])
                                + 100.0 * (rec["along_hit"] - baselines["along_hit"])
                                + (baselines["shin_clusters"] - rec["shin_clusters"]) / 10.0
                                + (baselines["across_clusters"] - rec["across_clusters"]) / 10.0
                                + (baselines["along_clusters"] - rec["along_clusters"]) / 20.0
                            )
                            rows.append(rec)

    rows.sort(
        key=lambda r: (
            bool(r["passes_brain_floor"]),
            bool(r["passes_auc_floor"]),
            float(r["composite_score"]),
        ),
        reverse=True,
    )
    top_rows = rows[: int(args.top_k)]
    payload = {
        "counts": {
            "brain_open": len(brain_pairs["open"]),
            "brain_skullor": len(brain_pairs["skullor"]),
            "shin": len(shin_pairs),
            "twinkling_along": len(twink_pairs["along"]),
            "twinkling_across": len(twink_pairs["across"]),
        },
        "top_rows": top_rows,
        "search_space": {
            "guard_thresholds": guard_thresholds,
            "alias_thresholds": alias_thresholds,
            "delta_thresholds": delta_thresholds,
            "z_u_thresholds": z_u_thresholds,
            "logic": ["or", "and"],
        "mode": ["boost", "choose", "calibrated_max"],
        },
    }
    _write_csv(args.out_csv, rows)
    _write_json(args.out_json, payload)

    print(json.dumps(payload["counts"], indent=2, sort_keys=True))
    for idx, row in enumerate(top_rows[: int(args.top_k)], start=1):
        print(f"[{idx}] score={row['composite_score']:.6f} logic={row['logic']} mode={row['mode']} guard={row['guard_thr']:.6g} alias={row['alias_thr']:.6g} delta={row['delta_thr']:.6g} zu={row['z_u_thr']:.6g}")
        print(
            f"    brain open={row['brain_open_tpr']:.4f} skullor={row['brain_skullor_tpr']:.4f} | "
            f"shin auc={row['shin_auc']:.4f} hit={row['shin_hit']:.4f} cls={row['shin_clusters']:.2f} | "
            f"across auc={row['across_auc']:.4f} hit={row['across_hit']:.4f} cls={row['across_clusters']:.2f} | "
            f"along auc={row['along_auc']:.4f} hit={row['along_hit']:.6f} cls={row['along_clusters']:.2f}"
        )


if __name__ == "__main__":
    main()
