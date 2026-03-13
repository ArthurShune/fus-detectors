#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.brain_whitening_policy_validation import _eval_at_tau, _tau_for_fpr
from scripts.fair_filter_comparison import _bundle_map_by_window
from scripts.shin_map_routing_analysis import _score_metrics


FEATURE_NAMES = (
    "guard",
    "alias",
    "peak_freq",
    "z_w",
    "z_u",
    "delta_w_minus_u",
)


@dataclass(frozen=True)
class PairRecord:
    scenario: str
    score_w: np.ndarray
    score_u: np.ndarray
    score_base: np.ndarray
    mask_flow: np.ndarray
    mask_bg: np.ndarray
    guard: np.ndarray
    alias: np.ndarray
    peak_freq: np.ndarray
    z_w: np.ndarray
    z_u: np.ndarray


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


def _load_pair(whitened_dir: Path, unwhitened_dir: Path, scenario: str) -> PairRecord | None:
    req = [
        whitened_dir / "score_stap_preka.npy",
        unwhitened_dir / "score_stap_preka.npy",
        whitened_dir / "score_base.npy",
        whitened_dir / "mask_flow.npy",
        whitened_dir / "mask_bg.npy",
        whitened_dir / "base_guard_frac_map.npy",
        whitened_dir / "base_m_alias_map.npy",
        whitened_dir / "base_peak_freq_map.npy",
    ]
    if not all(p.exists() for p in req):
        return None
    score_w = np.load(whitened_dir / "score_stap_preka.npy", allow_pickle=False).astype(np.float64, copy=False)
    score_u = np.load(unwhitened_dir / "score_stap_preka.npy", allow_pickle=False).astype(np.float64, copy=False)
    mask_bg = np.load(whitened_dir / "mask_bg.npy", allow_pickle=False).astype(bool, copy=False)
    return PairRecord(
        scenario=scenario,
        score_w=score_w,
        score_u=score_u,
        score_base=np.load(whitened_dir / "score_base.npy", allow_pickle=False).astype(np.float64, copy=False),
        mask_flow=np.load(whitened_dir / "mask_flow.npy", allow_pickle=False).astype(bool, copy=False),
        mask_bg=mask_bg,
        guard=np.load(whitened_dir / "base_guard_frac_map.npy", allow_pickle=False).astype(np.float64, copy=False),
        alias=np.load(whitened_dir / "base_m_alias_map.npy", allow_pickle=False).astype(np.float64, copy=False),
        peak_freq=np.load(whitened_dir / "base_peak_freq_map.npy", allow_pickle=False).astype(np.float64, copy=False),
        z_w=_robust_norm(score_w, mask_bg),
        z_u=_robust_norm(score_u, mask_bg),
    )


def _load_brain_pairs(whitened_root: Path, unwhitened_root: Path, scenario: str) -> list[PairRecord]:
    w_bundles = _bundle_map_by_window(whitened_root, 64)
    u_bundles = _bundle_map_by_window(unwhitened_root, 64)
    out: list[PairRecord] = []
    for offset in sorted(set(w_bundles) & set(u_bundles)):
        rec = _load_pair(w_bundles[offset], u_bundles[offset], scenario)
        if rec is not None:
            out.append(rec)
    return out


def _load_suffix_pairs(root: Path, scenario: str, suffix_w: str, suffix_u: str) -> list[PairRecord]:
    w_dirs = {
        p.name[: -len(suffix_w)]: p
        for p in root.iterdir()
        if p.is_dir() and p.name.endswith(suffix_w)
    }
    u_dirs = {
        p.name[: -len(suffix_u)]: p
        for p in root.iterdir()
        if p.is_dir() and p.name.endswith(suffix_u)
    }
    out: list[PairRecord] = []
    for stem in sorted(set(w_dirs) & set(u_dirs)):
        rec = _load_pair(w_dirs[stem], u_dirs[stem], scenario)
        if rec is not None:
            out.append(rec)
    return out


def _load_twinkling_pairs(whitened_root: Path, unwhitened_root: Path, scenario: str) -> list[PairRecord]:
    w_dirs = {p.name: p for p in whitened_root.iterdir() if p.is_dir()}
    u_dirs = {p.name: p for p in unwhitened_root.iterdir() if p.is_dir()}
    out: list[PairRecord] = []
    for stem in sorted(set(w_dirs) & set(u_dirs)):
        rec = _load_pair(w_dirs[stem], u_dirs[stem], scenario)
        if rec is not None:
            out.append(rec)
    return out


def _load_simus_pairs(setting_root: Path, scenario: str) -> list[PairRecord]:
    all_dirs = [p for p in setting_root.iterdir() if p.is_dir()]
    by_suffix: dict[str, Path] = {}
    out: list[PairRecord] = []
    by_stem: dict[str, dict[str, Path]] = {}
    for p in all_dirs:
        name = p.name
        if name.endswith("_huber_trim8"):
            stem = name[: -len("_huber_trim8")]
            by_stem.setdefault(stem, {})["w"] = p
        elif name.endswith("_unwhitened_ref"):
            stem = name[: -len("_unwhitened_ref")]
            by_stem.setdefault(stem, {})["u"] = p
    for stem, pair in sorted(by_stem.items()):
        if "w" not in pair or "u" not in pair:
            continue
        rec = _load_pair(pair["w"], pair["u"], scenario)
        if rec is not None:
            out.append(rec)
    return out


def _sample_training_matrix(pairs: list[PairRecord], per_scenario_cap: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    scenarios = sorted({p.scenario for p in pairs})
    for scenario in scenarios:
        scen_pairs = [p for p in pairs if p.scenario == scenario]
        X_rows: list[np.ndarray] = []
        y_rows: list[np.ndarray] = []
        for pair in scen_pairs:
            flow = pair.mask_flow
            bg = pair.mask_bg
            better_flow_whiten = pair.score_w >= pair.score_u
            better_bg_whiten = pair.score_w <= pair.score_u
            for mask, label_src in ((flow, better_flow_whiten), (bg, better_bg_whiten)):
                idx = np.flatnonzero(mask.ravel())
                if idx.size == 0:
                    continue
                if idx.size > per_scenario_cap:
                    idx = rng.choice(idx, size=per_scenario_cap, replace=False)
                feat = np.stack(
                    [
                        pair.guard.ravel()[idx],
                        pair.alias.ravel()[idx],
                        pair.peak_freq.ravel()[idx],
                        pair.z_w.ravel()[idx],
                        pair.z_u.ravel()[idx],
                        (pair.z_w - pair.z_u).ravel()[idx],
                    ],
                    axis=1,
                )
                lab = label_src.ravel()[idx].astype(np.int32, copy=False)
                X_rows.append(feat)
                y_rows.append(lab)
        if X_rows:
            xs.append(np.concatenate(X_rows, axis=0))
            ys.append(np.concatenate(y_rows, axis=0))
    if not xs:
        raise RuntimeError("No training samples collected")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _route_score(pair: PairRecord, model: Pipeline, threshold: float) -> np.ndarray:
    feat = np.stack(
        [
            pair.guard.ravel(),
            pair.alias.ravel(),
            pair.peak_freq.ravel(),
            pair.z_w.ravel(),
            pair.z_u.ravel(),
            (pair.z_w - pair.z_u).ravel(),
        ],
        axis=1,
    )
    prob = model.predict_proba(feat)[:, 1].reshape(pair.score_w.shape)
    use_w = prob >= float(threshold)
    return np.where(use_w, pair.score_w, pair.score_u)


def _eval_brain_tpr(pairs: list[PairRecord], model: Pipeline, threshold: float, alpha: float) -> float:
    vals: list[float] = []
    for pair in pairs:
        score = _route_score(pair, model, threshold)
        tau = _tau_for_fpr(score[pair.mask_bg], alpha)
        tpr, _ = _eval_at_tau(score[pair.mask_flow], score[pair.mask_bg], tau)
        vals.append(float(tpr))
    return float(np.median(np.asarray(vals, dtype=np.float64))) if vals else float("nan")


def _eval_score_metrics(pairs: list[PairRecord], model: Pipeline, threshold: float, alpha: float) -> dict[str, float]:
    aucs: list[float] = []
    hits: list[float] = []
    clusters: list[float] = []
    for pair in pairs:
        score = _route_score(pair, model, threshold)
        met = _score_metrics(score, pair.mask_flow, pair.mask_bg, alpha=alpha, connectivity=4)
        base = _score_metrics(pair.score_base, pair.mask_flow, pair.mask_bg, alpha=alpha, connectivity=4)
        auc = None
        if met.get("auc_flow_bg") is not None and base.get("auc_flow_bg") is not None:
            auc = float(met["auc_flow_bg"]) - float(base["auc_flow_bg"])
        aucs.append(float("nan") if auc is None else float(auc))
        hits.append(float("nan") if met.get("hit_flow") is None else float(met["hit_flow"]))
        clusters.append(float("nan") if met.get("bg_clusters") is None else float(met["bg_clusters"]))
    return {
        "auc": float(np.nanmean(np.asarray(aucs, dtype=np.float64))),
        "hit": float(np.nanmean(np.asarray(hits, dtype=np.float64))),
        "clusters": float(np.nanmean(np.asarray(clusters, dtype=np.float64))),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Learn a frozen cross-scenario hybrid router from paired whitened/unwhitened bundles.")
    ap.add_argument("--brain-open-whitened-root", type=Path, default=Path("runs/pilot/brain_whitening_policy_validation/open_seed1_huber_trim8"))
    ap.add_argument("--brain-open-unwhitened-root", type=Path, default=Path("runs/pilot/brain_whitening_policy_validation/open_seed1_unwhitened_ref"))
    ap.add_argument("--brain-skullor-whitened-root", type=Path, default=Path("runs/pilot/brain_whitening_policy_validation/skullor_seed2_huber_trim8"))
    ap.add_argument("--brain-skullor-unwhitened-root", type=Path, default=Path("runs/pilot/stap_whitening_regime_sweep/skullor_seed2_gamma0p00"))
    ap.add_argument("--shin-root", type=Path, default=Path("runs/shin_whitening_hybrid_allclips"))
    ap.add_argument("--twinkling-along-whitened-root", type=Path, default=Path("runs/real/twinkling_gammex_alonglinear17_prf2500_str6_msd_ratio_fast/data_twinkling_artifact_Flow_in_Gammex_phantom_Flow_in_Gammex_phantom__along_-_linear_probe___RawBCFCine"))
    ap.add_argument("--twinkling-along-unwhitened-root", type=Path, default=Path("runs/real/twinkling_gammex_alonglinear17_prf2500_str6_unwhitened_ratio/data_twinkling_artifact_Flow_in_Gammex_phantom_Flow_in_Gammex_phantom__along_-_linear_probe___RawBCFCine"))
    ap.add_argument("--twinkling-across-whitened-root", type=Path, default=Path("runs/real/twinkling_gammex_across17_prf2500_str4_msd_ratio_fast/data_twinkling_artifact_Flow_in_Gammex_phantom_Flow_in_Gammex_phantom__across_-_linear_probe___RawBCFCine_08062017_145434_17"))
    ap.add_argument("--twinkling-across-unwhitened-root", type=Path, default=Path("runs/real/twinkling_gammex_across17_prf2500_str4_unwhitened_ratio/data_twinkling_artifact_Flow_in_Gammex_phantom_Flow_in_Gammex_phantom__across_-_linear_probe___RawBCFCine_08062017_145434_17"))
    ap.add_argument("--simus-mobile-root", type=Path, default=Path("runs/sim_eval/simus_structural_candidate_compare_hybrid/bundles/Mobile"))
    ap.add_argument("--simus-intra-root", type=Path, default=Path("runs/sim_eval/simus_structural_candidate_compare_hybrid/bundles/Intra-operative_parenchymal"))
    ap.add_argument("--alpha-brain", type=float, default=1e-4)
    ap.add_argument("--alpha-other", type=float, default=1e-3)
    ap.add_argument("--per-scenario-cap", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--out-json", type=Path, default=Path("reports/hybrid_router_learn.json"))
    ap.add_argument("--out-csv", type=Path, default=Path("reports/hybrid_router_learn.csv"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    rng = np.random.default_rng(int(args.seed))

    scenario_pairs = {
        "brain_open": _load_brain_pairs(args.brain_open_whitened_root, args.brain_open_unwhitened_root, "brain_open"),
        "brain_skullor": _load_brain_pairs(args.brain_skullor_whitened_root, args.brain_skullor_unwhitened_root, "brain_skullor"),
        "shin": _load_suffix_pairs(args.shin_root, "shin", "_msd_ratio", "_unwhitened_ratio"),
        "twinkling_along": _load_twinkling_pairs(args.twinkling_along_whitened_root, args.twinkling_along_unwhitened_root, "twinkling_along"),
        "twinkling_across": _load_twinkling_pairs(args.twinkling_across_whitened_root, args.twinkling_across_unwhitened_root, "twinkling_across"),
        "simus_mobile": _load_simus_pairs(args.simus_mobile_root, "simus_mobile"),
        "simus_intra": _load_simus_pairs(args.simus_intra_root, "simus_intra"),
    }
    all_pairs = [pair for pairs in scenario_pairs.values() for pair in pairs]
    X, y = _sample_training_matrix(all_pairs, int(args.per_scenario_cap), rng)

    Cs = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    baselines = {
        "brain_open_tpr": 0.8688965782391388,
        "brain_skullor_tpr": 0.9165705497885429,
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
    for C in Cs:
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l1",
                        solver="liblinear",
                        C=float(C),
                        class_weight="balanced",
                        max_iter=400,
                        random_state=int(args.seed),
                    ),
                ),
            ]
        )
        model.fit(X, y)

        coef = model.named_steps["clf"].coef_[0]
        intercept = float(model.named_steps["clf"].intercept_[0])
        nonzero = int(np.count_nonzero(np.abs(coef) > 1e-8))

        for thr in thresholds:
            rec: dict[str, Any] = {
                "C": float(C),
                "threshold": float(thr),
                "nonzero_coef": nonzero,
                "intercept": intercept,
            }
            for idx, name in enumerate(FEATURE_NAMES):
                rec[f"coef_{name}"] = float(coef[idx])

            rec["brain_open_tpr"] = _eval_brain_tpr(scenario_pairs["brain_open"], model, thr, float(args.alpha_brain))
            rec["brain_skullor_tpr"] = _eval_brain_tpr(scenario_pairs["brain_skullor"], model, thr, float(args.alpha_brain))

            shin = _eval_score_metrics(scenario_pairs["shin"], model, thr, float(args.alpha_other))
            rec["shin_auc"] = shin["auc"]
            rec["shin_hit"] = shin["hit"]
            rec["shin_clusters"] = shin["clusters"]

            across = _eval_score_metrics(scenario_pairs["twinkling_across"], model, thr, float(args.alpha_other))
            rec["across_auc"] = across["auc"]
            rec["across_hit"] = across["hit"]
            rec["across_clusters"] = across["clusters"]

            along = _eval_score_metrics(scenario_pairs["twinkling_along"], model, thr, float(args.alpha_other))
            rec["along_auc"] = along["auc"]
            rec["along_hit"] = along["hit"]
            rec["along_clusters"] = along["clusters"]

            simus_mobile = _eval_score_metrics(scenario_pairs["simus_mobile"], model, thr, float(args.alpha_other))
            rec["simus_mobile_auc"] = simus_mobile["auc"]
            simus_intra = _eval_score_metrics(scenario_pairs["simus_intra"], model, thr, float(args.alpha_other))
            rec["simus_intra_auc"] = simus_intra["auc"]

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
                + 0.5 * rec["simus_mobile_auc"]
                + 0.5 * rec["simus_intra_auc"]
            )
            rows.append(rec)

    rows.sort(key=lambda r: float(r["composite_score"]), reverse=True)
    top_rows = rows[: int(args.top_k)]
    payload = {
        "counts": {k: len(v) for k, v in scenario_pairs.items()},
        "feature_names": list(FEATURE_NAMES),
        "top_rows": top_rows,
    }
    _write_csv(args.out_csv, rows)
    _write_json(args.out_json, payload)

    print(json.dumps(payload["counts"], indent=2, sort_keys=True))
    for idx, row in enumerate(top_rows, start=1):
        print(
            f"[{idx}] score={row['composite_score']:.6f} C={row['C']:.3g} thr={row['threshold']:.2f} "
            f"brain_open={row['brain_open_tpr']:.4f} brain_skullor={row['brain_skullor_tpr']:.4f} "
            f"shin_auc={row['shin_auc']:.4f} across_auc={row['across_auc']:.4f} along_auc={row['along_auc']:.4f} "
            f"simus_mobile_auc={row['simus_mobile_auc']:.4f} simus_intra_auc={row['simus_intra_auc']:.4f}"
        )
        print(
            f"    coefs guard={row['coef_guard']:.3f} alias={row['coef_alias']:.3f} peak={row['coef_peak_freq']:.3f} "
            f"z_w={row['coef_z_w']:.3f} z_u={row['coef_z_u']:.3f} delta={row['coef_delta_w_minus_u']:.3f}"
        )


if __name__ == "__main__":
    main()
