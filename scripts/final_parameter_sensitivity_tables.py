#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.stap.ka_contract_v2 import (
    KaContractV2Config,
    derive_score_shrink_v2_tile_scales,
    evaluate_ka_contract_v2,
)
from scripts.brain_whitening_policy_validation import _eval_at_tau, _tau_for_fpr
from scripts.fair_filter_comparison import _bundle_map_by_window
from scripts.shin_map_routing_analysis import _score_metrics
from sim.kwave.common import _hybrid_choose_advanced_tile_mask, _tile_iter, _tile_scores_to_map


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npy(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def _finite(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def _median(values: list[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    return float(np.median(arr)) if arr.size else float("nan")


def _iter_paired_frame_dirs(msd_root: Path, unw_root: Path) -> list[tuple[str, Path, Path]]:
    msd = {p.name: p for p in msd_root.iterdir() if p.is_dir()}
    unw = {p.name: p for p in unw_root.iterdir() if p.is_dir()}
    stems = sorted(set(msd) & set(unw))
    return [(stem, msd[stem], unw[stem]) for stem in stems]


def _load_tile_params(bundle_dir: Path) -> tuple[tuple[int, int], int]:
    meta = _load_json(bundle_dir / "meta.json")
    tile_hw = tuple(int(x) for x in meta["tile_hw"])
    stride = int(meta["tile_stride"])
    return (tile_hw[0], tile_hw[1]), stride


def _adaptive_score_map(
    whitened_dir: Path,
    unwhitened_dir: Path,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    score_w = _load_npy(whitened_dir / "score_stap_preka.npy").astype(np.float32, copy=False)
    score_u = _load_npy(unwhitened_dir / "score_stap_preka.npy").astype(np.float32, copy=False)
    feat = _load_npy(whitened_dir / "base_guard_frac_map.npy").astype(np.float32, copy=False)
    tile_hw, stride = _load_tile_params(whitened_dir)
    choose_advanced, _tile_promote = _hybrid_choose_advanced_tile_mask(
        feat,
        tile_hw=tile_hw,
        stride=stride,
        direction=">=",
        threshold=float(threshold),
        reduction="tile_mean",
        prefer_advanced_on_invalid=False,
    )
    hybrid = np.where(choose_advanced, score_w, score_u).astype(np.float32, copy=False)
    return hybrid, choose_advanced


def _brain_metric(bundle_dir: Path, score_map: np.ndarray, alpha: float) -> float:
    mask_flow = _load_npy(bundle_dir / "mask_flow.npy").astype(bool, copy=False)
    mask_bg = _load_npy(bundle_dir / "mask_bg.npy").astype(bool, copy=False)
    pos = score_map[mask_flow].astype(np.float64, copy=False).ravel()
    neg = score_map[mask_bg].astype(np.float64, copy=False).ravel()
    tau = _tau_for_fpr(neg, alpha)
    tpr, _fpr = _eval_at_tau(pos, neg, tau)
    return float(tpr)


def _adaptive_tau_sensitivity() -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    default_thr = 0.1453727245330811
    factors = [0.9, 1.0, 1.1]

    brain_specs = {
        "open_skull_tpr_1e4": (
            ROOT / "runs/pilot/brain_whitening_policy_validation/open_seed1_huber_trim8",
            ROOT / "runs/pilot/brain_whitening_policy_validation/open_seed1_unwhitened_ref",
        ),
        "structured_clutter_tpr_1e4": (
            ROOT / "runs/pilot/brain_whitening_policy_validation/skullor_seed2_huber_trim8",
            ROOT / "runs/pilot/stap_whitening_regime_sweep/skullor_seed2_gamma0p00",
        ),
    }
    shin_root = ROOT / "runs/shin_whitening_allclips_S_Lt64_cuda"
    gammex_across = (
        ROOT
        / "runs/real/twinkling_gammex_across17_prf2500_str4_msd_ratio_fast"
        / "data_twinkling_artifact_Flow_in_Gammex_phantom_Flow_in_Gammex_phantom__across_-_linear_probe___RawBCFCine_08062017_145434_17",
        ROOT
        / "runs/real/twinkling_gammex_across17_prf2500_str4_unwhitened_ratio"
        / "data_twinkling_artifact_Flow_in_Gammex_phantom_Flow_in_Gammex_phantom__across_-_linear_probe___RawBCFCine_08062017_145434_17",
    )

    rows: list[dict[str, Any]] = []
    out_rows: list[dict[str, Any]] = []
    for factor in factors:
        thr = float(default_thr * factor)
        row: dict[str, Any] = {"tau_g": thr, "multiplier": factor}

        for key, (w_root, u_root) in brain_specs.items():
            tprs: list[float] = []
            promote_fracs: list[float] = []
            w_bundles = _bundle_map_by_window(w_root, 64)
            u_bundles = _bundle_map_by_window(u_root, 64)
            for offset in sorted(set(w_bundles) & set(u_bundles)):
                score_map, choose_advanced = _adaptive_score_map(w_bundles[offset], u_bundles[offset], thr)
                tprs.append(_brain_metric(w_bundles[offset], score_map, 1e-4))
                promote_fracs.append(float(np.mean(choose_advanced)))
            row[key] = _median(tprs)
            row[f"{key}_promote_frac"] = _median(promote_fracs)

        shin_pairs = _iter_paired_frame_dirs(shin_root, shin_root)
        if not shin_pairs:
            raise RuntimeError(f"No paired Shin bundles under {shin_root}")
        shin_hits: list[float] = []
        shin_clusters: list[float] = []
        shin_promote: list[float] = []
        for _stem, w_dir, u_dir in shin_pairs:
            score_map, choose_advanced = _adaptive_score_map(w_dir, u_dir, thr)
            flow = _load_npy(w_dir / "mask_flow.npy").astype(bool, copy=False)
            bg = _load_npy(w_dir / "mask_bg.npy").astype(bool, copy=False)
            met = _score_metrics(score_map, flow, bg, alpha=1e-3, connectivity=4)
            if met.get("hit_flow") is not None:
                shin_hits.append(float(met["hit_flow"]))
            if met.get("bg_clusters") is not None:
                shin_clusters.append(float(met["bg_clusters"]))
            shin_promote.append(float(np.mean(choose_advanced)))
        row["shin_flow_hit_1e3"] = _median(shin_hits)
        row["shin_bg_clusters_1e3"] = _median(shin_clusters)
        row["shin_promote_frac"] = _median(shin_promote)

        across_pairs = _iter_paired_frame_dirs(*gammex_across)
        if not across_pairs:
            raise RuntimeError("No paired Gammex across-view bundles found.")
        across_hits: list[float] = []
        across_promote: list[float] = []
        for _stem, w_dir, u_dir in across_pairs:
            score_map, choose_advanced = _adaptive_score_map(w_dir, u_dir, thr)
            flow = _load_npy(w_dir / "mask_flow.npy").astype(bool, copy=False)
            bg = _load_npy(w_dir / "mask_bg.npy").astype(bool, copy=False)
            met = _score_metrics(score_map, flow, bg, alpha=1e-3, connectivity=4)
            if met.get("hit_flow") is not None:
                across_hits.append(float(met["hit_flow"]))
            across_promote.append(float(np.mean(choose_advanced)))
        row["gammex_across_tpr_1e3"] = _median(across_hits)
        row["gammex_across_promote_frac"] = _median(across_promote)

        rows.append(row)
        out_rows.append(dict(row))

    tex = [
        "% AUTO-GENERATED by scripts/final_parameter_sensitivity_tables.py; DO NOT EDIT BY HAND.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}lcccc@{}}",
        "\\hline",
        "$\\tau_g$ change & Open-skull TPR@$10^{-4}$ & Structured-clutter TPR@$10^{-4}$ & Shin flow hit@$10^{-3}$ & Gammex across TPR@$10^{-3}$ \\\\",
        "\\hline",
    ]
    for row in rows:
        delta_pct = int(round((float(row["multiplier"]) - 1.0) * 100.0))
        label = "default" if delta_pct == 0 else f"{delta_pct:+d}\\%"
        tex.append(
            f"{label} & "
            f"{row['open_skull_tpr_1e4']:.3f} & "
            f"{row['structured_clutter_tpr_1e4']:.3f} & "
            f"{row['shin_flow_hit_1e3']:.3f} & "
            f"{row['gammex_across_tpr_1e3']:.3f} \\\\"
        )
    tex.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\caption{Sensitivity of the frozen guard-triggered adaptive threshold $\\tau_g$ to a $\\pm 10\\%$ perturbation around the deployed tile-mean guard-energy threshold. Reported endpoints use each regime's headline metric: within-window TPR on the labeled brain stress tests, flow-proxy hit rate on the Shin real-IQ audit, and TPR on the structurally labeled Gammex across-view phantom. The detector behavior changes smoothly rather than collapsing under these perturbations.}",
            "\\label{tab:adaptive_tau_sensitivity}",
            "\\end{table}",
            "",
        ]
    )
    payload = {
        "default_tau_g": default_thr,
        "rows": rows,
    }
    return payload, out_rows, "\n".join(tex)


def _tile_means(arr: np.ndarray, tile_hw: tuple[int, int], stride: int, reducer: str = "mean") -> np.ndarray:
    vals: list[float] = []
    arr = np.asarray(arr, dtype=np.float64)
    for y0, x0 in _tile_iter(arr.shape, tile_hw, stride):
        tile = arr[y0 : y0 + tile_hw[0], x0 : x0 + tile_hw[1]]
        finite = tile[np.isfinite(tile)]
        if finite.size == 0:
            vals.append(float("nan"))
        elif reducer == "median":
            vals.append(float(np.median(finite)))
        else:
            vals.append(float(np.mean(finite)))
    return np.asarray(vals, dtype=np.float64)


def _recompute_ka_score(bundle_dir: Path, c_flow_scale: float) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    meta = _load_json(bundle_dir / "meta.json")
    report0 = meta.get("ka_contract_v2") or {}
    cfg_dict = dict(report0.get("config") or {})
    cfg_dict["c_flow"] = float(float(cfg_dict.get("c_flow", 0.20)) * float(c_flow_scale))
    cfg = KaContractV2Config(**cfg_dict)

    tile_hw = tuple(int(x) for x in meta["tile_hw"])
    stride = int(meta["tile_stride"])
    score_pre = _load_npy(bundle_dir / "score_stap_preka.npy").astype(np.float32, copy=False)
    flow_mask = _load_npy(bundle_dir / "mask_flow.npy").astype(bool, copy=False)
    bg_mask = _load_npy(bundle_dir / "mask_bg.npy").astype(bool, copy=False)
    m_alias_map = _load_npy(bundle_dir / "base_m_alias_map.npy").astype(np.float32, copy=False)
    guard_map = _load_npy(bundle_dir / "base_guard_frac_map.npy").astype(np.float32, copy=False)
    peak_freq_map = _load_npy(bundle_dir / "base_peak_freq_map.npy").astype(np.float32, copy=False)

    s_base_tiles = _tile_means(score_pre, tile_hw, stride, reducer="mean")
    m_alias_tiles = _tile_means(m_alias_map, tile_hw, stride, reducer="mean")
    r_guard_tiles = _tile_means(guard_map, tile_hw, stride, reducer="mean")
    tile_cov_flow = _tile_means(flow_mask.astype(np.float32), tile_hw, stride, reducer="mean")
    peak_freq_tiles = _tile_means(peak_freq_map, tile_hw, stride, reducer="median")
    valid_tiles = np.isfinite(s_base_tiles) & np.isfinite(m_alias_tiles) & np.isfinite(r_guard_tiles) & np.isfinite(tile_cov_flow)
    flow_low = float((((meta.get("stap_fallback_telemetry") or {}).get("band_ratio_spec") or {}).get("flow_low_hz")) or (((meta.get("stap_fallback_telemetry") or {}).get("band_ratio_stats") or {}).get("flow_low_hz")) or 150.0)
    flow_high = float((((meta.get("stap_fallback_telemetry") or {}).get("band_ratio_spec") or {}).get("flow_high_hz")) or (((meta.get("stap_fallback_telemetry") or {}).get("band_ratio_stats") or {}).get("flow_high_hz")) or 450.0)
    pf_peak_tiles = np.isfinite(peak_freq_tiles) & (peak_freq_tiles >= flow_low) & (peak_freq_tiles <= flow_high)

    report = evaluate_ka_contract_v2(
        s_base=s_base_tiles,
        m_alias=m_alias_tiles,
        r_guard=r_guard_tiles,
        pf_peak=pf_peak_tiles,
        c_flow=tile_cov_flow,
        valid_mask=valid_tiles,
        config=cfg,
    )
    score_post = score_pre
    gate_map = np.zeros_like(score_pre, dtype=bool)
    if str(report.get("reason")) == "ok":
        state = str(report.get("state") or "C0_OFF")
        apply_mode = "uplift" if state == "C2_UPLIFT" else "safety"
        risk_mode = str(((report.get("metrics") or {}).get("risk_mode")) or "alias").strip().lower()
        risk_tiles = r_guard_tiles if risk_mode == "guard" else m_alias_tiles
        shrink = derive_score_shrink_v2_tile_scales(
            report=report,
            s_base=s_base_tiles,
            m_alias=risk_tiles,
            c_flow=tile_cov_flow,
            valid_mask=valid_tiles,
            mode=apply_mode,
        )
        if bool(shrink.get("apply")):
            scale_tiles = np.asarray(shrink["scale_tiles"], dtype=np.float32)
            gated_tiles = np.asarray(shrink["gated_tiles"], dtype=bool)
            scale_map = _tile_scores_to_map(scale_tiles, score_pre.shape, tile_hw, stride)
            gate_union = np.zeros_like(score_pre, dtype=bool)
            idx = 0
            for y0, x0 in _tile_iter(score_pre.shape, tile_hw, stride):
                if idx < gated_tiles.size and gated_tiles[idx]:
                    gate_union[y0 : y0 + tile_hw[0], x0 : x0 + tile_hw[1]] = True
                idx += 1
            scale_final = np.ones_like(scale_map, dtype=np.float32)
            scale_final[gate_union] = scale_map[gate_union].astype(np.float32, copy=False)
            cfg_live = report.get("config") or {}
            protect_hi_by_score = bool(cfg_live.get("protect_hi_by_score", True))
            prot_pix = flow_mask.copy()
            if protect_hi_by_score:
                q_hi = float(cfg_live.get("q_hi_protect", 0.99999))
                finite = np.isfinite(score_pre)
                if np.any(finite):
                    thr_hi = float(np.quantile(score_pre[finite], q_hi))
                    prot_pix |= score_pre >= thr_hi
            scale_final[prot_pix] = 1.0
            score_post = (score_pre / np.maximum(scale_final, 1e-12)).astype(np.float32, copy=False)
            gate_map = gate_union

    protected_abs_delta = np.abs(score_post.astype(np.float64) - score_pre.astype(np.float64))[flow_mask]
    metrics = {
        "state": str(report.get("state") or "C0_OFF"),
        "reason": str(report.get("reason") or ""),
        "p_shrink": float(((report.get("metrics") or {}).get("p_shrink"))) if ((report.get("metrics") or {}).get("p_shrink")) is not None else None,
        "protected_max_abs_delta": float(np.max(protected_abs_delta)) if protected_abs_delta.size else 0.0,
        "gate_fraction": float(np.mean(gate_map)) if gate_map.size else 0.0,
    }
    return metrics, score_pre, score_post


def _ka_cflow_sensitivity() -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    root = (
        ROOT
        / "runs/real/twinkling_calculi_calcifications_prf500_str4_Lt8_msd_ka_scm_dl015_f050_bandsv2"
        / "data_twinkling_artifact_Twinkling_artifact_on_calculi_Twinkling_and_Flash_artifacts_on_artificial_calculi__calcifications"
    )
    bundle_dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "meta.json").is_file())
    if not bundle_dirs:
        raise RuntimeError(f"No calculi bundles found under {root}")

    rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for scale in (0.9, 1.0, 1.1):
        bundle_metrics: list[dict[str, Any]] = []
        pre_bg_all: list[np.ndarray] = []
        n_bg_total = 0
        n_flow_total = 0
        area_post = 0
        clust_post = 0
        flow_hits_post = 0
        max_protected_delta = 0.0
        score_posts: list[np.ndarray] = []
        masks_bg: list[np.ndarray] = []
        masks_flow: list[np.ndarray] = []
        for bundle_dir in bundle_dirs:
            metrics, score_pre, score_post = _recompute_ka_score(bundle_dir, scale)
            bg = _load_npy(bundle_dir / "mask_bg.npy").astype(bool, copy=False)
            flow = _load_npy(bundle_dir / "mask_flow.npy").astype(bool, copy=False)
            bundle_metrics.append(metrics)
            pre_bg_all.append(score_pre[bg].astype(np.float64, copy=False).ravel())
            score_posts.append(score_post.astype(np.float64, copy=False))
            masks_bg.append(bg)
            masks_flow.append(flow)
            max_protected_delta = max(max_protected_delta, float(metrics["protected_max_abs_delta"]))

        bg_pool = _finite(np.concatenate(pre_bg_all, axis=0))
        tau = _tau_for_fpr(bg_pool, 1e-3)
        for score_post, bg, flow in zip(score_posts, masks_bg, masks_flow):
            hit_bg = bg & (score_post >= tau)
            hit_flow = flow & (score_post >= tau)
            area_post += int(np.sum(hit_bg))
            clust_post += int(_score_metrics(score_post, flow, bg, alpha=1e-3, connectivity=4)["bg_clusters"] or 0)
            flow_hits_post += int(np.sum(hit_flow))
            n_bg_total += int(np.sum(bg))
            n_flow_total += int(np.sum(flow))

        states = [m["state"] for m in bundle_metrics]
        row = {
            "c_flow": 0.20 * scale,
            "multiplier": scale,
            "active_bundles": int(sum(s != "C0_OFF" for s in states)),
            "inert_bundles": int(sum(s == "C0_OFF" for s in states)),
            "median_p_shrink": _median([float(m["p_shrink"]) for m in bundle_metrics if m["p_shrink"] is not None]),
            "bg_tail_rate_post_1e3": (area_post / float(n_bg_total)) if n_bg_total > 0 else float("nan"),
            "bg_tail_clusters_post_1e3": float(clust_post),
            "flow_hit_rate_post_1e3": (flow_hits_post / float(n_flow_total)) if n_flow_total > 0 else float("nan"),
            "max_protected_abs_delta": float(max_protected_delta),
        }
        rows.append(row)
        csv_rows.append(dict(row))

    default_row = next(row for row in rows if abs(float(row["multiplier"]) - 1.0) < 1e-9)
    default_tail = float(default_row["bg_tail_rate_post_1e3"])
    default_clusters = float(default_row["bg_tail_clusters_post_1e3"])
    for row in rows:
        row["bg_tail_rate_rel_default"] = (
            float(row["bg_tail_rate_post_1e3"]) / default_tail if default_tail > 0 else float("nan")
        )
        row["bg_tail_clusters_rel_default"] = (
            float(row["bg_tail_clusters_post_1e3"]) / default_clusters if default_clusters > 0 else float("nan")
        )

    tex = [
        "% AUTO-GENERATED by scripts/final_parameter_sensitivity_tables.py; DO NOT EDIT BY HAND.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}lcccc@{}}",
        "\\hline",
        "$c_{\\mathrm{flow}}$ change & Active / inert bundles & median $p_{\\mathrm{shrink}}$ & bg-tail rate / default & bg-tail clusters / default \\\\",
        "\\hline",
    ]
    for row in rows:
        delta_pct = int(round((float(row["multiplier"]) - 1.0) * 100.0))
        label = "default" if delta_pct == 0 else f"{delta_pct:+d}\\%"
        tex.append(
            f"{label} & "
            f"{int(row['active_bundles'])}/{int(row['inert_bundles'])} & "
            f"{row['median_p_shrink']:.3f} & "
            f"{row['bg_tail_rate_rel_default']:.3f} & "
            f"{row['bg_tail_clusters_rel_default']:.3f} \\\\"
        )
    tex.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\caption{Sensitivity of the shrink-only regularizer to a $\\pm 10\\%$ perturbation of the flow-support proxy threshold $c_{\\mathrm{flow}}$ on the artifact-heavy Twinkling calculi audit. Tail-rate and cluster columns are normalized to the default frozen-bundle audit so that the table emphasizes robustness to threshold perturbation rather than re-reporting the headline calculi numbers. Activation, shrink coverage, and background-tail compression vary modestly rather than collapsing under these perturbations, while the protected-set invariance remains exact to numerical precision.}",
            "\\label{tab:ka_cflow_sensitivity}",
            "\\end{table}",
            "",
        ]
    )
    payload = {"default_c_flow": 0.20, "rows": rows}
    return payload, csv_rows, "\n".join(tex)


def main() -> None:
    tau_payload, tau_rows, tau_tex = _adaptive_tau_sensitivity()
    ka_payload, ka_rows, ka_tex = _ka_cflow_sensitivity()

    _write_json(ROOT / "reports/final_parameter_sensitivity.json", {"adaptive_tau_g": tau_payload, "ka_c_flow": ka_payload})
    _write_csv(ROOT / "reports/final_parameter_sensitivity.csv", tau_rows + ka_rows)
    (ROOT / "reports/companion/adaptive_tau_sensitivity_table.tex").write_text(tau_tex, encoding="utf-8")
    (ROOT / "reports/companion/ka_cflow_sensitivity_table.tex").write_text(ka_tex, encoding="utf-8")
    print(ROOT / "reports/final_parameter_sensitivity.json")


if __name__ == "__main__":
    main()
