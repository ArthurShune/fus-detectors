from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _iter_bundle_dirs(root: Path) -> list[Path]:
    out: list[Path] = []
    for meta_path in root.rglob("meta.json"):
        if meta_path.is_file():
            out.append(meta_path.parent)
    out.sort()
    return out


def _load_meta(bundle_dir: Path) -> dict[str, Any] | None:
    try:
        meta = json.loads((bundle_dir / "meta.json").read_text())
    except Exception:
        return None
    return meta if isinstance(meta, dict) else None


def _load_npy(bundle_dir: Path, name: str) -> np.ndarray | None:
    path = bundle_dir / f"{name}.npy"
    if not path.is_file():
        return None
    return np.load(path, allow_pickle=False)


def _finite_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return x[np.isfinite(x)]


def _right_tail_threshold(bg_scores: np.ndarray, alpha: float) -> tuple[float | None, float | None]:
    """
    Choose a threshold tau for right-tail scoring (higher = more positive) so that
    mean(bg >= tau) is approximately alpha on the provided sample.

    Returns (tau, realized_fpr). Uses a discrete rank rule (no interpolation).
    """
    bg = _finite_1d(bg_scores)
    n = int(bg.size)
    if n <= 0:
        return None, None
    a = float(alpha)
    if not np.isfinite(a) or a <= 0.0:
        tau = float("inf")
        return tau, 0.0
    if a >= 1.0:
        tau = float(np.min(bg))
        return tau, 1.0

    k = int(np.ceil(a * n))
    k = max(1, min(k, n))
    # tau = k-th largest value (partition on (n-k)-th smallest).
    tau = float(np.partition(bg, n - k)[n - k])
    realized = float(np.mean(bg >= tau))
    return tau, realized


def _connected_components(binary: np.ndarray, connectivity: int = 4) -> int:
    binary = np.asarray(binary, dtype=bool)
    if binary.size == 0:
        return 0
    try:
        import scipy.ndimage as ndi  # type: ignore

        if connectivity == 8:
            structure = np.ones((3, 3), dtype=int)
        else:
            structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
        _, n = ndi.label(binary, structure=structure)
        return int(n)
    except Exception:
        # Fallback BFS (slow but robust).
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


def _safe_mean(x: np.ndarray) -> float | None:
    xx = _finite_1d(x)
    if xx.size == 0:
        return None
    return float(np.mean(xx))


def _summary_stats(values: list[float | None]) -> dict[str, Any]:
    """Robust summary stats for a list that may contain None."""
    n_total = int(len(values))
    finite = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    out: dict[str, Any] = {
        "count_total": n_total,
        "count_finite": int(finite.size),
        "count_none_or_nonfinite": int(n_total - finite.size),
        "median": None,
        "p25": None,
        "p75": None,
        "min": None,
        "max": None,
        "frac_gt0": None,
    }
    if finite.size:
        out.update(
            {
                "median": float(np.median(finite)),
                "p25": float(np.quantile(finite, 0.25)),
                "p75": float(np.quantile(finite, 0.75)),
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "frac_gt0": float(np.mean(finite > 0.0)),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate structural-label ROC and KA tail-hygiene metrics on Twinkling/Gammex bundles.\n"
            "Uses mask_flow.npy/mask_bg.npy derived from B-mode-only tube segmentation.\n"
            "Scores are assumed right-tail (higher = more flow evidence): score_base, score_stap_preka, score_stap."
        )
    )
    parser.add_argument("--root", type=Path, required=True, help="Root directory containing bundles.")
    parser.add_argument("--out-csv", type=Path, required=True, help="Output per-bundle ROC CSV.")
    parser.add_argument(
        "--out-summary-json",
        type=Path,
        required=True,
        help="Output summary JSON (pooled ROC + hygiene).",
    )
    parser.add_argument(
        "--fprs",
        type=float,
        nargs="+",
        default=[1e-4, 3e-4, 1e-3],
        help="FPR targets for ROC points (default: %(default)s).",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        default=4,
        choices=[4, 8],
        help="Connectivity for cluster counts (default: %(default)s).",
    )
    parser.add_argument(
        "--frame-bootstrap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute frame-bootstrap CIs for pooled ROC points (default: %(default)s).",
    )
    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=2000,
        help="Number of frame-bootstrap replicates (default: %(default)s).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=1337,
        help="Seed for frame-bootstrap CIs (default: %(default)s).",
    )
    args = parser.parse_args()

    bundle_dirs = _iter_bundle_dirs(args.root)
    if not bundle_dirs:
        raise SystemExit(f"No bundles found under: {args.root}")

    rows: list[dict[str, Any]] = []

    pooled: dict[str, dict[str, list[np.ndarray]]] = {
        "base": {"bg": [], "flow": []},
        "base_pdlog": {"bg": [], "flow": []},
        "base_kasai": {"bg": [], "flow": []},
        "stap_preka": {"bg": [], "flow": []},
        "stap": {"bg": [], "flow": []},
    }

    ka_applied_count = 0
    ka_scale_present_count = 0
    max_abs_delta_flow = 0.0
    ka_state_counts: dict[str, int] = {}
    ka_reason_counts: dict[str, int] = {}
    ka_state_reason_counts: dict[str, dict[str, int]] = {}
    ka_risk_mode_counts: dict[str, int] = {}
    ka_metric_values: dict[str, list[float | None]] = {
        # Actionability / integrity metrics
        "delta_tail": [],
        "delta_bg_flow_median": [],
        "guard_q90": [],
        "iqr_alias_bg": [],
        "pf_peak_flow": [],
        # Coverage / effect metrics
        "p_shrink": [],
        "iqr_logw_gated": [],
        # Sample support
        "n_tiles": [],
        "n_bg_proxy": [],
        "n_flow_proxy": [],
        "n_tail": [],
        "n_mid": [],
        # Risk thresholding
        "tau_alias": [],
    }
    ka_metric_bool_counts: dict[str, dict[str, int]] = {
        "uplift_eligible_raw": {"true": 0, "false": 0, "missing": 0},
        "uplift_eligible": {"true": 0, "false": 0, "missing": 0},
        "uplift_vetoed_by_guard": {"true": 0, "false": 0, "missing": 0},
        "uplift_vetoed_by_pf_peak": {"true": 0, "false": 0, "missing": 0},
    }

    for bundle_dir in bundle_dirs:
        meta = _load_meta(bundle_dir) or {}
        ka_rep = meta.get("ka_contract_v2") if isinstance(meta, dict) else None
        if isinstance(ka_rep, dict):
            state = ka_rep.get("state")
            reason = ka_rep.get("reason")
            metrics = ka_rep.get("metrics") if isinstance(ka_rep.get("metrics"), dict) else {}
            risk_mode = metrics.get("risk_mode")
            if isinstance(state, str) and state:
                ka_state_counts[state] = ka_state_counts.get(state, 0) + 1
                if isinstance(reason, str) and reason:
                    ka_state_reason_counts.setdefault(state, {})
                    ka_state_reason_counts[state][reason] = ka_state_reason_counts[state].get(reason, 0) + 1
            if isinstance(reason, str) and reason:
                ka_reason_counts[reason] = ka_reason_counts.get(reason, 0) + 1
            if isinstance(risk_mode, str) and risk_mode:
                ka_risk_mode_counts[risk_mode] = ka_risk_mode_counts.get(risk_mode, 0) + 1
            # Numeric metric collection (best-effort).
            for k in ka_metric_values.keys():
                v = metrics.get(k)
                try:
                    fv = float(v) if v is not None else None
                except Exception:
                    fv = None
                ka_metric_values[k].append(fv)
            # Boolean metric counts.
            for k in ka_metric_bool_counts.keys():
                v = metrics.get(k)
                if v is True:
                    ka_metric_bool_counts[k]["true"] += 1
                elif v is False:
                    ka_metric_bool_counts[k]["false"] += 1
                else:
                    ka_metric_bool_counts[k]["missing"] += 1
        mask_flow = _load_npy(bundle_dir, "mask_flow")
        mask_bg = _load_npy(bundle_dir, "mask_bg")
        if mask_flow is None:
            continue
        if mask_bg is None:
            mask_bg = ~mask_flow.astype(bool)
        flow = np.asarray(mask_flow, dtype=bool)
        bg = np.asarray(mask_bg, dtype=bool)

        score_base = _load_npy(bundle_dir, "score_base")
        score_base_pdlog = _load_npy(bundle_dir, "score_base_pdlog")
        score_base_kasai = _load_npy(bundle_dir, "score_base_kasai")
        score_stap_preka = _load_npy(bundle_dir, "score_stap_preka")
        score_stap = _load_npy(bundle_dir, "score_stap")
        if score_base is None or score_stap_preka is None:
            continue
        if score_stap is None:
            score_stap = score_stap_preka

        ka_scale = _load_npy(bundle_dir, "ka_scale_map")
        tele = meta.get("stap_fallback_telemetry") or {}
        if isinstance(tele, dict):
            if bool(tele.get("score_ka_v2_applied", False)):
                ka_applied_count += 1
        if ka_scale is not None:
            ka_scale_present_count += 1
            # Protected-set invariance: flow pixels should be unchanged when KA is enabled.
            # This is a best-effort check even if KA is not applied (scale_map absent).
            delta = np.abs(
                np.asarray(score_stap, dtype=np.float64) - np.asarray(score_stap_preka, dtype=np.float64)
            )
            if flow.any():
                max_abs_delta_flow = max(max_abs_delta_flow, float(np.max(delta[flow])))

        # Collect pooled scores.
        pooled["base"]["bg"].append(np.asarray(score_base, dtype=np.float64)[bg])
        pooled["base"]["flow"].append(np.asarray(score_base, dtype=np.float64)[flow])
        if score_base_pdlog is not None:
            pooled["base_pdlog"]["bg"].append(np.asarray(score_base_pdlog, dtype=np.float64)[bg])
            pooled["base_pdlog"]["flow"].append(np.asarray(score_base_pdlog, dtype=np.float64)[flow])
        if score_base_kasai is not None:
            pooled["base_kasai"]["bg"].append(np.asarray(score_base_kasai, dtype=np.float64)[bg])
            pooled["base_kasai"]["flow"].append(np.asarray(score_base_kasai, dtype=np.float64)[flow])
        pooled["stap_preka"]["bg"].append(np.asarray(score_stap_preka, dtype=np.float64)[bg])
        pooled["stap_preka"]["flow"].append(np.asarray(score_stap_preka, dtype=np.float64)[flow])
        pooled["stap"]["bg"].append(np.asarray(score_stap, dtype=np.float64)[bg])
        pooled["stap"]["flow"].append(np.asarray(score_stap, dtype=np.float64)[flow])

        # Per-bundle ROC points (threshold per bundle, per method).
        method_scores: list[tuple[str, np.ndarray]] = [
            ("base", score_base),
        ]
        if score_base_pdlog is not None:
            method_scores.append(("base_pdlog", score_base_pdlog))
        if score_base_kasai is not None:
            method_scores.append(("base_kasai", score_base_kasai))
        method_scores.extend([("stap_preka", score_stap_preka), ("stap", score_stap)])

        for method, scores in method_scores:
            bg_scores = np.asarray(scores, dtype=np.float64)[bg]
            flow_scores = np.asarray(scores, dtype=np.float64)[flow]
            n_bg = int(np.isfinite(bg_scores).sum())
            n_flow = int(np.isfinite(flow_scores).sum())
            for alpha in args.fprs:
                tau, fpr_real = _right_tail_threshold(bg_scores, float(alpha))
                if tau is None or fpr_real is None:
                    continue
                tpr = float(np.mean(flow_scores >= float(tau))) if n_flow > 0 else None
                rows.append(
                    {
                        "bundle": str(bundle_dir),
                        "method": method,
                        "fpr_target": float(alpha),
                        "thr": float(tau),
                        "fpr_realized": float(fpr_real),
                        "tpr": tpr,
                        "n_bg": n_bg,
                        "n_flow": n_flow,
                        "fpr_min": (1.0 / float(n_bg)) if n_bg > 0 else None,
                        "ka_state": (meta.get("ka_contract_v2") or {}).get("state"),
                        "ka_reason": (meta.get("ka_contract_v2") or {}).get("reason"),
                        "ka_applied": bool(tele.get("score_ka_v2_applied", False))
                        if isinstance(tele, dict)
                        else None,
                        "ka_scaled_pixel_fraction": tele.get("score_ka_v2_scaled_pixel_fraction")
                        if isinstance(tele, dict)
                        else None,
                        "stap_fast_path_used": tele.get("stap_fast_path_used")
                        if isinstance(tele, dict)
                        else None,
                        "stap_total_ms": tele.get("stap_total_ms") if isinstance(tele, dict) else None,
                    }
                )

    # Write per-bundle ROC table.
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    # Pooled (all bundles concatenated) ROC points.
    pooled_summary: dict[str, Any] = {"methods": {}, "fprs": list(map(float, args.fprs))}
    for method in ("base", "base_pdlog", "base_kasai", "stap_preka", "stap"):
        if not pooled.get(method) or not pooled[method]["bg"]:
            continue
        bg_all = _finite_1d(np.concatenate(pooled[method]["bg"], axis=0)) if pooled[method]["bg"] else np.array([])
        flow_all = _finite_1d(np.concatenate(pooled[method]["flow"], axis=0)) if pooled[method]["flow"] else np.array([])
        n_bg = int(bg_all.size)
        n_flow = int(flow_all.size)
        pooled_summary["methods"][method] = {
            "n_bg": n_bg,
            "n_flow": n_flow,
            "fpr_min": (1.0 / float(n_bg)) if n_bg > 0 else None,
            "bg_mean": _safe_mean(bg_all),
            "flow_mean": _safe_mean(flow_all),
            "roc": [],
        }
        for alpha in args.fprs:
            tau, fpr_real = _right_tail_threshold(bg_all, float(alpha))
            if tau is None or fpr_real is None:
                continue
            tpr = float(np.mean(flow_all >= float(tau))) if n_flow > 0 else None
            pooled_summary["methods"][method]["roc"].append(
                {
                    "fpr_target": float(alpha),
                    "thr": float(tau),
                    "fpr_realized": float(fpr_real),
                    "tpr": tpr,
                }
            )

    # Frame-bootstrap CIs for pooled ROC points at the fixed pooled thresholds.
    # This avoids the "millions of pixels => tiny CI" fallacy by treating cine
    # frames (bundles) as the resampling unit.
    if bool(args.frame_bootstrap):
        rng = np.random.default_rng(int(args.bootstrap_seed))
        n_boot = int(max(50, args.bootstrap_n))
        for method, md in pooled_summary.get("methods", {}).items():
            if not md.get("roc"):
                continue
            bg_per_bundle = pooled.get(method, {}).get("bg") or []
            flow_per_bundle = pooled.get(method, {}).get("flow") or []
            if not bg_per_bundle or not flow_per_bundle:
                continue
            # Precompute per-frame counts/hits at each pooled threshold.
            n_frames = int(len(bg_per_bundle))
            for roc_pt in md["roc"]:
                tau = float(roc_pt["thr"])
                n_bg = np.array([int(np.isfinite(b).sum()) for b in bg_per_bundle], dtype=np.int64)
                n_flow = np.array([int(np.isfinite(f).sum()) for f in flow_per_bundle], dtype=np.int64)
                hit_bg = np.array([int(np.sum(np.asarray(b, dtype=np.float64) >= tau)) for b in bg_per_bundle], dtype=np.int64)
                hit_flow = np.array([int(np.sum(np.asarray(f, dtype=np.float64) >= tau)) for f in flow_per_bundle], dtype=np.int64)

                idx = rng.integers(0, n_frames, size=(n_boot, n_frames))
                sum_n_bg = np.sum(n_bg[idx], axis=1).astype(np.float64)
                sum_n_flow = np.sum(n_flow[idx], axis=1).astype(np.float64)
                sum_hit_bg = np.sum(hit_bg[idx], axis=1).astype(np.float64)
                sum_hit_flow = np.sum(hit_flow[idx], axis=1).astype(np.float64)

                fpr_boot = np.divide(sum_hit_bg, sum_n_bg, out=np.full_like(sum_n_bg, np.nan), where=sum_n_bg > 0)
                tpr_boot = np.divide(sum_hit_flow, sum_n_flow, out=np.full_like(sum_n_flow, np.nan), where=sum_n_flow > 0)
                fpr_boot = fpr_boot[np.isfinite(fpr_boot)]
                tpr_boot = tpr_boot[np.isfinite(tpr_boot)]
                if tpr_boot.size:
                    roc_pt["tpr_ci95"] = [float(np.quantile(tpr_boot, 0.025)), float(np.quantile(tpr_boot, 0.975))]
                else:
                    roc_pt["tpr_ci95"] = [None, None]
                if fpr_boot.size:
                    roc_pt["fpr_ci95"] = [float(np.quantile(fpr_boot, 0.025)), float(np.quantile(fpr_boot, 0.975))]
                else:
                    roc_pt["fpr_ci95"] = [None, None]
            md["frame_bootstrap"] = {"n": n_boot, "seed": int(args.bootstrap_seed), "unit": "bundle/frame"}

    # Tail-hygiene: compare score_stap_preka vs score_stap at fixed thresholds chosen on bg(preka).
    hygiene: list[dict[str, Any]] = []
    stap_bg_all = _finite_1d(np.concatenate(pooled["stap_preka"]["bg"], axis=0)) if pooled["stap_preka"]["bg"] else np.array([])
    if stap_bg_all.size:
        for alpha in args.fprs:
            tau_pre, fpr_pre = _right_tail_threshold(stap_bg_all, float(alpha))
            if tau_pre is None:
                continue
            area_pre = 0
            area_post = 0
            clust_pre = 0
            clust_post = 0
            flow_hits_pre = 0
            flow_hits_post = 0
            n_bg_total = 0
            n_flow_total = 0

            for bundle_dir in bundle_dirs:
                mask_flow = _load_npy(bundle_dir, "mask_flow")
                if mask_flow is None:
                    continue
                mask_bg = _load_npy(bundle_dir, "mask_bg")
                if mask_bg is None:
                    mask_bg = ~mask_flow.astype(bool)
                flow = np.asarray(mask_flow, dtype=bool)
                bg = np.asarray(mask_bg, dtype=bool)
                s_pre = _load_npy(bundle_dir, "score_stap_preka")
                s_post = _load_npy(bundle_dir, "score_stap")
                if s_pre is None:
                    continue
                if s_post is None:
                    s_post = s_pre
                s_pre = np.asarray(s_pre, dtype=np.float64)
                s_post = np.asarray(s_post, dtype=np.float64)
                hit_pre = bg & (s_pre >= float(tau_pre))
                hit_post = bg & (s_post >= float(tau_pre))
                area_pre += int(np.sum(hit_pre))
                area_post += int(np.sum(hit_post))
                clust_pre += _connected_components(hit_pre, connectivity=int(args.connectivity))
                clust_post += _connected_components(hit_post, connectivity=int(args.connectivity))
                flow_hits_pre += int(np.sum(flow & (s_pre >= float(tau_pre))))
                flow_hits_post += int(np.sum(flow & (s_post >= float(tau_pre))))
                n_bg_total += int(np.sum(bg))
                n_flow_total += int(np.sum(flow))

            hygiene.append(
                {
                    "fpr_target": float(alpha),
                    "thr_preka": float(tau_pre),
                    "bg_tail_rate_preka": (area_pre / float(n_bg_total)) if n_bg_total > 0 else None,
                    "bg_tail_rate_post": (area_post / float(n_bg_total)) if n_bg_total > 0 else None,
                    "bg_tail_area_preka": int(area_pre),
                    "bg_tail_area_post": int(area_post),
                    "bg_tail_clusters_preka": int(clust_pre),
                    "bg_tail_clusters_post": int(clust_post),
                    "flow_hit_rate_preka": (flow_hits_pre / float(n_flow_total)) if n_flow_total > 0 else None,
                    "flow_hit_rate_post": (flow_hits_post / float(n_flow_total)) if n_flow_total > 0 else None,
                    "n_bg_total": int(n_bg_total),
                    "n_flow_total": int(n_flow_total),
                    "ka_scale_present_bundles": int(ka_scale_present_count),
                    "ka_applied_bundles": int(ka_applied_count),
                    "max_abs_delta_flow": float(max_abs_delta_flow),
                }
            )

    summary = {
        "root": str(args.root),
        "bundle_count": int(len(bundle_dirs)),
        "ka_applied_bundles": int(ka_applied_count),
        "ka_scale_present_bundles": int(ka_scale_present_count),
        "max_abs_delta_flow": float(max_abs_delta_flow),
        "ka_state_counts": ka_state_counts,
        "ka_reason_counts": ka_reason_counts,
        "ka_state_reason_counts": ka_state_reason_counts,
        "ka_risk_mode_counts": ka_risk_mode_counts,
        "ka_metric_summary": {k: _summary_stats(v) for k, v in ka_metric_values.items()},
        "ka_metric_bool_counts": ka_metric_bool_counts,
        "pooled_roc": pooled_summary,
        "hygiene": hygiene,
    }
    args.out_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
