#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "clinical_translation"

SHIN_ROOT = ROOT / "runs" / "shin_ratbrain_baseline_matrix_shinU_e970_Lt64_nomaskunion_k80"
ULM_ROOTS = {
    "ulm_brainlike": ROOT / "runs" / "real" / "ulm7883227_motion_sweep_ULM_brainlike_e975",
    "ulm_elastic": ROOT / "runs" / "real" / "ulm7883227_motion_sweep_ULM_elastic_e975",
}

FPR = 1e-3


def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return x[np.isfinite(x)]


def _bootstrap_ci(
    xs: list[float],
    *,
    stat: str = "mean",
    n_boot: int = 4000,
    seed: int = 0,
) -> dict[str, float | int | None]:
    arr = _finite(np.asarray(xs, dtype=np.float64))
    if arr.size == 0:
        return {"center": None, "lo": None, "hi": None, "n": 0}
    if stat == "median":
        fn = np.median
        center = float(np.median(arr))
    else:
        fn = np.mean
        center = float(np.mean(arr))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boots = fn(arr[idx], axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"center": center, "lo": float(lo), "hi": float(hi), "n": int(arr.size)}


def _win_rate(xs: list[float], *, better: str = "positive") -> dict[str, float | int]:
    arr = _finite(np.asarray(xs, dtype=np.float64))
    if arr.size == 0:
        return {"wins": 0, "ties": 0, "losses": 0, "win_rate": float("nan"), "n": 0}
    if better == "negative":
        wins = int(np.sum(arr < 0))
        losses = int(np.sum(arr > 0))
    else:
        wins = int(np.sum(arr > 0))
        losses = int(np.sum(arr < 0))
    ties = int(np.sum(arr == 0))
    return {
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": float(wins / arr.size),
        "n": int(arr.size),
    }


def _connected_components(binary: np.ndarray) -> int:
    binary = np.asarray(binary, dtype=bool)
    if binary.size == 0:
        return 0
    try:
        import scipy.ndimage as ndi  # type: ignore

        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
        _, n = ndi.label(binary, structure=structure)
        return int(n)
    except Exception:
        h, w = binary.shape
        visited = np.zeros_like(binary, dtype=bool)
        n = 0
        neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for y in range(h):
            for x in range(w):
                if not binary[y, x] or visited[y, x]:
                    continue
                n += 1
                stack = [(y, x)]
                visited[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    for dy, dx in neigh:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
        return int(n)


def _auc_pos_vs_neg(pos: np.ndarray, neg: np.ndarray) -> float | None:
    pos = _finite(pos)
    neg = _finite(neg)
    if pos.size == 0 or neg.size == 0:
        return None
    neg_sorted = np.sort(neg)
    less = np.searchsorted(neg_sorted, pos, side="left")
    right = np.searchsorted(neg_sorted, pos, side="right")
    equal = right - less
    auc = (float(np.sum(less)) + 0.5 * float(np.sum(equal))) / float(pos.size * neg.size)
    return float(auc)


def _threshold_at_fpr(score: np.ndarray, neg_mask: np.ndarray, fpr: float) -> float | None:
    vals = _finite(score[np.asarray(neg_mask, dtype=bool)])
    if vals.size == 0:
        return None
    return float(np.quantile(vals, 1.0 - float(fpr)))


def _bundle_family(name: str) -> str:
    if "adaptive_local_svd" in name:
        return "adaptive_local_svd"
    if "local_svd" in name:
        return "local_svd"
    if "svd_similarity" in name:
        return "adaptive_global_svd"
    if "mcsvd" in name:
        return "mc_svd"
    if "rpca" in name:
        return "rpca"
    if "hosvd" in name:
        return "hosvd"
    if "raw" in name:
        return "raw"
    return "unknown"


def _score_metrics(score: np.ndarray, mask_flow: np.ndarray, mask_bg: np.ndarray) -> dict[str, float | int | None]:
    flow = np.asarray(mask_flow, dtype=bool)
    bg = np.asarray(mask_bg, dtype=bool)
    pos = score[flow]
    neg = score[bg]
    auc = _auc_pos_vs_neg(pos, neg)
    thr = _threshold_at_fpr(score, bg, FPR)
    if thr is None:
        return {
            "auc_flow_bg": auc,
            "tpr_at_fpr1e3": None,
            "bg_clusters_at_fpr1e3": None,
            "bg_tail_mean_at_fpr1e3": None,
        }
    hit_flow = flow & (score >= thr)
    hit_bg = bg & (score >= thr)
    bg_vals = _finite(score[hit_bg])
    return {
        "auc_flow_bg": auc,
        "tpr_at_fpr1e3": float(np.mean(hit_flow[flow])) if np.any(flow) else None,
        "bg_clusters_at_fpr1e3": int(_connected_components(hit_bg)),
        "bg_excess_ratio_at_fpr1e3": (
            float(np.mean((bg_vals - thr) / (abs(thr) + 1e-12))) if bg_vals.size else 0.0
        ),
    }


def _load_bundle_scores(bundle_dir: Path) -> dict[str, np.ndarray] | None:
    required = [
        "score_base.npy",
        "score_base_kasai.npy",
        "score_stap_preka.npy",
        "mask_flow.npy",
        "mask_bg.npy",
    ]
    if not all((bundle_dir / name).exists() for name in required):
        return None
    data = {name: np.load(bundle_dir / name, allow_pickle=False) for name in required}
    if float(np.nanmax(np.abs(data["score_stap_preka.npy"]))) <= 1e-8:
        return None
    return data


def _collect_shin_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bundle_dir in sorted(SHIN_ROOT.iterdir()):
        if not bundle_dir.is_dir():
            continue
        data = _load_bundle_scores(bundle_dir)
        if data is None:
            continue
        family = _bundle_family(bundle_dir.name)
        for head, score_name in [
            ("pd", "score_base.npy"),
            ("kasai", "score_base_kasai.npy"),
            ("stap", "score_stap_preka.npy"),
        ]:
            metrics = _score_metrics(
                np.asarray(data[score_name], dtype=np.float64),
                data["mask_flow.npy"],
                data["mask_bg.npy"],
            )
            rows.append(
                {
                    "dataset": "shin",
                    "group": f"shin_{family}",
                    "family": family,
                    "bundle": bundle_dir.name,
                    "head": head,
                    **metrics,
                }
            )
    return rows


def _collect_ulm_rows(tag: str, root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bundle_dir in sorted(root.iterdir()):
        if not bundle_dir.is_dir():
            continue
        data = _load_bundle_scores(bundle_dir)
        if data is None:
            continue
        for head, score_name in [
            ("pd", "score_base.npy"),
            ("kasai", "score_base_kasai.npy"),
            ("stap", "score_stap_preka.npy"),
        ]:
            metrics = _score_metrics(
                np.asarray(data[score_name], dtype=np.float64),
                data["mask_flow.npy"],
                data["mask_bg.npy"],
            )
            rows.append(
                {
                    "dataset": "ulm",
                    "group": tag,
                    "family": "mc_svd",
                    "bundle": bundle_dir.name,
                    "head": head,
                    **metrics,
                }
            )
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_group_bundle: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["group"]), str(row["bundle"]))
        by_group_bundle.setdefault(key, {})[str(row["head"])] = row

    summaries: dict[str, Any] = {}
    for group in sorted({str(r["group"]) for r in rows}):
        pairs = [heads for (g, _), heads in by_group_bundle.items() if g == group]
        deltas: dict[str, dict[str, list[float]]] = {}
        for heads in pairs:
            if "stap" not in heads:
                continue
            for baseline in ["pd", "kasai"]:
                if baseline not in heads:
                    continue
                k = f"stap_minus_{baseline}"
                deltas.setdefault(k, {})
                for metric, better in [
                    ("auc_flow_bg", "positive"),
                    ("tpr_at_fpr1e3", "positive"),
                    ("bg_clusters_at_fpr1e3", "negative"),
                    ("bg_excess_ratio_at_fpr1e3", "negative"),
                ]:
                    val = heads["stap"].get(metric)
                    ref = heads[baseline].get(metric)
                    if val is None or ref is None:
                        continue
                    deltas[k].setdefault(metric, []).append(float(val) - float(ref))
        group_sec: dict[str, Any] = {
            "n_bundles": sum(1 for (g, _), heads in by_group_bundle.items() if g == group and "stap" in heads),
            "heads_present": sorted({str(r["head"]) for r in rows if str(r["group"]) == group}),
        }
        for cmp_name, metric_map in deltas.items():
            cmp_sec: dict[str, Any] = {}
            for metric, vals in metric_map.items():
                better = "negative" if metric in {"bg_clusters_at_fpr1e3", "bg_tail_mean_at_fpr1e3"} else "positive"
                cmp_sec[metric] = {
                    "delta": _bootstrap_ci(vals),
                    "win_rate": _win_rate(vals, better=better),
                }
            group_sec[cmp_name] = cmp_sec
        summaries[group] = group_sec
    return summaries


def _rows_from_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for group, sec in summary.items():
        for cmp_name in ["stap_minus_pd", "stap_minus_kasai"]:
            if cmp_name not in sec:
                continue
            for metric, payload in sec[cmp_name].items():
                delta = payload["delta"]
                win = payload["win_rate"]
                flat.append(
                    {
                        "group": group,
                        "comparison": cmp_name,
                        "metric": metric,
                        "center": delta["center"],
                        "lo": delta["lo"],
                        "hi": delta["hi"],
                        "n": delta["n"],
                        "wins": win["wins"],
                        "ties": win["ties"],
                        "losses": win["losses"],
                        "win_rate": win["win_rate"],
                    }
                )
    return flat


def _md(summary: dict[str, Any]) -> str:
    out = ["# Real-Data Same-Residual Detector-Head Audit", ""]
    out.append("Coverage is limited to bundles where `score_stap_preka.npy` is non-zero and therefore STAP was actually active on the residual cube.")
    out.append("")
    for group, sec in summary.items():
        out.append(f"## {group}")
        out.append(f"- Bundles with active STAP: {sec['n_bundles']}")
        for cmp_name, label in [("stap_minus_pd", "STAP vs PD"), ("stap_minus_kasai", "STAP vs Kasai")]:
            cmp = sec.get(cmp_name)
            if not cmp:
                continue
            out.append(f"- {label}:")
            for metric in ["auc_flow_bg", "tpr_at_fpr1e3", "bg_clusters_at_fpr1e3", "bg_excess_ratio_at_fpr1e3"]:
                pretty = metric
                if metric == "bg_excess_ratio_at_fpr1e3":
                    pretty = "bg_excess_ratio_at_fpr1e3"
                if metric not in cmp:
                    continue
                delta = cmp[metric]["delta"]
                win = cmp[metric]["win_rate"]
                out.append(
                    f"  - {pretty}: Δ={delta['center']:.4f} "
                    f"(95% CI {delta['lo']:.4f} to {delta['hi']:.4f}; "
                    f"win rate {100*win['win_rate']:.1f}% = {win['wins']}/{win['n']})"
                )
        out.append("")
    return "\n".join(out)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _collect_shin_rows()
    for tag, root in ULM_ROOTS.items():
        rows.extend(_collect_ulm_rows(tag, root))

    summary = _aggregate(rows)
    flat = _rows_from_summary(summary)

    with (OUT_DIR / "realdata_detector_head_audit.csv").open("w", newline="") as f:
        cols = ["group", "comparison", "metric", "center", "lo", "hi", "n", "wins", "ties", "losses", "win_rate"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in flat:
            w.writerow(row)

    (OUT_DIR / "realdata_detector_head_audit.json").write_text(
        json.dumps({"summary": summary}, indent=2, sort_keys=True, allow_nan=True)
    )
    (OUT_DIR / "realdata_detector_head_audit.md").write_text(_md(summary))

    print("wrote", OUT_DIR / "realdata_detector_head_audit.csv")
    print("wrote", OUT_DIR / "realdata_detector_head_audit.json")
    print("wrote", OUT_DIR / "realdata_detector_head_audit.md")


if __name__ == "__main__":
    main()
