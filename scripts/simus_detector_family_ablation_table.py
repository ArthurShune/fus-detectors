#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ADAPTIVE_GUARD_RULE = {
    "feature": "base_guard_frac_map",
    "direction": ">=",
    "threshold": 0.1453727245330811,
    "aggregation": "tile_mean",
    "prefer_advanced_on_invalid": False,
}


def _setting_label(run_dir: Path) -> str:
    name = run_dir.name.lower()
    if "clin_mobile" in name:
        return "Mobile"
    if "clin_intraop_parenchyma" in name:
        return "Intra-operative parenchymal"
    return run_dir.name


def _dataset_name(run_dir: Path, key: str) -> str:
    return f"{run_dir.name}_{key}"


def _seed_from_run_dir(run_dir: Path) -> int | None:
    match = re.search(r"seed(\d+)", run_dir.name)
    return int(match.group(1)) if match else None


def _tile_iter(shape: tuple[int, int], tile_hw: tuple[int, int], stride: int):
    height, width = shape
    tile_h, tile_w = tile_hw
    for y0 in range(0, height - tile_h + 1, stride):
        for x0 in range(0, width - tile_w + 1, stride):
            yield y0, x0


def _tile_scores_to_map(
    tile_scores: np.ndarray,
    shape: tuple[int, int],
    tile_hw: tuple[int, int],
    stride: int,
) -> np.ndarray:
    """Scatter per-tile values back to image space with overlap averaging."""
    height, width = shape
    tile_h, tile_w = tile_hw
    accum = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    idx = 0
    for y0, x0 in _tile_iter(shape, tile_hw, stride):
        if idx >= tile_scores.size:
            raise RuntimeError("Insufficient tile scores for overlap-add reconstruction.")
        value = float(tile_scores[idx])
        accum[y0 : y0 + tile_h, x0 : x0 + tile_w] += value
        counts[y0 : y0 + tile_h, x0 : x0 + tile_w] += 1.0
        idx += 1
    if idx != tile_scores.size:
        raise RuntimeError("Excess tile scores provided for overlap-add reconstruction.")
    out = np.zeros_like(accum, dtype=np.float32)
    covered = counts > 0.0
    out[covered] = accum[covered] / counts[covered]
    return out


def _hybrid_choose_advanced_tile_mask(
    feature_map: np.ndarray,
    *,
    tile_hw: tuple[int, int],
    stride: int,
    direction: str,
    threshold: float,
    reduction: str,
    prefer_advanced_on_invalid: bool,
) -> tuple[np.ndarray, np.ndarray]:
    feat = np.asarray(feature_map, dtype=np.float32)
    reduction_norm = {"tile_mean": "mean", "tile_max": "max"}.get(str(reduction).strip().lower(), str(reduction).strip().lower())
    if reduction_norm not in {"mean", "max"}:
        raise ValueError(f"Unsupported tile reduction {reduction!r}")
    tile_promote: list[bool] = []
    for y0, x0 in _tile_iter(feat.shape, tile_hw, stride):
        tile = feat[y0 : y0 + tile_hw[0], x0 : x0 + tile_hw[1]]
        finite = np.isfinite(tile)
        if not np.any(finite):
            promote = bool(prefer_advanced_on_invalid)
        else:
            vals = np.asarray(tile[finite], dtype=np.float32)
            stat = float(np.max(vals)) if reduction_norm == "max" else float(np.mean(vals))
            if str(direction).strip() == ">=":
                promote = stat >= float(threshold)
            elif str(direction).strip() == "<=":
                promote = stat <= float(threshold)
            else:
                raise ValueError(f"Unsupported direction {direction!r}")
        tile_promote.append(bool(promote))
    tile_promote_arr = np.asarray(tile_promote, dtype=np.bool_)
    choose_advanced = _tile_scores_to_map(
        tile_promote_arr.astype(np.float32),
        feat.shape,
        tile_hw,
        stride,
    ) > 0.0
    return choose_advanced.astype(np.bool_, copy=False), tile_promote_arr


def _finite(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
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


def _threshold_from_neg(neg: np.ndarray, fpr: float) -> tuple[float | None, float | None]:
    neg = _finite(neg)
    if neg.size == 0:
        return None, None
    q = float(np.clip(1.0 - float(fpr), 0.0, 1.0))
    thr = float(np.quantile(neg, q))
    fpr_emp = float(np.mean(neg >= thr))
    return thr, fpr_emp


def _threshold_from_pos(pos: np.ndarray, tpr: float) -> tuple[float | None, float | None]:
    pos = _finite(pos)
    if pos.size == 0:
        return None, None
    q = float(np.clip(1.0 - float(tpr), 0.0, 1.0))
    thr = float(np.quantile(pos, q))
    tpr_emp = float(np.mean(pos >= thr))
    return thr, tpr_emp


def _rate_tag(rate: float) -> str:
    return f"{float(rate):.3f}".rstrip("0").rstrip(".").replace(".", "p")


def evaluate_structural_metrics(
    *,
    score: np.ndarray,
    mask_h1_pf_main: np.ndarray,
    mask_h0_bg: np.ndarray,
    mask_h0_nuisance_pa: np.ndarray | None,
    mask_h1_alias_qc: np.ndarray | None,
    fprs: list[float],
    match_tprs: list[float] | None = None,
) -> dict[str, Any]:
    score = np.asarray(score, dtype=np.float64)
    pos_main = score[np.asarray(mask_h1_pf_main, dtype=bool)]
    neg_bg = score[np.asarray(mask_h0_bg, dtype=bool)]
    neg_nuisance = (
        score[np.asarray(mask_h0_nuisance_pa, dtype=bool)] if mask_h0_nuisance_pa is not None else np.asarray([], dtype=np.float64)
    )
    pos_alias = (
        score[np.asarray(mask_h1_alias_qc, dtype=bool)] if mask_h1_alias_qc is not None else np.asarray([], dtype=np.float64)
    )
    out: dict[str, Any] = {
        "n_h1_pf_main": int(np.asarray(mask_h1_pf_main, dtype=bool).sum()),
        "n_h0_bg": int(np.asarray(mask_h0_bg, dtype=bool).sum()),
        "n_h0_nuisance_pa": int(np.asarray(mask_h0_nuisance_pa, dtype=bool).sum()) if mask_h0_nuisance_pa is not None else 0,
        "n_h1_alias_qc": int(np.asarray(mask_h1_alias_qc, dtype=bool).sum()) if mask_h1_alias_qc is not None else 0,
        "auc_main_vs_bg": _auc_pos_vs_neg(pos_main, neg_bg),
        "auc_main_vs_nuisance": _auc_pos_vs_neg(pos_main, neg_nuisance) if neg_nuisance.size else None,
        "fpr_floor_bg": (1.0 / float(max(1, neg_bg.size))) if neg_bg.size else None,
        "fpr_floor_nuisance": (1.0 / float(max(1, neg_nuisance.size))) if neg_nuisance.size else None,
    }
    for fpr in fprs:
        tag = f"{float(fpr):.0e}"
        thr, bg_emp = _threshold_from_neg(neg_bg, fpr)
        if thr is None:
            out[f"thr@{tag}"] = None
            out[f"tpr_main@{tag}"] = None
            out[f"fpr_bg@{tag}"] = None
            out[f"fpr_nuisance@{tag}"] = None
            out[f"tpr_alias_qc@{tag}"] = None
            continue
        out[f"thr@{tag}"] = thr
        out[f"tpr_main@{tag}"] = float(np.mean(pos_main >= thr)) if pos_main.size else None
        out[f"fpr_bg@{tag}"] = bg_emp
        out[f"fpr_nuisance@{tag}"] = float(np.mean(neg_nuisance >= thr)) if neg_nuisance.size else None
        out[f"tpr_alias_qc@{tag}"] = float(np.mean(pos_alias >= thr)) if pos_alias.size else None
    for tpr in match_tprs or []:
        tag = _rate_tag(tpr)
        thr, tpr_emp = _threshold_from_pos(pos_main, tpr)
        if thr is None:
            out[f"thr_match_tpr@{tag}"] = None
            out[f"tpr_main_match@{tag}"] = None
            out[f"fpr_bg_match@{tag}"] = None
            out[f"fpr_nuisance_match@{tag}"] = None
            out[f"tpr_alias_qc_match@{tag}"] = None
            continue
        out[f"thr_match_tpr@{tag}"] = thr
        out[f"tpr_main_match@{tag}"] = tpr_emp
        out[f"fpr_bg_match@{tag}"] = float(np.mean(neg_bg >= thr)) if neg_bg.size else None
        out[f"fpr_nuisance_match@{tag}"] = float(np.mean(neg_nuisance >= thr)) if neg_nuisance.size else None
        out[f"tpr_alias_qc_match@{tag}"] = float(np.mean(pos_alias >= thr)) if pos_alias.size else None
    return out


def _bundle_dir(out_root: Path, run_dir: Path, key: str) -> Path:
    return Path(out_root) / _dataset_name(run_dir, key)


def _load_array(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path).astype(np.float32, copy=False)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _adaptive_score_from_cached_bundles(
    *,
    advanced_bundle_dir: Path,
    rescue_bundle_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    advanced_score = _load_array(advanced_bundle_dir / "score_stap_preka.npy")
    rescue_score = _load_array(rescue_bundle_dir / "score_stap_preka.npy")
    feature_map = _load_array(advanced_bundle_dir / f"{ADAPTIVE_GUARD_RULE['feature']}.npy")
    meta = _load_json(advanced_bundle_dir / "meta.json")
    tile_hw = tuple(int(x) for x in meta["tile_hw"])
    stride = int(meta["tile_stride"])
    choose_advanced, promote_tiles = _hybrid_choose_advanced_tile_mask(
        feature_map,
        tile_hw=tile_hw,
        stride=stride,
        direction=str(ADAPTIVE_GUARD_RULE["direction"]),
        threshold=float(ADAPTIVE_GUARD_RULE["threshold"]),
        reduction=str(ADAPTIVE_GUARD_RULE["aggregation"]),
        prefer_advanced_on_invalid=bool(ADAPTIVE_GUARD_RULE["prefer_advanced_on_invalid"]),
    )
    adaptive_score = np.where(choose_advanced, advanced_score, rescue_score).astype(np.float32, copy=False)
    stats = {
        "tile_hw": list(tile_hw),
        "tile_stride": stride,
        "advanced_fraction": float(np.mean(choose_advanced)) if choose_advanced.size else None,
        "advanced_pixels": int(np.count_nonzero(choose_advanced)),
        "advanced_tile_fraction": float(np.mean(promote_tiles)) if promote_tiles.size else None,
        "advanced_tiles": int(np.count_nonzero(promote_tiles)),
        "rule": dict(ADAPTIVE_GUARD_RULE),
    }
    return adaptive_score, stats


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no rows to write")
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


def _format3(x: float | None) -> str:
    if x is None or not np.isfinite(float(x)):
        return "--"
    return f"{float(x):.3f}"


def _format_mean_minmax(mean: float | None, lo: float | None, hi: float | None) -> str:
    if mean is None or lo is None or hi is None:
        return "--"
    vals = [mean, lo, hi]
    if not all(np.isfinite(float(v)) for v in vals):
        return "--"
    return f"{float(mean):.3f} [{float(lo):.3f},{float(hi):.3f}]"


def _build_table(summary_rows: list[dict[str, Any]]) -> str:
    order = [
        "Baseline (power Doppler)",
        "Baseline (Kasai lag-1 magnitude)",
        "Fixed matched-subspace detector",
        "Adaptive detector",
        "Fully whitened detector",
    ]
    settings = ["Mobile", "Intra-operative parenchymal"]
    keyed = {(r["method_label"], r["setting"]): r for r in summary_rows}

    lines: list[str] = []
    lines.append("% AUTO-GENERATED by scripts/simus_detector_family_ablation_table.py; DO NOT EDIT BY HAND.")
    lines.append("\\begin{center}")
    lines.append("\\captionsetup{type=table}")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append("\\begin{tabular}{@{}lccc ccc@{}}")
    lines.append("\\hline")
    lines.append(
        "Method & \\multicolumn{3}{c}{Mobile} & \\multicolumn{3}{c}{Intra-operative parenchymal} \\\\"
    )
    lines.append("\\cline{2-4} \\cline{5-7}")
    lines.append(
        " & AUC$_{\\mathrm{main/bg}}$ & AUC$_{\\mathrm{main/nuis}}$ & "
        "FPR$_{\\mathrm{nuis}}$ @ TPR$_{\\mathrm{main}}=0.5$ & "
        "AUC$_{\\mathrm{main/bg}}$ & AUC$_{\\mathrm{main/nuis}}$ & "
        "FPR$_{\\mathrm{nuis}}$ @ TPR$_{\\mathrm{main}}=0.5$ \\\\"
    )
    lines.append("\\hline")
    for method in order:
        row = [method]
        for setting in settings:
            item = keyed.get((method, setting), {})
            row.extend(
                [
                    _format_mean_minmax(
                        item.get("auc_main_vs_bg_mean"),
                        item.get("auc_main_vs_bg_min"),
                        item.get("auc_main_vs_bg_max"),
                    ),
                    _format_mean_minmax(
                        item.get("auc_main_vs_nuisance_mean"),
                        item.get("auc_main_vs_nuisance_min"),
                        item.get("auc_main_vs_nuisance_max"),
                    ),
                    _format_mean_minmax(
                        item.get("fpr_nuisance_match_0p5_mean"),
                        item.get("fpr_nuisance_match_0p5_min"),
                        item.get("fpr_nuisance_match_0p5_max"),
                    ),
                ]
            )
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(
        "\\caption{Same-residual detector-family ablation on SIMUS-Struct using a common MC--SVD residual. "
        "Rows differ only in the downstream score head: power Doppler, Kasai lag-1 magnitude, the fixed "
        "flow-band matched-subspace detector without whitening ($R=I$), the adaptive detector that promotes "
        "clutter-heavy tiles onto a whitened branch, and the fully whitened detector. Values are means with "
        "[min,max] over held-out evaluation seeds 127 and 128 for each setting. Lower nuisance FPR at matched "
        "TPR is better.}"
    )
    lines.append("\\label{tab:simus_detector_family_ablation}")
    lines.append("\\end{center}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Detector-family ablation on held-out SIMUS structural runs.")
    ap.add_argument(
        "--run",
        type=Path,
        action="append",
        default=None,
        help="Explicit SIMUS run directory (repeatable). Defaults to held-out mobile/intra-op seeds 127 and 128.",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("runs/sim_eval/simus_detector_family_ablation"),
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/simus_v2/simus_detector_family_ablation.csv"),
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/simus_v2/simus_detector_family_ablation.json"),
    )
    ap.add_argument(
        "--out-tex",
        type=Path,
        default=Path("reports/paper/simus_detector_family_ablation_table.tex"),
    )
    ap.add_argument("--stap-profile", type=str, default="Brain-SIMUS-Clin-MotionRobust-v0")
    ap.add_argument("--stap-device", type=str, default="cuda")
    ap.add_argument(
        "--reuse-bundles",
        action="store_true",
        help="Retained for CLI compatibility. This script now reads cached bundles directly.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    runs = args.run or [
        Path("runs/sim/simus_v2_acceptance_clin_mobile_pf_v2_phase3_seed127"),
        Path("runs/sim/simus_v2_acceptance_clin_mobile_pf_v2_phase3_seed128"),
        Path("runs/sim/simus_v2_acceptance_clin_intraop_parenchyma_pf_v3_phase3_seed127"),
        Path("runs/sim/simus_v2_acceptance_clin_intraop_parenchyma_pf_v3_phase3_seed128"),
    ]

    methods = [
        {
            "key": "pd",
            "method_label": "Baseline (power Doppler)",
            "bundle_key": "pd",
            "score_name": "score_pd_base.npy",
        },
        {
            "key": "kasai",
            "method_label": "Baseline (Kasai lag-1 magnitude)",
            "bundle_key": "pd",
            "score_name": "score_base_kasai.npy",
        },
        {
            "key": "fixed",
            "method_label": "Fixed matched-subspace detector",
            "bundle_key": "unwhitened_ratio",
            "score_name": "score_stap_preka.npy",
        },
        {
            "key": "adaptive",
            "method_label": "Adaptive detector",
            "score_name": "score_stap_preka.npy",
        },
        {
            "key": "fully_whitened",
            "method_label": "Fully whitened detector",
            "bundle_key": "stap",
            "score_name": "score_stap_preka.npy",
        },
    ]

    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run_dir in runs:
        run_dir = Path(run_dir)
        ds = run_dir / "dataset"
        mask_h1_pf_main = np.load(ds / "mask_h1_pf_main.npy")
        mask_h0_bg = np.load(ds / "mask_h0_bg.npy")
        mask_h0_nuisance_pa = np.load(ds / "mask_h0_nuisance_pa.npy")
        mask_h1_alias_qc = np.load(ds / "mask_h1_alias_qc.npy")
        setting = _setting_label(run_dir)

        for method in methods:
            extra_row: dict[str, Any] = {}
            if str(method["key"]) == "adaptive":
                fully_whitened_bundle_dir = _bundle_dir(Path(args.out_root), run_dir, "stap")
                rescue_bundle_dir = _bundle_dir(Path(args.out_root), run_dir, "unwhitened_ratio")
                score, adaptive_stats = _adaptive_score_from_cached_bundles(
                    advanced_bundle_dir=fully_whitened_bundle_dir,
                    rescue_bundle_dir=rescue_bundle_dir,
                )
                bundle_dir = fully_whitened_bundle_dir
                extra_row = {
                    "bundle_dir": f"{fully_whitened_bundle_dir} :: rescue={rescue_bundle_dir}",
                    "adaptive_guard_rule": adaptive_stats["rule"],
                    "adaptive_guard_advanced_fraction": adaptive_stats["advanced_fraction"],
                    "adaptive_guard_advanced_pixels": adaptive_stats["advanced_pixels"],
                    "adaptive_guard_advanced_tile_fraction": adaptive_stats["advanced_tile_fraction"],
                    "adaptive_guard_advanced_tiles": adaptive_stats["advanced_tiles"],
                }
            else:
                bundle_dir = _bundle_dir(Path(args.out_root), run_dir, str(method["bundle_key"]))
                score = _load_array(Path(bundle_dir) / str(method["score_name"]))
            metrics = evaluate_structural_metrics(
                score=score,
                mask_h1_pf_main=mask_h1_pf_main,
                mask_h0_bg=mask_h0_bg,
                mask_h0_nuisance_pa=mask_h0_nuisance_pa,
                mask_h1_alias_qc=mask_h1_alias_qc,
                fprs=[1e-4, 1e-3],
                match_tprs=[0.5],
            )
            row = {
                "run": run_dir.name,
                "seed": _seed_from_run_dir(run_dir),
                "setting": setting,
                "method_key": method["key"],
                "method_label": method["method_label"],
                "bundle_dir": str(bundle_dir),
                "stap_profile": str(args.stap_profile),
                "auc_main_vs_bg": metrics.get("auc_main_vs_bg"),
                "auc_main_vs_nuisance": metrics.get("auc_main_vs_nuisance"),
                "tpr_main@1e-03": metrics.get("tpr_main@1e-03"),
                "fpr_nuisance@1e-03": metrics.get("fpr_nuisance@1e-03"),
                "fpr_nuisance_match@0p5": metrics.get("fpr_nuisance_match@0p5"),
                "fpr_bg_match@0p5": metrics.get("fpr_bg_match@0p5"),
            }
            row.update(extra_row)
            rows.append(row)
            grouped[(setting, method["method_label"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (setting, method_label), items in sorted(grouped.items()):
        auc_main_vs_bg_vals = [float(x["auc_main_vs_bg"]) for x in items]
        auc_main_vs_nuisance_vals = [float(x["auc_main_vs_nuisance"]) for x in items]
        fpr_nuisance_match_0p5_vals = [float(x["fpr_nuisance_match@0p5"]) for x in items]
        tpr_main_1e3_vals = [float(x["tpr_main@1e-03"]) for x in items]
        fpr_nuisance_1e3_vals = [float(x["fpr_nuisance@1e-03"]) for x in items]
        summary_rows.append(
            {
                "setting": setting,
                "method_label": method_label,
                "count": len(items),
                "seeds": [int(x["seed"]) for x in items if x.get("seed") is not None],
                "auc_main_vs_bg_mean": float(np.mean(auc_main_vs_bg_vals)),
                "auc_main_vs_bg_min": float(np.min(auc_main_vs_bg_vals)),
                "auc_main_vs_bg_max": float(np.max(auc_main_vs_bg_vals)),
                "auc_main_vs_nuisance_mean": float(np.mean(auc_main_vs_nuisance_vals)),
                "auc_main_vs_nuisance_min": float(np.min(auc_main_vs_nuisance_vals)),
                "auc_main_vs_nuisance_max": float(np.max(auc_main_vs_nuisance_vals)),
                "fpr_nuisance_match_0p5_mean": float(np.mean(fpr_nuisance_match_0p5_vals)),
                "fpr_nuisance_match_0p5_min": float(np.min(fpr_nuisance_match_0p5_vals)),
                "fpr_nuisance_match_0p5_max": float(np.max(fpr_nuisance_match_0p5_vals)),
                "tpr_main_1e3_mean": float(np.mean(tpr_main_1e3_vals)),
                "tpr_main_1e3_min": float(np.min(tpr_main_1e3_vals)),
                "tpr_main_1e3_max": float(np.max(tpr_main_1e3_vals)),
                "fpr_nuisance_1e3_mean": float(np.mean(fpr_nuisance_1e3_vals)),
                "fpr_nuisance_1e3_min": float(np.min(fpr_nuisance_1e3_vals)),
                "fpr_nuisance_1e3_max": float(np.max(fpr_nuisance_1e3_vals)),
            }
        )

    _write_csv(Path(args.out_csv), rows)
    _write_json(
        Path(args.out_json),
        {
            "schema_version": "simus_detector_family_ablation.v1",
            "runs": [str(p) for p in runs],
            "stap_profile": str(args.stap_profile),
            "rows": rows,
            "summary": summary_rows,
        },
    )
    Path(args.out_tex).write_text(_build_table(summary_rows), encoding="utf-8")
    print(f"[simus-detector-family-ablation] wrote {args.out_csv}")
    print(f"[simus-detector-family-ablation] wrote {args.out_json}")
    print(f"[simus-detector-family-ablation] wrote {args.out_tex}")


if __name__ == "__main__":
    main()
