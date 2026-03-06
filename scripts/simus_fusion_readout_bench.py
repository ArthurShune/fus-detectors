#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.simus_eval_structural import evaluate_structural_metrics
from sim.simus.bundle import load_canonical_run


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


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


def _split_csv_list(spec: str) -> list[str]:
    return [s.strip() for s in (spec or "").split(",") if s.strip()]


def _rate_tag(rate: float) -> str:
    return f"{float(rate):.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _parse_case(spec: str) -> dict[str, Path]:
    parts = str(spec).split("::")
    if len(parts) != 4:
        raise ValueError(
            f"--case expects name::run_dir::baseline_bundle::stap_bundle, got {spec!r}"
        )
    return {
        "name": parts[0],
        "run_dir": Path(parts[1]),
        "baseline_bundle": Path(parts[2]),
        "stap_bundle": Path(parts[3]),
    }


def _load_score(bundle_dir: Path, name: str) -> np.ndarray:
    path = bundle_dir / f"{name}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    return np.load(path).astype(np.float32, copy=False)


def _quantile_rank_map(score: np.ndarray) -> np.ndarray:
    vals = np.asarray(score, dtype=np.float64).ravel()
    out = np.full(vals.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return out.reshape(score.shape).astype(np.float32)
    vf = vals[finite]
    order = np.argsort(vf, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1, dtype=np.float64)
    out[finite] = ranks / float(order.size + 1)
    return out.reshape(score.shape).astype(np.float32)


def _robust_z_map(score: np.ndarray) -> np.ndarray:
    vals = np.asarray(score, dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return np.zeros_like(vals, dtype=np.float32)
    med = float(np.median(finite))
    q25 = float(np.quantile(finite, 0.25))
    q75 = float(np.quantile(finite, 0.75))
    iqr = max(q75 - q25, 1e-6)
    return ((vals - med) / iqr).astype(np.float32)


def _tail_log_map(rank_map: np.ndarray) -> np.ndarray:
    q = np.asarray(rank_map, dtype=np.float64)
    q = np.clip(q, 1e-6, 1.0 - 1e-6)
    return (-np.log1p(-q)).astype(np.float32)


def _candidate_scores(
    score_base: np.ndarray,
    score_stap: np.ndarray,
) -> dict[str, np.ndarray]:
    q_base = _quantile_rank_map(score_base)
    q_stap = _quantile_rank_map(score_stap)
    z_base = _robust_z_map(score_base)
    z_stap = _robust_z_map(score_stap)
    tail_base = _tail_log_map(q_base)
    tail_stap = _tail_log_map(q_stap)
    z_stap_pos = np.maximum(z_stap, 0.0).astype(np.float32, copy=False)
    return {
        "baseline_score": score_base.astype(np.float32, copy=False),
        "stap_detector": score_stap.astype(np.float32, copy=False),
        # Parameter-free fusion candidates. These use global self-normalization only.
        "fusion_rank_product": (q_base * q_stap).astype(np.float32, copy=False),
        "fusion_rank_min": np.minimum(q_base, q_stap).astype(np.float32, copy=False),
        "fusion_tail_fisher": (tail_base + tail_stap).astype(np.float32, copy=False),
        "fusion_tail_min": np.minimum(tail_base, tail_stap).astype(np.float32, copy=False),
        "fusion_zsum": (z_base + z_stap).astype(np.float32, copy=False),
        # Asymmetric candidates: preserve baseline score scale, gate it with the STAP detector.
        "fusion_base_x_rankstap": (score_base * q_stap).astype(np.float32, copy=False),
        "fusion_base_x_tailstap": (score_base * tail_stap).astype(np.float32, copy=False),
        "fusion_base_x_zstap_pos": (score_base * z_stap_pos).astype(np.float32, copy=False),
    }


def _region_mean(score: np.ndarray, mask: np.ndarray | None) -> float | None:
    if mask is None:
        return None
    vals = np.asarray(score, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Benchmark parameter-free baseline+STAP readout fusions on existing SIMUS bundles."
        )
    )
    ap.add_argument(
        "--case",
        type=str,
        action="append",
        required=True,
        help="Case spec: name::run_dir::baseline_bundle::stap_bundle",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/simus_motion/simus_fusion_readout_bench.csv"),
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/simus_motion/simus_fusion_readout_bench.json"),
    )
    ap.add_argument("--fprs", type=str, default="1e-4,1e-3")
    ap.add_argument("--match-tprs", type=str, default="0.25,0.5,0.75")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    fprs = [float(x) for x in _split_csv_list(str(args.fprs))]
    match_tprs = [float(x) for x in _split_csv_list(str(args.match_tprs))]
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {"schema_version": "simus_fusion_readout_bench.v1", "cases": {}}

    for case_spec in args.case:
        case = _parse_case(case_spec)
        run_dir = case["run_dir"]
        _, masks, meta = load_canonical_run(run_dir)
        mask_h1_pf_main = masks.get("mask_h1_pf_main")
        mask_h0_bg = masks.get("mask_h0_bg")
        if mask_h1_pf_main is None or mask_h0_bg is None:
            raise ValueError(f"{run_dir}: missing mask_h1_pf_main.npy or mask_h0_bg.npy")
        mask_h0_nuisance_pa = masks.get("mask_h0_nuisance_pa")
        mask_h1_alias_qc = masks.get("mask_h1_alias_qc")

        score_base = _load_score(case["baseline_bundle"], "score_base")
        score_stap = _load_score(case["stap_bundle"], "score_stap_preka")
        candidates = _candidate_scores(score_base, score_stap)
        best_auc_nuis: tuple[str, float] | None = None
        best_fpr_match: tuple[str, float] | None = None
        match_key = f"fpr_nuisance_match@{_rate_tag(match_tprs[0])}"

        case_rows: list[dict[str, Any]] = []
        for name, score in candidates.items():
            metrics = evaluate_structural_metrics(
                score=score,
                mask_h1_pf_main=mask_h1_pf_main,
                mask_h0_bg=mask_h0_bg,
                mask_h0_nuisance_pa=mask_h0_nuisance_pa,
                mask_h1_alias_qc=mask_h1_alias_qc,
                fprs=fprs,
                match_tprs=match_tprs,
            )
            row = {
                "case": case["name"],
                "profile": meta.get("simus", {}).get("profile"),
                "tier": meta.get("simus", {}).get("tier"),
                "run_dir": str(run_dir),
                "baseline_bundle": str(case["baseline_bundle"]),
                "stap_bundle": str(case["stap_bundle"]),
                "candidate": name,
                "motion_disp_rms_px": meta.get("motion", {}).get("telemetry", {}).get("disp_rms_px"),
                "phase_rms_rad": meta.get("phase_screen", {}).get("telemetry", {}).get("phase_rms_rad"),
                "mean_h1": _region_mean(score, mask_h1_pf_main),
                "mean_bg": _region_mean(score, mask_h0_bg),
                "mean_nuisance": _region_mean(score, mask_h0_nuisance_pa),
            }
            row.update(metrics)
            rows.append(row)
            case_rows.append(row)

            auc_nuis = row.get("auc_main_vs_nuisance")
            if auc_nuis is not None:
                val = float(auc_nuis)
                if best_auc_nuis is None or val > best_auc_nuis[1]:
                    best_auc_nuis = (name, val)
            fpr_match = row.get(match_key)
            if fpr_match is not None:
                val = float(fpr_match)
                if best_fpr_match is None or val < best_fpr_match[1]:
                    best_fpr_match = (name, val)

        details["cases"][case["name"]] = {
            "run_dir": str(run_dir),
            "profile": meta.get("simus", {}).get("profile"),
            "tier": meta.get("simus", {}).get("tier"),
            "motion_disp_rms_px": meta.get("motion", {}).get("telemetry", {}).get("disp_rms_px"),
            "phase_rms_rad": meta.get("phase_screen", {}).get("telemetry", {}).get("phase_rms_rad"),
            "best_auc_main_vs_nuisance": {
                "candidate": best_auc_nuis[0] if best_auc_nuis else None,
                "value": best_auc_nuis[1] if best_auc_nuis else None,
            },
            f"best_{match_key}": {
                "candidate": best_fpr_match[0] if best_fpr_match else None,
                "value": best_fpr_match[1] if best_fpr_match else None,
            },
            "rows": case_rows,
        }

    _write_csv(Path(args.out_csv), rows)
    _write_json(Path(args.out_json), {"rows": rows, "details": details})
    print(f"[simus-fusion-readout-bench] wrote {args.out_csv}")
    print(f"[simus-fusion-readout-bench] wrote {args.out_json}")


if __name__ == "__main__":
    main()
