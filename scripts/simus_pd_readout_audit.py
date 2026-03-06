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


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    finite = np.isfinite(aa) & np.isfinite(bb)
    if finite.sum() < 2:
        return None
    aa = aa[finite]
    bb = bb[finite]
    if float(np.std(aa)) <= 0.0 or float(np.std(bb)) <= 0.0:
        return None
    return float(np.corrcoef(aa, bb)[0, 1])


def _quantile_stats(arr: np.ndarray) -> dict[str, float]:
    vals = np.asarray(arr, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "min": float("nan"),
            "p01": float("nan"),
            "p10": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }
    return {
        "min": float(np.min(vals)),
        "p01": float(np.quantile(vals, 0.01)),
        "p10": float(np.quantile(vals, 0.10)),
        "p50": float(np.quantile(vals, 0.50)),
        "p90": float(np.quantile(vals, 0.90)),
        "p99": float(np.quantile(vals, 0.99)),
        "max": float(np.max(vals)),
    }


def _region_stats(
    score: np.ndarray,
    mask_h1_pf_main: np.ndarray,
    mask_h0_bg: np.ndarray,
    mask_h0_nuisance_pa: np.ndarray | None,
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    regions = {
        "h1": np.asarray(mask_h1_pf_main, dtype=bool),
        "bg": np.asarray(mask_h0_bg, dtype=bool),
        "nuisance": (
            np.asarray(mask_h0_nuisance_pa, dtype=bool)
            if mask_h0_nuisance_pa is not None
            else np.zeros_like(mask_h1_pf_main, dtype=bool)
        ),
    }
    arr = np.asarray(score, dtype=np.float64)
    for label, mask in regions.items():
        vals = arr[mask]
        vals = vals[np.isfinite(vals)]
        out[f"{label}_median"] = float(np.median(vals)) if vals.size else None
        out[f"{label}_mean"] = float(np.mean(vals)) if vals.size else None
    return out


def _load_source_run(bundle_dir: Path) -> tuple[Path, dict[str, np.ndarray], dict[str, Any]]:
    meta = json.loads((bundle_dir / "meta.json").read_text(encoding="utf-8"))
    source_run = meta.get("source_run")
    if not source_run:
        raise ValueError(f"{bundle_dir}: meta.json missing source_run")
    run_dir = Path(str(source_run))
    _, masks, run_meta = load_canonical_run(run_dir)
    return run_dir, masks, meta | {"source_meta": run_meta}


def _candidate_maps(bundle_dir: Path) -> dict[str, np.ndarray]:
    pd_base = np.load(bundle_dir / "pd_base.npy").astype(np.float32, copy=False)
    pd_stap = np.load(bundle_dir / "pd_stap.npy").astype(np.float32, copy=False)
    score = np.load(bundle_dir / "score_stap_preka.npy").astype(np.float32, copy=False)
    band_fraction = np.divide(pd_stap, np.maximum(pd_base, 1e-12), dtype=np.float32)
    band_fraction = np.clip(band_fraction, 1e-12, None)
    neg_log_band = (-np.log(band_fraction)).astype(np.float32, copy=False)
    inv_band = np.divide(1.0, band_fraction, dtype=np.float32)
    score_x_neg_log = (score * neg_log_band).astype(np.float32, copy=False)
    return {
        "pd_stap": pd_stap,
        "score_stap_preka": score,
        "band_fraction": band_fraction,
        "band_suppression": (1.0 - band_fraction).astype(np.float32, copy=False),
        "inv_band_fraction": inv_band,
        "neg_log_band_fraction": neg_log_band,
        "score_x_neg_log_band_fraction": score_x_neg_log,
    }


def _metrics_for_candidate(
    name: str,
    score: np.ndarray,
    *,
    mask_h1_pf_main: np.ndarray,
    mask_h0_bg: np.ndarray,
    mask_h0_nuisance_pa: np.ndarray | None,
    mask_h1_alias_qc: np.ndarray | None,
    fprs: list[float],
    match_tprs: list[float],
) -> dict[str, Any]:
    row = {"candidate": name}
    row.update(
        evaluate_structural_metrics(
            score=score,
            mask_h1_pf_main=mask_h1_pf_main,
            mask_h0_bg=mask_h0_bg,
            mask_h0_nuisance_pa=mask_h0_nuisance_pa,
            mask_h1_alias_qc=mask_h1_alias_qc,
            fprs=fprs,
            match_tprs=match_tprs,
        )
    )
    row.update(_region_stats(score, mask_h1_pf_main, mask_h0_bg, mask_h0_nuisance_pa))
    return row


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit STAP PD readout collapse on SIMUS bundles.")
    ap.add_argument("--bundle", type=Path, action="append", required=True, help="Bundle directory containing pd_base.npy / pd_stap.npy / score_stap_preka.npy.")
    ap.add_argument("--out-csv", type=Path, default=Path("reports/simus_motion/simus_phase4_pd_readout_audit.csv"))
    ap.add_argument("--out-json", type=Path, default=Path("reports/simus_motion/simus_phase4_pd_readout_audit.json"))
    ap.add_argument("--fprs", type=str, default="1e-4,1e-3")
    ap.add_argument("--match-tprs", type=str, default="0.25,0.5,0.75")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    fprs = [float(x) for x in _split_csv_list(str(args.fprs))]
    match_tprs = [float(x) for x in _split_csv_list(str(args.match_tprs))]

    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {"schema_version": "simus_pd_readout_audit.v1", "bundles": {}}

    for bundle_dir in args.bundle:
        bundle_dir = Path(bundle_dir)
        run_dir, masks, meta = _load_source_run(bundle_dir)
        mask_h1_pf_main = masks.get("mask_h1_pf_main")
        mask_h0_bg = masks.get("mask_h0_bg")
        if mask_h1_pf_main is None or mask_h0_bg is None:
            raise ValueError(f"{run_dir}: missing mask_h1_pf_main.npy or mask_h0_bg.npy")
        mask_h0_nuisance_pa = masks.get("mask_h0_nuisance_pa")
        mask_h1_alias_qc = masks.get("mask_h1_alias_qc")
        eval_mask_bg = masks.get("mask_bg")

        candidates = _candidate_maps(bundle_dir)
        pd_base = np.load(bundle_dir / "pd_base.npy").astype(np.float32, copy=False)
        pd_stap = np.load(bundle_dir / "pd_stap.npy").astype(np.float32, copy=False)
        score = candidates["score_stap_preka"]
        band_fraction = candidates["band_fraction"]

        bundle_key = f"{bundle_dir.parent.name}__{bundle_dir.name}"
        bg_identity_mask = np.isclose(pd_stap, pd_base, rtol=0.0, atol=1e-7)
        detail = {
            "bundle_dir": str(bundle_dir),
            "run_dir": str(run_dir),
            "profile": meta.get("source_meta", {}).get("simus", {}).get("profile"),
            "tier": meta.get("source_meta", {}).get("simus", {}).get("tier"),
            "motion_disp_rms_px": meta.get("source_meta", {}).get("motion", {}).get("telemetry", {}).get("disp_rms_px"),
            "phase_rms_rad": meta.get("source_meta", {}).get("phase_screen", {}).get("telemetry", {}).get("phase_rms_rad"),
            "corr_pd_stap_vs_score": _safe_corr(pd_stap, score),
            "corr_band_fraction_vs_score": _safe_corr(band_fraction, score),
            "corr_neg_log_band_vs_score": _safe_corr(candidates["neg_log_band_fraction"], score),
            "bg_identity_fraction_all": float(np.mean(bg_identity_mask)),
            "bg_identity_fraction_h0_bg": float(np.mean(bg_identity_mask[np.asarray(mask_h0_bg, dtype=bool)])),
            "band_fraction_eq1_fraction_all": float(np.mean(band_fraction >= 1.0 - 1e-7)),
            "band_fraction_eq1_fraction_h0_bg": float(np.mean(band_fraction[np.asarray(mask_h0_bg, dtype=bool)] >= 1.0 - 1e-7)),
            "band_fraction_eq1_fraction_h1_pf_main": float(np.mean(band_fraction[np.asarray(mask_h1_pf_main, dtype=bool)] >= 1.0 - 1e-7)),
            "band_fraction_stats": _quantile_stats(band_fraction),
            "score_stats": _quantile_stats(score),
        }
        if mask_h0_nuisance_pa is not None:
            nuisance_mask = np.asarray(mask_h0_nuisance_pa, dtype=bool)
            detail["bg_identity_fraction_h0_nuisance_pa"] = float(np.mean(bg_identity_mask[nuisance_mask]))
            detail["band_fraction_eq1_fraction_h0_nuisance_pa"] = float(
                np.mean(band_fraction[nuisance_mask] >= 1.0 - 1e-7)
            )
        if eval_mask_bg is not None:
            eval_bg_mask = np.asarray(eval_mask_bg, dtype=bool)
            detail["mask_h0_bg_in_mask_bg_fraction"] = float(
                np.mean(eval_bg_mask[np.asarray(mask_h0_bg, dtype=bool)])
            )
            detail["mask_bg_coverage"] = float(np.mean(eval_bg_mask))

        candidate_rows: list[dict[str, Any]] = []
        for name, score_map in candidates.items():
            row = {
                "bundle": bundle_key,
                "profile": detail["profile"],
                "tier": detail["tier"],
                "motion_disp_rms_px": detail["motion_disp_rms_px"],
                "phase_rms_rad": detail["phase_rms_rad"],
                "bundle_dir": str(bundle_dir),
                "run_dir": str(run_dir),
            }
            row.update(
                _metrics_for_candidate(
                    name,
                    score_map,
                    mask_h1_pf_main=mask_h1_pf_main,
                    mask_h0_bg=mask_h0_bg,
                    mask_h0_nuisance_pa=mask_h0_nuisance_pa,
                    mask_h1_alias_qc=mask_h1_alias_qc,
                    fprs=fprs,
                    match_tprs=match_tprs,
                )
            )
            if name == "pd_stap":
                row["corr_vs_score_stap_preka"] = detail["corr_pd_stap_vs_score"]
            elif name == "band_fraction":
                row["corr_vs_score_stap_preka"] = detail["corr_band_fraction_vs_score"]
            elif name == "neg_log_band_fraction":
                row["corr_vs_score_stap_preka"] = detail["corr_neg_log_band_vs_score"]
            else:
                row["corr_vs_score_stap_preka"] = _safe_corr(score_map, score)
            candidate_rows.append(row)
            rows.append(row)

        detail["candidates"] = {row["candidate"]: row for row in candidate_rows}
        details["bundles"][bundle_key] = detail

    _write_csv(Path(args.out_csv), rows)
    _write_json(Path(args.out_json), {"rows": rows, "details": details})
    print(f"[simus-pd-readout-audit] wrote {args.out_csv}")
    print(f"[simus-pd-readout-audit] wrote {args.out_json}")


if __name__ == "__main__":
    main()
