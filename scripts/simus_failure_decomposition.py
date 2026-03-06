#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.simus_eval_structural import _baseline_label, evaluate_structural_metrics
from sim.kwave.icube_bundle import write_acceptance_bundle_from_icube
from sim.simus.bundle import bundle_profile_kwargs, load_canonical_run


@dataclass(frozen=True)
class VariantSpec:
    key: str
    baseline_type: str
    reg_enable: bool
    run_stap: bool


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


def _prf_from_meta(meta: dict[str, Any]) -> float:
    return float(meta.get("acquisition", {}).get("prf_hz", meta.get("config", {}).get("prf_hz", 0.0)))


def _seed_from_meta(meta: dict[str, Any]) -> int:
    return int(meta.get("config", {}).get("seed", meta.get("seed", 0)) or 0)


def _variants() -> list[VariantSpec]:
    return [
        VariantSpec(key="raw_regoff_base", baseline_type="raw", reg_enable=False, run_stap=False),
        VariantSpec(key="raw_regon_base", baseline_type="raw", reg_enable=True, run_stap=False),
        VariantSpec(key="raw_regoff_stap", baseline_type="raw", reg_enable=False, run_stap=True),
        VariantSpec(key="raw_regon_stap", baseline_type="raw", reg_enable=True, run_stap=True),
        VariantSpec(key="mcsvd_regoff_base", baseline_type="mc_svd", reg_enable=False, run_stap=False),
        VariantSpec(key="mcsvd_regon_base", baseline_type="mc_svd", reg_enable=True, run_stap=False),
        VariantSpec(key="mcsvd_regoff_stap", baseline_type="mc_svd", reg_enable=False, run_stap=True),
        VariantSpec(key="mcsvd_regon_stap", baseline_type="mc_svd", reg_enable=True, run_stap=True),
    ]


def _load_score(bundle_dir: Path, *, eval_score: str, run_stap: bool) -> tuple[Path, np.ndarray]:
    if eval_score == "pd":
        candidates = [bundle_dir / "score_pd_stap.npy"] if run_stap else [bundle_dir / "score_pd_base.npy"]
        if run_stap:
            candidates.append(bundle_dir / "score_pd_base.npy")
        else:
            candidates.append(bundle_dir / "score_base.npy")
    else:
        candidates = [bundle_dir / "score_stap_preka.npy", bundle_dir / "score_stap.npy"] if run_stap else [bundle_dir / "score_base.npy"]
    path = next((p for p in candidates if p.is_file()), candidates[0])
    return path, np.load(path).astype(np.float32, copy=False)


def infer_primary_cause(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def metric(variant: str, eval_score: str, key: str) -> float | None:
        for row in rows:
            if row["variant"] == variant and row["eval_score"] == eval_score:
                value = row.get(key)
                return None if value in (None, "", "None") else float(value)
        return None

    reg_delta = None
    raw_delta = None
    detector_gap = None
    mcsvd_stap_vs_base = None

    m_on = metric("mcsvd_regon_stap", "pd", "auc_main_vs_nuisance")
    m_off = metric("mcsvd_regoff_stap", "pd", "auc_main_vs_nuisance")
    if m_on is not None and m_off is not None:
        reg_delta = m_off - m_on

    raw_on = metric("raw_regon_stap", "pd", "auc_main_vs_nuisance")
    if raw_on is not None and m_on is not None:
        raw_delta = raw_on - m_on

    m_on_vnext = metric("mcsvd_regon_stap", "vnext", "auc_main_vs_nuisance")
    if m_on is not None and m_on_vnext is not None:
        detector_gap = m_on_vnext - m_on

    m_base = metric("mcsvd_regon_base", "pd", "auc_main_vs_nuisance")
    if m_on is not None and m_base is not None:
        mcsvd_stap_vs_base = m_on - m_base

    cause = "mixed_or_unclear"
    reason = "No single axis dominates."
    if raw_delta is not None and raw_delta > 0.10:
        cause = "upstream_mcsvd"
        reason = "Replacing MC-SVD with raw input improves STAP nuisance separation materially."
    elif reg_delta is not None and reg_delta > 0.05:
        cause = "registration"
        reason = "Disabling registration improves STAP nuisance separation materially."
    elif detector_gap is not None and detector_gap > 0.10:
        cause = "pd_readout_or_postfilter"
        reason = "STAP detector score is materially better than PD-after-STAP on the same bundle."
    elif mcsvd_stap_vs_base is not None and mcsvd_stap_vs_base < -0.05:
        cause = "frozen_stap_profile"
        reason = "Adding STAP to MC-SVD is directly harmful even after upstream and registration controls."

    return {
        "primary_cause": cause,
        "reason": reason,
        "delta_raw_regon_stap_minus_mcsvd_regon_stap_auc_nuis_pd": raw_delta,
        "delta_mcsvd_regoff_stap_minus_mcsvd_regon_stap_auc_nuis_pd": reg_delta,
        "delta_mcsvd_regon_stap_vnext_minus_pd_auc_nuis": detector_gap,
        "delta_mcsvd_regon_stap_minus_mcsvd_regon_base_auc_nuis_pd": mcsvd_stap_vs_base,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Decompose SIMUS STAP failures into registration, upstream baseline, and detector-readout effects.")
    ap.add_argument("--run", type=Path, action="append", required=True, help="Canonical SIMUS run directory.")
    ap.add_argument("--out-root", type=Path, default=Path("runs/sim_eval/simus_failure_decomposition"))
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/simus_motion/simus_phase4_failure_decomposition.csv"),
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/simus_motion/simus_phase4_failure_decomposition.json"),
    )
    ap.add_argument("--stap-profile", type=str, default="Brain-SIMUS-Clin", choices=["Brain-SIMUS-Clin"])
    ap.add_argument("--stap-device", type=str, default="cuda")
    ap.add_argument("--fprs", type=str, default="1e-4,1e-3")
    ap.add_argument("--match-tprs", type=str, default="0.25,0.5,0.75")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    fprs = [float(x) for x in _split_csv_list(str(args.fprs))]
    match_tprs = [float(x) for x in _split_csv_list(str(args.match_tprs))]
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {"schema_version": "simus_failure_decomposition.v1", "runs": {}}

    for run_dir in args.run:
        run_dir = Path(run_dir)
        icube, masks, meta = load_canonical_run(run_dir)
        prf = _prf_from_meta(meta)
        seed = _seed_from_meta(meta)
        mask_h1_pf_main = masks.get("mask_h1_pf_main")
        mask_h0_bg = masks.get("mask_h0_bg")
        if mask_h1_pf_main is None or mask_h0_bg is None:
            raise ValueError(f"{run_dir}: dataset is missing mask_h1_pf_main.npy or mask_h0_bg.npy")
        mask_h0_nuisance_pa = masks.get("mask_h0_nuisance_pa")
        mask_h1_alias_qc = masks.get("mask_h1_alias_qc")

        run_key = run_dir.name
        details["runs"][run_key] = {
            "run_dir": str(run_dir),
            "profile": meta.get("simus", {}).get("profile"),
            "tier": meta.get("simus", {}).get("tier"),
            "shape": list(icube.shape),
            "motion_disp_rms_px": meta.get("motion", {}).get("telemetry", {}).get("disp_rms_px"),
            "phase_rms_rad": meta.get("phase_screen", {}).get("telemetry", {}).get("phase_rms_rad"),
            "variants": {},
        }

        per_run_rows: list[dict[str, Any]] = []
        for variant in _variants():
            kwargs = bundle_profile_kwargs(str(args.stap_profile), T=int(icube.shape[0]), baseline_type=str(variant.baseline_type))
            kwargs["reg_enable"] = bool(variant.reg_enable)
            kwargs["baseline_type"] = str(variant.baseline_type)
            bundle_dir = Path(args.out_root) / run_key / variant.key
            bundle_paths = write_acceptance_bundle_from_icube(
                out_root=bundle_dir.parent,
                dataset_name=bundle_dir.name,
                Icube=icube,
                prf_hz=float(prf),
                seed=int(seed),
                mask_flow_override=np.asarray(masks.get("mask_flow"), dtype=bool),
                mask_bg_override=np.asarray(masks.get("mask_bg"), dtype=bool),
                run_stap=bool(variant.run_stap),
                stap_device=str(args.stap_device),
                meta_extra={
                    "simus_failure_decomposition": True,
                    "source_run": str(run_dir),
                    "variant_key": variant.key,
                },
                **kwargs,
            )
            bundle_final = Path(bundle_paths["meta"]).parent

            for eval_score in ("pd", "vnext"):
                score_path, score = _load_score(bundle_final, eval_score=eval_score, run_stap=bool(variant.run_stap))
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
                    "run": run_key,
                    "profile": meta.get("simus", {}).get("profile"),
                    "tier": meta.get("simus", {}).get("tier"),
                    "variant": variant.key,
                    "baseline_type": variant.baseline_type,
                    "baseline_label": _baseline_label(variant.baseline_type),
                    "reg_enable": int(variant.reg_enable),
                    "run_stap": int(variant.run_stap),
                    "pipeline_label": f"{_baseline_label(variant.baseline_type)} -> STAP" if variant.run_stap else _baseline_label(variant.baseline_type),
                    "eval_score": eval_score,
                    "bundle_dir": str(bundle_final),
                    "score_file": score_path.name,
                    "motion_disp_rms_px": meta.get("motion", {}).get("telemetry", {}).get("disp_rms_px"),
                    "phase_rms_rad": meta.get("phase_screen", {}).get("telemetry", {}).get("phase_rms_rad"),
                }
                row.update(metrics)
                rows.append(row)
                per_run_rows.append(row)
            details["runs"][run_key]["variants"][variant.key] = {
                "bundle_dir": str(bundle_final),
                "baseline_type": variant.baseline_type,
                "reg_enable": bool(variant.reg_enable),
                "run_stap": bool(variant.run_stap),
            }

        details["runs"][run_key]["inference"] = infer_primary_cause(per_run_rows)

    _write_csv(Path(args.out_csv), rows)
    _write_json(Path(args.out_json), {"rows": rows, "details": details})
    print(f"[simus-failure-decomposition] wrote {args.out_csv}")
    print(f"[simus-failure-decomposition] wrote {args.out_json}")


if __name__ == "__main__":
    main()
