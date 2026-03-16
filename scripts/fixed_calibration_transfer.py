#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.simus_eval_structural import _threshold_from_neg
from scripts.ulm7883227_structural_roc import (
    _bootstrap_ci,
    _compute_reference_map,
    _derive_structural_masks,
    _load_bundle_scores,
    _parse_blocks,
    _score_specs,
    _window_starts,
    load_ulm_block_iq,
    load_ulm_zenodo_7883227_params,
)
from sim.kwave.icube_bundle import write_acceptance_bundle_from_icube
from sim.simus.bundle import derive_bundle_from_run


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _finite(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def _fmt(x: float | None, digits: int = 3) -> str:
    if x is None or not np.isfinite(float(x)):
        return "--"
    return f"{float(x):.{digits}f}"


def _fmt_fixed_seed_band(values: list[float], digits: int = 3) -> str:
    arr = np.asarray([float(v) for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return "--"
    center = float(np.mean(arr))
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    return f"{center:.{digits}f} [{lo:.{digits}f},{hi:.{digits}f}]"


def _alpha_tex(alpha: float) -> str:
    exponent = int(round(math.log10(float(alpha))))
    return rf"10^{{{exponent}}}"


def _simus_methods() -> list[dict[str, Any]]:
    return [
        {
            "key": "pd",
            "label": "Power Doppler on MC--SVD residual",
            "run_stap": False,
            "bundle_overrides": {},
            "score_name": "score_pd_base.npy",
        },
        {
            "key": "fixed_head",
            "label": "Fixed matched-subspace head",
            "run_stap": True,
            "bundle_overrides": {"stap_detector_variant": "unwhitened_ratio"},
            "score_name": "score_stap_preka.npy",
        },
    ]


def _simus_setting_specs() -> list[dict[str, Any]]:
    return [
        {
            "setting": "Mobile",
            "calibration_runs": [
                Path("runs/sim/simus_v2_acceptance_clin_mobile_pf_v2_phase3_seed125"),
                Path("runs/sim/simus_v2_acceptance_clin_mobile_pf_v2_phase3_seed126"),
            ],
            "eval_runs": [
                Path("runs/sim/simus_v2_acceptance_clin_mobile_pf_v2_phase3_seed127"),
                Path("runs/sim/simus_v2_acceptance_clin_mobile_pf_v2_phase3_seed128"),
            ],
        },
        {
            "setting": "Intra-operative parenchymal",
            "calibration_runs": [
                Path("runs/sim/simus_v2_acceptance_clin_intraop_parenchyma_pf_v3_phase3_seed125"),
                Path("runs/sim/simus_v2_acceptance_clin_intraop_parenchyma_pf_v3_phase3_seed126"),
            ],
            "eval_runs": [
                Path("runs/sim/simus_v2_acceptance_clin_intraop_parenchyma_pf_v3_phase3_seed127"),
                Path("runs/sim/simus_v2_acceptance_clin_intraop_parenchyma_pf_v3_phase3_seed128"),
            ],
        },
    ]


def _ensure_simus_bundle(
    *,
    run_dir: Path,
    dataset_name: str,
    out_root: Path,
    stap_profile: str,
    stap_device: str,
    method: dict[str, Any],
) -> Path:
    bundle_dir = out_root / dataset_name
    if (bundle_dir / "meta.json").is_file():
        return bundle_dir
    return derive_bundle_from_run(
        run_dir=run_dir,
        out_root=out_root,
        dataset_name=dataset_name,
        stap_profile=str(stap_profile),
        baseline_type="mc_svd",
        run_stap=bool(method["run_stap"]),
        stap_device=str(stap_device),
        bundle_overrides=dict(method["bundle_overrides"]),
        meta_extra={
            "fixed_calibration_transfer": {
                "domain": "simus",
                "method_key": str(method["key"]),
            }
        },
    )


def _load_simus_eval_unit(
    *,
    run_dir: Path,
    bundle_dir: Path,
    score_name: str,
) -> dict[str, np.ndarray]:
    ds = run_dir / "dataset"
    return {
        "score": np.load(bundle_dir / score_name).astype(np.float32, copy=False),
        "mask_h1_pf_main": np.load(ds / "mask_h1_pf_main.npy").astype(bool),
        "mask_h0_bg": np.load(ds / "mask_h0_bg.npy").astype(bool),
        "mask_h0_nuisance_pa": np.load(ds / "mask_h0_nuisance_pa.npy").astype(bool),
    }


def _run_simus_transfer(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    methods = _simus_methods()
    for spec in _simus_setting_specs():
        setting = str(spec["setting"])
        for method in methods:
            calib_negatives: list[np.ndarray] = []
            for run_dir in spec["calibration_runs"]:
                if not run_dir.is_dir():
                    raise FileNotFoundError(run_dir)
                bundle_dir = _ensure_simus_bundle(
                    run_dir=run_dir,
                    dataset_name=f"{run_dir.name}_{method['key']}",
                    out_root=Path(args.simus_out_root),
                    stap_profile=str(args.simus_stap_profile),
                    stap_device=str(args.stap_device),
                    method=method,
                )
                unit = _load_simus_eval_unit(run_dir=run_dir, bundle_dir=bundle_dir, score_name=str(method["score_name"]))
                calib_negatives.append(unit["score"][unit["mask_h0_bg"]])
            calib_neg = _finite(np.concatenate(calib_negatives, axis=0))
            thr, calib_fpr = _threshold_from_neg(calib_neg, float(args.alpha))
            if thr is None:
                raise RuntimeError(f"{setting}/{method['key']}: failed to calibrate threshold")
            heldout_vals: list[dict[str, Any]] = []
            for run_dir in spec["eval_runs"]:
                bundle_dir = _ensure_simus_bundle(
                    run_dir=run_dir,
                    dataset_name=f"{run_dir.name}_{method['key']}",
                    out_root=Path(args.simus_out_root),
                    stap_profile=str(args.simus_stap_profile),
                    stap_device=str(args.stap_device),
                    method=method,
                )
                unit = _load_simus_eval_unit(run_dir=run_dir, bundle_dir=bundle_dir, score_name=str(method["score_name"]))
                pos = _finite(unit["score"][unit["mask_h1_pf_main"]])
                neg_bg = _finite(unit["score"][unit["mask_h0_bg"]])
                neg_nuis = _finite(unit["score"][unit["mask_h0_nuisance_pa"]])
                row = {
                    "domain": "simus",
                    "setting": setting,
                    "method_key": str(method["key"]),
                    "method_label": str(method["label"]),
                    "run": run_dir.name,
                    "alpha_nominal": float(args.alpha),
                    "threshold": float(thr),
                    "calibration_bg_fpr": float(calib_fpr),
                    "calibration_negatives": int(calib_neg.size),
                    "heldout_tpr_main": float(np.mean(pos >= thr)) if pos.size else None,
                    "heldout_fpr_bg": float(np.mean(neg_bg >= thr)) if neg_bg.size else None,
                    "heldout_fpr_nuisance": float(np.mean(neg_nuis >= thr)) if neg_nuis.size else None,
                    "n_h1_main": int(pos.size),
                    "n_h0_bg": int(neg_bg.size),
                    "n_h0_nuisance": int(neg_nuis.size),
                    "calibration_runs": [p.name for p in spec["calibration_runs"]],
                }
                rows.append(row)
                heldout_vals.append(row)
            summary_rows.append(
                {
                    "domain": "simus",
                    "setting": setting,
                    "method_key": str(method["key"]),
                    "method_label": str(method["label"]),
                    "alpha_nominal": float(args.alpha),
                    "transfer_design": "separate calibration seeds 125-126 -> held-out seeds 127-128",
                    "calibration_runs": [p.name for p in spec["calibration_runs"]],
                    "heldout_runs": [p.name for p in spec["eval_runs"]],
                    "heldout_tpr_main": _fmt_fixed_seed_band([r["heldout_tpr_main"] for r in heldout_vals]),
                    "heldout_fpr_bg": _fmt_fixed_seed_band([r["heldout_fpr_bg"] for r in heldout_vals]),
                    "heldout_fpr_nuisance": _fmt_fixed_seed_band([r["heldout_fpr_nuisance"] for r in heldout_vals]),
                }
            )
    return rows, summary_rows


def _build_simus_table(summary_rows: list[dict[str, Any]], *, alpha: float) -> str:
    order = ["Power Doppler on MC--SVD residual", "Fixed matched-subspace head"]
    by_key = {(r["setting"], r["method_label"]): r for r in summary_rows if r["domain"] == "simus"}
    settings = ["Mobile", "Intra-operative parenchymal"]
    lines: list[str] = []
    lines.append("% AUTO-GENERATED by scripts/fixed_calibration_transfer.py; DO NOT EDIT BY HAND.")
    lines.append("\\begin{center}")
    lines.append("\\captionsetup{type=table}")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append("\\begin{tabular}{@{}lccc ccc@{}}")
    lines.append("\\hline")
    lines.append("Method & \\multicolumn{3}{c}{Mobile} & \\multicolumn{3}{c}{Intra-operative parenchymal} \\\\")
    lines.append("\\cline{2-4} \\cline{5-7}")
    lines.append(
        " & TPR$_{\\mathrm{main}}$ & realized FPR$_{\\mathrm{bg}}$ & realized FPR$_{\\mathrm{nuis}}$"
        " & TPR$_{\\mathrm{main}}$ & realized FPR$_{\\mathrm{bg}}$ & realized FPR$_{\\mathrm{nuis}}$ \\\\"
    )
    lines.append("\\hline")
    for method in order:
        row = [method]
        for setting in settings:
            sec = by_key.get((setting, method), {})
            row.extend(
                [
                    str(sec.get("heldout_tpr_main", "--")),
                    str(sec.get("heldout_fpr_bg", "--")),
                    str(sec.get("heldout_fpr_nuisance", "--")),
                ]
            )
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(
        "\\caption{Fixed-calibration transfer on the prespecified held-out SIMUS/PyMUST structural benchmark at "
        f"nominal $\\alpha={_alpha_tex(alpha)}$. For each setting, a single threshold is learned once from pooled background negatives on separate calibration seeds 125--126 and then applied unchanged to the held-out evaluation seeds 127--128. Entries report mean [min,max] across the two held-out seeds, so this table is a direct non-retrospective threshold-transfer check rather than the per-window oracle calibration used for ROC reporting elsewhere.}}"
    )
    lines.append("\\label{tab:simus_fixed_calibration_transfer}")
    lines.append("\\end{center}")
    lines.append("")
    return "\n".join(lines)


def _ulm_score_display(score_key: str) -> str:
    return {
        "pd": "Baseline (power Doppler)",
        "kasai": "Baseline (Kasai lag-1 magnitude)",
        "matched_subspace": "Whitened matched-subspace specialist",
    }[score_key]


def _run_ulm_transfer(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    params = load_ulm_zenodo_7883227_params(args.ulm_data_root)
    prf_hz = float(args.ulm_prf_hz) if args.ulm_prf_hz is not None else float(params.frame_rate_hz)
    lt_eff = min(int(args.ulm_lt), int(args.ulm_window_frames) - 1)
    ref_blocks = _parse_blocks(args.ulm_ref_blocks, root=args.ulm_data_root)
    eval_blocks = _parse_blocks(args.ulm_eval_blocks, root=args.ulm_data_root)
    pala_powdop_blocks = _parse_blocks(args.ulm_pala_powdop_blocks, root=args.ulm_data_root)

    ref_maps: dict[int, np.ndarray] = {}
    support_maps: dict[int, np.ndarray] = {}
    for block_id in ref_blocks:
        ref_map, tele = _compute_reference_map(
            int(block_id),
            root=Path(args.ulm_data_root),
            cache_dir=Path(args.ulm_cache_root),
            reg_enable=False,
            reg_subpixel=int(args.ulm_reg_subpixel),
            svd_energy_frac=float(args.ulm_svd_energy_frac),
            device=str(args.stap_device),
            mode="pala_example_matout",
            local_density_quantile=float(args.ulm_reference_local_density_quantile),
            local_density_peak_size=int(args.ulm_reference_local_density_peak_size),
            local_density_sigma=float(args.ulm_reference_local_density_sigma),
            pala_example_root=Path(args.ulm_pala_example_root),
            pala_powdop_blocks=[int(b) for b in pala_powdop_blocks],
            pala_svd_cutoff_start=int(args.ulm_pala_svd_cutoff_start),
            pala_trim_sr_border=int(args.ulm_pala_trim_sr_border),
        )
        ref_maps[int(block_id)] = ref_map
        support_path = str(tele.get("support_mask_path", "")).strip()
        if support_path:
            support_maps[int(block_id)] = np.load(support_path, allow_pickle=False).astype(bool)
        else:
            support_maps[int(block_id)] = np.isfinite(ref_map)

    split_defs = [
        {"split": "blocks1to5_to_6to10", "train": [1, 2, 3, 4, 5], "eval": [6, 7, 8, 9, 10]},
        {"split": "blocks6to10_to_1to5", "train": [6, 7, 8, 9, 10], "eval": [1, 2, 3, 4, 5]},
    ]
    per_window_rows: list[dict[str, Any]] = []
    bundle_cleanup: list[Path] = []
    keep_bundles = bool(args.ulm_keep_bundles)
    for split in split_defs:
        train_blocks = [b for b in split["train"] if b in eval_blocks]
        eval_split_blocks = [b for b in split["eval"] if b in eval_blocks]
        if not train_blocks or not eval_split_blocks:
            continue
        train_negs: dict[str, list[np.ndarray]] = defaultdict(list)
        eval_cache: list[dict[str, Any]] = []
        for block_id in sorted(set(train_blocks + eval_split_blocks)):
            loo_ids = [bid for bid in ref_blocks if int(bid) != int(block_id)]
            if not loo_ids:
                loo_ids = list(ref_blocks)
            ref_map = np.mean(np.stack([ref_maps[int(bid)] for bid in loo_ids], axis=0), axis=0).astype(np.float32, copy=False)
            support_mask = (np.mean(np.stack([support_maps[int(bid)].astype(np.float32) for bid in loo_ids], axis=0), axis=0) >= 0.5)
            mask_flow, mask_bg, qc = _derive_structural_masks(
                ref_map,
                support_mask=support_mask,
                vessel_quantile=float(args.ulm_vessel_quantile),
                bg_quantile=float(args.ulm_background_quantile),
                erode_iters=int(args.ulm_mask_erode_iters),
                guard_dilate_iters=int(args.ulm_guard_dilate_iters),
                edge_margin=int(args.ulm_edge_margin),
                vessel_mask_mode=str(args.ulm_vessel_mask_mode),
                peak_size=int(args.ulm_peak_size),
                peak_dilate_iters=int(args.ulm_peak_dilate_iters),
                background_mask_mode=str(args.ulm_background_mask_mode),
                shell_inner_dilate_iters=int(args.ulm_shell_inner_dilate_iters),
                shell_outer_dilate_iters=int(args.ulm_shell_outer_dilate_iters),
            )
            if int(mask_bg.sum()) < 1000:
                raise RuntimeError(f"ULM block {block_id}: background mask too small ({int(mask_bg.sum())})")
            iq_full = load_ulm_block_iq(int(block_id), frames=None, root=Path(args.ulm_data_root))
            starts = _window_starts(int(iq_full.shape[0]), int(args.ulm_window_frames), int(args.ulm_window_stride))
            if args.ulm_max_windows_per_block is not None:
                starts = starts[: int(args.ulm_max_windows_per_block)]
            for win_idx, start in enumerate(starts):
                stop = int(start) + int(args.ulm_window_frames)
                cube = iq_full[int(start):int(stop)]
                bundle_root = Path(args.ulm_out_root) if keep_bundles else Path(
                    tempfile.mkdtemp(prefix="ulm_fixed_cal_", dir=Path(args.ulm_bundle_tmp_root))
                )
                if not keep_bundles:
                    bundle_cleanup.append(bundle_root)
                paths = write_acceptance_bundle_from_icube(
                    out_root=bundle_root,
                    dataset_name=f"ulm_fixedcal_block{int(block_id):03d}_{int(start):04d}_{int(stop):04d}",
                    Icube=cube,
                    prf_hz=float(prf_hz),
                    tile_hw=(int(args.ulm_tile_h), int(args.ulm_tile_w)),
                    tile_stride=int(args.ulm_tile_stride),
                    Lt=int(lt_eff),
                    diag_load=float(args.ulm_diag_load),
                    cov_estimator=str(args.ulm_cov_estimator),
                    stap_device=str(args.stap_device),
                    baseline_type="mc_svd",
                    reg_enable=False,
                    reg_subpixel=int(args.ulm_reg_subpixel),
                    svd_energy_frac=float(args.ulm_svd_energy_frac),
                    run_stap=True,
                    stap_detector_variant="msd_ratio",
                    score_ka_v2_enable=False,
                    stap_conditional_enable=False,
                    flow_mask_mode="pd_auto",
                    mask_flow_override=mask_flow,
                    mask_bg_override=mask_bg,
                    band_ratio_flow_low_hz=float(args.ulm_flow_low_hz),
                    band_ratio_flow_high_hz=float(args.ulm_flow_high_hz),
                    band_ratio_alias_center_hz=float(args.ulm_alias_center_hz),
                    band_ratio_alias_width_hz=float(args.ulm_alias_width_hz),
                    meta_extra={
                        "fixed_calibration_transfer": {
                            "domain": "ulm",
                            "split": str(split["split"]),
                            "block_id": int(block_id),
                            "window_start": int(start),
                            "window_stop": int(stop),
                            "reference_block_ids": [int(b) for b in loo_ids],
                            "mask_qc": qc,
                        }
                    },
                )
                scores = _load_bundle_scores(Path(paths["meta"]).parent, stap_variant="msd_ratio", include_postka=False)
                cache_rec = {
                    "split": str(split["split"]),
                    "block_id": int(block_id),
                    "window_index": int(win_idx),
                    "window_start": int(start),
                    "window_stop": int(stop),
                    "mask_flow": mask_flow,
                    "mask_bg": mask_bg,
                    "scores": {
                        key: scores[key]
                        for key in ("pd", "kasai", "matched_subspace")
                        if key in scores
                    },
                }
                if block_id in train_blocks:
                    for score_key, score in cache_rec["scores"].items():
                        train_negs[score_key].append(score[mask_bg])
                if block_id in eval_split_blocks:
                    eval_cache.append(cache_rec)

        thresholds: dict[str, tuple[float, float]] = {}
        for score_key in ("pd", "kasai", "matched_subspace"):
            negs = train_negs.get(score_key) or []
            thr, realized = _threshold_from_neg(_finite(np.concatenate(negs, axis=0)), float(args.alpha))
            if thr is None:
                raise RuntimeError(f"ULM split {split['split']}: failed to calibrate {score_key}")
            thresholds[score_key] = (float(thr), float(realized))

        for rec in eval_cache:
            for score_key, score in rec["scores"].items():
                thr, calib_fpr = thresholds[score_key]
                pos = _finite(score[rec["mask_flow"]])
                neg = _finite(score[rec["mask_bg"]])
                per_window_rows.append(
                    {
                        "domain": "ulm",
                        "split": str(split["split"]),
                        "block_id": int(rec["block_id"]),
                        "window_index": int(rec["window_index"]),
                        "window_start": int(rec["window_start"]),
                        "window_stop": int(rec["window_stop"]),
                        "score_key": str(score_key),
                        "score_label": _ulm_score_display(str(score_key)),
                        "alpha_nominal": float(args.alpha),
                        "threshold": float(thr),
                        "calibration_shell_fpr": float(calib_fpr),
                        "heldout_tpr_core": float(np.mean(pos >= thr)) if pos.size else None,
                        "heldout_fpr_shell": float(np.mean(neg >= thr)) if neg.size else None,
                        "n_core": int(pos.size),
                        "n_shell": int(neg.size),
                    }
                )

    if not keep_bundles:
        for path in bundle_cleanup:
            shutil.rmtree(path, ignore_errors=True)

    summary_rows: list[dict[str, Any]] = []
    for score_key in ("pd", "kasai", "matched_subspace"):
        rows = [r for r in per_window_rows if r["score_key"] == score_key]
        summary_rows.append(
            {
                "domain": "ulm",
                "score_key": str(score_key),
                "score_label": _ulm_score_display(str(score_key)),
                "alpha_nominal": float(args.alpha),
                "transfer_design": "complementary 5-block fixed calibration banks",
                "heldout_tpr_core": _bootstrap_ci([r["heldout_tpr_core"] for r in rows], n_boot=int(args.ulm_bootstrap_n), seed=int(args.ulm_bootstrap_seed) + 11),
                "heldout_fpr_shell": _bootstrap_ci([r["heldout_fpr_shell"] for r in rows], n_boot=int(args.ulm_bootstrap_n), seed=int(args.ulm_bootstrap_seed) + 23),
                "n_windows": int(len(rows)),
                "n_blocks": int(len({int(r["block_id"]) for r in rows})),
            }
        )
    return per_window_rows, summary_rows


def _build_ulm_table(summary_rows: list[dict[str, Any]], *, alpha: float) -> str:
    lines: list[str] = []
    lines.append("% AUTO-GENERATED by scripts/fixed_calibration_transfer.py; DO NOT EDIT BY HAND.")
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\hline")
    lines.append("Score & core TPR & realized shell FPR \\\\")
    lines.append("\\hline")
    for key in ("pd", "kasai", "matched_subspace"):
        sec = next(r for r in summary_rows if r["domain"] == "ulm" and r["score_key"] == key)
        lines.append(
            f"{sec['score_label']} & "
            f"{_fmt(sec['heldout_tpr_core']['center'])} [{_fmt(sec['heldout_tpr_core']['lo'])},{_fmt(sec['heldout_tpr_core']['hi'])}] & "
            f"{_fmt(sec['heldout_fpr_shell']['center'])} [{_fmt(sec['heldout_fpr_shell']['lo'])},{_fmt(sec['heldout_fpr_shell']['hi'])}] \\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append(
        "\\caption{Fixed-calibration transfer on the PALA-backed ULM structural audit at "
        f"nominal $\\alpha={_alpha_tex(alpha)}$. A threshold is learned once from pooled perivascular-shell negatives on one 5-block calibration bank and then applied unchanged to the complementary 5 held-out blocks; the reported values pool the two complementary block splits and show window-level means with 95\\% bootstrap CIs over held-out windows. This is therefore a genuine non-retrospective threshold-transfer check on real in-vivo IQ rather than per-block retuning.}}"
    )
    lines.append("\\label{tab:ulm_fixed_calibration_transfer}")
    lines.append("\\end{table}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fixed-calibration deployment-transfer checks on SIMUS and PALA-backed ULM.")
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--stap-device", type=str, default="cuda")

    ap.add_argument("--simus-stap-profile", type=str, default="Brain-SIMUS-Clin-MotionRobust-v0")
    ap.add_argument("--simus-out-root", type=Path, default=ROOT / "runs" / "sim_eval" / "simus_fixed_calibration_transfer")
    ap.add_argument("--simus-out-csv", type=Path, default=ROOT / "reports" / "simus_v2" / "simus_fixed_calibration_transfer.csv")
    ap.add_argument("--simus-out-tex", type=Path, default=ROOT / "reports" / "simus_fixed_calibration_transfer_table.tex")

    ap.add_argument("--ulm-data-root", type=Path, default=ROOT / "data" / "ulm_zenodo_7883227")
    ap.add_argument("--ulm-cache-root", type=Path, default=ROOT / "tmp" / "ulm7883227_structural_roc_cache")
    ap.add_argument("--ulm-bundle-tmp-root", type=Path, default=ROOT / "tmp" / "ulm7883227_fixed_calibration_bundles")
    ap.add_argument("--ulm-out-root", type=Path, default=ROOT / "runs" / "real" / "ulm7883227_fixed_calibration_transfer")
    ap.add_argument("--ulm-ref-blocks", type=str, default="1-10")
    ap.add_argument("--ulm-eval-blocks", type=str, default="1-10")
    ap.add_argument("--ulm-window-frames", type=int, default=64)
    ap.add_argument("--ulm-window-stride", type=int, default=128)
    ap.add_argument("--ulm-max-windows-per-block", type=int, default=2)
    ap.add_argument("--ulm-tile-h", type=int, default=8)
    ap.add_argument("--ulm-tile-w", type=int, default=8)
    ap.add_argument("--ulm-tile-stride", type=int, default=3)
    ap.add_argument("--ulm-lt", type=int, default=64)
    ap.add_argument("--ulm-prf-hz", type=float, default=None)
    ap.add_argument("--ulm-svd-energy-frac", type=float, default=0.975)
    ap.add_argument("--ulm-flow-low-hz", type=float, default=10.0)
    ap.add_argument("--ulm-flow-high-hz", type=float, default=150.0)
    ap.add_argument("--ulm-alias-center-hz", type=float, default=350.0)
    ap.add_argument("--ulm-alias-width-hz", type=float, default=150.0)
    ap.add_argument("--ulm-pala-example-root", type=Path, default=Path("/tmp/PALA_repo_1073521"))
    ap.add_argument("--ulm-pala-powdop-blocks", type=str, default="1,3,5,7,9,11,13,15,17,19")
    ap.add_argument("--ulm-pala-svd-cutoff-start", type=int, default=5)
    ap.add_argument("--ulm-pala-trim-sr-border", type=int, default=1)
    ap.add_argument("--ulm-diag-load", type=float, default=0.07)
    ap.add_argument("--ulm-cov-estimator", type=str, default="scm")
    ap.add_argument("--ulm-reg-subpixel", type=int, default=4)
    ap.add_argument("--ulm-reference-local-density-quantile", type=float, default=0.9995)
    ap.add_argument("--ulm-reference-local-density-peak-size", type=int, default=3)
    ap.add_argument("--ulm-reference-local-density-sigma", type=float, default=1.0)
    ap.add_argument("--ulm-vessel-quantile", type=float, default=0.92)
    ap.add_argument("--ulm-background-quantile", type=float, default=0.50)
    ap.add_argument("--ulm-mask-erode-iters", type=int, default=1)
    ap.add_argument("--ulm-guard-dilate-iters", type=int, default=3)
    ap.add_argument("--ulm-edge-margin", type=int, default=4)
    ap.add_argument("--ulm-vessel-mask-mode", type=str, default="peaks", choices=["area", "peaks"])
    ap.add_argument("--ulm-peak-size", type=int, default=3)
    ap.add_argument("--ulm-peak-dilate-iters", type=int, default=1)
    ap.add_argument("--ulm-background-mask-mode", type=str, default="shell", choices=["global_low", "shell"])
    ap.add_argument("--ulm-shell-inner-dilate-iters", type=int, default=4)
    ap.add_argument("--ulm-shell-outer-dilate-iters", type=int, default=10)
    ap.add_argument("--ulm-bootstrap-n", type=int, default=2000)
    ap.add_argument("--ulm-bootstrap-seed", type=int, default=1337)
    ap.add_argument("--ulm-keep-bundles", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--ulm-out-csv", type=Path, default=ROOT / "reports" / "ulm7883227_pala_fixed_calibration_transfer.csv")
    ap.add_argument("--ulm-out-tex", type=Path, default=ROOT / "reports" / "ulm7883227_pala_fixed_calibration_transfer_table.tex")

    ap.add_argument("--out-json", type=Path, default=ROOT / "reports" / "fixed_calibration_transfer.json")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.simus_out_root).mkdir(parents=True, exist_ok=True)
    Path(args.ulm_cache_root).mkdir(parents=True, exist_ok=True)
    Path(args.ulm_bundle_tmp_root).mkdir(parents=True, exist_ok=True)
    Path(args.ulm_out_root).mkdir(parents=True, exist_ok=True)
    rows_simus, summary_simus = _run_simus_transfer(args)
    rows_ulm, summary_ulm = _run_ulm_transfer(args)

    _write_csv(Path(args.simus_out_csv), rows_simus)
    Path(args.simus_out_tex).write_text(_build_simus_table(summary_simus, alpha=float(args.alpha)), encoding="utf-8")
    _write_csv(Path(args.ulm_out_csv), rows_ulm)
    Path(args.ulm_out_tex).write_text(_build_ulm_table(summary_ulm, alpha=float(args.alpha)), encoding="utf-8")

    payload = {
        "alpha_nominal": float(args.alpha),
        "simus": {
            "rows": rows_simus,
            "summary": summary_simus,
            "stap_profile": str(args.simus_stap_profile),
        },
        "ulm": {
            "rows": rows_ulm,
            "summary": summary_ulm,
            "window_frames": int(args.ulm_window_frames),
            "reference_mode": "pala_example_matout",
            "stap_detector_variant": "msd_ratio",
        },
    }
    _write_json(Path(args.out_json), payload)
    print(f"[fixed-calibration-transfer] wrote {args.out_json}")


if __name__ == "__main__":
    main()
