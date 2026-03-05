#!/usr/bin/env python3
"""
Compare STAP vs baseline clutter filters on a canonical Icube dataset.

This script is intended for simulation backends that produce ground-truth
flow/background masks (e.g. SIMUS/PyMUST or physical_doppler surrogates).

It:
  1) derives acceptance bundles for a set of baseline methods (run_stap=False),
  2) derives one STAP bundle (typically MC-SVD + STAP),
  3) evaluates AUC and tail hit@alpha / TPR@FPR on the score maps using the
     provided masks, and
  4) writes a consolidated CSV + JSON under reports/.

It does not change detector math; it just exercises existing baseline+STAP code.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from sim.kwave.icube_bundle import write_acceptance_bundle_from_icube


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no rows")
    fieldnames: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[np.isfinite(x)]


def _auc_pos_vs_neg(pos: np.ndarray, neg: np.ndarray) -> float | None:
    """
    AUC = P{S_pos > S_neg} + 0.5 P{S_pos == S_neg} (Mann-Whitney).
    """
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


def _tpr_at_fpr(pos: np.ndarray, neg: np.ndarray, fpr: float) -> tuple[float | None, float | None, float | None]:
    """
    Right-tail scoring: classify as positive when score >= thr.
    thr is set as the (1-fpr) quantile of the negative distribution.
    Returns (thr, tpr, fpr_emp).
    """
    pos = _finite(pos)
    neg = _finite(neg)
    if pos.size == 0 or neg.size == 0:
        return None, None, None
    q = float(np.clip(1.0 - float(fpr), 0.0, 1.0))
    thr = float(np.quantile(neg, q))
    fpr_emp = float(np.mean(neg >= thr))
    tpr = float(np.mean(pos >= thr))
    return thr, tpr, fpr_emp


@dataclass(frozen=True)
class MethodSpec:
    key: str
    baseline_type: str
    run_stap: bool
    role: str  # "baseline" | "stap"


def _load_dataset(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    ds = Path(run_dir) / "dataset"
    meta = _load_json(ds / "meta.json")
    Icube = np.load(ds / "icube.npy").astype(np.complex64, copy=False)
    mask_flow = np.load(ds / "mask_flow.npy").astype(bool, copy=False)
    mask_bg = np.load(ds / "mask_bg.npy").astype(bool, copy=False)
    mask_alias = None
    alias_path = ds / "mask_alias_expected.npy"
    if alias_path.is_file():
        mask_alias = np.load(alias_path).astype(bool, copy=False)
    if Icube.ndim != 3:
        raise ValueError(f"Icube must have shape (T,H,W), got {Icube.shape}")
    if mask_flow.shape != Icube.shape[1:] or mask_bg.shape != Icube.shape[1:]:
        raise ValueError("mask shape mismatch vs Icube")
    if mask_alias is not None and mask_alias.shape != Icube.shape[1:]:
        raise ValueError("mask_alias_expected shape mismatch vs Icube")
    return Icube, mask_flow, mask_bg, mask_alias, meta


def _derive_bundle(
    *,
    out_root: Path,
    dataset_name: str,
    Icube: np.ndarray,
    prf_hz: float,
    seed: int,
    mask_flow: np.ndarray,
    mask_bg: np.ndarray,
    method: MethodSpec,
    stap_device: str,
    # Fixed "paper-style" profile knobs (should match clinical Brain-* defaults).
    tile_hw: tuple[int, int],
    tile_stride: int,
    Lt: int,
    svd_energy_frac: float,
    diag_load: float,
) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    paths = write_acceptance_bundle_from_icube(
        out_root=out_root,
        dataset_name=dataset_name,
        Icube=Icube,
        prf_hz=prf_hz,
        seed=seed,
        tile_hw=tile_hw,
        tile_stride=tile_stride,
        Lt=Lt,
        diag_load=diag_load,
        cov_estimator="tyler_pca",
        huber_c=5.0,
        mvdr_load_mode="auto",
        mvdr_auto_kappa=120.0,
        constraint_ridge=0.18,
        fd_span_mode="fixed",
        fd_fixed_span_hz=250.0,
        fd_span_rel=(0.30, 1.10),
        constraint_mode="exp+deriv",
        grid_step_rel=0.20,
        fd_min_pts=9,
        fd_max_pts=15,
        msd_lambda=0.05,
        msd_ridge=0.10,
        msd_agg_mode="median",
        msd_ratio_rho=0.05,
        msd_contrast_alpha=0.6,
        baseline_type=str(method.baseline_type),
        reg_enable=True,
        reg_method="phasecorr",
        reg_subpixel=4,
        reg_reference="median",
        svd_energy_frac=float(svd_energy_frac),
        mask_flow_override=mask_flow,
        mask_bg_override=mask_bg,
        stap_conditional_enable=False,
        run_stap=bool(method.run_stap),
        stap_device=str(stap_device),
        feasibility_mode="updated",
        meta_extra={
            "icube_baseline_compare": True,
            "compare_method_key": str(method.key),
            "compare_role": str(method.role),
        },
    )
    # Bundle writer returns a dict of paths; meta.json path is under key "meta".
    meta_path = Path(paths["meta"])
    return meta_path.parent


def _eval_score_map(
    score: np.ndarray,
    mask_flow: np.ndarray,
    mask_bg: np.ndarray,
    fprs: list[float],
) -> dict[str, Any]:
    score = np.asarray(score, dtype=np.float64)
    flow = np.asarray(mask_flow, dtype=bool)
    bg = np.asarray(mask_bg, dtype=bool)
    pos = score[flow]
    neg = score[bg]
    out: dict[str, Any] = {
        "n_flow": int(flow.sum()),
        "n_bg": int(bg.sum()),
        "auc": _auc_pos_vs_neg(pos, neg),
        "fpr_min": (1.0 / float(int(bg.sum()))) if int(bg.sum()) > 0 else None,
    }
    for fpr in fprs:
        tag = f"{float(fpr):.0e}"
        thr, tpr, fpr_emp = _tpr_at_fpr(pos, neg, fpr)
        out[f"thr@{tag}"] = thr
        out[f"tpr@{tag}"] = tpr
        out[f"fpr@{tag}"] = fpr_emp
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Baseline vs STAP comparison on a canonical Icube dataset.")
    ap.add_argument("--run", type=Path, action="append", required=True, help="Run dir containing dataset/.")
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("runs/sim_eval/icube_baseline_compare"),
        help="Root for derived acceptance bundles (default: %(default)s).",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/simus_baseline_compare/icube_baseline_compare.csv"),
        help="Output CSV path (tracked) (default: %(default)s).",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/simus_baseline_compare/icube_baseline_compare.json"),
        help="Output JSON path (tracked) (default: %(default)s).",
    )
    ap.add_argument("--tag", type=str, default=None, help="Optional tag to include in dataset_name.")
    ap.add_argument(
        "--eval-score",
        type=str,
        default="pd",
        choices=["pd", "vnext"],
        help=(
            "Which score map to evaluate (paper-style). "
            "'pd' uses score_pd_{base,stap}.npy (PD after filtering); "
            "'vnext' uses score_base.npy / score_stap_preka.npy (STAP detector score)."
        ),
    )

    ap.add_argument(
        "--baselines",
        type=str,
        default="mc_svd,svd_similarity,local_svd,rpca,hosvd",
        help="Comma-separated baseline_type list (default: %(default)s).",
    )
    ap.add_argument(
        "--stap-baseline",
        type=str,
        default="mc_svd",
        help="Baseline type used for the STAP row (default: %(default)s).",
    )
    ap.add_argument(
        "--stap-device",
        type=str,
        default="cpu",
        help="STAP device used when deriving bundles (default: %(default)s).",
    )
    ap.add_argument("--tile-hw", type=int, nargs=2, default=(8, 8))
    ap.add_argument("--tile-stride", type=int, default=3)
    ap.add_argument("--Lt", type=int, default=8)
    ap.add_argument("--svd-energy-frac", type=float, default=0.90)
    ap.add_argument("--diag-load", type=float, default=0.07)
    ap.add_argument(
        "--fprs",
        type=str,
        default="1e-4,3e-4,1e-3",
        help="Comma-separated FPR targets (default: %(default)s).",
    )
    return ap.parse_args()


def _split_csv_list(spec: str) -> list[str]:
    return [s.strip() for s in (spec or "").split(",") if s.strip()]


def main() -> None:
    args = parse_args()
    runs = [Path(p) for p in (args.run or [])]
    out_root = Path(args.out_root)
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)

    baselines = _split_csv_list(str(args.baselines))
    stap_baseline = str(args.stap_baseline).strip()
    tag = str(args.tag).strip() if args.tag else ""
    fprs = [float(x) for x in _split_csv_list(str(args.fprs))]
    eval_score = str(args.eval_score).strip()

    tile_hw = (int(args.tile_hw[0]), int(args.tile_hw[1]))
    tile_stride = int(args.tile_stride)
    Lt = int(args.Lt)
    svd_energy_frac = float(args.svd_energy_frac)
    diag_load = float(args.diag_load)
    stap_device = str(args.stap_device)

    methods: list[MethodSpec] = []
    for b in baselines:
        key = f"baseline_{b}"
        methods.append(MethodSpec(key=key, baseline_type=b, run_stap=False, role="baseline"))
    methods.append(MethodSpec(key="stap", baseline_type=stap_baseline, run_stap=True, role="stap"))

    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {"schema_version": "icube_baseline_compare.v2", "runs": {}}

    # Keep CPU deterministic by default (no CUDA graph etc).
    os.environ.setdefault("STAP_FAST_CUDA_GRAPH", "0")

    for run_dir in runs:
        Icube, mask_flow, mask_bg, mask_alias, meta = _load_dataset(run_dir)
        prf = float(meta.get("acquisition", {}).get("prf_hz", meta.get("config", {}).get("prf_hz", 0.0)))
        if prf <= 0.0:
            raise ValueError(f"{run_dir}: dataset meta missing prf_hz")
        seed = int(meta.get("config", {}).get("seed", meta.get("seed", 0)) or 0)
        run_key = str(run_dir.name)
        alias_in_flow = int((mask_flow & mask_alias).sum()) if mask_alias is not None else 0
        alias_frac_in_flow = float(alias_in_flow / max(1, int(mask_flow.sum()))) if mask_alias is not None else None
        details["runs"][run_key] = {
            "run_dir": str(run_dir),
            "dataset_meta_rel": "dataset/meta.json",
            "prf_hz": prf,
            "seed": seed,
            "shape": list(Icube.shape),
            "alias_in_flow": alias_in_flow if mask_alias is not None else None,
            "alias_frac_in_flow": alias_frac_in_flow,
        }

        label_modes: list[tuple[str, np.ndarray]] = [("flow_all", mask_flow)]
        if mask_alias is not None and bool((mask_flow & mask_alias).any()):
            label_modes.append(("flow_unaliased", mask_flow & (~mask_alias)))
            label_modes.append(("flow_aliased", mask_flow & mask_alias))

        for method in methods:
            name_parts = [run_dir.name]
            if tag:
                name_parts.append(tag)
            name_parts.append(method.key)
            dataset_name = "_".join(name_parts)
            bundle_dir = _derive_bundle(
                out_root=out_root / run_dir.name,
                dataset_name=dataset_name,
                Icube=Icube,
                prf_hz=prf,
                seed=seed,
                mask_flow=mask_flow,
                mask_bg=mask_bg,
                method=method,
                stap_device=stap_device,
                tile_hw=tile_hw,
                tile_stride=tile_stride,
                Lt=Lt,
                svd_energy_frac=svd_energy_frac,
                diag_load=diag_load,
            )

            score_path: Path
            if eval_score == "pd":
                if method.role == "baseline":
                    candidates = [bundle_dir / "score_pd_base.npy", bundle_dir / "score_base.npy"]
                else:
                    candidates = [bundle_dir / "score_pd_stap.npy", bundle_dir / "score_pd_base.npy"]
            else:
                if method.role == "baseline":
                    candidates = [bundle_dir / "score_base.npy"]
                else:
                    candidates = [bundle_dir / "score_stap_preka.npy", bundle_dir / "score_stap.npy"]
            score_path = next((p for p in candidates if p.is_file()), candidates[0])

            score = np.load(score_path).astype(np.float32, copy=False)
            for label_mode, pos_mask in label_modes:
                metrics = _eval_score_map(score, pos_mask, mask_bg, fprs=fprs)
                row = {
                    "run": run_key,
                    "method": method.key,
                    "baseline_type": method.baseline_type,
                    "run_stap": int(method.run_stap),
                    "role": method.role,
                    "eval_score": eval_score,
                    "label_mode": label_mode,
                    "bundle_dir": str(bundle_dir),
                    "score_file": str(score_path.name),
                    "T": int(Icube.shape[0]),
                    "H": int(Icube.shape[1]),
                    "W": int(Icube.shape[2]),
                    "prf_hz": prf,
                    "Lt": Lt,
                    "tile_hw": f"{tile_hw[0]}x{tile_hw[1]}",
                    "tile_stride": tile_stride,
                    "diag_load": diag_load,
                    "svd_energy_frac": svd_energy_frac,
                    "alias_in_flow": alias_in_flow if mask_alias is not None else None,
                    "alias_frac_in_flow": alias_frac_in_flow,
                }
                row.update(metrics)
                rows.append(row)

                details["runs"][run_key].setdefault("methods", {}).setdefault(method.key, {})[label_mode] = {
                    "baseline_type": method.baseline_type,
                    "run_stap": bool(method.run_stap),
                    "role": method.role,
                    "eval_score": eval_score,
                    "label_mode": label_mode,
                    "bundle_dir": str(bundle_dir),
                    "score_file": str(score_path.name),
                    "metrics": metrics,
                }

    _write_csv(out_csv, rows)
    _write_json(out_json, {"rows": rows, "details": details})
    print(f"[icube-baseline-compare] wrote {out_csv}")
    print(f"[icube-baseline-compare] wrote {out_json}")


if __name__ == "__main__":
    main()
