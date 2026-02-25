#!/usr/bin/env python3
"""
Cross-mask audit for Shin RatBrain baseline-matrix runs.

Goal: reduce "proxy gaming" concerns by evaluating all methods under
multiple deterministic proxy-mask sources.

Currently supports:
  - mask_src=mc_svd: use the stored masks from the MC-SVD+STAP bundle
  - mask_src=hosvd: derive pd_auto masks from the HOSVD baseline score map
  - mask_src=consensus3: majority-vote flow mask from pd_auto masks of
    (MC-SVD, RPCA, HOSVD) baseline score maps (then bg = default_bg & ~flow)

Outputs a long-form CSV with one row per (scenario, mask_src, method).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sim.kwave import common as kw
from sim.kwave.icube_bundle import _default_masks_generic

from scripts.shin_ratbrain_baseline_matrix import _auc_pos_vs_neg, _tail_metrics


def _load(path: Path) -> np.ndarray:
    return np.load(str(path), allow_pickle=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-mask audit for Shin baseline matrix CSV.")
    ap.add_argument(
        "--in-csv",
        type=Path,
        default=Path("reports/shin_ratbrain_baseline_matrix_shinU_e970_Lt64_nomaskunion_k80.csv"),
        help="Input CSV produced by scripts/shin_ratbrain_baseline_matrix.py.",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/shin_ratbrain_crossmask_audit.csv"),
        help="Output CSV path (long-form).",
    )
    ap.add_argument(
        "--mask-srcs",
        type=str,
        default="mc_svd,consensus3",
        help="Comma-separated mask sources to evaluate (default: %(default)s).",
    )
    ap.add_argument(
        "--alphas",
        type=str,
        default="1e-1,1e-2,3e-3,1e-3,3e-4,1e-4",
        help="Comma-separated alpha grid (must match CSV columns) (default: %(default)s).",
    )
    ap.add_argument("--connectivity", type=int, default=4, choices=[4, 8])
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    if df.empty:
        raise SystemExit(f"Empty CSV: {in_csv}")

    mask_srcs = [s.strip() for s in str(args.mask_srcs).split(",") if s.strip()]
    alphas = [float(s.strip()) for s in str(args.alphas).split(",") if s.strip()]
    if not mask_srcs:
        raise SystemExit("No mask sources specified.")
    if not alphas:
        raise SystemExit("No alphas specified.")

    # One row per scenario in the input corresponds to one method; collapse to scenarios.
    scenarios = (
        df[["iq_file", "frames_spec", "bundle_dir_stap", "bundle_dir_hosvd"]]
        .drop_duplicates()
        .to_dict(orient="records")
    )
    if not scenarios:
        raise SystemExit("No scenarios found in input CSV.")

    methods = [
        ("mc_svd", "score_base.npy", "bundle_dir_stap"),
        ("rpca", "score_base.npy", "bundle_dir_rpca"),
        ("hosvd", "score_base.npy", "bundle_dir_hosvd"),
        ("stap", "score_stap_preka.npy", "bundle_dir_stap"),
        ("stap_raw", "score_stap_preka.npy", "bundle_dir_stap_raw"),
    ]

    rows: list[dict[str, Any]] = []

    # Pre-index by scenario to locate method bundle directories.
    for scen in scenarios:
        iq_file = str(scen["iq_file"])
        frames_spec = str(scen["frames_spec"])

        df_s = df[(df["iq_file"] == iq_file) & (df["frames_spec"] == frames_spec)].copy()
        if df_s.empty:
            continue

        # Bundle dirs are repeated in each method row; pull from any row.
        dirs = {}
        for col in ["bundle_dir_stap", "bundle_dir_stap_raw", "bundle_dir_rpca", "bundle_dir_hosvd"]:
            if col in df_s.columns:
                v = df_s[col].dropna().astype(str)
                if not v.empty:
                    dirs[col] = Path(v.iloc[0])

        if "bundle_dir_stap" not in dirs or "bundle_dir_hosvd" not in dirs:
            continue

        # Load the canonical stored masks (mc_svd source).
        mask_flow_mcsvd = _load(dirs["bundle_dir_stap"] / "mask_flow.npy").astype(bool, copy=False)
        mask_bg_mcsvd = _load(dirs["bundle_dir_stap"] / "mask_bg.npy").astype(bool, copy=False)

        # Derive alternative masks.
        # For HOSVD source, recompute pd_auto from the HOSVD baseline score map (same defaults).
        H, W = mask_flow_mcsvd.shape
        mask_flow_default, mask_bg_default = _default_masks_generic(H, W)

        masks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for src in mask_srcs:
            if src == "mc_svd":
                masks[src] = (mask_flow_mcsvd, mask_bg_mcsvd)
            elif src == "hosvd":
                score_hosvd = _load(dirs["bundle_dir_hosvd"] / "score_base.npy")
                mf, mb, _stats = kw._resolve_flow_mask(
                    score_hosvd,
                    mask_flow_default,
                    mask_bg_default,
                    mode="pd_auto",
                    pd_quantile=0.995,
                    depth_min_frac=0.25,
                    depth_max_frac=0.85,
                    erode_iters=0,
                    dilate_iters=2,
                    min_pixels=64,
                    min_coverage_frac=0.0,
                    union_with_default=False,
                )
                masks[src] = (mf.astype(bool, copy=False), mb.astype(bool, copy=False))
            elif src == "consensus3":
                # Majority-vote proxy: build pd_auto flow masks from each baseline score map,
                # then vote >= 2. This avoids a single method defining the proxy labels.
                score_mcsvd = _load(dirs["bundle_dir_stap"] / "score_base.npy")
                score_rpca = _load(dirs["bundle_dir_rpca"] / "score_base.npy") if "bundle_dir_rpca" in dirs else None
                score_hosvd = _load(dirs["bundle_dir_hosvd"] / "score_base.npy")
                if score_rpca is None:
                    raise SystemExit("consensus3 requires bundle_dir_rpca in the input CSV")
                m1, _ = kw._flow_mask_from_pd(
                    score_mcsvd,
                    percentile=0.995,
                    depth_min_frac=0.25,
                    depth_max_frac=0.85,
                    erode_iters=0,
                    dilate_iters=2,
                )
                m2, _ = kw._flow_mask_from_pd(
                    score_rpca,
                    percentile=0.995,
                    depth_min_frac=0.25,
                    depth_max_frac=0.85,
                    erode_iters=0,
                    dilate_iters=2,
                )
                m3, _ = kw._flow_mask_from_pd(
                    score_hosvd,
                    percentile=0.995,
                    depth_min_frac=0.25,
                    depth_max_frac=0.85,
                    erode_iters=0,
                    dilate_iters=2,
                )
                mf = (m1.astype(np.int8) + m2.astype(np.int8) + m3.astype(np.int8)) >= 2
                mb = mask_bg_default & (~mf)
                if int(mb.sum()) < 64:
                    mb = (~mf).copy()
                masks[src] = (mf.astype(bool, copy=False), mb.astype(bool, copy=False))
            else:
                raise SystemExit(f"Unknown mask source: {src!r}")

        # Evaluate all methods under each mask source.
        for mask_src, (mask_flow, mask_bg) in masks.items():
            # Precompute n_bg/n_flow and tail floor once.
            n_flow = int(mask_flow.sum())
            n_bg = int(mask_bg.sum())
            fpr_min = (1.0 / float(n_bg)) if n_bg > 0 else float("nan")

            for method_key, score_name, dir_col in methods:
                if dir_col not in dirs:
                    continue
                score_path = dirs[dir_col] / score_name
                if not score_path.is_file():
                    continue
                score = _load(score_path)
                auc = _auc_pos_vs_neg(score[mask_flow], score[mask_bg])

                row: dict[str, Any] = {
                    "iq_file": iq_file,
                    "frames_spec": frames_spec,
                    "mask_src": mask_src,
                    "method_key": method_key,
                    "auc_flow_vs_bg": auc,
                    "n_flow": n_flow,
                    "n_bg": n_bg,
                    "fpr_min": fpr_min,
                }
                row.update(
                    _tail_metrics(
                        score=score,
                        mask_flow=mask_flow,
                        mask_bg=mask_bg,
                        alphas=alphas,
                        connectivity=int(args.connectivity),
                    )
                )
                rows.append(row)

    if not rows:
        raise SystemExit("No rows written (check input CSV and bundle paths).")

    # Write CSV in stable key order.
    fieldnames: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"[shin-crossmask] wrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
