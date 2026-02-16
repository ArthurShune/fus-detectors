#!/usr/bin/env python3
"""
Sweep Macé/Urban scans and planes and summarize tile-level ROC metrics.

For each scan and coronal plane, this script:
  - maps PD pixels into Allen atlas coordinates,
  - labels 8x8 tiles as H1/H0 using atlas ROIs (VIS/SC/LGd vs CP/CA/DG),
  - computes tile-level scores:
      * PD-z (max z-score over time of tile-averaged PD),
      * hemo-STAP BR and GLRT (hemodynamic STAP on tile-averaged PD),
      * GLM t-stat (minimal GLM cross-check),
  - and records ROC summaries for each score:
      partial AUC at FPR <= 0.05 and TPR at empirical FPR_min = 1/N_neg.

Results are written to a CSV file with one row per (scan, plane, score_name).

Usage
-----
    PYTHONPATH=. python scripts/mace_tiles_sweep.py \
        --out-csv reports/mace_mini_roc.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from eval.metrics import partial_auc, roc_curve, tpr_at_fpr_target
from pipeline.realdata import extract_tile_stack, iter_pd_slices, tile_iter
from pipeline.realdata.mace_wholebrain import (
    MaceAtlas,
    MaceRegionInfo,
    build_mace_transform_matrix,
    load_all_mace_scans,
    load_mace_atlas,
    load_mace_region_info,
    load_mace_transform,
    scan_plane_to_atlas_indices,
)
from pipeline.stap.hemo import HemoStapConfig, hemo_stap_scores_for_tiles


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sweep Macé tiles and summarize ROC metrics.")
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root directory containing whole-brain-fUS (defaults to data/whole-brain-fUS).",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("mace_tile_roc_summary.csv"),
        help="Output CSV path for per-plane ROC summary.",
    )
    ap.add_argument(
        "--tile-h",
        type=int,
        default=8,
        help="Tile height (pixels).",
    )
    ap.add_argument(
        "--tile-w",
        type=int,
        default=8,
        help="Tile width (pixels).",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=3,
        help="Tile stride (pixels).",
    )
    ap.add_argument(
        "--label-alpha",
        type=float,
        default=0.6,
        help="Minimum fraction of labeled pixels (H1 or H0) required to assign a tile label.",
    )
    return ap.parse_args()


def _default_pos_prefixes() -> List[str]:
    return ["VIS", "SC", "LGd"]


def _default_neg_prefixes() -> List[str]:
    return ["CP", "CA", "DG"]


def _build_label_sets(
    region_info: MaceRegionInfo,
    pos_prefixes: Sequence[str],
    neg_prefixes: Sequence[str],
) -> Tuple[List[int], List[int], Dict[str, List[str]]]:
    acr_list = region_info.acronyms
    label_for_acr = region_info.label_for_acr

    def select_by_prefix(prefixes: Sequence[str]) -> List[str]:
        out: List[str] = []
        for acr in acr_list:
            for p in prefixes:
                if acr.startswith(p):
                    out.append(acr)
                    break
        seen: Dict[str, None] = {}
        uniq: List[str] = []
        for a in out:
            if a not in seen:
                seen[a] = None
                uniq.append(a)
        return uniq

    pos_acr = select_by_prefix(pos_prefixes)
    neg_acr = select_by_prefix(neg_prefixes)

    pos_labels = [label_for_acr[a] for a in pos_acr if a in label_for_acr]
    neg_labels = [label_for_acr[a] for a in neg_acr if a in label_for_acr]

    info = {"pos_acr": pos_acr, "neg_acr": neg_acr}
    return pos_labels, neg_labels, info


def _atlas_masks_from_labels(
    atlas: MaceAtlas,
    pos_labels: Sequence[int],
    neg_labels: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    regions = atlas.regions
    pos_mask = np.isin(regions, np.asarray(pos_labels, dtype=regions.dtype))
    neg_mask = np.isin(regions, np.asarray(neg_labels, dtype=regions.dtype))
    neg_mask = neg_mask & ~pos_mask
    return pos_mask, neg_mask


def _tile_labels_from_pixel_masks(
    H: int,
    W: int,
    pos_mask_flat: np.ndarray,
    neg_mask_flat: np.ndarray,
    tile_hw: Tuple[int, int],
    stride: int,
    alpha: float,
) -> List[int]:
    """
    Compute tile labels (+1 for H1, 0 for H0, -1 for discarded).
    """

    labels: List[int] = []
    th, tw = tile_hw
    pos_img = pos_mask_flat.reshape(H, W)
    neg_img = neg_mask_flat.reshape(H, W)

    for y0, x0 in tile_iter((H, W), tile_hw, stride):
        y1 = y0 + th
        x1 = x0 + tw
        pos_tile = pos_img[y0:y1, x0:x1]
        neg_tile = neg_img[y0:y1, x0:x1]
        n_pos = int(pos_tile.sum())
        n_neg = int(neg_tile.sum())
        n_lab = n_pos + n_neg
        if n_lab == 0:
            labels.append(-1)
            continue
        if n_pos >= alpha * n_lab:
            labels.append(+1)
        elif n_neg >= alpha * n_lab:
            labels.append(0)
        else:
            labels.append(-1)
    return labels


def _pd_z_score(p_t: np.ndarray, eps: float = 1e-12) -> float:
    mu = float(p_t.mean())
    sigma = float(p_t.std())
    sigma_safe = sigma if sigma > 0.0 else 1.0
    z = (p_t - mu) / sigma_safe
    return float(z.max())


def _glm_boxcar_tstat(p_t: np.ndarray, eps: float = 1e-12) -> float:
    y = p_t.astype(np.float64)
    T = y.size
    if T < 5:
        return 0.0
    t_idx = np.arange(T, dtype=np.float64)
    intercept = np.ones_like(t_idx)
    trend = (t_idx - t_idx.mean()) / (t_idx.std() + eps)
    t0 = int(0.3 * T)
    t1 = int(0.7 * T)
    box = np.zeros_like(t_idx)
    box[t0:t1] = 1.0
    X = np.stack([intercept, trend, box], axis=1)
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.pinv(XtX)
    except np.linalg.LinAlgError:
        return 0.0
    beta = XtX_inv @ (X.T @ y)
    y_hat = X @ beta
    resid = y - y_hat
    dof = max(T - X.shape[1], 1)
    sigma2 = float((resid @ resid) / float(dof))
    var_beta = np.diag(XtX_inv) * sigma2
    if not np.isfinite(var_beta[2]) or var_beta[2] <= 0.0:
        return 0.0
    t_stat = beta[2] / np.sqrt(var_beta[2])
    return float(t_stat)


def _pd_z_score_baseline(p_t: np.ndarray, baseline_frac: float = 0.3, eps: float = 1e-12) -> float:
    """
    PD-z score using an early baseline segment for mean/std.

    The first baseline_frac of frames are used to estimate μ/σ; the score
    is the maximum z over the remaining frames.
    """

    T = p_t.size
    if T <= 1:
        return 0.0
    n_base = max(1, int(baseline_frac * T))
    n_base = min(n_base, T - 1)
    base = p_t[:n_base]
    mu = float(base.mean())
    sigma = float(base.std())
    sigma_safe = sigma if sigma > 0.0 else 1.0
    z = (p_t - mu) / sigma_safe
    z_det = z[n_base:]
    if z_det.size == 0:
        z_det = z
    return float(z_det.max())


def main() -> None:
    args = parse_args()
    from pipeline.realdata import mace_data_root

    data_root = mace_data_root() if args.data_root is None else args.data_root
    out_csv = args.out_csv

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    atlas = load_mace_atlas(data_root)
    region_info = load_mace_region_info(data_root)
    transf = load_mace_transform(data_root)
    A, t = build_mace_transform_matrix(transf)

    pos_prefixes = _default_pos_prefixes()
    neg_prefixes = _default_neg_prefixes()
    pos_labels, neg_labels, _ = _build_label_sets(region_info, pos_prefixes, neg_prefixes)
    if not pos_labels or not neg_labels:
        raise RuntimeError("Empty positive or negative label set; adjust prefixes.")

    pos_mask_atlas, neg_mask_atlas = _atlas_masks_from_labels(atlas, pos_labels, neg_labels)

    scans = load_all_mace_scans(data_root)
    if not scans:
        raise RuntimeError("No Macé scans found.")

    tile_hw = (int(args.tile_h), int(args.tile_w))
    stride = int(args.stride)
    alpha = float(args.label_alpha)

    hemo_cfg = HemoStapConfig(
        dt=0.1,
        L_t=40,
        pf_band=(0.0, 0.5),
        pa_band=(1.0, 3.0),
        pg_band=(0.5, 1.0),
        bg_beta=0.0,
    )

    fieldnames = [
        "scan_idx",
        "scan_name",
        "plane_idx",
        "n_pos_tiles",
        "n_neg_tiles",
        "score_name",
        "partial_auc_fpr_le_1e-3",
        "tpr_at_empirical_fpr_min",
        "fpr_min",
    ]

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for s_idx, scan in enumerate(scans):
            for z in range(scan.n_planes):
                # PD plane and mapping to atlas
                _, pd_T_HW = list(iter_pd_slices(scan))[z]
                T_pd, H, W = pd_T_HW.shape
                Ha, Wa, Za = atlas.regions.shape
                i_dv, i_ap, i_lr, inside = scan_plane_to_atlas_indices(H, W, z, A, t, (Ha, Wa, Za))

                pd_flat = pd_T_HW.mean(axis=0).ravel()
                pos_mask_flat = np.zeros_like(pd_flat, dtype=bool)
                neg_mask_flat = np.zeros_like(pd_flat, dtype=bool)

                inside_idx = np.nonzero(inside)[0]
                atlas_pos = pos_mask_atlas[i_dv[inside_idx], i_ap[inside_idx], i_lr[inside_idx]]
                atlas_neg = neg_mask_atlas[i_dv[inside_idx], i_ap[inside_idx], i_lr[inside_idx]]
                pos_mask_flat[inside_idx] = atlas_pos
                neg_mask_flat[inside_idx] = atlas_neg
                neg_mask_flat = neg_mask_flat & ~pos_mask_flat

                labels = _tile_labels_from_pixel_masks(
                    H, W, pos_mask_flat, neg_mask_flat, tile_hw, stride, alpha
                )

                # Collect tile time-series and scores for labeled tiles.
                tile_series: List[np.ndarray] = []
                pdz_scores: List[float] = []
                glm_scores: List[float] = []
                lbl_filtered: List[int] = []

                for idx, (y0, x0) in enumerate(tile_iter((H, W), tile_hw, stride)):
                    label = labels[idx]
                    if label not in (0, 1):
                        continue
                    tile = extract_tile_stack(pd_T_HW, y0, x0, tile_hw)
                    p_t = tile.mean(axis=(1, 2))
                    tile_series.append(p_t.astype(np.float32))
                    pdz_scores.append(_pd_z_score_baseline(p_t))
                    glm_scores.append(_glm_boxcar_tstat(p_t))
                    lbl_filtered.append(label)

                if not tile_series:
                    continue

                p_tiles = np.stack(tile_series, axis=0)
                pdz_scores = np.asarray(pdz_scores, dtype=np.float64)
                glm_scores = np.asarray(glm_scores, dtype=np.float64)
                labels_arr = np.asarray(lbl_filtered, dtype=int)
                hemo_scores = hemo_stap_scores_for_tiles(p_tiles, hemo_cfg)
                hemo_br_scores = hemo_scores["hemo_br"].astype(np.float64)
                hemo_glrt_scores = hemo_scores["hemo_glrt"].astype(np.float64)

                pos_mask_tiles = labels_arr == 1
                neg_mask_tiles = labels_arr == 0
                n_pos_tiles = int(pos_mask_tiles.sum())
                n_neg_tiles = int(neg_mask_tiles.sum())
                if n_pos_tiles == 0 or n_neg_tiles == 0:
                    continue

                for name, scores in (
                    ("pd_z", pdz_scores),
                    ("hemo_br", hemo_br_scores),
                    ("hemo_glrt", hemo_glrt_scores),
                    ("glm_t", glm_scores),
                ):
                    s_pos = scores[pos_mask_tiles]
                    s_neg = scores[neg_mask_tiles]
                    fpr, tpr, _ = roc_curve(s_pos, s_neg, num_thresh=4096)
                    fpr_min = 1.0 / float(n_neg_tiles)
                    fpr_min_clipped = float(np.clip(fpr_min, 1e-8, 1.0))
                    pauc_max = float(max(0.05, fpr_min_clipped))
                    auc = partial_auc(fpr, tpr, fpr_max=pauc_max)
                    tpr_emp = tpr_at_fpr_target(fpr, tpr, target_fpr=fpr_min_clipped)
                    writer.writerow(
                        {
                            "scan_idx": s_idx,
                            "scan_name": scan.name,
                            "plane_idx": z,
                            "n_pos_tiles": n_pos_tiles,
                            "n_neg_tiles": n_neg_tiles,
                            "score_name": name,
                            "partial_auc_fpr_le_0.05": auc,
                            "pauc_max": pauc_max,
                            "tpr_at_empirical_fpr_min": tpr_emp,
                            "fpr_min": fpr_min_clipped,
                        }
                    )


if __name__ == "__main__":
    main()
