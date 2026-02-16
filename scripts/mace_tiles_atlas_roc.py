#!/usr/bin/env python3
"""
Tile-level PD ROC against Allen atlas ROIs for the Macé/Urban dataset.

This script moves the Macé real-data harness closer to the clinical profile by:
  - operating on 8x8 tiles with a configurable stride,
  - labeling tiles as H1/H0 using atlas-derived ROI masks, and
  - computing ROC curves for three tile-level scores:
      * PD-z: max z-score over time of tile-averaged PD,
      * hemo-STAP: hemodynamic band-ratio and GLRT-style scores from a
        miniature temporal STAP core on tile-averaged PD,
      * GLM: a minimal GLM t-stat on the same PD time-series as a cross-check.

ROI sets are defined heuristically via acronym prefixes:
  - positives: acronyms starting with VIS, SC, LGd (visual cortex, SC, LGN),
  - negatives: acronyms starting with CP, CA, DG (striatum/hippocampus).

Usage
-----
    PYTHONPATH=. python scripts/mace_tiles_atlas_roc.py \
        --scan-index 0 \
        --plane-index 10
"""

from __future__ import annotations

import argparse
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
    ap = argparse.ArgumentParser(
        description="Tile-level PD ROC vs atlas ROIs on Macé whole-brain fUS"
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root directory containing whole-brain-fUS (defaults to data/whole-brain-fUS).",
    )
    ap.add_argument(
        "--scan-index",
        type=int,
        default=0,
        help="Index of scan to analyze (0-based).",
    )
    ap.add_argument(
        "--plane-index",
        type=int,
        default=10,
        help="Coronal plane index (0-based) to analyze.",
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
    ap.add_argument(
        "--pos-prefix",
        action="append",
        default=None,
        help="Acronym prefix for positive ROIs (can be given multiple times).",
    )
    ap.add_argument(
        "--neg-prefix",
        action="append",
        default=None,
        help="Acronym prefix for negative ROIs (can be given multiple times).",
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
) -> Tuple[List[int], List[float]]:
    """
    Compute tile labels and coverage fractions.

    Returns
    -------
    labels : list[int]
        +1 for H1 tiles, 0 for H0 tiles, -1 for discarded tiles.
    coverages : list[float]
        Fraction of labeled pixels (H1 or H0) contributing to each tile.
    """

    labels: List[int] = []
    coverages: List[float] = []
    th, tw = tile_hw
    pos_mask_img = pos_mask_flat.reshape(H, W)
    neg_mask_img = neg_mask_flat.reshape(H, W)

    for y0, x0 in tile_iter((H, W), tile_hw, stride):
        y1 = y0 + th
        x1 = x0 + tw
        pos_tile = pos_mask_img[y0:y1, x0:x1]
        neg_tile = neg_mask_img[y0:y1, x0:x1]
        n_pos = int(pos_tile.sum())
        n_neg = int(neg_tile.sum())
        n_lab = n_pos + n_neg
        if n_lab == 0:
            labels.append(-1)
            coverages.append(0.0)
            continue
        frac = n_lab / float(th * tw)
        coverages.append(frac)
        if n_pos >= alpha * n_lab:
            labels.append(+1)
        elif n_neg >= alpha * n_lab:
            labels.append(0)
        else:
            labels.append(-1)
    return labels, coverages


def _pd_z_score(p_t: np.ndarray, baseline_frac: float = 0.3, eps: float = 1e-12) -> float:
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


def _hemo_band_ratio(
    p_t: np.ndarray,
    dt: float,
    pf_lo: float = 0.05,
    pf_hi: float = 0.3,
    pa_lo: float = 0.6,
    pa_hi: float = 1.5,
    eps: float = 1e-12,
) -> float:
    """
    Simple hemodynamic band-ratio score based on the PSD of the tile-averaged PD.
    """

    x = p_t - float(p_t.mean())
    T = x.size
    if T <= 1:
        return 0.0
    freqs = np.fft.rfftfreq(T, d=dt)
    fft_vals = np.fft.rfft(x)
    psd = (fft_vals.conj() * fft_vals).real
    pf_mask = (freqs >= pf_lo) & (freqs <= pf_hi)
    pa_mask = (freqs >= pa_lo) & (freqs <= pa_hi)
    E_pf = float(psd[pf_mask].sum()) if pf_mask.any() else 0.0
    E_pa = float(psd[pa_mask].sum()) if pa_mask.any() else 0.0
    return E_pf / (E_pa + eps)


def _glm_boxcar_tstat(p_t: np.ndarray, eps: float = 1e-12) -> float:
    """
    Minimal GLM cross-check: intercept + linear trend + broad boxcar regressor.

    The boxcar covers the central portion of the time-series (roughly one
    stimulus window), and we return the t-statistic associated with that
    regressor.
    """

    y = p_t.astype(np.float64)
    T = y.size
    if T < 5:
        return 0.0
    t_idx = np.arange(T, dtype=np.float64)
    # Design: intercept, linear trend, central boxcar
    intercept = np.ones_like(t_idx)
    trend = (t_idx - t_idx.mean()) / (t_idx.std() + eps)
    # Boxcar over central 40% of frames
    t0 = int(0.3 * T)
    t1 = int(0.7 * T)
    box = np.zeros_like(t_idx)
    box[t0:t1] = 1.0
    X = np.stack([intercept, trend, box], axis=1)  # (T,3)
    # OLS fit
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


def main() -> None:
    args = parse_args()
    from pipeline.realdata import mace_data_root

    data_root = mace_data_root() if args.data_root is None else args.data_root

    # Atlas + transform
    atlas = load_mace_atlas(data_root)
    region_info = load_mace_region_info(data_root)
    transf = load_mace_transform(data_root)
    A, t = build_mace_transform_matrix(transf)

    pos_prefixes = args.pos_prefix or _default_pos_prefixes()
    neg_prefixes = args.neg_prefix or _default_neg_prefixes()
    pos_labels, neg_labels, info = _build_label_sets(region_info, pos_prefixes, neg_prefixes)
    if not pos_labels or not neg_labels:
        raise RuntimeError("Empty positive or negative label set; adjust prefixes.")

    pos_mask_atlas, neg_mask_atlas = _atlas_masks_from_labels(atlas, pos_labels, neg_labels)

    # Scan and plane
    scans = load_all_mace_scans(data_root)
    if not scans:
        raise RuntimeError("No Macé scans found.")
    if args.scan_index < 0 or args.scan_index >= len(scans):
        raise ValueError(f"scan_index {args.scan_index} out of range for {len(scans)} scans")
    scan = scans[args.scan_index]
    if args.plane_index < 0 or args.plane_index >= scan.n_planes:
        raise ValueError(f"plane_index {args.plane_index} out of range for {scan.n_planes} planes")

    _, pd_T_HW = list(iter_pd_slices(scan))[args.plane_index]
    T, H, W = pd_T_HW.shape

    # Map plane to atlas
    Ha, Wa, Za = atlas.regions.shape
    i_dv, i_ap, i_lr, inside = scan_plane_to_atlas_indices(
        H, W, args.plane_index, A, t, (Ha, Wa, Za)
    )

    # Build pixel-level masks on the scan grid
    pd_flat = pd_T_HW.mean(axis=0).ravel()
    pos_mask_flat = np.zeros_like(pd_flat, dtype=bool)
    neg_mask_flat = np.zeros_like(pd_flat, dtype=bool)

    inside_idx = np.nonzero(inside)[0]
    atlas_pos = pos_mask_atlas[i_dv[inside_idx], i_ap[inside_idx], i_lr[inside_idx]]
    atlas_neg = neg_mask_atlas[i_dv[inside_idx], i_ap[inside_idx], i_lr[inside_idx]]

    pos_mask_flat[inside_idx] = atlas_pos
    neg_mask_flat[inside_idx] = atlas_neg
    neg_mask_flat = neg_mask_flat & ~pos_mask_flat

    # Tile labels
    tile_hw = (int(args.tile_h), int(args.tile_w))
    stride = int(args.stride)
    alpha = float(args.label_alpha)

    labels, coverages = _tile_labels_from_pixel_masks(
        H, W, pos_mask_flat, neg_mask_flat, tile_hw, stride, alpha
    )

    # Tile scores and series
    pdz_scores: List[float] = []
    glm_scores: List[float] = []
    tile_series: List[np.ndarray] = []
    lbl_filtered: List[int] = []

    for idx, (y0, x0) in enumerate(tile_iter((H, W), tile_hw, stride)):
        label = labels[idx]
        if label not in (0, 1):
            continue
        tile = extract_tile_stack(pd_T_HW, y0, x0, tile_hw)  # (T, th, tw)
        p_t = tile.mean(axis=(1, 2))
        tile_series.append(p_t.astype(np.float32))
        pdz = _pd_z_score(p_t)
        # Minimal GLM cross-check: intercept + linear trend + broad boxcar.
        glm = _glm_boxcar_tstat(p_t)
        pdz_scores.append(pdz)
        glm_scores.append(glm)
        lbl_filtered.append(label)

    pdz_scores = np.asarray(pdz_scores, dtype=np.float64)
    glm_scores = np.asarray(glm_scores, dtype=np.float64)
    labels_arr = np.asarray(lbl_filtered, dtype=int)

    # Hemodynamic STAP scores on tile-averaged PD (no background blending in main analysis)
    if tile_series:
        p_tiles = np.stack(tile_series, axis=0)  # (N_tiles, T_pd)
        hemo_cfg = HemoStapConfig(
            dt=float(scan.dt),
            L_t=40,
            pf_band=(0.0, 0.5),
            pa_band=(1.0, 3.0),
            pg_band=(0.5, 1.0),
            bg_beta=0.0,
        )
        hemo_scores = hemo_stap_scores_for_tiles(p_tiles, hemo_cfg)
        hemo_br_scores = hemo_scores["hemo_br"].astype(np.float64)
        hemo_glrt_scores = hemo_scores["hemo_glrt"].astype(np.float64)
    else:
        hemo_br_scores = np.zeros((0,), dtype=np.float64)
        hemo_glrt_scores = np.zeros((0,), dtype=np.float64)

    pos_mask_tiles = labels_arr == 1
    neg_mask_tiles = labels_arr == 0

    n_pos_tiles = int(pos_mask_tiles.sum())
    n_neg_tiles = int(neg_mask_tiles.sum())
    if n_pos_tiles == 0 or n_neg_tiles == 0:
        raise RuntimeError(
            f"No positive ({n_pos_tiles}) or negative ({n_neg_tiles}) tiles after labeling."
        )

    def _roc_summary(scores: np.ndarray, name: str) -> None:
        s_pos = scores[pos_mask_tiles]
        s_neg = scores[neg_mask_tiles]
        fpr, tpr, _ = roc_curve(s_pos, s_neg, num_thresh=4096)
        fpr_min = 1.0 / float(n_neg_tiles)
        fpr_min_clipped = float(np.clip(fpr_min, 1e-8, 1.0))
        pauc_max = float(max(0.05, fpr_min_clipped))
        auc = partial_auc(fpr, tpr, fpr_max=pauc_max)
        tpr_emp = tpr_at_fpr_target(fpr, tpr, target_fpr=fpr_min_clipped)
        print(f"  [{name}] partial AUC (FPR<={pauc_max:.3f}): {auc:.4f}")
        print(f"  [{name}] TPR at empirical FPR_min={fpr_min_clipped:.3e}: {tpr_emp:.4f}")

    frac_inside = float(inside.mean())
    print(f"Macé scan '{scan.name}', plane {args.plane_index}:")
    print(f"  PD shape T,H,W = ({T}, {H}, {W}), tile_hw={tile_hw}, stride={stride}")
    print(f"  Atlas shape DV,AP,LR = {atlas.regions.shape}")
    print(f"  Transform in-bounds fraction: {frac_inside:.3f}")
    print(f"  Positive ROI prefixes: {pos_prefixes}")
    print(f"  Negative ROI prefixes: {neg_prefixes}")
    print(f"  Positive acronyms (first 10): {info['pos_acr'][:10]}")
    print(f"  Negative acronyms (first 10): {info['neg_acr'][:10]}")
    print(f"  Tiles labeled H1 = {n_pos_tiles}, H0 = {n_neg_tiles}")

    _roc_summary(pdz_scores, "PD-z")
    _roc_summary(hemo_br_scores, "hemo-STAP BR")
    _roc_summary(hemo_glrt_scores, "hemo-STAP GLRT")
    _roc_summary(glm_scores, "GLM")


if __name__ == "__main__":
    main()
