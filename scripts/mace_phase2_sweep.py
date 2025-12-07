#!/usr/bin/env python3
"""
Phase 2.1: systematic Macé/Urban tile-level ROC + alias-gated PD-z sweep.

For each scan and coronal plane with both H1/H0 tiles, this script:
  - maps PD pixels into Allen CCFv3 atlas space,
  - labels 8x8 tiles as H1/H0 using atlas ROIs (VIS/SC/LGd vs CP/CA/DG),
  - computes tile-level scores on tile-averaged PD time series:
      * PD-z (baseline segment, max z over detection segment),
      * hemodynamic STAP band-ratio (BR),
  - computes hemodynamic band energies Ef/Ea/Eg and an alias metric Ea/Ef,
  - applies a fixed alias gate (alias <= global quantile) to PD-z and recomputes ROC,
  - records, per (scan, plane):
      * Pf/Pa/Po peak-band fractions in H1/H0,
      * alias log(Ea/Ef) medians in H1/H0,
      * PD-z and hemo-BR pAUC/TPR at empirical FPR_min,
      * PD-z+alias-gate pAUC/TPR at empirical FPR_min,
      * hit/FP counts at a fixed target TPR before/after gating.

Output
------
CSV with one row per (scan, plane) satisfying n_pos_tiles>0 and n_neg_tiles>0.

Usage
-----
    PYTHONPATH=. python scripts/mace_phase2_sweep.py \
        --out-csv reports/mace_phase2_summary.csv
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
    ap = argparse.ArgumentParser(
        description="Phase 2.1 Macé sweep: PD-z / hemo-STAP ROC and alias-gated PD-z per plane."
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root directory containing whole-brain-fUS (defaults to data/whole-brain-fUS).",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("mace_phase2_summary.csv"),
        help="Output CSV path for per-plane summary.",
    )
    ap.add_argument("--tile-h", type=int, default=8, help="Tile height (pixels).")
    ap.add_argument("--tile-w", type=int, default=8, help="Tile width (pixels).")
    ap.add_argument("--stride", type=int, default=3, help="Tile stride (pixels).")
    ap.add_argument(
        "--label-alpha",
        type=float,
        default=0.6,
        help="Minimum fraction of labeled pixels (H1 or H0) required to assign a tile label.",
    )
    ap.add_argument(
        "--alias-quantile",
        type=float,
        default=0.8,
        help="Alias threshold quantile for PD-z gating (Ea/Ef <= quantile).",
    )
    ap.add_argument(
        "--tpr-target",
        type=float,
        default=0.5,
        help="Target TPR for hit/FP count comparison.",
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


def _roc_summary(pos_scores: np.ndarray, neg_scores: np.ndarray) -> Tuple[float, float, float]:
    """Compute pAUC and TPR@FPR_min given positive/negative score vectors."""
    fpr, tpr, _ = roc_curve(pos_scores, neg_scores, num_thresh=4096)
    pauc = partial_auc(fpr, tpr, fpr_max=1e-3)
    fpr_min = 1.0 / float(len(neg_scores))
    fpr_min_clipped = float(np.clip(fpr_min, 1e-8, 1.0))
    tpr_emp = tpr_at_fpr_target(fpr, tpr, target_fpr=fpr_min_clipped)
    return float(pauc), float(tpr_emp), fpr_min_clipped


def _counts_at_tpr(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    target_tpr: float,
) -> Tuple[int, int, float]:
    """Return (hits_pos, fp_neg, threshold) at the threshold closest to target_tpr."""
    fpr, tpr, thr = roc_curve(pos_scores, neg_scores, num_thresh=None)
    idx = int(np.argmin(np.abs(tpr - target_tpr)))
    thr_val = float(thr[idx])
    hits = int((pos_scores >= thr_val).sum())
    fps = int((neg_scores >= thr_val).sum())
    return hits, fps, thr_val


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
        "frac_pf_peak_pos",
        "frac_pf_peak_neg",
        "median_log_alias_pos",
        "median_log_alias_neg",
        "delta_log_alias",
        "pd_pauc",
        "pd_tpr_at_fpr_min",
        "pd_fpr_min",
        "br_pauc",
        "br_tpr_at_fpr_min",
        "br_fpr_min",
        "alias_quantile",
        "alias_threshold",
        "gate_kept_frac",
        "gated_pd_pauc",
        "gated_pd_tpr_at_fpr_min",
        "gated_pd_fpr_min",
        "tpr_target",
        "pd_hits_at_tpr",
        "pd_fp_at_tpr",
        "gated_pd_hits_at_tpr",
        "gated_pd_fp_at_tpr",
    ]

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for s_idx, scan in enumerate(scans):
            for z, (_, pd_T_HW) in enumerate(iter_pd_slices(scan)):
                T_pd, H, W = pd_T_HW.shape

                # Map pixels in this plane to atlas indices.
                Ha, Wa, Za = atlas.regions.shape
                i_dv, i_ap, i_lr, inside = scan_plane_to_atlas_indices(
                    H, W, z, A, t, (Ha, Wa, Za)
                )

                # Build per-pixel H1/H0 masks in image space.
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

                tile_series: List[np.ndarray] = []
                lbl_filtered: List[int] = []
                pdz_scores: List[float] = []

                for idx, (y0, x0) in enumerate(tile_iter((H, W), tile_hw, stride)):
                    label = labels[idx]
                    if label not in (0, 1):
                        continue
                    tile = extract_tile_stack(pd_T_HW, y0, x0, tile_hw)
                    p_t = tile.mean(axis=(1, 2)).astype(np.float32)
                    tile_series.append(p_t)
                    lbl_filtered.append(label)
                    pdz_scores.append(_pd_z_score_baseline(p_t))

                if not tile_series:
                    continue

                p_tiles = np.stack(tile_series, axis=0)
                labels_arr = np.asarray(lbl_filtered, dtype=int)
                pdz_scores_arr = np.asarray(pdz_scores, dtype=np.float64)

                pos_mask_tiles = labels_arr == 1
                neg_mask_tiles = labels_arr == 0
                n_pos_tiles = int(pos_mask_tiles.sum())
                n_neg_tiles = int(neg_mask_tiles.sum())
                if n_pos_tiles == 0 or n_neg_tiles == 0:
                    continue

                # Hemodynamic STAP band energies and BR scores.
                hemo_scores = hemo_stap_scores_for_tiles(p_tiles, hemo_cfg)
                Ef = hemo_scores["Ef"]
                Ea = hemo_scores["Ea"]
                Eg = hemo_scores["Eg"]
                hemo_br = hemo_scores["hemo_br"].astype(np.float64)

                alias = (Ea + 1e-8) / (Ef + 1e-8)
                alias_log = np.log(alias + 1e-12)
                bands = np.stack([Ef, Ea, Eg], axis=1)
                peak_idx = np.argmax(bands, axis=1)  # 0=Pf, 1=Pa, 2=Po

                def _frac(mask: np.ndarray, idx: np.ndarray, val: int) -> float:
                    subset = idx[mask]
                    return float((subset == val).mean()) if subset.size else 0.0

                frac_pf_peak_pos = _frac(pos_mask_tiles, peak_idx, 0)
                frac_pf_peak_neg = _frac(neg_mask_tiles, peak_idx, 0)

                def _median(x: np.ndarray, mask: np.ndarray) -> float:
                    vals = x[mask]
                    if vals.size == 0:
                        return 0.0
                    return float(np.median(vals))

                median_log_alias_pos = _median(alias_log, pos_mask_tiles)
                median_log_alias_neg = _median(alias_log, neg_mask_tiles)
                delta_log_alias = median_log_alias_pos - median_log_alias_neg

                # Ungated PD-z ROC.
                pd_pauc, pd_tpr, pd_fpr_min = _roc_summary(
                    pdz_scores_arr[pos_mask_tiles], pdz_scores_arr[neg_mask_tiles]
                )

                # Hemodynamic BR ROC.
                br_pauc, br_tpr, br_fpr_min = _roc_summary(
                    hemo_br[pos_mask_tiles], hemo_br[neg_mask_tiles]
                )

                # Alias-gated PD-z ROC.
                alpha_thr = float(np.quantile(alias, args.alias_quantile))
                gate = alias <= alpha_thr
                pos_g = pos_mask_tiles & gate
                neg_g = neg_mask_tiles & gate
                kept_frac = float(gate.mean())

                if pos_g.any() and neg_g.any():
                    gated_pd_pauc, gated_pd_tpr, gated_pd_fpr_min = _roc_summary(
                        pdz_scores_arr[pos_g], pdz_scores_arr[neg_g]
                    )
                    pd_hits_at_tpr, pd_fp_at_tpr, _ = _counts_at_tpr(
                        pdz_scores_arr[pos_mask_tiles],
                        pdz_scores_arr[neg_mask_tiles],
                        target_tpr=float(args.tpr_target),
                    )
                    gated_pd_hits_at_tpr, gated_pd_fp_at_tpr, _ = _counts_at_tpr(
                        pdz_scores_arr[pos_g],
                        pdz_scores_arr[neg_g],
                        target_tpr=float(args.tpr_target),
                    )
                else:
                    gated_pd_pauc = float("nan")
                    gated_pd_tpr = float("nan")
                    gated_pd_fpr_min = float("nan")
                    pd_hits_at_tpr = 0
                    pd_fp_at_tpr = 0
                    gated_pd_hits_at_tpr = 0
                    gated_pd_fp_at_tpr = 0

                writer.writerow(
                    {
                        "scan_idx": s_idx,
                        "scan_name": scan.name,
                        "plane_idx": z,
                        "n_pos_tiles": n_pos_tiles,
                        "n_neg_tiles": n_neg_tiles,
                        "frac_pf_peak_pos": frac_pf_peak_pos,
                        "frac_pf_peak_neg": frac_pf_peak_neg,
                        "median_log_alias_pos": median_log_alias_pos,
                        "median_log_alias_neg": median_log_alias_neg,
                        "delta_log_alias": delta_log_alias,
                        "pd_pauc": pd_pauc,
                        "pd_tpr_at_fpr_min": pd_tpr,
                        "pd_fpr_min": pd_fpr_min,
                        "br_pauc": br_pauc,
                        "br_tpr_at_fpr_min": br_tpr,
                        "br_fpr_min": br_fpr_min,
                        "alias_quantile": float(args.alias_quantile),
                        "alias_threshold": alpha_thr,
                        "gate_kept_frac": kept_frac,
                        "gated_pd_pauc": gated_pd_pauc,
                        "gated_pd_tpr_at_fpr_min": gated_pd_tpr,
                        "gated_pd_fpr_min": gated_pd_fpr_min,
                        "tpr_target": float(args.tpr_target),
                        "pd_hits_at_tpr": pd_hits_at_tpr,
                        "pd_fp_at_tpr": pd_fp_at_tpr,
                        "gated_pd_hits_at_tpr": gated_pd_hits_at_tpr,
                        "gated_pd_fp_at_tpr": gated_pd_fp_at_tpr,
                    }
                )


if __name__ == "__main__":
    main()

