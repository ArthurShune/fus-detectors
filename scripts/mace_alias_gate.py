#!/usr/bin/env python3
"""
Alias-gated PD-z ROC on Macé/Urban tiles for a given scan/plane.

This script:
  - loads one Macé scan and plane,
  - builds atlas-based H1/H0 tile labels (VIS/SC/LGd vs CP/CA/DG),
  - computes per-tile PD-z baseline scores,
  - computes hemo-STAP scores and an alias metric log(Ea/Ef),
  - applies an alias gate (alias <= quantile) and recomputes PD-z ROC on gated tiles,
  - reports pAUC@1e-3 and TPR@FPR_min for PD-z and PD-z+gate, plus gating stats.

Usage:
    PYTHONPATH=. python scripts/mace_alias_gate.py --scan-index 0 --plane-index 3 --alias-quantile 0.6
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
    ap = argparse.ArgumentParser(description="Alias-gated PD-z ROC on Macé tiles.")
    ap.add_argument("--data-root", type=Path, default=None, help="Root for whole-brain-fUS data.")
    ap.add_argument("--scan-index", type=int, default=0, help="Scan index (0-based).")
    ap.add_argument("--plane-index", type=int, default=3, help="Plane index (0-based).")
    ap.add_argument("--tile-h", type=int, default=8)
    ap.add_argument("--tile-w", type=int, default=8)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--label-alpha", type=float, default=0.6, help="Min fraction to assign tile label.")
    ap.add_argument("--alias-quantile", type=float, default=0.6, help="Alias threshold quantile for gating.")
    ap.add_argument("--tpr-target", type=float, default=0.5, help="Target TPR for FP count comparison.")
    ap.add_argument("--out-png", type=Path, default=None, help="Optional path to save a quick-look figure.")
    return ap.parse_args()


def _default_pos_prefixes() -> List[str]:
    return ["VIS", "SC", "LGd"]


def _default_neg_prefixes() -> List[str]:
    return ["CP", "CA", "DG"]


def _build_label_sets(
    region_info: MaceRegionInfo,
    pos_prefixes: Sequence[str],
    neg_prefixes: Sequence[str],
) -> Tuple[List[int], List[int]]:
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
    return pos_labels, neg_labels


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


def _pd_z_score(p_t: np.ndarray, baseline_frac: float = 0.3, eps: float = 1e-12) -> float:
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
    tpr_emp = tpr_at_fpr_target(fpr, tpr, target_fpr=float(np.clip(fpr_min, 1e-8, 1.0)))
    return pauc, tpr_emp, fpr_min


def _threshold_for_tpr(pos_scores: np.ndarray, neg_scores: np.ndarray, target_tpr: float) -> float:
    """Return the score threshold whose TPR is closest to target_tpr."""
    fpr, tpr, thr = roc_curve(pos_scores, neg_scores, num_thresh=None)
    idx = int(np.argmin(np.abs(tpr - target_tpr)))
    return float(thr[idx])


def _counts_at_tpr(pos_scores: np.ndarray, neg_scores: np.ndarray, target_tpr: float) -> Tuple[int, int, float]:
    """Return (hits_pos, fp_neg, threshold) at the threshold closest to target_tpr."""
    thr = _threshold_for_tpr(pos_scores, neg_scores, target_tpr)
    hits = int((pos_scores >= thr).sum())
    fps = int((neg_scores >= thr).sum())
    return hits, fps, thr


def main() -> None:
    args = parse_args()
    from pipeline.realdata import mace_data_root

    data_root = mace_data_root() if args.data_root is None else args.data_root
    atlas = load_mace_atlas(data_root)
    region_info = load_mace_region_info(data_root)
    transf = load_mace_transform(data_root)
    A, t = build_mace_transform_matrix(transf)

    pos_labels, neg_labels = _build_label_sets(region_info, _default_pos_prefixes(), _default_neg_prefixes())
    pos_mask_atlas, neg_mask_atlas = _atlas_masks_from_labels(atlas, pos_labels, neg_labels)

    scans = load_all_mace_scans(data_root)
    if args.scan_index < 0 or args.scan_index >= len(scans):
        raise ValueError(f"scan_index {args.scan_index} out of range")
    scan = scans[args.scan_index]
    if args.plane_index < 0 or args.plane_index >= scan.n_planes:
        raise ValueError(f"plane_index {args.plane_index} out of range")

    _, pd_T_HW = list(iter_pd_slices(scan))[args.plane_index]
    T, H, W = pd_T_HW.shape
    Ha, Wa, Za = atlas.regions.shape
    i_dv, i_ap, i_lr, inside = scan_plane_to_atlas_indices(H, W, args.plane_index, A, t, (Ha, Wa, Za))

    pd_flat = pd_T_HW.mean(axis=0).ravel()
    pos_mask_flat = np.zeros_like(pd_flat, dtype=bool)
    neg_mask_flat = np.zeros_like(pd_flat, dtype=bool)
    inside_idx = np.nonzero(inside)[0]
    atlas_pos = pos_mask_atlas[i_dv[inside_idx], i_ap[inside_idx], i_lr[inside_idx]]
    atlas_neg = neg_mask_atlas[i_dv[inside_idx], i_ap[inside_idx], i_lr[inside_idx]]
    pos_mask_flat[inside_idx] = atlas_pos
    neg_mask_flat[inside_idx] = atlas_neg
    neg_mask_flat = neg_mask_flat & ~pos_mask_flat

    tile_hw = (args.tile_h, args.tile_w)
    stride = args.stride
    alpha = args.label_alpha

    labels = _tile_labels_from_pixel_masks(H, W, pos_mask_flat, neg_mask_flat, tile_hw, stride, alpha)

    tile_series: List[np.ndarray] = []
    lbl_filtered: List[int] = []
    pdz_scores: List[float] = []
    for idx, (y0, x0) in enumerate(tile_iter((H, W), tile_hw, stride)):
        label = labels[idx]
        if label not in (0, 1):
            continue
        tile = extract_tile_stack(pd_T_HW, y0, x0, tile_hw)
        p_t = tile.mean(axis=(1, 2))
        tile_series.append(p_t.astype(np.float32))
        lbl_filtered.append(label)
        pdz_scores.append(_pd_z_score(p_t))

    if not tile_series:
        raise RuntimeError("No labeled tiles found.")

    p_tiles = np.stack(tile_series, axis=0)
    labels_arr = np.asarray(lbl_filtered, dtype=int)
    pdz_scores = np.asarray(pdz_scores, dtype=np.float64)

    pos_mask = labels_arr == 1
    neg_mask = labels_arr == 0
    if not pos_mask.any() or not neg_mask.any():
        raise RuntimeError("No positive or negative tiles after labeling.")

    hemo_cfg = HemoStapConfig(
        dt=0.1,
        L_t=40,
        pf_band=(0.0, 0.5),
        pa_band=(1.0, 3.0),
        pg_band=(0.5, 1.0),
        bg_beta=0.0,
    )
    hemo_scores = hemo_stap_scores_for_tiles(p_tiles, hemo_cfg)
    Ef = hemo_scores["Ef"]
    Ea = hemo_scores["Ea"]
    Eo = hemo_scores["Eo"]
    alias = (Ea + 1e-8) / (Ef + 1e-8)
    alias_log = np.log(alias + 1e-12)
    pf_frac = Ef / (Ef + Ea + Eo + 1e-8)
    pf_peak = (Ef > Ea) & (Ef > Eo)

    # Base ROC
    pauc_pd, tpr_pd, fpr_min = _roc_summary(pdz_scores[pos_mask], pdz_scores[neg_mask])
    pos_hit_tgt, neg_fp_tgt, thr_tgt = _counts_at_tpr(
        pdz_scores[pos_mask], pdz_scores[neg_mask], target_tpr=float(args.tpr_target)
    )

    # Gated ROC
    alpha_thr = np.quantile(alias, args.alias_quantile)
    gate = alias <= alpha_thr
    pos_g = pos_mask & gate
    neg_g = neg_mask & gate
    if pos_g.any() and neg_g.any():
        pauc_g, tpr_g, fpr_min_g = _roc_summary(pdz_scores[pos_g], pdz_scores[neg_g])
        pos_hit_tgt_g, neg_fp_tgt_g, thr_tgt_g = _counts_at_tpr(
            pdz_scores[pos_g], pdz_scores[neg_g], target_tpr=float(args.tpr_target)
        )
    else:
        pauc_g = np.nan
        tpr_g = np.nan
        fpr_min_g = np.nan
        pos_hit_tgt_g = np.nan
        neg_fp_tgt_g = np.nan
        thr_tgt_g = np.nan

    kept_frac = float(gate.mean())
    print(f"[mace-gate] scan={scan.name} plane={args.plane_index}")
    print(f"[mace-gate] tiles: pos={pos_mask.sum()}, neg={neg_mask.sum()}, kept={kept_frac:.3f}")
    print(f"[mace-gate] alias quantile={args.alias_quantile:.3f}, thr={alpha_thr:.3e}")
    print(f"[mace-gate] alias log median pos/neg: {np.median(alias_log[pos_mask]):.3f} / {np.median(alias_log[neg_mask]):.3f}")
    # Pf-fraction diagnostic
    pauc_pf, tpr_pf, fpr_min_pf = _roc_summary(pf_frac[pos_mask], pf_frac[neg_mask])
    print(f"[mace-gate] Pf-fraction: pAUC@1e-3={pauc_pf:.6f}, TPR@FPR_min={tpr_pf:.3f} (FPR_min={fpr_min_pf:.3e})")
    print(f"[mace-gate] PD-z: pAUC@1e-3={pauc_pd:.6f}, TPR@FPR_min={tpr_pd:.3f} (FPR_min={fpr_min:.3e})")
    print(f"[mace-gate] PD-z:      hits@TPR={args.tpr_target:.2f}: {pos_hit_tgt}, FP_H0: {neg_fp_tgt} at thr={thr_tgt:.3e}")
    print(f"[mace-gate] PD-z + gate: pAUC@1e-3={pauc_g:.6f}, TPR@FPR_min={tpr_g:.3f} (FPR_min={fpr_min_g:.3e})")
    print(f"[mace-gate] PD-z + gate: hits@TPR={args.tpr_target:.2f}: {pos_hit_tgt_g}, FP_H0: {neg_fp_tgt_g} at thr={thr_tgt_g:.3e}")

    if args.out_png:
        import matplotlib.pyplot as plt

        # Tile maps (simple max aggregation over overlapping tiles)
        def fill_map(values: np.ndarray) -> np.ndarray:
            out = np.full((H, W), np.nan, dtype=np.float32)
            for (y0, x0), val in zip(tile_iter((H, W), tile_hw, stride), values):
                if np.isnan(val):
                    continue
                y1, x1 = y0 + tile_hw[0], x0 + tile_hw[1]
                block = out[y0:y1, x0:x1]
                if np.isnan(block).all():
                    block[...] = val
                else:
                    block[...] = np.fmax(block, val if np.isscalar(val) else val)
            return out

        pd_map = fill_map(pdz_scores)
        det_map = fill_map((pdz_scores >= thr_tgt).astype(float))
        det_gate_map = fill_map(((pdz_scores >= thr_tgt_g) & gate).astype(float))

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        # Pf peak fraction bar
        ax = axes[0, 0]
        frac_pos = pf_peak[pos_mask].mean()
        frac_neg = pf_peak[neg_mask].mean()
        ax.bar(["H1", "H0"], [frac_pos, frac_neg], color=["tab:blue", "tab:gray"])
        ax.set_ylim(0, 1)
        ax.set_title("Pf peak fraction")
        for i, v in enumerate([frac_pos, frac_neg]):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        # Alias box
        ax = axes[0, 1]
        ax.boxplot([alias_log[pos_mask], alias_log[neg_mask]], labels=["H1", "H0"])
        ax.set_title("Alias log(Ea/Ef)")
        ax.set_ylabel("log alias")
        # Maps
        # Occupied band heuristic: where pd_map is finite
        y_has_data = np.any(np.isfinite(pd_map), axis=1)
        if y_has_data.any():
            y_indices = np.nonzero(y_has_data)[0]
            y_min = int(y_indices.min())
            y_max = int(y_indices.max())
        else:
            y_min, y_max = 0, H - 1

        im0 = axes[1, 0].imshow(pd_map, cmap="viridis", origin="upper")
        axes[1, 0].set_title("PD-z (tile-filled)")
        axes[1, 0].set_xlabel("x (px)")
        axes[1, 0].set_ylabel("y (px)")
        axes[1, 0].set_ylim(y_max + 0.5, y_min - 0.5)
        fig.colorbar(im0, ax=axes[1, 0], fraction=0.046, pad=0.04)
        im1 = axes[1, 1].imshow(det_map, cmap="Reds", origin="upper")
        axes[1, 1].set_xlabel("x (px)")
        axes[1, 1].set_ylabel("y (px)")
        axes[1, 1].set_ylim(y_max + 0.5, y_min - 0.5)
        axes[1, 1].contour(det_gate_map, levels=[0.5], colors="cyan", linewidths=1.0)
        axes[1, 1].set_title(f"Detections @ TPR={args.tpr_target:.2f} (cyan = gated)")
        fig.tight_layout()
        fig.savefig(args.out_png, dpi=200)
        plt.close(fig)
        print(f"[mace-gate] saved figure to {args.out_png}")


if __name__ == "__main__":
    main()
