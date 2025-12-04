#!/usr/bin/env python3
"""
Pixel-level PD ROC against Allen atlas ROIs for the Macé/Urban dataset.

This script demonstrates the next integration step:
  - use the affine transform in Transformation.mat to map scan voxels into
    Allen atlas coordinates,
  - define simple positive/negative ROI sets from atlas region acronyms, and
  - compute a PD-based ROC curve between those ROI sets on a single PD plane.

For now, ROI sets are defined heuristically via acronym prefixes:
  - positives: acronyms starting with VIS, SC, LGd (visual cortex, SC, LGN),
  - negatives: acronyms starting with CP, CA, DG (striatum/hippocampus).

This is intended as a wiring / sanity harness; ROI definitions can be refined
later in line with the methods text.

Usage
-----
    PYTHONPATH=. python scripts/mace_pd_atlas_roc.py \
        --scan-index 0 \
        --plane-index 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from eval.metrics import partial_auc, roc_curve, tpr_at_fpr_target
from pipeline.realdata import iter_pd_slices, mace_data_root
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="PD ROC vs atlas ROIs on Macé whole-brain fUS")
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
    # Visual cortex, superior colliculus, lateral geniculate
    return ["VIS", "SC", "LGd"]


def _default_neg_prefixes() -> List[str]:
    # Striatum / hippocampus as null ROIs for a visual task
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
        # Deduplicate, preserve order
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
    # Exclude overlaps from negatives
    neg_mask = neg_mask & ~pos_mask
    return pos_mask, neg_mask


def main() -> None:
    args = parse_args()
    data_root = mace_data_root() if args.data_root is None else args.data_root

    # Load atlas + region metadata + transform
    atlas = load_mace_atlas(data_root)
    region_info = load_mace_region_info(data_root)
    transf = load_mace_transform(data_root)
    A, t = build_mace_transform_matrix(transf)

    # Build ROI label sets
    pos_prefixes = args.pos_prefix or _default_pos_prefixes()
    neg_prefixes = args.neg_prefix or _default_neg_prefixes()
    pos_labels, neg_labels, info = _build_label_sets(region_info, pos_prefixes, neg_prefixes)
    if not pos_labels or not neg_labels:
        raise RuntimeError("Empty positive or negative label set; adjust prefixes.")

    pos_mask_atlas, neg_mask_atlas = _atlas_masks_from_labels(atlas, pos_labels, neg_labels)

    # Load PD scan and select plane
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

    # Map PD plane voxels into atlas indices
    Ha, Wa, Za = atlas.regions.shape
    i_dv, i_ap, i_lr, inside = scan_plane_to_atlas_indices(
        H, W, args.plane_index, A, t, (Ha, Wa, Za)
    )

    # Flatten PD plane and atlas masks
    pd_mean = pd_T_HW.mean(axis=0).ravel()
    pos_mask_flat = np.zeros_like(pd_mean, dtype=bool)
    neg_mask_flat = np.zeros_like(pd_mean, dtype=bool)

    inside_idx = np.nonzero(inside)[0]
    atlas_pos = pos_mask_atlas[i_dv[inside_idx], i_ap[inside_idx], i_lr[inside_idx]]
    atlas_neg = neg_mask_atlas[i_dv[inside_idx], i_ap[inside_idx], i_lr[inside_idx]]

    pos_mask_flat[inside_idx] = atlas_pos
    neg_mask_flat[inside_idx] = atlas_neg

    # Ensure disjoint and non-empty
    neg_mask_flat = neg_mask_flat & ~pos_mask_flat
    scores_pos = pd_mean[pos_mask_flat]
    scores_neg = pd_mean[neg_mask_flat]

    n_pos = int(scores_pos.size)
    n_neg = int(scores_neg.size)
    if n_pos == 0 or n_neg == 0:
        raise RuntimeError(f"No positives ({n_pos}) or negatives ({n_neg}) after ROI mapping.")

    # ROC at pixel level
    fpr, tpr, _ = roc_curve(scores_pos, scores_neg, num_thresh=4096)
    auc = partial_auc(fpr, tpr, fpr_max=1e-3)
    fpr_min = 1.0 / float(n_neg)
    fpr_min = float(np.clip(fpr_min, 1e-8, 1.0))
    tpr_emp = tpr_at_fpr_target(fpr, tpr, target_fpr=fpr_min)

    # Diagnostics
    frac_inside = float(inside.mean())
    print(f"Macé scan '{scan.name}', plane {args.plane_index}:")
    print(f"  PD shape T,H,W = ({T}, {H}, {W})")
    print(f"  Atlas shape DV,AP,LR = {atlas.regions.shape}")
    print(f"  Transform in-bounds fraction: {frac_inside:.3f}")
    print(f"  Positive ROI prefixes: {pos_prefixes}")
    print(f"  Negative ROI prefixes: {neg_prefixes}")
    print(f"  Positive acronyms (first 10): {info['pos_acr'][:10]}")
    print(f"  Negative acronyms (first 10): {info['neg_acr'][:10]}")
    print(f"  Positives (pixels) = {n_pos}, negatives (pixels) = {n_neg}")
    print(f"  partial AUC (FPR<=1e-3): {auc:.4f}")
    print(f"  TPR at empirical FPR_min={fpr_min:.3e}: {tpr_emp:.4f}")


if __name__ == "__main__":
    main()
