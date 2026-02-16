#!/usr/bin/env python3
"""
Phase 5: Macé PD-only holdout evaluation for alias-gated PD-z.

This script addresses two review risks in the Macé PD-only section:

  1) Plane selection bias: selecting "nice" planes and reporting gating effects
     on the same data.
  2) Label-tuned thresholds: choosing a PD-z threshold per plane using atlas H1/H0
     labels (e.g., forcing TPR≈0.5).

Protocol (lean, defensible)
---------------------------
We treat atlas labels (VIS/SC/LGd vs CP/CA/DG) as *offline evaluation labels*.
We do not use them to set thresholds.

Per plane we compute:
  - PD-z score per tile (baseline_frac early segment, max z after baseline).
  - Hemodynamic band energies (Ef/Eg/Ea/Eo) and alias metric:
        m_alias = log((Ea+eps)/(Ef+eps)).
  - A mean-PD-derived flow proxy mask (top q_flow within atlas in-bounds).
    Per-tile flow coverage c_flow is the fraction of pixels in the tile covered
    by this mask.
  - Background proxy tiles: c_flow <= c_bg (label-free).

Thresholds (label-free)
-----------------------
  - Detection threshold: choose tau_pd_z so that
        P{PD-z >= tau_pd_z | bg_proxy} = alpha_pd.
  - Alias veto: choose tau_alias as the q_alias quantile of m_alias on bg_proxy,
    and keep tiles with m_alias <= tau_alias (hard veto).

Holdout split
-------------
By default we deduplicate the dataset (scan2==scan1, scan6==scan3) and use:
  - train scan_groups: scan1/2, scan3/6
  - test scan_groups:  scan4, scan5

Plane selection (label-free, fixed from train split)
---------------------------------------------------
We define a label-free telemetry statistic per plane:
  pf_peak_flow := fraction of flow-proxy tiles (c_flow >= c_flow_thr) whose
                  pooled hemo spectrum is Pf-dominant (Ef >= max(Eg,Ea,Eo)).

We set a selection threshold as the q_select quantile of pf_peak_flow over the
training split, then evaluate the alias gate on test planes that satisfy:
  pf_peak_flow >= threshold.

Outputs
-------
  - CSV with per-plane metrics and pre/post-gate hit/FP counts.
  - JSON summary with selection threshold and aggregate test-set deltas.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from pipeline.realdata import extract_tile_stack, iter_pd_slices, mace_data_root, tile_iter
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
    ap = argparse.ArgumentParser(description="Macé PD-only holdout evaluation for alias-gated PD-z.")
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root directory containing whole-brain-fUS (defaults to data/whole-brain-fUS).",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/mace_alias_gate_holdout.csv"),
        help="Output per-plane CSV path.",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/mace_alias_gate_holdout.json"),
        help="Output summary JSON path.",
    )
    ap.add_argument("--tile-h", type=int, default=8)
    ap.add_argument("--tile-w", type=int, default=8)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--inside-min", type=float, default=0.5, help="Min atlas in-bounds fraction per tile.")
    ap.add_argument("--baseline-frac", type=float, default=0.3, help="PD-z baseline fraction.")
    ap.add_argument(
        "--q-flow",
        type=float,
        default=0.95,
        help="Quantile on mean PD (within atlas in-bounds) for flow proxy mask.",
    )
    ap.add_argument("--c-bg", type=float, default=0.05, help="Background-proxy coverage threshold.")
    ap.add_argument("--c-flow", type=float, default=0.20, help="Flow-proxy coverage threshold.")
    ap.add_argument(
        "--alpha-pd",
        type=float,
        default=0.01,
        help="Label-free bg-proxy FPR target for PD-z threshold calibration.",
    )
    ap.add_argument(
        "--alias-quantile",
        type=float,
        default=0.8,
        help="Alias veto threshold quantile (on bg-proxy m_alias).",
    )
    ap.add_argument(
        "--q-select",
        type=float,
        default=0.95,
        help="Quantile on train pf_peak_flow used to select planes in test split.",
    )
    ap.add_argument(
        "--train-scan-groups",
        nargs="+",
        type=str,
        default=["scan1/2", "scan3/6"],
        help="Train scan groups (deduplicated).",
    )
    ap.add_argument(
        "--test-scan-groups",
        nargs="+",
        type=str,
        default=["scan4", "scan5"],
        help="Test scan groups (deduplicated).",
    )
    ap.add_argument(
        "--label-alpha",
        type=float,
        default=0.6,
        help="Min labeled-pixel fraction to assign atlas H1/H0 tile label.",
    )
    # Hemo telemetry configuration.
    ap.add_argument("--Lt", type=int, default=40, help="Hemo Hankel length (frames).")
    ap.add_argument("--pf-band", type=float, nargs=2, default=(0.0, 0.5))
    ap.add_argument("--pg-band", type=float, nargs=2, default=(0.5, 1.0))
    ap.add_argument("--pa-band", type=float, nargs=2, default=(1.0, 3.0))
    return ap.parse_args()


def _scan_group(scan_name: str) -> str:
    name = str(scan_name)
    if name in {"scan1", "scan2"}:
        return "scan1/2"
    if name in {"scan3", "scan6"}:
        return "scan3/6"
    return name


def _is_group_representative(scan_name: str) -> bool:
    name = str(scan_name)
    return name not in {"scan2", "scan6"}


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
) -> np.ndarray:
    """Tile labels: +1=H1, 0=H0, -1=discard/mixed/unlabeled."""
    th, tw = tile_hw
    pos_img = pos_mask_flat.reshape(H, W)
    neg_img = neg_mask_flat.reshape(H, W)
    out: List[int] = []
    for y0, x0 in tile_iter((H, W), tile_hw, stride):
        y1 = y0 + th
        x1 = x0 + tw
        pos_tile = pos_img[y0:y1, x0:x1]
        neg_tile = neg_img[y0:y1, x0:x1]
        n_pos = int(pos_tile.sum())
        n_neg = int(neg_tile.sum())
        n_lab = n_pos + n_neg
        if n_lab == 0:
            out.append(-1)
            continue
        if n_pos >= alpha * n_lab:
            out.append(+1)
        elif n_neg >= alpha * n_lab:
            out.append(0)
        else:
            out.append(-1)
    return np.asarray(out, dtype=int)


def _pd_z_score(p_t: np.ndarray, baseline_frac: float, eps: float = 1e-12) -> float:
    p_t = np.asarray(p_t, dtype=np.float64).ravel()
    T = int(p_t.size)
    if T <= 1:
        return 0.0
    frac = float(np.clip(baseline_frac, 0.0, 1.0))
    n_base = max(1, int(frac * T))
    n_base = min(n_base, T - 1)
    base = p_t[:n_base]
    mu = float(np.mean(base))
    sigma = float(np.std(base))
    sigma_safe = sigma if sigma > 0.0 else 1.0
    z = (p_t - mu) / (sigma_safe + float(eps))
    z_det = z[n_base:]
    if z_det.size == 0:
        z_det = z
    return float(np.max(z_det))


def main() -> None:
    args = parse_args()
    data_root = mace_data_root() if args.data_root is None else args.data_root
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    tile_hw = (int(args.tile_h), int(args.tile_w))
    stride = int(args.stride)
    inside_min = float(args.inside_min)
    baseline_frac = float(args.baseline_frac)
    q_flow = float(args.q_flow)
    c_bg = float(args.c_bg)
    c_flow_thr = float(args.c_flow)
    alpha_pd = float(args.alpha_pd)
    alias_q = float(args.alias_quantile)
    q_select = float(args.q_select)
    label_alpha = float(args.label_alpha)

    atlas = load_mace_atlas(data_root)
    region_info = load_mace_region_info(data_root)
    transf = load_mace_transform(data_root)
    A, t = build_mace_transform_matrix(transf)

    pos_labels, neg_labels = _build_label_sets(
        region_info, _default_pos_prefixes(), _default_neg_prefixes()
    )
    if not pos_labels or not neg_labels:
        raise RuntimeError("Empty positive or negative label set; adjust prefixes.")
    pos_mask_atlas, neg_mask_atlas = _atlas_masks_from_labels(atlas, pos_labels, neg_labels)

    scans = load_all_mace_scans(data_root)
    scans = [s for s in scans if _is_group_representative(s.name)]
    if not scans:
        raise RuntimeError("No Macé scans found.")

    rows: List[Dict[str, Any]] = []

    for scan in scans:
        scan_group = _scan_group(scan.name)
        dt = float(scan.dt)

        hemo_cfg = HemoStapConfig(
            dt=dt,
            L_t=int(args.Lt),
            pf_band=(float(args.pf_band[0]), float(args.pf_band[1])),
            pg_band=(float(args.pg_band[0]), float(args.pg_band[1])),
            pa_band=(float(args.pa_band[0]), float(args.pa_band[1])),
            bg_beta=0.0,
        )

        for plane_idx, pd_T_HW in iter_pd_slices(scan):
            T_pd, H, W = pd_T_HW.shape

            # Atlas in-bounds mapping.
            Ha, Wa, Za = atlas.regions.shape
            i_dv, i_ap, i_lr, inside = scan_plane_to_atlas_indices(
                H, W, int(plane_idx), A, t, (Ha, Wa, Za)
            )
            inside_img = inside.reshape(H, W)
            inside_flat = inside_img.ravel()
            inside_idx = np.nonzero(inside_flat)[0]
            if inside_idx.size < 10:
                continue

            # Flow proxy mask from mean PD (label-free).
            mean_pd = pd_T_HW.mean(axis=0).astype(np.float64, copy=False)
            mean_pd_in = mean_pd.ravel()[inside_flat]
            thr_flow = float(np.quantile(mean_pd_in, np.clip(q_flow, 0.0, 1.0)))
            flow_mask_img = inside_img & (mean_pd >= thr_flow)

            # Offline atlas-defined H1/H0 pixel masks in scan space.
            pos_mask_flat = np.zeros((H * W,), dtype=bool)
            neg_mask_flat = np.zeros((H * W,), dtype=bool)
            atlas_pos = pos_mask_atlas[i_dv[inside_idx], i_ap[inside_idx], i_lr[inside_idx]]
            atlas_neg = neg_mask_atlas[i_dv[inside_idx], i_ap[inside_idx], i_lr[inside_idx]]
            pos_mask_flat[inside_idx] = atlas_pos
            neg_mask_flat[inside_idx] = atlas_neg
            neg_mask_flat = neg_mask_flat & ~pos_mask_flat

            tile_labels = _tile_labels_from_pixel_masks(
                H, W, pos_mask_flat, neg_mask_flat, tile_hw, stride, label_alpha
            )

            # Per-tile arrays (full length).
            th, tw = tile_hw
            tile_coords: List[Tuple[int, int]] = list(tile_iter((H, W), tile_hw, stride))
            n_tiles = len(tile_coords)

            valid = np.zeros((n_tiles,), dtype=bool)
            inside_frac = np.zeros((n_tiles,), dtype=np.float32)
            c_flow_tile = np.zeros((n_tiles,), dtype=np.float32)
            pdz = np.full((n_tiles,), np.nan, dtype=np.float64)

            valid_idx: List[int] = []
            tile_series: List[np.ndarray] = []

            inside_img_bool = inside_img.astype(bool, copy=False)
            for idx, (y0, x0) in enumerate(tile_coords):
                y1 = y0 + th
                x1 = x0 + tw
                inside_tile = inside_img_bool[y0:y1, x0:x1]
                frac_in = float(np.mean(inside_tile))
                inside_frac[idx] = frac_in
                if frac_in < inside_min:
                    continue
                valid[idx] = True

                flow_tile = flow_mask_img[y0:y1, x0:x1]
                c_flow_tile[idx] = float(np.mean(flow_tile))

                tile = extract_tile_stack(pd_T_HW, y0, x0, tile_hw)
                p_t = tile.mean(axis=(1, 2)).astype(np.float32)
                pdz[idx] = _pd_z_score(p_t, baseline_frac=baseline_frac)
                valid_idx.append(idx)
                tile_series.append(p_t)

            if not valid_idx:
                continue

            p_tiles = np.stack(tile_series, axis=0)  # (n_valid, T)
            hemo_scores = hemo_stap_scores_for_tiles(p_tiles, hemo_cfg)
            Ef_v = hemo_scores["Ef"].astype(np.float64, copy=False)
            Eg_v = hemo_scores["Eg"].astype(np.float64, copy=False)
            Ea_v = hemo_scores["Ea"].astype(np.float64, copy=False)
            Eo_v = hemo_scores["Eo"].astype(np.float64, copy=False)

            Ef = np.full((n_tiles,), np.nan, dtype=np.float64)
            Eg = np.full((n_tiles,), np.nan, dtype=np.float64)
            Ea = np.full((n_tiles,), np.nan, dtype=np.float64)
            Eo = np.full((n_tiles,), np.nan, dtype=np.float64)
            Ef[valid_idx] = Ef_v
            Eg[valid_idx] = Eg_v
            Ea[valid_idx] = Ea_v
            Eo[valid_idx] = Eo_v

            eps = 1e-12
            m_alias = np.log((Ea + 1e-8) / (Ef + 1e-8) + eps)
            pf_peak = np.zeros((n_tiles,), dtype=bool)
            # Pf dominant among (Ef,Eg,Ea,Eo).
            denom = np.nanmax(np.stack([Eg, Ea, Eo], axis=0), axis=0)
            pf_peak[valid] = Ef[valid] >= denom[valid]

            bg_proxy = valid & (c_flow_tile <= c_bg)
            flow_proxy = valid & (c_flow_tile >= c_flow_thr)

            if not np.any(bg_proxy):
                continue

            # Label-free thresholds calibrated on bg proxy.
            tau_pd = float(np.quantile(pdz[bg_proxy], 1.0 - np.clip(alpha_pd, 0.0, 1.0)))
            tau_alias = float(np.quantile(m_alias[bg_proxy], np.clip(alias_q, 0.0, 1.0)))
            gate_keep = m_alias <= tau_alias

            # Label-free plane selection metric.
            pf_peak_flow = float(np.mean(pf_peak[flow_proxy])) if np.any(flow_proxy) else float("nan")
            n_flow_proxy = int(np.sum(flow_proxy))
            n_bg_proxy = int(np.sum(bg_proxy))

            # Offline (atlas-labeled) evaluation counts.
            pos_tiles = valid & (tile_labels == 1)
            neg_tiles = valid & (tile_labels == 0)
            hits_pre = int(np.sum(pos_tiles & (pdz >= tau_pd)))
            fp_pre = int(np.sum(neg_tiles & (pdz >= tau_pd)))
            hits_post = int(np.sum(pos_tiles & (pdz >= tau_pd) & gate_keep))
            fp_post = int(np.sum(neg_tiles & (pdz >= tau_pd) & gate_keep))

            row: Dict[str, Any] = {
                "dataset": "mace_pdonly",
                "scan_group": scan_group,
                "scan_name": scan.name,
                "plane_idx": int(plane_idx),
                "T_pd": int(T_pd),
                "H": int(H),
                "W": int(W),
                "dt_s": float(dt),
                "Lt": int(hemo_cfg.L_t),
                "tile_h": int(th),
                "tile_w": int(tw),
                "tile_stride": int(stride),
                "pf_band_hz": f"{hemo_cfg.pf_band[0]:g}-{hemo_cfg.pf_band[1]:g}",
                "pg_band_hz": f"{hemo_cfg.pg_band[0]:g}-{hemo_cfg.pg_band[1]:g}",
                "pa_band_hz": f"{hemo_cfg.pa_band[0]:g}-{hemo_cfg.pa_band[1]:g}",
                "q_flow": float(q_flow),
                "thr_flow": float(thr_flow),
                "inside_min": float(inside_min),
                "baseline_frac": float(baseline_frac),
                "c_bg": float(c_bg),
                "c_flow": float(c_flow_thr),
                "alpha_pd": float(alpha_pd),
                "alias_quantile": float(alias_q),
                "tau_pd": float(tau_pd),
                "tau_alias": float(tau_alias),
                "n_tiles": int(n_tiles),
                "n_valid": int(np.sum(valid)),
                "n_bg_proxy": int(n_bg_proxy),
                "n_flow_proxy": int(n_flow_proxy),
                "pf_peak_flow": pf_peak_flow,
                "hits_pre": int(hits_pre),
                "fp_pre": int(fp_pre),
                "hits_post": int(hits_post),
                "fp_post": int(fp_post),
                "hit_retention": float(hits_post / hits_pre) if hits_pre > 0 else float("nan"),
                "fp_reduction": float(fp_pre - fp_post),
            }
            rows.append(row)

    if not rows:
        raise SystemExit("No rows produced; check dataset path and settings.")

    # Build train/test selection threshold from training split.
    train_groups = set(str(s) for s in args.train_scan_groups)
    test_groups = set(str(s) for s in args.test_scan_groups)
    train_vals = np.array(
        [r["pf_peak_flow"] for r in rows if r["scan_group"] in train_groups and np.isfinite(r["pf_peak_flow"])],
        dtype=np.float64,
    )
    if train_vals.size == 0:
        raise SystemExit("No finite pf_peak_flow values in training split.")
    pf_thr = float(np.quantile(train_vals, np.clip(q_select, 0.0, 1.0)))

    for r in rows:
        r["split"] = "train" if r["scan_group"] in train_groups else ("test" if r["scan_group"] in test_groups else "other")
        r["selected"] = bool(np.isfinite(r["pf_peak_flow"]) and (r["pf_peak_flow"] >= pf_thr))

    # Aggregate held-out evaluation.
    def _agg(filter_fn) -> Dict[str, Any]:
        rs = [r for r in rows if filter_fn(r)]
        if not rs:
            return {"n_planes": 0}
        hits_pre = int(sum(int(r["hits_pre"]) for r in rs))
        fp_pre = int(sum(int(r["fp_pre"]) for r in rs))
        hits_post = int(sum(int(r["hits_post"]) for r in rs))
        fp_post = int(sum(int(r["fp_post"]) for r in rs))
        return {
            "n_planes": int(len(rs)),
            "hits_pre": hits_pre,
            "fp_pre": fp_pre,
            "hits_post": hits_post,
            "fp_post": fp_post,
            "hit_retention": float(hits_post / hits_pre) if hits_pre > 0 else float("nan"),
            "fp_reduction": int(fp_pre - fp_post),
        }

    summary: Dict[str, Any] = {
        "train_scan_groups": sorted(train_groups),
        "test_scan_groups": sorted(test_groups),
        "q_select": float(q_select),
        "pf_peak_flow_threshold": float(pf_thr),
        "alpha_pd": float(alpha_pd),
        "alias_quantile": float(alias_q),
        "counts_test_all": _agg(lambda r: r["split"] == "test"),
        "counts_test_selected": _agg(lambda r: r["split"] == "test" and r["selected"]),
        "counts_train_selected": _agg(lambda r: r["split"] == "train" and r["selected"]),
    }

    # Write CSV (stable union of keys).
    keys: Dict[str, None] = {}
    for r in rows:
        for k in r.keys():
            keys.setdefault(k, None)
    preferred = [
        "dataset",
        "scan_group",
        "scan_name",
        "split",
        "plane_idx",
        "selected",
        "pf_peak_flow",
        "alpha_pd",
        "tau_pd",
        "alias_quantile",
        "tau_alias",
        "hits_pre",
        "fp_pre",
        "hits_post",
        "fp_post",
        "hit_retention",
        "fp_reduction",
    ]
    fieldnames: List[str] = []
    for k in preferred:
        if k in keys:
            fieldnames.append(k)
            keys.pop(k, None)
    fieldnames.extend(sorted(keys.keys()))

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with out_json.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"[mace-holdout] wrote {out_csv} rows={len(rows)}")
    print(f"[mace-holdout] wrote {out_json}")
    print(f"[mace-holdout] pf_peak_flow threshold (train q={q_select:g}): {pf_thr:.6f}")
    print(f"[mace-holdout] test/all: {summary['counts_test_all']}")
    print(f"[mace-holdout] test/selected: {summary['counts_test_selected']}")


if __name__ == "__main__":
    main()

