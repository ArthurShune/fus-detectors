#!/usr/bin/env python3
"""
Phase 2: Macé/Urban PD-only KA Contract v2 sweep (per plane).

This script runs the label-free KA Contract v2 state machine on PD-only planes
using hemodynamic band telemetry (Ef/Eg/Ea/Eo from whitened Hankel tiles) and
PD-derived proxy sets (flow/background) based on mean-PD vascularity.

Contract inputs (runtime / label-free)
--------------------------------------
- s_base: per-tile PD-z score (max z after an early baseline segment)
- m_alias: log((Ea+eps)/(Ef+eps))
- r_guard: Eg / (Ef+Eg+Ea+Eo+eps)
- c_flow: per-tile flow coverage fraction from a mean-PD flow proxy mask
- pf_peak: per-tile Pf dominance flag (Ef >= max(Eg,Ea,Eo))
- valid_mask: tiles with sufficient atlas in-bounds coverage

Offline (atlas-labeled) descriptive fields
-----------------------------------------
We also compute atlas-defined H1/H0 tile labels (VIS/SC/LGd vs CP/CA/DG) and
report descriptive metrics prefixed with `offline_`. These are NOT contract
inputs and are intended only for reporting discipline and consistency checks.

Output
------
Writes a CSV (default: reports/mace_pdonly_contract_v2.csv) with one row per
unique (scan_group, plane). Under our loader, scan1/scan2 and scan3/scan6 are
identical PD tensors, so we deduplicate by processing scan1 and scan3 only and
annotate their rows with scan_group=scan1/2 or scan3/6.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from eval.metrics import roc_curve
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
from pipeline.stap.ka_contract_v2 import KaContractV2Config, evaluate_ka_contract_v2


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Macé PD-only KA Contract v2 sweep (per plane).")
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root directory containing whole-brain-fUS (defaults to data/whole-brain-fUS).",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/mace_pdonly_contract_v2.csv"),
        help="Output CSV path (one row per unique scan_group/plane).",
    )
    ap.add_argument("--tile-h", type=int, default=8, help="Tile height (pixels).")
    ap.add_argument("--tile-w", type=int, default=8, help="Tile width (pixels).")
    ap.add_argument("--stride", type=int, default=3, help="Tile stride (pixels).")
    ap.add_argument(
        "--inside-min",
        type=float,
        default=0.5,
        help="Minimum atlas in-bounds fraction per tile to mark it valid.",
    )
    ap.add_argument(
        "--baseline-frac",
        type=float,
        default=0.3,
        help="Fraction of initial frames used for PD-z baseline.",
    )
    ap.add_argument(
        "--q-flow",
        type=float,
        default=0.95,
        help="Quantile on mean PD (within atlas in-bounds mask) used for flow proxy mask.",
    )
    ap.add_argument(
        "--label-alpha",
        type=float,
        default=0.6,
        help="Min fraction of labeled pixels required to assign atlas H1/H0 tile label.",
    )
    ap.add_argument(
        "--alias-quantile",
        type=float,
        default=0.8,
        help="Offline Macé alias veto threshold quantile (within labeled tiles).",
    )
    ap.add_argument(
        "--tpr-target",
        type=float,
        default=0.5,
        help="Offline hit/FP comparison target TPR (ungated) on atlas-labeled H1/H0 tiles.",
    )
    # Hemodynamic STAP settings (telemetry only).
    ap.add_argument("--Lt", type=int, default=40, help="Hemo Hankel length (frames).")
    ap.add_argument("--pf-band", type=float, nargs=2, default=(0.0, 0.5), help="Pf band (Hz).")
    ap.add_argument("--pg-band", type=float, nargs=2, default=(0.5, 1.0), help="Guard band (Hz).")
    ap.add_argument("--pa-band", type=float, nargs=2, default=(1.0, 3.0), help="Pa band (Hz).")
    # Contract v2 config overrides (minimal set; keep deterministic defaults).
    ap.add_argument("--c-bg", type=float, default=None, help="Override contract c_bg (background proxy).")
    ap.add_argument("--c-flow", type=float, default=None, help="Override contract c_flow (flow proxy).")
    ap.add_argument("--q-risk", type=float, default=None, help="Override contract q_risk (risk quantile).")
    ap.add_argument(
        "--alias-iqr-min",
        type=float,
        default=None,
        help="Override contract alias_iqr_min (flatness sentinel).",
    )
    ap.add_argument("--n-min", type=int, default=None, help="Override contract n_min (valid tiles).")
    ap.add_argument("--n-bg-min", type=int, default=None, help="Override contract n_bg_min (bg proxy).")
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
    if name == "scan2":
        return False
    if name == "scan6":
        return False
    return True


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
    """
    Compute atlas tile labels (+1 for H1, 0 for H0, -1 for discarded/mixed).
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


def _pd_z_score(p_t: np.ndarray, baseline_frac: float = 0.3, eps: float = 1e-12) -> float:
    """
    PD-z score using an early baseline segment for mean/std.

    The first baseline_frac of frames are used to estimate μ/σ; the score is the
    maximum z over the remaining frames.
    """

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


def _counts_at_tpr(pos_scores: np.ndarray, neg_scores: np.ndarray, target_tpr: float) -> Tuple[int, int, float]:
    """
    Return (hits_pos, fp_neg, threshold) at the threshold closest to target_tpr.
    """

    fpr, tpr, thr = roc_curve(pos_scores, neg_scores, num_thresh=None)
    idx = int(np.argmin(np.abs(tpr - float(target_tpr))))
    thr_val = float(thr[idx])
    hits = int(np.sum(pos_scores >= thr_val))
    fps = int(np.sum(neg_scores >= thr_val))
    return hits, fps, thr_val


def _flatten_report(report: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten a v2 contract report into a row dict with stable prefixes.
    """

    row: dict[str, Any] = {
        "ka_contract_v2_state": report.get("state"),
        "ka_contract_v2_reason": report.get("reason"),
    }
    cfg = report.get("config") or {}
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            row[f"ka_contract_v2_cfg_{k}"] = v
    metrics = report.get("metrics") or {}
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            row[f"ka_contract_v2_{k}"] = v
    return row


def main() -> None:
    args = parse_args()

    data_root = mace_data_root() if args.data_root is None else args.data_root
    out_csv = args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)

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
    if not scans:
        raise RuntimeError("No Macé scans found.")

    # Deduplicate scan tensors (scan2 duplicates scan1, scan6 duplicates scan3).
    scans = [s for s in scans if _is_group_representative(s.name)]

    tile_hw = (int(args.tile_h), int(args.tile_w))
    stride = int(args.stride)
    inside_min = float(args.inside_min)
    baseline_frac = float(args.baseline_frac)
    q_flow = float(args.q_flow)
    label_alpha = float(args.label_alpha)
    alias_quantile = float(args.alias_quantile)
    tpr_target = float(args.tpr_target)

    hemo_cfg = HemoStapConfig(
        dt=0.1,
        L_t=int(args.Lt),
        pf_band=(float(args.pf_band[0]), float(args.pf_band[1])),
        pg_band=(float(args.pg_band[0]), float(args.pg_band[1])),
        pa_band=(float(args.pa_band[0]), float(args.pa_band[1])),
        bg_beta=0.0,
    )

    cfg_kwargs: dict[str, Any] = {}
    if args.c_bg is not None:
        cfg_kwargs["c_bg"] = float(args.c_bg)
    if args.c_flow is not None:
        cfg_kwargs["c_flow"] = float(args.c_flow)
    if args.q_risk is not None:
        cfg_kwargs["q_risk"] = float(args.q_risk)
    if args.alias_iqr_min is not None:
        cfg_kwargs["alias_iqr_min"] = float(args.alias_iqr_min)
    if args.n_min is not None:
        cfg_kwargs["n_min"] = int(args.n_min)
    if args.n_bg_min is not None:
        cfg_kwargs["n_bg_min"] = int(args.n_bg_min)
    contract_cfg = KaContractV2Config(**cfg_kwargs) if cfg_kwargs else KaContractV2Config()

    rows: List[Dict[str, Any]] = []
    state_counts: Counter[tuple[str | None, str | None]] = Counter()

    for scan in scans:
        scan_group = _scan_group(scan.name)
        dt = float(scan.dt)
        fs_hz = 1.0 / dt if dt > 0 else float("nan")

        hemo_cfg_plane = HemoStapConfig(
            **{**asdict(hemo_cfg), "dt": dt}
        )

        for plane_idx, pd_T_HW in iter_pd_slices(scan):
            T_pd, H, W = pd_T_HW.shape

            # Map pixels in this plane to atlas indices (for in-bounds mask and offline labels).
            Ha, Wa, Za = atlas.regions.shape
            i_dv, i_ap, i_lr, inside = scan_plane_to_atlas_indices(
                H, W, int(plane_idx), A, t, (Ha, Wa, Za)
            )
            inside_img = inside.reshape(H, W)
            inside_vals = inside_img.ravel()

            # Mean PD and flow proxy mask (label-free).
            mean_pd = pd_T_HW.mean(axis=0).astype(np.float64, copy=False)  # (H,W)
            mean_pd_in = mean_pd.ravel()[inside_vals]
            if mean_pd_in.size < 10:
                continue
            thr_flow = float(np.quantile(mean_pd_in, np.clip(q_flow, 0.0, 1.0)))
            flow_mask_img = inside_img & (mean_pd >= thr_flow)

            # Offline atlas-defined H1/H0 pixel masks in scan space.
            pos_mask_flat = np.zeros((H * W,), dtype=bool)
            neg_mask_flat = np.zeros((H * W,), dtype=bool)
            inside_idx = np.nonzero(inside_vals)[0]
            atlas_pos = pos_mask_atlas[i_dv[inside_idx], i_ap[inside_idx], i_lr[inside_idx]]
            atlas_neg = neg_mask_atlas[i_dv[inside_idx], i_ap[inside_idx], i_lr[inside_idx]]
            pos_mask_flat[inside_idx] = atlas_pos
            neg_mask_flat[inside_idx] = atlas_neg
            neg_mask_flat = neg_mask_flat & ~pos_mask_flat

            labels = _tile_labels_from_pixel_masks(
                H, W, pos_mask_flat, neg_mask_flat, tile_hw, stride, label_alpha
            )

            # Collect per-tile PD series and proxy/validity stats (all tiles).
            coords = list(tile_iter((H, W), tile_hw, stride))
            n_tiles = len(coords)
            if n_tiles == 0:
                continue

            p_tiles = np.empty((n_tiles, T_pd), dtype=np.float32)
            s_base = np.empty((n_tiles,), dtype=np.float64)
            c_flow = np.empty((n_tiles,), dtype=np.float64)
            inside_frac = np.empty((n_tiles,), dtype=np.float64)
            for i, (y0, x0) in enumerate(coords):
                tile = extract_tile_stack(pd_T_HW, y0, x0, tile_hw)
                p_t = tile.mean(axis=(1, 2)).astype(np.float32, copy=False)
                p_tiles[i] = p_t
                s_base[i] = _pd_z_score(p_t, baseline_frac=baseline_frac)
                y1, x1 = y0 + tile_hw[0], x0 + tile_hw[1]
                c_flow[i] = float(flow_mask_img[y0:y1, x0:x1].mean())
                inside_frac[i] = float(inside_img[y0:y1, x0:x1].mean())

            valid_mask = inside_frac >= inside_min

            # Hemodynamic telemetry: Ef/Eg/Ea/Eo.
            hemo_scores = hemo_stap_scores_for_tiles(p_tiles, hemo_cfg_plane)
            Ef = hemo_scores["Ef"].astype(np.float64, copy=False)
            Eg = hemo_scores["Eg"].astype(np.float64, copy=False)
            Ea = hemo_scores["Ea"].astype(np.float64, copy=False)
            Eo = hemo_scores["Eo"].astype(np.float64, copy=False)
            eps = 1e-12
            m_alias = np.log((Ea + eps) / (Ef + eps))
            r_guard = Eg / (Ef + Eg + Ea + Eo + eps)
            pf_peak = (Ef >= Eg) & (Ef >= Ea) & (Ef >= Eo)

            report = evaluate_ka_contract_v2(
                s_base=s_base,
                m_alias=m_alias,
                r_guard=r_guard,
                pf_peak=pf_peak,
                c_flow=c_flow,
                valid_mask=valid_mask,
                config=contract_cfg,
            )

            row: Dict[str, Any] = {
                "dataset": "mace_pdonly",
                "scan_name": scan.name,
                "scan_group": scan_group,
                "plane_idx": int(plane_idx),
                "T_pd": int(T_pd),
                "H": int(H),
                "W": int(W),
                "dt_s": float(dt),
                "fs_hz": float(fs_hz),
                "Lt": int(hemo_cfg_plane.L_t),
                "tile_h": int(tile_hw[0]),
                "tile_w": int(tile_hw[1]),
                "tile_stride": int(stride),
                "inside_min": float(inside_min),
                "baseline_frac": float(baseline_frac),
                "q_flow": float(q_flow),
                "thr_flow": float(thr_flow),
                "label_alpha": float(label_alpha),
                "alias_quantile": float(alias_quantile),
                "tpr_target": float(tpr_target),
            }
            row.update(
                {
                    "pf_band_hz": f"{hemo_cfg_plane.pf_band[0]:g}-{hemo_cfg_plane.pf_band[1]:g}",
                    "pg_band_hz": f"{hemo_cfg_plane.pg_band[0]:g}-{hemo_cfg_plane.pg_band[1]:g}",
                    "pa_band_hz": f"{hemo_cfg_plane.pa_band[0]:g}-{hemo_cfg_plane.pa_band[1]:g}",
                }
            )
            row.update(_flatten_report(report))

            # Offline descriptive metrics (atlas-labeled), kept separate by prefix.
            labels_arr = np.asarray(labels, dtype=int)
            pos_tiles = labels_arr == 1
            neg_tiles = labels_arr == 0
            row["offline_n_H1"] = int(np.sum(pos_tiles))
            row["offline_n_H0"] = int(np.sum(neg_tiles))
            if np.any(pos_tiles):
                row["offline_pf_peak_frac_H1"] = float(np.mean(pf_peak[pos_tiles]))
                row["offline_m_alias_median_H1"] = float(np.median(m_alias[pos_tiles]))
            else:
                row["offline_pf_peak_frac_H1"] = float("nan")
                row["offline_m_alias_median_H1"] = float("nan")
            if np.any(neg_tiles):
                row["offline_pf_peak_frac_H0"] = float(np.mean(pf_peak[neg_tiles]))
                row["offline_m_alias_median_H0"] = float(np.median(m_alias[neg_tiles]))
            else:
                row["offline_pf_peak_frac_H0"] = float("nan")
                row["offline_m_alias_median_H0"] = float("nan")
            if np.any(pos_tiles) and np.any(neg_tiles):
                row["offline_delta_logEaEf"] = float(
                    row["offline_m_alias_median_H1"] - row["offline_m_alias_median_H0"]
                )
            else:
                row["offline_delta_logEaEf"] = float("nan")

            # Offline hit/FP counts (PD-z threshold fixed by ungated target TPR).
            if np.any(pos_tiles) and np.any(neg_tiles):
                pos_scores = s_base[pos_tiles]
                neg_scores = s_base[neg_tiles]
                hits_pre, fp_pre, thr_pd = _counts_at_tpr(pos_scores, neg_scores, tpr_target)
                row["offline_pd_hits_pre"] = int(hits_pre)
                row["offline_pd_fp_pre"] = int(fp_pre)
                row["offline_pd_thr_pre"] = float(thr_pd)

                alias_ratio = (Ea + 1e-8) / (Ef + 1e-8)
                alias_thr = float(np.quantile(alias_ratio[pos_tiles | neg_tiles], np.clip(alias_quantile, 0.0, 1.0)))
                gate_keep = alias_ratio <= alias_thr
                hits_post = int(np.sum((pos_scores >= thr_pd) & gate_keep[pos_tiles]))
                fp_post = int(np.sum((neg_scores >= thr_pd) & gate_keep[neg_tiles]))
                row["offline_alias_thr"] = float(alias_thr)
                row["offline_pd_hits_post"] = int(hits_post)
                row["offline_pd_fp_post"] = int(fp_post)
                row["offline_hit_retention"] = float(hits_post / hits_pre) if hits_pre > 0 else float("nan")
            else:
                row["offline_pd_hits_pre"] = float("nan")
                row["offline_pd_fp_pre"] = float("nan")
                row["offline_pd_thr_pre"] = float("nan")
                row["offline_alias_thr"] = float("nan")
                row["offline_pd_hits_post"] = float("nan")
                row["offline_pd_fp_post"] = float("nan")
                row["offline_hit_retention"] = float("nan")

            rows.append(row)
            state_counts[(report.get("state"), report.get("reason"))] += 1

    if not rows:
        raise SystemExit("No rows produced; check dataset path and settings.")

    # Write CSV with a stable union of fields.
    fieldnames: List[str] = []
    keys: Dict[str, None] = {}
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys[k] = None
    # Prefer identifier columns first.
    preferred = [
        "dataset",
        "scan_group",
        "scan_name",
        "plane_idx",
        "T_pd",
        "H",
        "W",
        "dt_s",
        "fs_hz",
        "Lt",
        "tile_h",
        "tile_w",
        "tile_stride",
        "pf_band_hz",
        "pg_band_hz",
        "pa_band_hz",
        "q_flow",
        "thr_flow",
        "inside_min",
        "baseline_frac",
        "label_alpha",
        "alias_quantile",
        "tpr_target",
        "ka_contract_v2_state",
        "ka_contract_v2_reason",
    ]
    for k in preferred:
        if k in keys:
            fieldnames.append(k)
            keys.pop(k, None)
    # Then the remaining columns in sorted order for determinism.
    fieldnames.extend(sorted(keys.keys()))

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[mace-pdonly-v2] wrote {out_csv} rows={len(rows)}")
    print("[mace-pdonly-v2] top (state, reason):")
    for (state, reason), count in state_counts.most_common(12):
        print(f"  {count:4d}  {state}/{reason}")


if __name__ == "__main__":
    main()

