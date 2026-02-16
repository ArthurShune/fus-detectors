#!/usr/bin/env python3
"""
Pixel-level real-data quantitative check on Macé/Urban whole-brain fUS using an
independent structural label source: the Allen atlas micro-Doppler vascular map
(`atlas.Vascular`) shipped with the dataset.

Motivation
----------
Many Macé analyses in this repository are intentionally "PD-only telemetry"
because the dataset does not provide Doppler-side (IQ/RF) labels. However, the
dataset *does* include an independently derived vascular atlas volume aligned to
the Allen CCF, and an affine transform (Transformation.mat) that maps scan voxels
into atlas coordinates.

This script maps `atlas.Vascular` into each scan plane, thresholds it to define
vascular vs non-vascular pixels, and reports low-FPR operating points for:
  - PD-z (change-from-baseline z-score on log-PD),
  - hemodynamic band telemetry (simple FFT band energies on log-PD),
  - a label-free "alias veto" applied as a shrink-only score veto on PD-z.

This is a structural separability check (vascular vs non-vascular), not a
clinical efficacy claim.

Usage
-----
    PYTHONPATH=. python scripts/mace_vascular_pixel_eval.py \
      --out-csv reports/mace_vascular_pixel_eval.csv \
      --out-json reports/mace_vascular_pixel_eval.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from pipeline.realdata import iter_pd_slices, mace_data_root
from pipeline.realdata.mace_wholebrain import (
    build_mace_transform_matrix,
    load_all_mace_scans,
    load_mace_atlas,
    load_mace_transform,
    scan_plane_to_atlas_indices,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Macé pixel-level evaluation against atlas.Vascular (independent structural labels)."
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
        default=Path("reports/mace_vascular_pixel_eval.csv"),
        help="Output CSV path (per-plane rows).",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/mace_vascular_pixel_eval.json"),
        help="Output JSON path (aggregate summary).",
    )
    ap.add_argument(
        "--vascular-threshold",
        type=float,
        default=0.5,
        help="Threshold on atlas.Vascular to define vascular pixels (>=thr => positive).",
    )
    ap.add_argument(
        "--inside-min-frac",
        type=float,
        default=0.80,
        help="Skip planes with < this fraction of pixels mapping inside atlas+brain.",
    )
    ap.add_argument(
        "--min-neg",
        type=int,
        default=5000,
        help="Skip planes with fewer than this many negative (non-vascular) pixels.",
    )
    ap.add_argument(
        "--min-pos",
        type=int,
        default=500,
        help="Skip planes with fewer than this many positive (vascular) pixels.",
    )
    ap.add_argument(
        "--baseline-frac",
        type=float,
        default=0.30,
        help="Fraction of early frames used as baseline for PD-z.",
    )
    ap.add_argument(
        "--eps-pd",
        type=float,
        default=1e-8,
        help="Epsilon added inside log(PD+eps) for stability.",
    )
    ap.add_argument(
        "--eps-energy",
        type=float,
        default=1e-12,
        help="Epsilon used in band-energy ratios.",
    )
    ap.add_argument(
        "--pf-band",
        type=float,
        nargs=2,
        default=(0.0, 0.5),
        help="Pf band in Hz (inclusive).",
    )
    ap.add_argument(
        "--pg-band",
        type=float,
        nargs=2,
        default=(0.5, 1.0),
        help="Guard band in Hz (inclusive).",
    )
    ap.add_argument(
        "--pa-band",
        type=float,
        nargs=2,
        default=(1.0, 3.0),
        help="Pa band in Hz (inclusive).",
    )
    ap.add_argument(
        "--fprs",
        type=str,
        default="1e-4,3e-4,1e-3",
        help="Comma-separated FPR targets for TPR reporting (right-tail scores).",
    )
    ap.add_argument(
        "--q-flow",
        type=float,
        default=0.95,
        help="Quantile of mean-PD (inside pixels) used to define a flow proxy mask.",
    )
    ap.add_argument(
        "--alias-quantile",
        type=float,
        default=0.90,
        help="Quantile of m_alias on bg-proxy pixels used as alias veto threshold.",
    )
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


def _parse_fprs(s: str) -> List[float]:
    out: List[float] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("No FPR targets provided.")
    for f in out:
        if not (0.0 < f < 1.0):
            raise ValueError(f"FPR must be in (0,1), got {f}")
    return out


def _tpr_at_fpr(scores_pos: np.ndarray, scores_neg: np.ndarray, fpr: float) -> tuple[float, float]:
    """Return (threshold, TPR) for right-tail scores at requested FPR."""
    pos = np.asarray(scores_pos, dtype=np.float64).ravel()
    neg = np.asarray(scores_neg, dtype=np.float64).ravel()
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan"), float("nan")
    thr = float(np.quantile(neg, 1.0 - float(fpr), method="linear"))
    tpr = float(np.mean(pos >= thr))
    return thr, tpr


def _quantiles(arr: np.ndarray, qs: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(arr, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {f"q{int(100*q):02d}": float("nan") for q in qs}
    return {f"q{int(100*q):02d}": float(np.quantile(a, q)) for q in qs}


def _summarize_rows(rows: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    vals: List[float] = []
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if np.isfinite(fv):
            vals.append(fv)
    if not vals:
        return {"n": 0, "median": float("nan"), "q25": float("nan"), "q75": float("nan")}
    a = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(a.size),
        "median": float(np.quantile(a, 0.5)),
        "q25": float(np.quantile(a, 0.25)),
        "q75": float(np.quantile(a, 0.75)),
    }


def main() -> None:
    args = parse_args()
    data_root = mace_data_root() if args.data_root is None else args.data_root
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    vascular_thr = float(args.vascular_threshold)
    inside_min = float(args.inside_min_frac)
    min_neg = int(args.min_neg)
    min_pos = int(args.min_pos)
    baseline_frac = float(args.baseline_frac)
    eps_pd = float(args.eps_pd)
    eps_e = float(args.eps_energy)
    pf_lo, pf_hi = float(args.pf_band[0]), float(args.pf_band[1])
    pg_lo, pg_hi = float(args.pg_band[0]), float(args.pg_band[1])
    pa_lo, pa_hi = float(args.pa_band[0]), float(args.pa_band[1])
    fprs = _parse_fprs(args.fprs)
    q_flow = float(args.q_flow)
    alias_q = float(args.alias_quantile)

    atlas = load_mace_atlas(data_root)
    transf = load_mace_transform(data_root)
    A, t = build_mace_transform_matrix(transf)

    scans = load_all_mace_scans(data_root)
    scans = [s for s in scans if _is_group_representative(s.name)]
    if not scans:
        raise RuntimeError("No Macé scans found.")

    Ha, Wa, Za = atlas.regions.shape
    rows: List[Dict[str, Any]] = []

    for scan in scans:
        dt = float(scan.dt)
        fs = 1.0 / dt
        scan_group = _scan_group(scan.name)

        for plane_idx, pd_T_HW in iter_pd_slices(scan):
            T, H, W = pd_T_HW.shape

            # Map scan pixels to atlas indices; then restrict to atlas bounds + brain (region != 0).
            i_dv, i_ap, i_lr, inside = scan_plane_to_atlas_indices(
                H, W, int(plane_idx), A, t, (Ha, Wa, Za)
            )
            inside = inside.astype(bool, copy=False)
            inside_idx = np.nonzero(inside)[0]
            if inside_idx.size == 0:
                continue

            # Exclude atlas background (region 0) from the "inside" set.
            reg_vals = atlas.regions[i_dv[inside_idx], i_ap[inside_idx], i_lr[inside_idx]]
            brain_idx = inside_idx[reg_vals != 0]
            inside_brain = np.zeros((H * W,), dtype=bool)
            inside_brain[brain_idx] = True

            inside_frac = float(np.mean(inside_brain))
            if inside_frac < inside_min:
                continue

            vascular_vals = atlas.vascular[i_dv[brain_idx], i_ap[brain_idx], i_lr[brain_idx]].astype(
                np.float32, copy=False
            )
            vascular_flat = np.zeros((H * W,), dtype=np.float32)
            vascular_flat[brain_idx] = vascular_vals

            pos = inside_brain & (vascular_flat >= vascular_thr)
            neg = inside_brain & ~pos
            n_pos = int(np.sum(pos))
            n_neg = int(np.sum(neg))
            if n_neg < min_neg or n_pos < min_pos:
                continue

            fpr_min = 1.0 / float(n_neg)

            # ===== Pixel-level PD-z on log-PD =====
            pd_T_N = pd_T_HW.reshape(T, -1).astype(np.float32, copy=False)  # (T, Npix)
            u_T_N = np.log(pd_T_N + eps_pd, dtype=np.float64)  # promote for stability
            frac = float(np.clip(baseline_frac, 0.0, 1.0))
            n_base = max(1, int(frac * T))
            n_base = min(n_base, T - 1)
            mu = u_T_N[:n_base].mean(axis=0)
            sigma = u_T_N[:n_base].std(axis=0)
            sigma_safe = np.where(sigma > 0.0, sigma, 1.0)
            z_T_N = (u_T_N - mu) / (sigma_safe + 1e-12)
            z_det = z_T_N[n_base:] if n_base < T else z_T_N
            pdz = np.max(z_det, axis=0)  # (Npix,)

            # ===== Simple hemo-band telemetry via FFT band energies on log-PD =====
            u0_T_N = u_T_N - u_T_N.mean(axis=0, keepdims=True)
            F = np.fft.rfft(u0_T_N, axis=0)
            P = (F.real * F.real + F.imag * F.imag).astype(np.float64, copy=False)  # (Fbins, Npix)
            freqs = np.fft.rfftfreq(T, d=dt)
            # Exclude DC from Pf by requiring freq > 0.
            pf_mask = (freqs > 0.0) & (freqs >= pf_lo) & (freqs <= pf_hi)
            pg_mask = (freqs >= pg_lo) & (freqs <= pg_hi)
            pa_mask = (freqs >= pa_lo) & (freqs <= pa_hi)
            Ef = P[pf_mask].sum(axis=0) if np.any(pf_mask) else np.zeros((H * W,), dtype=np.float64)
            Eg = P[pg_mask].sum(axis=0) if np.any(pg_mask) else np.zeros((H * W,), dtype=np.float64)
            Ea = P[pa_mask].sum(axis=0) if np.any(pa_mask) else np.zeros((H * W,), dtype=np.float64)
            Etot = P.sum(axis=0)
            Eo = np.maximum(0.0, Etot - Ef - Eg - Ea)
            m_alias = np.log((Ea + eps_e) / (Ef + eps_e))
            score_logbr = -m_alias  # = log((Ef+eps)/(Ea+eps)), right-tail "vascular-likeness"

            # ===== Label-free "alias veto" (shrink-only) applied to PD-z =====
            mean_pd = pd_T_N.mean(axis=0).astype(np.float64, copy=False)
            # Use a right-tail score convention for mean PD by taking the *left tail*
            # (higher score = lower mean PD). This is a structural baseline only.
            score_inv_mean_pd = -np.log(mean_pd + eps_pd)
            mean_pd_in = mean_pd[inside_brain]
            thr_flow = float(np.quantile(mean_pd_in, np.clip(q_flow, 0.0, 1.0)))
            flow_proxy = inside_brain & (mean_pd >= thr_flow)
            bg_proxy = inside_brain & ~flow_proxy

            tau_alias = float(np.quantile(m_alias[bg_proxy], np.clip(alias_q, 0.0, 1.0)))
            veto = inside_brain & (~flow_proxy) & (m_alias >= tau_alias)
            pdz_veto = pdz.copy()
            pdz_veto[veto] = -np.inf

            veto_frac_pos = float(np.mean(veto[pos])) if n_pos > 0 else float("nan")
            veto_frac_neg = float(np.mean(veto[neg])) if n_neg > 0 else float("nan")

            # ===== Low-FPR operating points (right tail) =====
            row: Dict[str, Any] = {
                "dataset": "mace_vascular_pixel_eval",
                "scan_group": scan_group,
                "scan_name": scan.name,
                "plane_idx": int(plane_idx),
                "T_pd": int(T),
                "H": int(H),
                "W": int(W),
                "dt_s": float(dt),
                "fs_hz": float(fs),
                "pf_band_hz": f"{pf_lo:g}-{pf_hi:g}",
                "pg_band_hz": f"{pg_lo:g}-{pg_hi:g}",
                "pa_band_hz": f"{pa_lo:g}-{pa_hi:g}",
                "vascular_thr": float(vascular_thr),
                "inside_frac": float(inside_frac),
                "n_pos": int(n_pos),
                "n_neg": int(n_neg),
                "fpr_min": float(fpr_min),
                "q_flow": float(q_flow),
                "thr_flow_mean_pd": float(thr_flow),
                "alias_quantile": float(alias_q),
                "tau_alias": float(tau_alias),
                "veto_frac_pos": float(veto_frac_pos),
                "veto_frac_neg": float(veto_frac_neg),
            }

            # Scores evaluated against vascular vs non-vascular masks.
            score_sets = {
                "inv_mean_pd": score_inv_mean_pd,
                "pdz": pdz,
                "hemo_m_alias": m_alias,
                "pdz_alias_veto": pdz_veto,
            }
            for name, s in score_sets.items():
                s_pos = s[pos]
                s_neg = s[neg]
                for fpr in fprs:
                    key = f"tpr_{name}_fpr{fpr:g}"
                    if fpr < fpr_min:
                        row[key] = float("nan")
                        continue
                    _, tpr = _tpr_at_fpr(s_pos, s_neg, fpr)
                    row[key] = float(tpr)

            rows.append(row)

    if not rows:
        raise RuntimeError("No valid planes found (all filtered); relax thresholds or check data root.")

    # Aggregate summary (median/IQR across planes) for the core paper table.
    summary: Dict[str, Any] = {
        "dataset": "mace_vascular_pixel_eval",
        "n_planes": int(len(rows)),
        "vascular_thr": float(vascular_thr),
        "inside_min_frac": float(inside_min),
        "min_neg": int(min_neg),
        "min_pos": int(min_pos),
        "baseline_frac": float(baseline_frac),
        "eps_pd": float(eps_pd),
        "pf_band_hz": [float(pf_lo), float(pf_hi)],
        "pg_band_hz": [float(pg_lo), float(pg_hi)],
        "pa_band_hz": [float(pa_lo), float(pa_hi)],
        "fprs": [float(f) for f in fprs],
        "q_flow": float(q_flow),
        "alias_quantile": float(alias_q),
        "inside_frac": _summarize_rows(rows, "inside_frac"),
        "n_neg": _summarize_rows(rows, "n_neg"),
        "n_pos": _summarize_rows(rows, "n_pos"),
        "fpr_min": _summarize_rows(rows, "fpr_min"),
        "veto_frac_pos": _summarize_rows(rows, "veto_frac_pos"),
        "veto_frac_neg": _summarize_rows(rows, "veto_frac_neg"),
    }

    for name in ("inv_mean_pd", "pdz", "hemo_m_alias", "pdz_alias_veto"):
        for fpr in fprs:
            k = f"tpr_{name}_fpr{fpr:g}"
            summary[k] = _summarize_rows(rows, k)

    # Write CSV with a stable column order.
    keys: Dict[str, None] = {}
    for r in rows:
        for k in r.keys():
            keys.setdefault(k, None)
    preferred: List[str] = [
        "dataset",
        "scan_group",
        "scan_name",
        "plane_idx",
        "T_pd",
        "H",
        "W",
        "dt_s",
        "fs_hz",
        "pf_band_hz",
        "pg_band_hz",
        "pa_band_hz",
        "vascular_thr",
        "inside_frac",
        "n_pos",
        "n_neg",
        "fpr_min",
        "q_flow",
        "thr_flow_mean_pd",
        "alias_quantile",
        "tau_alias",
        "veto_frac_pos",
        "veto_frac_neg",
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

    print(f"[mace-vascular] wrote {out_csv} rows={len(rows)}")
    print(f"[mace-vascular] wrote {out_json}")


if __name__ == "__main__":
    main()
