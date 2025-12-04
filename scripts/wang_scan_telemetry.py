#!/usr/bin/env python3
"""
Scan Wang IQData files over frame windows and report basic Doppler telemetry.

For each IQData file and for each frame window of a given length, this script:
  - loads the IQ subset (frames [start, stop]),
  - computes PD tiles and runs the Doppler STAP core (clinical bands),
  - derives vessel/background masks from PD mean (heuristic),
  - reports Pf/Pa/Po peak fractions, alias ratio medians, and PD-z ROC.

If a user-defined criterion is met, the scan exits early and prints the match.

Usage
-----
    PYTHONPATH=. python scripts/wang_scan_telemetry.py \
        --window 256 \
        --step 256 \
        --pf-min 0.05 \
        --pauc-min 0.01
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from eval.metrics import partial_auc, roc_curve, tpr_at_fpr_target
from pipeline.realdata import tile_iter
from pipeline.realdata.wang_ulm import (
    load_wang_iq,
    load_wang_metadata,
    list_wang_files,
)
from pipeline.stap.hemo import HemoStapConfig, hemo_stap_scores_for_tiles


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Scan Wang IQData files for Pf/ROC criteria.")
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root containing IQDataXXX.dat and IQSizeInfo.mat (default: data/wang_ulm).",
    )
    ap.add_argument(
        "--window",
        type=int,
        default=256,
        help="Frame window length to analyze (default: 256).",
    )
    ap.add_argument(
        "--step",
        type=int,
        default=256,
        help="Frame step between windows (default: 256, non-overlapping).",
    )
    ap.add_argument(
        "--tile-h",
        type=int,
        default=8,
        help="Tile height (default: 8).",
    )
    ap.add_argument(
        "--tile-w",
        type=int,
        default=8,
        help="Tile width (default: 8).",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=3,
        help="Tile stride (default: 3).",
    )
    ap.add_argument(
        "--lt",
        type=int,
        default=8,
        help="Hankel length L_t (default: 8).",
    )
    ap.add_argument(
        "--pf-min",
        type=float,
        default=0.05,
        help="Minimum Pf-peak fraction to consider a match (default: 0.05).",
    )
    ap.add_argument(
        "--pauc-min",
        type=float,
        default=0.01,
        help="Minimum PD-z pAUC@1e-3 to consider a match (default: 0.01).",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress for each file/window even if criteria not met.",
    )
    return ap.parse_args()


def _frame_windows(total: int, length: int, step: int) -> Iterable[Tuple[int, int]]:
    start = 0
    while start < total:
        stop = min(start + length - 1, total - 1)
        yield start, stop
        if stop == total - 1:
            break
        start += step


def _pd_z_score(p_t: np.ndarray, baseline_frac: float = 0.2) -> float:
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
    info = load_wang_metadata(args.data_root)
    files = list_wang_files(args.data_root)
    if not files:
        raise RuntimeError("No IQDataXXX.dat files found.")

    stap_cfg = HemoStapConfig(
        dt=0.001,
        L_t=int(args.lt),
        pf_band=(30.0, 200.0),
        pg_band=(200.0, 350.0),
        pa_band=(350.0, 500.0),
        bg_beta=0.0,
    )

    for f_idx, path in enumerate(files, start=1):
        for start, stop in _frame_windows(info.frames, args.window, args.step):
            idx = np.arange(start, stop + 1, dtype=int)
            iq = load_wang_iq(path, info, frames=idx, dtype=np.complex64)  # (T,H,W)
            pd_T_HW = (np.abs(iq) ** 2).astype(np.float32, copy=False)
            T, H, W = pd_T_HW.shape

            # Derive masks from PD mean for this window.
            pd_mean = pd_T_HW.mean(axis=0)
            v_thr = np.quantile(pd_mean, 0.99)
            b_thr = np.quantile(pd_mean, 0.2)
            vessel_mask = pd_mean >= v_thr
            bg_mask = (pd_mean <= b_thr) & (~vessel_mask)

            tile_hw = (args.tile_h, args.tile_w)
            stride = args.stride
            tile_series: List[np.ndarray] = []
            labels: List[int] = []
            for y0, x0 in tile_iter((H, W), tile_hw, stride):
                tile = pd_T_HW[:, y0 : y0 + tile_hw[0], x0 : x0 + tile_hw[1]]
                p_t = tile.mean(axis=(1, 2))
                tile_series.append(p_t.astype(np.float32))
                v_cov = vessel_mask[y0 : y0 + tile_hw[0], x0 : x0 + tile_hw[1]].mean()
                b_cov = bg_mask[y0 : y0 + tile_hw[0], x0 : x0 + tile_hw[1]].mean()
                if v_cov >= 0.6:
                    labels.append(1)
                elif b_cov >= 0.6:
                    labels.append(0)
                else:
                    labels.append(-1)

            p_tiles = np.stack(tile_series, axis=0)
            labels_arr = np.asarray(labels, dtype=int)
            pos_mask = labels_arr == 1
            neg_mask = labels_arr == 0
            if not pos_mask.any() or not neg_mask.any():
                continue

            scores = hemo_stap_scores_for_tiles(p_tiles, stap_cfg)
            Ef = scores["Ef"]
            Ea = scores["Ea"]
            peak_idx = np.argmax(np.stack([Ef, Ea, scores["Eg"]], axis=1), axis=1)
            pf_peak = float((peak_idx == 0).mean())
            pa_peak = float((peak_idx == 1).mean())
            alias_ratio = (Ea + 1e-8) / (Ef + 1e-8)
            alias_med_pos = float(np.median(np.log10(alias_ratio[pos_mask])))
            alias_med_neg = float(np.median(np.log10(alias_ratio[neg_mask])))

            pdz_scores = np.array([_pd_z_score(p) for p in p_tiles])
            fpr, tpr, _ = roc_curve(pdz_scores[pos_mask], pdz_scores[neg_mask], num_thresh=4096)
            pauc = partial_auc(fpr, tpr, fpr_max=1e-3)
            fpr_min = 1.0 / float(neg_mask.sum())
            tpr_emp = tpr_at_fpr_target(fpr, tpr, target_fpr=float(np.clip(fpr_min, 1e-8, 1.0)))

            msg = (
                f"[wang-scan] file={path.name} frames={start}-{stop} "
                f"pf_peak={pf_peak:.3f} pa_peak={pa_peak:.3f} "
                f"alias_med_pos={alias_med_pos:.3f} alias_med_neg={alias_med_neg:.3f} "
                f"pauc@1e-3={pauc:.4f} tpr@fpr_min={tpr_emp:.4f}"
            )
            if args.verbose:
                print(msg)

            if pf_peak >= args.pf_min and pauc >= args.pauc_min:
                if not args.verbose:
                    print(msg)
                print("[wang-scan] criteria met, exiting.")
                return


if __name__ == "__main__":
    main()
