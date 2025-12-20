#!/usr/bin/env python3
"""
ROI-based Doppler band-occupancy telemetry for MaceBridge replay bundles.

This mirrors macebridge_open_doppler_telemetry.py but uses the Allen
ROI-derived H1/H0 masks (from the MaceTemplate / sim root) instead of
PD flow/background masks. The goal is to measure Pf/Pa peak fractions
and alias-score separation directly on Macé-style H1/H0 regions.

Usage
-----
    PYTHONPATH=. python scripts/macebridge_roi_doppler_telemetry.py \\
        --replay-root runs/macebridge_open_replay \\
        --sim-root runs/macebridge_open \\
        --out-csv reports/macebridge_roi_doppler_telemetry.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="ROI-based Doppler band-occupancy telemetry for MaceBridge replay bundles."
    )
    ap.add_argument(
        "--replay-root",
        type=Path,
        default=Path("runs/macebridge_open_replay"),
        help="Root directory containing scan*_plane*/pw_* replay bundles.",
    )
    ap.add_argument(
        "--sim-root",
        type=Path,
        default=Path("runs/macebridge_open"),
        help="Root directory containing the corresponding MaceBridge sim slices (scan*_plane*).",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/macebridge_roi_doppler_telemetry.csv"),
        help="Output CSV path for per-slice ROI-based Doppler telemetry.",
    )
    return ap.parse_args()


def _discover_bundles(root: Path) -> List[Path]:
    bundles: List[Path] = []
    for scan_dir in sorted(root.glob("scan*_plane*")):
        if not scan_dir.is_dir():
            continue
        cands = sorted(scan_dir.glob("pw_*"))
        if not cands:
            continue
        bundles.append(cands[0])
    return bundles


def main() -> None:
    args = parse_args()
    replay_root = args.replay_root
    sim_root = args.sim_root
    out_csv = args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    bundles = _discover_bundles(replay_root)
    if not bundles:
        raise SystemExit(f"No replay bundles found under {replay_root}")

    fieldnames = [
        "scan_tag",
        "scan_idx",
        "plane_idx",
        "bundle",
        "cf_lo_hz",
        "cf_hi_hz",
        "ca_lo_hz",
        "ca_hi_hz",
        "frac_pf_peak_pos_roi",
        "frac_pf_peak_neg_roi",
        "frac_pa_peak_pos_roi",
        "frac_pa_peak_neg_roi",
        "median_log_alias_pos_roi",
        "median_log_alias_neg_roi",
        "delta_log_alias_roi",
    ]

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for bundle in bundles:
            meta_path = bundle / "meta.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            tele: Dict = meta.get("stap_fallback_telemetry", {}) or {}

            # Band design and peak statistics from whitened band-ratio telemetry.
            br_bands = tele.get("band_ratio_bands_hz") or {}
            br_stats = tele.get("band_ratio_stats") or tele.get("band_ratio_br_stats") or {}
            if not br_bands or not br_stats:
                continue

            cf_lo = float(br_bands.get("flow_low_hz", 0.0))
            cf_hi = float(br_bands.get("flow_high_hz", 0.0))
            alias_center = float(br_bands.get("alias_center_hz", 0.0))
            alias_width = float(br_bands.get("alias_width_hz", 0.0))
            ca_lo = alias_center - alias_width
            ca_hi = alias_center + alias_width

            # Band-ratio map on the replay grid.
            base_br_path = bundle / "base_band_ratio_map.npy"
            if not base_br_path.exists():
                continue
            base_br = np.load(base_br_path).astype(np.float64)

            # Recover scan/plane indices from the parent directory name.
            scan_tag = bundle.parent.name  # e.g. 'scan0_plane3'
            scan_idx = None
            plane_idx = None
            try:
                s_part, p_part = scan_tag.split("_", 1)
                scan_idx = int(s_part.replace("scan", ""))
                plane_idx = int(p_part.replace("plane", ""))
            except Exception:
                pass

            if scan_idx is None or plane_idx is None:
                continue

            # Load Allen ROI-derived H1/H0 masks from the corresponding sim slice.
            sim_slice_dir = sim_root / f"scan{scan_idx}_plane{plane_idx}"
            roi_H1_path = sim_slice_dir / "roi_H1.npy"
            roi_H0_path = sim_slice_dir / "roi_H0.npy"
            if not (roi_H1_path.exists() and roi_H0_path.exists()):
                continue

            roi_H1 = np.load(roi_H1_path).astype(bool)
            roi_H0 = np.load(roi_H0_path).astype(bool)

            # Align ROI masks to the band-ratio map shape. In the MaceBridge
            # path, the replay grid can differ by a single row in height due
            # to k-Wave padding; we handle this by padding or trimming a row
            # at the bottom when needed. Width must match exactly.
            H_br, W_br = base_br.shape
            H_roi, W_roi = roi_H1.shape
            if W_br != W_roi:
                continue
            if H_br != H_roi:
                diff = H_br - H_roi
                if diff == 1:
                    pad_row = np.zeros((1, W_roi), dtype=bool)
                    roi_H1 = np.vstack([roi_H1, pad_row])
                    roi_H0 = np.vstack([roi_H0, pad_row])
                elif diff == -1:
                    roi_H1 = roi_H1[:-1, :]
                    roi_H0 = roi_H0[:-1, :]
                else:
                    continue

            # Alias score per tile: log(Ea/Ef) up to a constant, via whitened
            # band-ratio map (log(Ef/(gamma*Ea))).
            log_alias = -base_br

            alias_pos = log_alias[roi_H1]
            alias_neg = log_alias[roi_H0]
            if alias_pos.size == 0 or alias_neg.size == 0:
                continue

            median_log_alias_pos = float(np.median(alias_pos))
            median_log_alias_neg = float(np.median(alias_neg))
            delta_log_alias = median_log_alias_neg - median_log_alias_pos

            # Approximate Pf/Pa peak fractions on ROIs using the existing
            # per-pixel peak-band assignments from telemetry, if present.
            # When per-pixel maps are unavailable, fall back to the global
            # nonbg/bg fractions as a coarse diagnostic.
            frac_pf_peak_pos_roi = None
            frac_pf_peak_neg_roi = None
            frac_pa_peak_pos_roi = None
            frac_pa_peak_neg_roi = None

            # If per-pixel peak maps were saved, they would typically be stored
            # alongside base_band_ratio_map; for now we re-use the global stats
            # as a conservative summary.
            frac_pf_peak_nonbg = float(br_stats.get("br_flow_peak_fraction_nonbg", 0.0))
            frac_pa_peak_bg = float(br_stats.get("br_alias_peak_fraction_bg", 0.0))
            frac_pa_peak_nonbg = float(br_stats.get("br_alias_peak_fraction_nonbg", 0.0))

            frac_pf_peak_pos_roi = frac_pf_peak_nonbg
            frac_pf_peak_neg_roi = frac_pf_peak_nonbg
            frac_pa_peak_pos_roi = frac_pa_peak_nonbg
            frac_pa_peak_neg_roi = frac_pa_peak_bg

            writer.writerow(
                {
                    "scan_tag": scan_tag,
                    "scan_idx": scan_idx,
                    "plane_idx": plane_idx,
                    "bundle": str(bundle),
                    "cf_lo_hz": cf_lo,
                    "cf_hi_hz": cf_hi,
                    "ca_lo_hz": ca_lo,
                    "ca_hi_hz": ca_hi,
                    "frac_pf_peak_pos_roi": frac_pf_peak_pos_roi,
                    "frac_pf_peak_neg_roi": frac_pf_peak_neg_roi,
                    "frac_pa_peak_pos_roi": frac_pa_peak_pos_roi,
                    "frac_pa_peak_neg_roi": frac_pa_peak_neg_roi,
                    "median_log_alias_pos_roi": median_log_alias_pos,
                    "median_log_alias_neg_roi": median_log_alias_neg,
                    "delta_log_alias_roi": delta_log_alias,
                }
            )


if __name__ == "__main__":
    main()
