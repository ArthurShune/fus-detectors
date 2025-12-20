#!/usr/bin/env python3
"""
Doppler band-occupancy telemetry for MaceBridge-OpenSkull replay bundles.

For each replayed MaceBridge slice (scan*_plane*/pw_*), this script:
  - loads the PD-based flow/background masks (mask_flow, mask_bg),
  - reads whitened band-ratio telemetry from stap_fallback_telemetry
    (flow/alias bands in Hz and peak-band fractions), and
  - computes a simple alias score log(Ea/Ef) from base_band_ratio_map
    and summarizes its medians separately for H1/H0 tiles.

The goal is to produce a Doppler-side analogue of the Macé hemodynamic
telemetry: Pf/Pa peak fractions and alias-score separation between
flow-mask (H1) and background (H0) tiles.

Usage
-----
    PYTHONPATH=. python scripts/macebridge_open_doppler_telemetry.py \\
        --replay-root runs/macebridge_open_replay \\
        --out-csv reports/macebridge_open_doppler_telemetry.csv
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
        description="Doppler band-occupancy telemetry for MaceBridge-OpenSkull replay bundles."
    )
    ap.add_argument(
        "--replay-root",
        type=Path,
        default=Path("runs/macebridge_open_replay"),
        help="Root directory containing scan*_plane*/pw_* replay bundles.",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/macebridge_open_doppler_telemetry.csv"),
        help="Output CSV path for per-slice Doppler telemetry.",
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
        "frac_pf_peak_pos",
        "frac_pa_peak_pos",
        "frac_pa_peak_neg",
        "median_log_alias_pos",
        "median_log_alias_neg",
        "delta_log_alias",
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

            # Flow (H1) / background (H0) Pf/Pa peak fractions from telemetry.
            frac_pf_peak_pos = float(br_stats.get("br_flow_peak_fraction_nonbg", 0.0))
            frac_pa_peak_neg = float(br_stats.get("br_alias_peak_fraction_bg", 0.0))
            frac_pa_peak_pos = float(br_stats.get("br_alias_peak_fraction_nonbg", 0.0))

            # Alias score per tile: log(Ea/Ef) up to a constant. The whitened
            # band-ratio map stores log(Ef/(gamma*Ea)); negating it (and ignoring
            # the gamma constant) yields a log-alias ratio whose class-median
            # differences match R-2 style checks.
            base_br_path = bundle / "base_band_ratio_map.npy"
            mask_flow_path = bundle / "mask_flow.npy"
            mask_bg_path = bundle / "mask_bg.npy"
            if not (base_br_path.exists() and mask_flow_path.exists() and mask_bg_path.exists()):
                continue

            base_br = np.load(base_br_path).astype(np.float64)
            mask_flow = np.load(mask_flow_path).astype(bool)
            mask_bg = np.load(mask_bg_path).astype(bool)

            if base_br.shape != mask_flow.shape or base_br.shape != mask_bg.shape:
                # Require spatial alignment between band-ratio map and masks.
                continue

            log_alias = -base_br  # proportional to log(Ea/Ef).

            alias_pos = log_alias[mask_flow]
            alias_neg = log_alias[mask_bg]
            if alias_pos.size == 0 or alias_neg.size == 0:
                continue

            median_log_alias_pos = float(np.median(alias_pos))
            median_log_alias_neg = float(np.median(alias_neg))
            delta_log_alias = median_log_alias_neg - median_log_alias_pos

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

            writer.writerow(
                {
                    "scan_tag": scan_tag,
                    "scan_idx": scan_idx if scan_idx is not None else "",
                    "plane_idx": plane_idx if plane_idx is not None else "",
                    "bundle": str(bundle),
                    "cf_lo_hz": cf_lo,
                    "cf_hi_hz": cf_hi,
                    "ca_lo_hz": ca_lo,
                    "ca_hi_hz": ca_hi,
                    "frac_pf_peak_pos": frac_pf_peak_pos,
                    "frac_pa_peak_pos": frac_pa_peak_pos,
                    "frac_pa_peak_neg": frac_pa_peak_neg,
                    "median_log_alias_pos": median_log_alias_pos,
                    "median_log_alias_neg": median_log_alias_neg,
                    "delta_log_alias": delta_log_alias,
                }
            )


if __name__ == "__main__":
    main()

