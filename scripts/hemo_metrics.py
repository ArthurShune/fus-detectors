#!/usr/bin/env python3
"""
Compute hemodynamic fidelity metrics for STAP bundles.

For each bundle we report:
  - PD-SNR (baseline vs STAP) using flow/background masks
  - Flow relative error vs baseline PD (mean absolute deviation / baseline)
  - Flow correlation (Pearson) between PD maps inside the flow mask
  - Background mean change (relative)

Usage:
    PYTHONPATH=. python scripts/hemo_metrics.py \
        runs/motion/alias/pw_7.5MHz_5ang_2ens_128T_seed1 \
        runs/motion/alias_pftrace/pw_7.5MHz_5ang_2ens_128T_seed1
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from eval.metrics import pd_snr_db


def load_bundle(bundle_path: Path) -> dict:
    pd_base = np.load(bundle_path / "pd_base.npy")
    pd_stap = np.load(bundle_path / "pd_stap.npy")
    mask_flow = np.load(bundle_path / "mask_flow.npy").astype(bool)
    mask_bg = np.load(bundle_path / "mask_bg.npy").astype(bool)
    return {
        "name": bundle_path.name,
        "label": bundle_path.parent.name,
        "pd_base": pd_base,
        "pd_stap": pd_stap,
        "mask_flow": mask_flow,
        "mask_bg": mask_bg,
    }


def hemo_metrics(bundle: dict) -> dict:
    pd_base = bundle["pd_base"]
    pd_stap = bundle["pd_stap"]
    mask_flow = bundle["mask_flow"]
    mask_bg = bundle["mask_bg"]
    eps = 1e-12

    base_flow = pd_base[mask_flow]
    stap_flow = pd_stap[mask_flow]
    base_bg = pd_base[mask_bg]
    stap_bg = pd_stap[mask_bg]

    flow_rel_error = float(np.mean(np.abs(stap_flow - base_flow) / (np.abs(base_flow) + eps)))
    flow_corr = float(np.corrcoef(base_flow.ravel(), stap_flow.ravel())[0, 1])
    bg_rel_change = float((np.mean(stap_bg) - np.mean(base_bg)) / (np.mean(base_bg) + eps))
    pd_snr_base = float(pd_snr_db(pd_base, mask_flow, mask_bg))
    pd_snr_stap = float(pd_snr_db(pd_stap, mask_flow, mask_bg))

    return {
        "bundle": bundle["name"],
        "label": bundle["label"],
        "flow_rel_error": flow_rel_error,
        "flow_corr": flow_corr,
        "bg_rel_change": bg_rel_change,
        "pd_snr_base": pd_snr_base,
        "pd_snr_stap": pd_snr_stap,
        "pd_snr_gain": pd_snr_stap - pd_snr_base,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Hemodynamic fidelity metrics for STAP bundles")
    ap.add_argument("bundles", nargs="+", type=Path, help="Paths to bundle directories (pw_*)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    results: List[dict] = []
    for bundle_dir in args.bundles:
        bundle = load_bundle(bundle_dir)
        metrics = hemo_metrics(bundle)
        results.append(metrics)
        print(
            f"{metrics['label']}/{metrics['bundle']}: "
            f"flow_rel_err={metrics['flow_rel_error']:.4f}, "
            f"flow_corr={metrics['flow_corr']:.4f}, "
            f"bg_rel_change={metrics['bg_rel_change']:.4f}, "
            f"SNR_base={metrics['pd_snr_base']:.2f} dB, "
            f"SNR_stap={metrics['pd_snr_stap']:.2f} dB, "
            f"SNR_gain={metrics['pd_snr_gain']:.2f} dB"
        )


if __name__ == "__main__":
    main()
