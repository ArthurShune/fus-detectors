#!/usr/bin/env python3
"""
Summarize STAP debug NPZ files for quick inspection of flow tiles, KA settings, and LCMV responses.

Usage:
    PYTHONPATH=. conda run -n stap-fus python scripts/analyze_stap_debug.py \
        --bundle runs/pilot/r1_real_psd_bg_guard095_coords/pw_7.5MHz_3ang_64T_seed0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def summarize_debug(bundle_dir: Path, limit: int) -> None:
    debug_dir = bundle_dir / "stap_debug"
    if not debug_dir.exists():
        print(f"[warn] No stap_debug directory under {bundle_dir}")
        return

    def safe_float(val) -> float:
        try:
            if val is None:
                return float("nan")
            return float(val)
        except Exception:
            return float("nan")

    rows = []
    for npz_path in sorted(debug_dir.glob("*.npz")):
        data = load_npz(npz_path)
        flow_ratio = safe_float(data.get("flow_mu_ratio"))
        diag_load = safe_float(data.get("diag_load"))
        ka_beta = safe_float(data.get("ka_beta"))
        ka_lambda = safe_float(data.get("ka_lambda_used"))
        alias_ratio = safe_float(data.get("psd_flow_alias_ratio"))
        y0 = safe_float(data.get("y0"))
        x0 = safe_float(data.get("x0"))
        weights = data.get("lcmv_weights")
        flow_resp = data.get("flow_response")
        motion_resp = data.get("motion_response")
        band_q = data.get("band_fraction_quantiles")
        score_q = data.get("score_quantiles")
        psd_ratio = safe_float(data.get("psd_flow_to_dc_ratio"))
        psd_peak = safe_float(data.get("psd_peak_hz"))
        row = {
            "file": npz_path.name,
            "y0": y0,
            "x0": x0,
            "flow_mu_ratio": flow_ratio,
            "bg_var_inflation": safe_float(data.get("bg_var_inflation")),
            "diag_load": diag_load,
            "load_mode": str(data.get("load_mode")),
            "ka_beta": ka_beta,
            "ka_lambda": ka_lambda,
            "psd_alias_ratio": alias_ratio,
            "psd_flow_to_dc_ratio": psd_ratio,
            "psd_peak_hz": psd_peak,
            "band_frac_q50": safe_float(
                band_q[1] if isinstance(band_q, np.ndarray) and band_q.size >= 2 else None
            ),
            "score_q50": safe_float(
                score_q[1] if isinstance(score_q, np.ndarray) and score_q.size >= 2 else None
            ),
            "kc_flow": int(data.get("kc_flow", 0)),
            "has_weights": weights is not None,
            "flow_resp_min": float(np.min(np.abs(flow_resp))) if flow_resp is not None else np.nan,
            "flow_resp_max": float(np.max(np.abs(flow_resp))) if flow_resp is not None else np.nan,
            "motion_resp_max": (
                float(np.max(np.abs(motion_resp))) if motion_resp is not None else np.nan
            ),
        }
        rows.append(row)
    rows.sort(key=lambda r: r["flow_mu_ratio"])
    print(f"Found {len(rows)} debug tiles")
    lim = max(1, limit)
    print("Lowest flow ratios:")
    for row in rows[: min(lim, len(rows))]:
        print(row)
    print("Highest flow ratios:")
    for row in rows[-min(lim, len(rows)) :]:
        print(row)
    rows_bg = [r for r in rows if np.isfinite(r["bg_var_inflation"])]
    rows_bg.sort(key=lambda r: r["bg_var_inflation"], reverse=True)
    if rows_bg:
        print("Highest background variance inflation:")
        for row in rows_bg[: min(lim, len(rows_bg))]:
            print(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze STAP debug NPZ payloads.")
    ap.add_argument(
        "--bundle", type=Path, required=True, help="Bundle directory containing stap_debug/"
    )
    ap.add_argument(
        "--limit", type=int, default=5, help="Number of entries to print per section (default: 5)"
    )
    args = ap.parse_args()
    summarize_debug(args.bundle, args.limit)


if __name__ == "__main__":
    main()
