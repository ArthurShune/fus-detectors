import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from sim.kwave.common import (
    SimGeom,
    _beamform_angle,
    _demod_iq,
    _doppler_psd_summary,
    _precompute_geometry,
    _synthesize_cube,
)


def parse_args() -> Path:
    parser = argparse.ArgumentParser(description="Inspect Doppler peaks across flow tiles.")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Pilot run directory (defaults to env RUN_ROOT or runs/pilot/r1_real_psd_v4).",
    )
    args = parser.parse_args()
    if args.run_root is not None:
        return args.run_root
    env_path = os.getenv("RUN_ROOT")
    if env_path:
        return Path(env_path)
    return Path("runs/pilot/r1_real_psd_v4")


RUN_ROOT = parse_args()
BUNDLE_DIR = RUN_ROOT / "pw_7.5MHz_3ang_64T_seed0"

meta = json.load(open(BUNDLE_DIR / "meta.json"))
geom = SimGeom(**meta["sim_geom"])
angle_dirs = sorted(d for d in RUN_ROOT.iterdir() if d.is_dir() and d.name.startswith("angle_"))

XX, ZZ, d_rx = _precompute_geometry(geom)
imgs = []
for d in angle_dirs:
    rf = np.load(d / "rf.npy")
    dt = float(np.load(d / "dt.npy"))
    angle_deg = float(d.name.split("_")[1])
    iq = _demod_iq(rf, dt, geom.f0)
    img = _beamform_angle(iq, angle_deg, dt, geom, XX, ZZ, d_rx)
    imgs.append(img)

image_sets = [np.stack(imgs, axis=0)]
pulses = int(meta["pulses_per_set"])
seed = int(meta["seed"])
Icube = _synthesize_cube(image_sets=image_sets, pulses_per_set=pulses, seed=seed).astype(
    np.complex64
)

mask_flow = np.load(BUNDLE_DIR / "mask_flow.npy")
mask_bg = np.load(BUNDLE_DIR / "mask_bg.npy")
pd_base = np.load(BUNDLE_DIR / "pd_base.npy")
pd_stap = np.load(BUNDLE_DIR / "pd_stap.npy")
Th, Tw = meta["tile_hw"]
stride = meta["tile_stride"]
Lt = int(meta["Lt"])
prf_hz = float(meta["prf_hz"])
fundamental = prf_hz / Lt

records = []
H, W = mask_flow.shape
for y0 in range(0, H - Th + 1, stride):
    for x0 in range(0, W - Tw + 1, stride):
        tile_mask = mask_flow[y0 : y0 + Th, x0 : x0 + Tw]
        if not tile_mask.any():
            continue
        cube_tile = Icube[:, y0 : y0 + Th, x0 : x0 + Tw]
        summary = _doppler_psd_summary(cube_tile, prf_hz, targets_hz=(0.0, fundamental))
        if not summary:
            continue

        base_tile = pd_base[y0 : y0 + Th, x0 : x0 + Tw]
        stap_tile = pd_stap[y0 : y0 + Th, x0 : x0 + Tw]
        flow_vals_base = base_tile[tile_mask]
        flow_vals_stap = stap_tile[tile_mask]
        flow_ratio = None
        if flow_vals_base.size > 0:
            mu_base = float(flow_vals_base.mean())
            mu_stap = float(flow_vals_stap.mean())
            flow_ratio = mu_stap / max(mu_base, 1e-12)

        bg_mask_tile = mask_bg[y0 : y0 + Th, x0 : x0 + Tw]
        bg_ratio = None
        if bg_mask_tile.any():
            base_bg = base_tile[bg_mask_tile]
            stap_bg = stap_tile[bg_mask_tile]
            if base_bg.size > 1:
                var_base = float(base_bg.var())
                var_stap = float(stap_bg.var())
                if var_base > 0.0:
                    bg_ratio = var_stap / max(var_base, 1e-12)

        flow_freq = float(summary.get("psd_power_flow_hz", 0.0))
        alias_ratio = abs(flow_freq) / fundamental if fundamental > 0 else None
        records.append(
            {
                "y0": y0,
                "x0": x0,
                "psd_peak_hz": float(summary.get("psd_peak_hz", 0.0)),
                "psd_peak_power": float(summary.get("psd_peak_power", 0.0)),
                "psd_flow_hz": flow_freq,
                "psd_flow_power": float(summary.get("psd_power_flow", 0.0)),
                "psd_flow_alias_ratio": alias_ratio,
                "psd_flow_alias": bool(alias_ratio is not None and alias_ratio > 1.05),
                "flow_mu_ratio": flow_ratio,
                "flow_coverage": float(tile_mask.mean()),
                "bg_var_inflation": bg_ratio,
            }
        )

abs_peaks = np.array([abs(r["psd_peak_hz"]) for r in records])
print(f"Total flow tiles analysed: {len(records)}")
print(f"Fraction |peak| > 630 Hz: {(abs_peaks > 630).mean():.3f}")
print(f"Fraction |peak| > 900 Hz: {(abs_peaks > 900).mean():.3f}")
print("Peak Hz percentiles:", np.percentile(abs_peaks, [10, 25, 50, 75, 90, 95, 99]))
flow_ratios = np.array([r["flow_mu_ratio"] for r in records if r["flow_mu_ratio"] is not None])
if flow_ratios.size:
    print("Flow mu ratio percentiles:", np.percentile(flow_ratios, [10, 25, 50, 75, 90]))
bg_ratios = np.array([r["bg_var_inflation"] for r in records if r["bg_var_inflation"] is not None])
if bg_ratios.size:
    print("BG var ratio percentiles:", np.percentile(bg_ratios, [10, 50, 90]))

out_csv = RUN_ROOT / "flow_tile_psd.csv"
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "y0",
            "x0",
            "psd_peak_hz",
            "psd_peak_power",
            "psd_flow_hz",
            "psd_flow_power",
            "psd_flow_alias_ratio",
            "psd_flow_alias",
            "flow_mu_ratio",
            "flow_coverage",
            "bg_var_inflation",
        ],
    )
    writer.writeheader()
    writer.writerows(records)
print(f"Wrote {out_csv}")
