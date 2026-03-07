#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates

from scripts.simus_eval_motion import make_motion_ladder_config
from scripts.simus_eval_structural import _split_csv_list
from sim.simus.bundle import estimate_simus_policy_features, load_canonical_run
from sim.simus.motion import build_motion_artifacts


PROFILE_TO_NOMOTION_RUN = {
    "ClinIntraOp-Pf-v1": Path("runs/sim_eval/simus_motion_ladder_intraop_paper_seed21/runs/simus_clinintraop_pf_v1_motionx0_phasex0_seed21"),
    "ClinMobile-Pf-v1": Path("runs/sim_eval/simus_motion_ladder_mobile_paper_seed21/runs/simus_clinmobile_pf_v1_motionx0_phasex0_seed21"),
}


def _warp_icube(icube: np.ndarray, dx_px: np.ndarray, dz_px: np.ndarray) -> np.ndarray:
    T, H, W = icube.shape
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
    out = np.empty_like(icube)
    for t in range(T):
        coords = np.stack([yy - np.asarray(dz_px[t], dtype=np.float32), xx - np.asarray(dx_px[t], dtype=np.float32)], axis=0)
        real = map_coordinates(np.asarray(icube[t].real, dtype=np.float32), coords, order=1, mode="nearest", prefilter=False)
        imag = map_coordinates(np.asarray(icube[t].imag, dtype=np.float32), coords, order=1, mode="nearest", prefilter=False)
        out[t] = real + 1j * imag
    return out.astype(np.complex64, copy=False)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Approximate corrected SIMUS motion-regime telemetry by warping existing no-motion beamformed IQ cubes.")
    ap.add_argument("--profiles", type=str, default="ClinIntraOp-Pf-v1,ClinMobile-Pf-v1")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--tier", type=str, default="paper", choices=["smoke", "paper"])
    ap.add_argument("--motion-scales", type=str, default="0.25,0.5,1.0")
    ap.add_argument(
        "--real-telemetry-json",
        type=Path,
        default=Path("reports/simus_sanity_link/real_motion_proxy_telemetry.json"),
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/simus_motion/simus_motion_proxy_warp_check_seed21.csv"),
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/simus_motion/simus_motion_proxy_warp_check_seed21.json"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    profiles = _split_csv_list(str(args.profiles))
    motion_scales = [float(x) for x in _split_csv_list(str(args.motion_scales))]
    real_payload = json.loads(Path(args.real_telemetry_json).read_text(encoding="utf-8"))
    real_max = float((real_payload.get("overall") or {}).get("reg_shift_p90_max"))
    rows: list[dict[str, Any]] = []

    for profile in profiles:
        run_dir = PROFILE_TO_NOMOTION_RUN.get(str(profile))
        if run_dir is None:
            raise ValueError(f"no default no-motion run registered for profile {profile!r}")
        icube, _, _ = load_canonical_run(run_dir)
        for motion_scale in motion_scales:
            cfg = make_motion_ladder_config(
                profile=str(profile),
                tier=str(args.tier),
                seed=int(args.seed),
                motion_scale=float(motion_scale),
                phase_scale=float(motion_scale),
            )
            art = build_motion_artifacts(cfg=cfg, seed=int(args.seed))
            moved = _warp_icube(icube, art.dx_px, art.dz_px)
            feat = estimate_simus_policy_features(moved, reg_subpixel=4, reg_reference="median")
            rows.append(
                {
                    "profile": str(profile),
                    "seed": int(args.seed),
                    "motion_scale": float(motion_scale),
                    "nomotion_run_dir": str(run_dir),
                    "disp_rms_px": float(art.telemetry.get("disp_rms_px") or 0.0),
                    "disp_p90_px": float(art.telemetry.get("disp_p90_px") or 0.0),
                    "rigid_rms_px": float(art.telemetry.get("rigid_rms_px") or 0.0),
                    "rigid_p90_px": float(art.telemetry.get("rigid_p90_px") or 0.0),
                    "reg_shift_rms": float(feat["reg_shift_rms"]),
                    "reg_shift_p90": float(feat["reg_shift_p90"]),
                    "reg_psr_median": float(feat["reg_psr_median"]),
                    "real_reg_shift_p90_max": float(real_max),
                    "within_real_reg_shift_envelope": bool(float(feat["reg_shift_p90"]) <= float(real_max)),
                }
            )

    payload = {
        "schema_version": "simus_motion_proxy_warp_check.v1",
        "seed": int(args.seed),
        "tier": str(args.tier),
        "motion_scales": motion_scales,
        "real_reg_shift_p90_max": float(real_max),
        "rows": rows,
    }
    _write_csv(Path(args.out_csv), rows)
    _write_json(Path(args.out_json), payload)
    print(f"[simus-motion-proxy-warp-check] wrote {args.out_csv}")
    print(f"[simus-motion-proxy-warp-check] wrote {args.out_json}")


if __name__ == "__main__":
    main()
