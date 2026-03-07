#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.simus_eval_motion import make_motion_ladder_config
from scripts.simus_eval_structural import _split_csv_list
from sim.simus.bundle import estimate_simus_policy_features, load_canonical_run
from sim.simus.pilot_pymust_simus import SUPPORTED_SIMUS_PROFILES, write_simus_run


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


def summarize_rows(rows: list[dict[str, Any]], *, real_reg_shift_p90_max: float | None) -> dict[str, Any]:
    by_profile: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_profile.setdefault(str(row["profile"]), []).append(row)
    out: dict[str, Any] = {}
    for profile, items in sorted(by_profile.items()):
        ordered = sorted(items, key=lambda r: (int(r["seed"]), float(r["motion_scale"])))
        inside = [r for r in ordered if bool(r["within_real_reg_shift_envelope"])]
        out[profile] = {
            "n": int(len(ordered)),
            "max_motion_scale_within_real_envelope": max((float(r["motion_scale"]) for r in inside), default=None),
            "min_motion_scale_above_real_envelope": min(
                (float(r["motion_scale"]) for r in ordered if not bool(r["within_real_reg_shift_envelope"])),
                default=None,
            ),
            "median_reg_shift_p90": float(np.median([float(r["reg_shift_p90"]) for r in ordered])),
            "median_motion_disp_rms_px": float(np.median([float(r["motion_disp_rms_px"]) for r in ordered])),
            "real_reg_shift_p90_max": real_reg_shift_p90_max,
        }
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Scan lower SIMUS motion scales and compare registration proxy telemetry to the real-data envelope.")
    ap.add_argument("--profiles", type=str, default="ClinIntraOp-Pf-v1,ClinMobile-Pf-v1")
    ap.add_argument("--tier", type=str, default="paper", choices=["smoke", "paper"])
    ap.add_argument("--seeds", type=str, default="21")
    ap.add_argument("--motion-scales", type=str, default="0,0.05,0.1,0.15,0.2,0.25")
    ap.add_argument("--out-root", type=Path, default=Path("runs/sim_eval/simus_motion_anchor_scan"))
    ap.add_argument("--out-csv", type=Path, default=Path("reports/simus_motion/simus_motion_anchor_scan.csv"))
    ap.add_argument("--out-json", type=Path, default=Path("reports/simus_motion/simus_motion_anchor_scan.json"))
    ap.add_argument(
        "--real-telemetry-json",
        type=Path,
        default=Path("reports/simus_sanity_link/real_motion_proxy_telemetry.json"),
    )
    ap.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    profiles = _split_csv_list(str(args.profiles))
    bad_profiles = [p for p in profiles if p not in SUPPORTED_SIMUS_PROFILES]
    if bad_profiles:
        raise ValueError(f"unknown SIMUS profiles: {bad_profiles}")
    seeds = [int(x) for x in _split_csv_list(str(args.seeds))]
    motion_scales = [float(x) for x in _split_csv_list(str(args.motion_scales))]

    real_reg_shift_p90_max: float | None = None
    if Path(args.real_telemetry_json).is_file():
        payload = json.loads(Path(args.real_telemetry_json).read_text(encoding="utf-8"))
        real_reg_shift_p90_max = (payload.get("overall") or {}).get("reg_shift_p90_max")
        real_reg_shift_p90_max = float(real_reg_shift_p90_max) if real_reg_shift_p90_max is not None else None

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for profile in profiles:
        for seed in seeds:
            for motion_scale in motion_scales:
                cfg = make_motion_ladder_config(
                    profile=str(profile),
                    tier=str(args.tier),
                    seed=int(seed),
                    motion_scale=float(motion_scale),
                    phase_scale=float(motion_scale),
                )
                run_dir = out_root / "runs" / cfg.profile.replace("+", "_plus_")
                if not bool(args.reuse_existing) or not (run_dir / "dataset" / "meta.json").is_file():
                    write_simus_run(out_root=run_dir, cfg=cfg, skip_bundle=True)
                icube, _, meta = load_canonical_run(run_dir)
                feat = estimate_simus_policy_features(icube, reg_subpixel=4, reg_reference="median")
                reg_shift_p90 = float(feat["reg_shift_p90"])
                row = {
                    "profile": str(profile),
                    "seed": int(seed),
                    "motion_scale": float(motion_scale),
                    "run_dir": str(run_dir),
                    "T": int(icube.shape[0]),
                    "H": int(icube.shape[1]),
                    "W": int(icube.shape[2]),
                    "motion_disp_rms_px": float(meta.get("motion", {}).get("telemetry", {}).get("disp_rms_px") or 0.0),
                    "phase_rms_rad": float(meta.get("phase_screen", {}).get("telemetry", {}).get("phase_rms_rad") or 0.0),
                    "reg_shift_rms": float(feat["reg_shift_rms"]),
                    "reg_shift_p90": reg_shift_p90,
                    "reg_psr_median": float(feat["reg_psr_median"]),
                    "reg_ms_prepass": float(feat["reg_ms_prepass"]),
                    "real_reg_shift_p90_max": real_reg_shift_p90_max,
                    "within_real_reg_shift_envelope": bool(
                        real_reg_shift_p90_max is not None and reg_shift_p90 <= float(real_reg_shift_p90_max)
                    ),
                }
                rows.append(row)

    payload = {
        "schema_version": "simus_motion_anchor_scan.v1",
        "profiles": profiles,
        "seeds": seeds,
        "motion_scales": motion_scales,
        "real_reg_shift_p90_max": real_reg_shift_p90_max,
        "rows": rows,
        "summary_by_profile": summarize_rows(rows, real_reg_shift_p90_max=real_reg_shift_p90_max),
    }
    _write_csv(Path(args.out_csv), rows)
    _write_json(Path(args.out_json), payload)
    print(f"[simus-motion-anchor-scan] wrote {args.out_csv}")
    print(f"[simus-motion-anchor-scan] wrote {args.out_json}")


if __name__ == "__main__":
    main()
