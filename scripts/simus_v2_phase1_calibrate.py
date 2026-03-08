#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path
from typing import Any

from scripts.simus_v2_acceptance import _combine_anchor_envelope, evaluate_run, _write_csv, _write_json
from scripts.simus_v2_anchor_envelopes import DEFAULT_ACCEPTANCE_METRICS
from scripts.physical_doppler_sanity_link import BandEdges, TileSpec
from sim.simus.config import default_profile_config
from sim.simus.pilot_pymust_simus import write_simus_run


CANDIDATES: dict[str, dict[str, Any]] = {
    "base": {
        "description": "Current frozen ClinIntraOp-Pf-v2 defaults.",
        "motion": {},
        "phase_screen": {},
    },
    "calA": {
        "description": "Balanced residual-motion increase with moderate phase drift.",
        "motion": {
            "random_walk_sigma_px": 0.03,
            "drift_x_px": 0.18,
            "drift_z_px": 0.08,
            "elastic_amp_px": 0.28,
            "elastic_sigma_px": 16.0,
            "elastic_temporal_rho": 0.85,
        },
        "phase_screen": {
            "std_rad": 0.45,
            "corr_len_elem": 8.0,
            "drift_rho": 0.97,
            "drift_sigma_rad": 0.02,
        },
    },
    "calB": {
        "description": "Stronger stochastic residual motion with faster elastic decorrelation.",
        "motion": {
            "random_walk_sigma_px": 0.05,
            "drift_x_px": 0.26,
            "drift_z_px": 0.11,
            "elastic_amp_px": 0.38,
            "elastic_sigma_px": 18.0,
            "elastic_temporal_rho": 0.72,
        },
        "phase_screen": {
            "std_rad": 0.45,
            "corr_len_elem": 7.0,
            "drift_rho": 0.95,
            "drift_sigma_rad": 0.025,
        },
    },
    "calC": {
        "description": "Phase-heavy decorrelation with moderate residual motion.",
        "motion": {
            "random_walk_sigma_px": 0.04,
            "drift_x_px": 0.20,
            "drift_z_px": 0.09,
            "elastic_amp_px": 0.24,
            "elastic_sigma_px": 14.0,
            "elastic_temporal_rho": 0.78,
        },
        "phase_screen": {
            "std_rad": 0.58,
            "corr_len_elem": 5.0,
            "drift_rho": 0.92,
            "drift_sigma_rad": 0.04,
        },
    },
    "calD": {
        "description": "High-motion candidate with moderate phase drift.",
        "motion": {
            "random_walk_sigma_px": 0.06,
            "pulse_jitter_sigma_px": 0.08,
            "drift_x_px": 0.30,
            "drift_z_px": 0.13,
            "elastic_amp_px": 0.46,
            "elastic_sigma_px": 18.0,
            "elastic_temporal_rho": 0.68,
        },
        "phase_screen": {
            "std_rad": 0.40,
            "corr_len_elem": 8.0,
            "drift_rho": 0.95,
            "drift_sigma_rad": 0.02,
        },
    },
    "calJ1": {
        "description": "Balanced candidate with explicit pulse-to-pulse residual jitter.",
        "motion": {
            "random_walk_sigma_px": 0.03,
            "pulse_jitter_sigma_px": 0.12,
            "drift_x_px": 0.18,
            "drift_z_px": 0.08,
            "elastic_amp_px": 0.28,
            "elastic_sigma_px": 16.0,
            "elastic_temporal_rho": 0.80,
            "elastic_mode_count": 1,
        },
        "phase_screen": {
            "std_rad": 0.45,
            "corr_len_elem": 7.0,
            "drift_rho": 0.95,
            "drift_sigma_rad": 0.02,
        },
    },
    "calJ2": {
        "description": "Stronger jitter-led candidate for aggressive background decorrelation.",
        "motion": {
            "random_walk_sigma_px": 0.04,
            "pulse_jitter_sigma_px": 0.20,
            "drift_x_px": 0.22,
            "drift_z_px": 0.10,
            "elastic_amp_px": 0.30,
            "elastic_sigma_px": 14.0,
            "elastic_temporal_rho": 0.75,
            "elastic_mode_count": 1,
        },
        "phase_screen": {
            "std_rad": 0.50,
            "corr_len_elem": 6.0,
            "drift_rho": 0.93,
            "drift_sigma_rad": 0.03,
        },
    },
    "calM1": {
        "description": "Multi-mode elastic residual with moderate pulse jitter.",
        "motion": {
            "random_walk_sigma_px": 0.03,
            "pulse_jitter_sigma_px": 0.08,
            "drift_x_px": 0.18,
            "drift_z_px": 0.08,
            "elastic_amp_px": 0.34,
            "elastic_sigma_px": 16.0,
            "elastic_temporal_rho": 0.78,
            "elastic_mode_count": 3,
        },
        "phase_screen": {
            "std_rad": 0.45,
            "corr_len_elem": 7.0,
            "drift_rho": 0.95,
            "drift_sigma_rad": 0.02,
        },
    },
    "calM2": {
        "description": "Stronger multi-mode elastic residual with lighter jitter for coherence balance.",
        "motion": {
            "random_walk_sigma_px": 0.04,
            "pulse_jitter_sigma_px": 0.06,
            "drift_x_px": 0.20,
            "drift_z_px": 0.09,
            "elastic_amp_px": 0.42,
            "elastic_sigma_px": 14.0,
            "elastic_temporal_rho": 0.72,
            "elastic_mode_count": 4,
        },
        "phase_screen": {
            "std_rad": 0.42,
            "corr_len_elem": 8.0,
            "drift_rho": 0.95,
            "drift_sigma_rad": 0.018,
        },
    },
    "calM3": {
        "description": "Higher-mode residual field with moderate jitter to reduce background rank concentration.",
        "motion": {
            "random_walk_sigma_px": 0.035,
            "pulse_jitter_sigma_px": 0.075,
            "drift_x_px": 0.20,
            "drift_z_px": 0.09,
            "elastic_amp_px": 0.50,
            "elastic_sigma_px": 13.0,
            "elastic_temporal_rho": 0.68,
            "elastic_mode_count": 8,
        },
        "phase_screen": {
            "std_rad": 0.40,
            "corr_len_elem": 8.0,
            "drift_rho": 0.95,
            "drift_sigma_rad": 0.016,
        },
    },
    "calL1": {
        "description": "Localized multi-mode residual field tuned from the near-pass M2 candidate.",
        "motion": {
            "random_walk_sigma_px": 0.04,
            "pulse_jitter_sigma_px": 0.06,
            "drift_x_px": 0.20,
            "drift_z_px": 0.09,
            "elastic_amp_px": 0.42,
            "elastic_sigma_px": 14.0,
            "elastic_temporal_rho": 0.72,
            "elastic_mode_count": 4,
        },
        "phase_screen": {
            "std_rad": 0.42,
            "corr_len_elem": 8.0,
            "drift_rho": 0.95,
            "drift_sigma_rad": 0.018,
        },
    },
}


def _parse_list(values: list[str] | None) -> list[str]:
    out: list[str] = []
    for value in values or []:
        for part in str(value).split(","):
            text = part.strip()
            if text:
                out.append(text)
    return out


def _norm_miss(value: float | None, lo: float | None, hi: float | None) -> float | None:
    if value is None or lo is None or hi is None:
        return None
    width = max(float(hi) - float(lo), 1e-6)
    if float(lo) <= float(value) <= float(hi):
        return 0.0
    if float(value) < float(lo):
        return float(lo - value) / width
    return float(value - hi) / width


def _candidate_cfg(profile: str, tier: str, seed: int, candidate_name: str):
    cfg = default_profile_config(profile=profile, tier=tier, seed=seed)  # type: ignore[arg-type]
    spec = CANDIDATES[candidate_name]
    motion = dataclasses.replace(cfg.motion, **spec["motion"])
    phase = dataclasses.replace(cfg.phase_screen, **spec["phase_screen"])
    return dataclasses.replace(cfg, motion=motion, phase_screen=phase)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run named Phase 1 calibration candidates against the frozen SIMUS v2 acceptance gate.")
    ap.add_argument("--profile", type=str, default="ClinIntraOp-Pf-v2")
    ap.add_argument("--tier", type=str, default="paper", choices=["smoke", "paper"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--candidate", action="append", default=None, help="Candidate name(s); comma-separated allowed.")
    ap.add_argument("--anchor-json", type=Path, default=Path("reports/simus_v2/anchors/simus_v2_anchor_envelopes.json"))
    ap.add_argument("--anchor-kind", action="append", default=None, help="Anchor kind(s); comma-separated allowed.")
    ap.add_argument("--out-root", type=Path, default=Path("runs/sim"))
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--reuse-existing", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    candidates = _parse_list(args.candidate) or ["base", "calA", "calB", "calC", "calD"]
    unknown = [name for name in candidates if name not in CANDIDATES]
    if unknown:
        raise SystemExit(f"unknown candidates: {', '.join(sorted(unknown))}")

    anchor_payload = json.loads(Path(args.anchor_json).read_text(encoding="utf-8"))
    anchor_kinds = _parse_list(args.anchor_kind) or ["shin", "gammex_along", "gammex_across", "ulm_7883227"]
    metrics = list(DEFAULT_ACCEPTANCE_METRICS)
    acceptance_env = _combine_anchor_envelope(
        anchor_payload,
        anchor_kinds,
        metrics,
        lower_q=0.10,
        upper_q=0.90,
    )
    bands = BandEdges(pf_lo_hz=30.0, pf_hi_hz=250.0, pg_lo_hz=250.0, pg_hi_hz=400.0, pa_lo_hz=400.0)
    tile = TileSpec(h=8, w=8, stride=3)

    stem = f"simus_v2_phase1_calibration_{args.profile.replace('-', '_').lower()}_{args.tier}_seed{args.seed}"
    out_csv = args.out_csv or Path("reports/simus_v2/acceptance") / f"{stem}.csv"
    out_json = args.out_json or Path("reports/simus_v2/acceptance") / f"{stem}.json"

    rows: list[dict[str, Any]] = []
    payload_runs: list[dict[str, Any]] = []
    for candidate_name in candidates:
        run_root = Path(args.out_root) / f"{stem}_{candidate_name}"
        if not (args.reuse_existing and (run_root / "dataset" / "meta.json").is_file()):
            cfg = _candidate_cfg(args.profile, args.tier, int(args.seed), candidate_name)
            write_simus_run(out_root=run_root, cfg=cfg, skip_bundle=True)
        summary, metric_rows = evaluate_run(
            run_dir=run_root,
            bands=bands,
            tile=tile,
            acceptance_env=acceptance_env,
            metrics=metrics,
        )
        mean_miss = 0.0
        max_miss = 0.0
        miss_n = 0
        for metric_row in metric_rows:
            miss = _norm_miss(metric_row.get("value"), metric_row.get("lo"), metric_row.get("hi"))
            metric_row["norm_miss"] = miss
            if miss is not None:
                mean_miss += float(miss)
                max_miss = max(max_miss, float(miss))
                miss_n += 1
        mean_miss = float(mean_miss / miss_n) if miss_n else 0.0
        row = {
            "candidate": candidate_name,
            "description": CANDIDATES[candidate_name]["description"],
            "run_dir": str(run_root),
            "passed_metrics": int(summary["passed_metrics"]),
            "required_metrics": int(summary["required_metrics"]),
            "failed_metrics": int(summary["failed_metrics"]),
            "pass_fraction": summary["pass_fraction"],
            "overall_pass": bool(summary["overall_pass"]),
            "mean_norm_miss": mean_miss,
            "max_norm_miss": max_miss,
        }
        for metric_row in metric_rows:
            metric = str(metric_row["metric"])
            row[f"{metric}__value"] = metric_row["value"]
            row[f"{metric}__status"] = metric_row["status"]
            row[f"{metric}__norm_miss"] = metric_row["norm_miss"]
        rows.append(row)
        payload_runs.append(
            {
                "candidate": candidate_name,
                "description": CANDIDATES[candidate_name]["description"],
                "summary": summary,
                "metrics": metric_rows,
            }
        )

    rows.sort(
        key=lambda r: (
            -int(bool(r["overall_pass"])),
            -int(r["passed_metrics"]),
            float(r["mean_norm_miss"]),
            float(r["max_norm_miss"]),
            str(r["candidate"]),
        )
    )
    best = rows[0] if rows else None
    out_payload = {
        "schema_version": "simus_v2_phase1_calibration.v1",
        "profile": args.profile,
        "tier": args.tier,
        "seed": int(args.seed),
        "anchor_json": str(args.anchor_json),
        "anchor_kinds": anchor_kinds,
        "metrics": metrics,
        "best_candidate": None if best is None else best["candidate"],
        "runs": payload_runs,
    }
    _write_csv(out_csv, rows)
    _write_json(out_json, out_payload)
    print(f"[simus-v2-phase1-calibrate] wrote {out_csv}")
    print(f"[simus-v2-phase1-calibrate] wrote {out_json}")
    if best is not None:
        print(
            "[simus-v2-phase1-calibrate] best",
            best["candidate"],
            f"pass={best['passed_metrics']}/{best['required_metrics']}",
            f"mean_norm_miss={best['mean_norm_miss']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
