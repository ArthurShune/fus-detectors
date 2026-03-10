#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import dataclasses
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

from scripts.simus_v2_acceptance import ANCHOR_PRESETS, _combine_anchor_envelope, evaluate_run, _write_csv, _write_json
from scripts.simus_v2_anchor_envelopes import DEFAULT_ACCEPTANCE_METRICS
from scripts.physical_doppler_sanity_link import BandEdges, TileSpec
from sim.simus.config import BackgroundCompartmentSpec, StructuredClutterSpec, VesselSpec, default_profile_config
from sim.simus.pilot_pymust_simus import write_simus_run


CANDIDATES: dict[str, dict[str, Any]] = {
    "base": {
        "description": "Current frozen ClinIntraOp-Pf-v2 defaults.",
        "motion": {},
        "phase_screen": {},
        "noise": {},
        "background": {},
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
        "noise": {},
        "background": {},
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
        "noise": {},
        "background": {},
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
        "noise": {},
        "background": {},
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
        "noise": {},
        "background": {},
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
        "noise": {},
        "background": {},
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
        "noise": {},
        "background": {},
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
        "noise": {},
        "background": {},
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
        "noise": {},
        "background": {},
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
        "noise": {},
        "background": {},
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
        "noise": {},
        "background": {},
    },
    "stabI1": {
        "description": "Lower-variance intra-op residuals with modest background-motion damping.",
        "motion": {
            "random_walk_sigma_px": 0.025,
            "pulse_jitter_sigma_px": 0.035,
            "drift_x_px": 0.16,
            "drift_z_px": 0.07,
            "elastic_amp_px": 0.34,
            "elastic_temporal_rho": 0.78,
            "elastic_mode_count": 3,
        },
        "phase_screen": {
            "std_rad": 0.40,
            "corr_len_elem": 8.0,
            "drift_rho": 0.95,
            "drift_sigma_rad": 0.018,
        },
        "noise": {
            "iq_rms_frac": 0.26,
        },
        "background": {
            "motion_amp_scale": 0.82,
            "motion_jitter_scale": 0.60,
            "motion_rho_delta": 0.04,
            "rc_scale": 0.95,
        },
    },
    "stabI2": {
        "description": "Stronger damping of background compartments with a slightly higher noise floor.",
        "motion": {
            "random_walk_sigma_px": 0.02,
            "pulse_jitter_sigma_px": 0.03,
            "drift_x_px": 0.15,
            "drift_z_px": 0.06,
            "elastic_amp_px": 0.32,
            "elastic_temporal_rho": 0.80,
            "elastic_mode_count": 3,
        },
        "phase_screen": {
            "std_rad": 0.38,
            "corr_len_elem": 8.0,
            "drift_rho": 0.96,
            "drift_sigma_rad": 0.016,
        },
        "noise": {
            "iq_rms_frac": 0.28,
        },
        "background": {
            "motion_amp_scale": 0.72,
            "motion_jitter_scale": 0.50,
            "motion_rho_delta": 0.06,
            "rc_scale": 0.90,
        },
    },
    "stabI3": {
        "description": "Reduced compartment dominance with preserved global motion and extra noise decorrelation.",
        "motion": {
            "random_walk_sigma_px": 0.03,
            "pulse_jitter_sigma_px": 0.04,
            "drift_x_px": 0.17,
            "drift_z_px": 0.07,
            "elastic_amp_px": 0.36,
            "elastic_temporal_rho": 0.78,
            "elastic_mode_count": 4,
        },
        "phase_screen": {
            "std_rad": 0.40,
            "corr_len_elem": 8.0,
            "drift_rho": 0.95,
            "drift_sigma_rad": 0.017,
        },
        "noise": {
            "iq_rms_frac": 0.30,
        },
        "background": {
            "motion_amp_scale": 0.68,
            "motion_jitter_scale": 0.45,
            "motion_rho_delta": 0.05,
            "rc_scale": 0.82,
            "scatterer_scale": 0.90,
        },
    },
    "stabI4": {
        "description": "Keep more deterministic motion while strongly narrowing stochastic background spread.",
        "motion": {
            "random_walk_sigma_px": 0.015,
            "pulse_jitter_sigma_px": 0.025,
            "drift_x_px": 0.18,
            "drift_z_px": 0.08,
            "elastic_amp_px": 0.30,
            "elastic_temporal_rho": 0.82,
            "elastic_mode_count": 2,
        },
        "phase_screen": {
            "std_rad": 0.41,
            "corr_len_elem": 8.0,
            "drift_rho": 0.96,
            "drift_sigma_rad": 0.016,
        },
        "noise": {
            "iq_rms_frac": 0.27,
        },
        "background": {
            "motion_amp_scale": 0.65,
            "motion_jitter_scale": 0.35,
            "motion_rho_delta": 0.08,
            "rc_scale": 0.88,
        },
    },
    "stabI5_coupledbg": {
        "description": "Coupled low-frequency tissue background with one dominant mass mode and weaker shear modes.",
        "motion": {},
        "phase_screen": {
            "drift_rho": 1.0,
            "drift_sigma_rad": 0.0,
        },
        "noise": {
            "iq_rms_frac": 0.22,
        },
        "background": {},
        "ordinary_background": {
            "mode": "coupled_mass_shear_v2",
            "global_cutoff_hz": 7.0,
            "shear1_cutoff_hz": 11.0,
            "shear2_cutoff_hz": 13.0,
            "residual_cutoff_hz": 9.0,
            "global_disp_px": 0.10,
            "shear1_disp_px": 0.036,
            "shear2_disp_px": 0.028,
            "residual_disp_px": 0.010,
            "global_dx_scale": 0.07,
            "global_dz_scale": 1.00,
            "shear_dx_scale": 0.20,
            "shear_dz_scale": 0.16,
            "deep_dz_scale": 0.10,
            "sector_mix_scale": 0.10,
            "anchor_motion_scale": 0.14,
            "structured_motion_scale": 1.10,
            "sectorize_mid": False,
        },
    },
    "stabI6_coupledbg_sector": {
        "description": "Coupled low-frequency background with left/right mid-depth sectors to relieve rank-1 dominance.",
        "motion": {},
        "phase_screen": {
            "drift_rho": 1.0,
            "drift_sigma_rad": 0.0,
        },
        "noise": {
            "iq_rms_frac": 0.22,
        },
        "background": {},
        "ordinary_background": {
            "mode": "coupled_mass_shear_v2",
            "global_cutoff_hz": 7.0,
            "shear1_cutoff_hz": 11.0,
            "shear2_cutoff_hz": 13.0,
            "residual_cutoff_hz": 9.0,
            "global_disp_px": 0.095,
            "shear1_disp_px": 0.040,
            "shear2_disp_px": 0.030,
            "residual_disp_px": 0.010,
            "global_dx_scale": 0.07,
            "global_dz_scale": 0.98,
            "shear_dx_scale": 0.22,
            "shear_dz_scale": 0.16,
            "deep_dz_scale": 0.10,
            "sector_mix_scale": 0.16,
            "anchor_motion_scale": 0.14,
            "structured_motion_scale": 1.10,
            "sectorize_mid": True,
        },
    },
    "mobileA": {
        "description": "Reduce additive noise and high-rate jitter while keeping mobile nuisance prevalence intact.",
        "motion": {
            "random_walk_sigma_px": 0.05,
            "pulse_jitter_sigma_px": 0.08,
            "drift_x_px": 0.40,
            "drift_z_px": 0.17,
            "elastic_amp_px": 0.60,
            "elastic_temporal_rho": 0.70,
        },
        "phase_screen": {
            "std_rad": 0.52,
            "corr_len_elem": 7.0,
            "drift_rho": 0.93,
            "drift_sigma_rad": 0.022,
        },
        "noise": {
            "iq_rms_frac": 0.22,
        },
    },
    "mobileB": {
        "description": "Stronger coherence recovery: less noise, less jitter, slightly slower phase drift.",
        "motion": {
            "random_walk_sigma_px": 0.045,
            "pulse_jitter_sigma_px": 0.07,
            "drift_x_px": 0.38,
            "drift_z_px": 0.16,
            "elastic_amp_px": 0.56,
            "elastic_temporal_rho": 0.72,
        },
        "phase_screen": {
            "std_rad": 0.48,
            "corr_len_elem": 8.0,
            "drift_rho": 0.94,
            "drift_sigma_rad": 0.018,
        },
        "noise": {
            "iq_rms_frac": 0.18,
        },
    },
    "mobileC": {
        "description": "Moderate coherence recovery with preserved motion spread and nuisance prevalence.",
        "motion": {
            "random_walk_sigma_px": 0.05,
            "pulse_jitter_sigma_px": 0.08,
            "drift_x_px": 0.42,
            "drift_z_px": 0.18,
            "elastic_amp_px": 0.58,
            "elastic_temporal_rho": 0.74,
        },
        "phase_screen": {
            "std_rad": 0.50,
            "corr_len_elem": 7.0,
            "drift_rho": 0.94,
            "drift_sigma_rad": 0.020,
        },
        "noise": {
            "iq_rms_frac": 0.20,
        },
        "background": {},
    },
    "v3a": {
        "description": "Deeper parenchymal competitive zone with shallower nuisance vessel and boundary geometry.",
        "config": {
            "parenchyma_zone_top_frac": 0.30,
            "surface_nuisance_zone_top_frac": 0.30,
            "bg_top_exclusion_frac": 0.12,
        },
        "motion": {
            "breathing_amp_x_px": 0.34,
            "breathing_amp_z_px": 0.16,
            "drift_x_px": 0.12,
            "drift_z_px": 0.05,
            "elastic_amp_px": 0.24,
        },
        "phase_screen": {},
        "noise": {
            "iq_rms_frac": 0.20,
        },
        "background": {},
        "vessels": {
            "nuisance_superficial": {
                "center_x_m": -6.2e-3,
                "radius_m": 1.20e-3,
                "z_min_m": 5.6e-3,
                "z_max_m": 9.4e-3,
                "blood_vmax_mps": 0.082,
            },
        },
        "structured_clutter": {
            "superficial_sheet": {
                "z0_m": 5.9e-3,
                "z1_m": 6.1e-3,
                "scatterer_count": 260,
            },
            "oblique_boundary": {
                "z0_m": 6.4e-3,
                "z1_m": 14.0e-3,
                "scatterer_count": 220,
            },
        },
    },
    "v3b": {
        "description": "More conservative parenchymal zone with weaker superficial nuisance and quieter intra-op dynamics.",
        "config": {
            "parenchyma_zone_top_frac": 0.34,
            "surface_nuisance_zone_top_frac": 0.34,
            "bg_top_exclusion_frac": 0.14,
        },
        "motion": {
            "breathing_amp_x_px": 0.30,
            "breathing_amp_z_px": 0.14,
            "drift_x_px": 0.10,
            "drift_z_px": 0.04,
            "elastic_amp_px": 0.20,
            "random_walk_sigma_px": 0.02,
            "pulse_jitter_sigma_px": 0.05,
        },
        "phase_screen": {},
        "noise": {
            "iq_rms_frac": 0.18,
        },
        "background": {},
        "vessels": {
            "nuisance_superficial": {
                "center_x_m": -6.4e-3,
                "radius_m": 1.10e-3,
                "z_min_m": 5.6e-3,
                "z_max_m": 9.0e-3,
                "blood_vmax_mps": 0.078,
            },
        },
        "structured_clutter": {
            "superficial_sheet": {
                "z0_m": 5.8e-3,
                "z1_m": 6.0e-3,
                "scatterer_count": 220,
            },
            "oblique_boundary": {
                "z0_m": 6.2e-3,
                "z1_m": 13.2e-3,
                "scatterer_count": 180,
            },
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


def _scale_background_compartments(
    compartments: tuple[BackgroundCompartmentSpec, ...],
    *,
    motion_amp_scale: float = 1.0,
    motion_jitter_scale: float = 1.0,
    motion_sigma_scale: float = 1.0,
    motion_rho_delta: float = 0.0,
    rc_scale: float = 1.0,
    scatterer_scale: float = 1.0,
) -> tuple[BackgroundCompartmentSpec, ...]:
    out: list[BackgroundCompartmentSpec] = []
    for comp in compartments:
        out.append(
            dataclasses.replace(
                comp,
                rc_scale=float(comp.rc_scale) * float(rc_scale),
                scatterer_count=max(1, int(round(float(comp.scatterer_count) * float(scatterer_scale)))),
                motion_amp_px=float(comp.motion_amp_px) * float(motion_amp_scale),
                motion_sigma_px=float(comp.motion_sigma_px) * float(motion_sigma_scale),
                motion_rho=float(min(0.999, max(0.0, float(comp.motion_rho) + float(motion_rho_delta)))),
                motion_jitter_sigma_px=float(comp.motion_jitter_sigma_px) * float(motion_jitter_scale),
            )
        )
    return tuple(out)


def _override_structured_clutter(
    clutter_specs: tuple[StructuredClutterSpec, ...],
    overrides: dict[str, dict[str, Any]],
) -> tuple[StructuredClutterSpec, ...]:
    out: list[StructuredClutterSpec] = []
    for spec in clutter_specs:
        patch = dict(overrides.get(str(spec.name), {}))
        out.append(dataclasses.replace(spec, **patch) if patch else spec)
    return tuple(out)


def _override_vessels(
    vessels: tuple[VesselSpec, ...],
    overrides: dict[str, dict[str, Any]],
) -> tuple[VesselSpec, ...]:
    out: list[VesselSpec] = []
    for vessel in vessels:
        patch = dict(overrides.get(str(vessel.name), {}))
        out.append(dataclasses.replace(vessel, **patch) if patch else vessel)
    return tuple(out)


def _candidate_cfg(profile: str, tier: str, seed: int, candidate_name: str):
    cfg = default_profile_config(profile=profile, tier=tier, seed=seed)  # type: ignore[arg-type]
    spec = CANDIDATES[candidate_name]
    motion = dataclasses.replace(cfg.motion, **spec["motion"])
    phase = dataclasses.replace(cfg.phase_screen, **spec["phase_screen"])
    noise = dataclasses.replace(cfg.noise, **spec.get("noise", {}))
    ordinary_background = dataclasses.replace(cfg.ordinary_background, **dict(spec.get("ordinary_background", {})))
    background = _scale_background_compartments(
        tuple(cfg.background_compartments),
        **dict(spec.get("background", {})),
    )
    vessels = _override_vessels(tuple(cfg.vessels), dict(spec.get("vessels", {})))
    structured_clutter = _override_structured_clutter(
        tuple(cfg.structured_clutter),
        dict(spec.get("structured_clutter", {})),
    )
    return dataclasses.replace(
        cfg,
        motion=motion,
        phase_screen=phase,
        noise=noise,
        background_compartments=background,
        ordinary_background=ordinary_background,
        vessels=vessels,
        structured_clutter=structured_clutter,
        **dict(spec.get("config", {})),
    )


def _prepare_thread_env(threads_per_worker: int | None) -> None:
    if threads_per_worker is None or int(threads_per_worker) <= 0:
        return
    value = str(int(threads_per_worker))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = value


def _run_candidate_seed(task: dict[str, Any]) -> dict[str, Any]:
    _prepare_thread_env(task.get("threads_per_worker"))
    profile = str(task["profile"])
    tier = str(task["tier"])
    seed = int(task["seed"])
    candidate_name = str(task["candidate"])
    run_root = Path(str(task["run_root"]))
    anchor_payload = dict(task["anchor_payload"])
    acceptance_env = task.get("acceptance_env")
    profile_gate = task.get("profile_gate")
    reuse_existing = bool(task.get("reuse_existing", False))
    bands = BandEdges(pf_lo_hz=30.0, pf_hi_hz=250.0, pg_lo_hz=250.0, pg_hi_hz=400.0, pa_lo_hz=400.0)
    tile = TileSpec(h=8, w=8, stride=3)

    if not (reuse_existing and (run_root / "dataset" / "meta.json").is_file()):
        cfg = _candidate_cfg(profile, tier, seed, candidate_name)
        write_simus_run(out_root=run_root, cfg=cfg, skip_bundle=True)
    if profile_gate:
        from scripts.simus_v2_acceptance import _evaluate_profile_gate, summarize_run

        summary, metric_rows = _evaluate_profile_gate(
            run_summary=summarize_run(run_dir=run_root, bands=bands, tile=tile),
            anchor_payload=anchor_payload,
            gate_name=str(profile_gate),
            lower_q=0.10,
            upper_q=0.90,
        )
    else:
        summary, metric_rows = evaluate_run(
            run_dir=run_root,
            bands=bands,
            tile=tile,
            acceptance_env=dict(acceptance_env or {}),
            metrics=list(DEFAULT_ACCEPTANCE_METRICS),
        )

    mean_miss = 0.0
    run_max_miss = 0.0
    miss_n = 0
    for metric_row in metric_rows:
        miss = _norm_miss(metric_row.get("value"), metric_row.get("lo"), metric_row.get("hi"))
        metric_row["norm_miss"] = miss
        if miss is not None:
            mean_miss += float(miss)
            run_max_miss = max(run_max_miss, float(miss))
            miss_n += 1
    mean_miss = float(mean_miss / miss_n) if miss_n else 0.0
    run_row = {
        "candidate": candidate_name,
        "seed": seed,
        "description": CANDIDATES[candidate_name]["description"],
        "run_dir": str(run_root),
        "passed_metrics": int(summary["passed_metrics"]),
        "required_metrics": int(summary["required_metrics"]),
        "failed_metrics": int(summary["failed_metrics"]),
        "pass_fraction": summary["pass_fraction"],
        "overall_pass": bool(summary["overall_pass"]),
        "mean_norm_miss": mean_miss,
        "max_norm_miss": run_max_miss,
    }
    for metric_row in metric_rows:
        metric = str(metric_row["metric"])
        run_row[f"{metric}__value"] = metric_row["value"]
        run_row[f"{metric}__status"] = metric_row["status"]
        run_row[f"{metric}__norm_miss"] = metric_row["norm_miss"]
    return {
        "candidate": candidate_name,
        "seed": seed,
        "summary": summary,
        "metrics": metric_rows,
        "run_row": run_row,
    }


def _default_max_workers(*, tier: str, tasks: int, threads_per_worker: int | None) -> int:
    if tasks <= 1:
        return 1
    cpu_total = max(os.cpu_count() or 1, 1)
    if threads_per_worker is not None and int(threads_per_worker) > 0:
        by_threads = max(cpu_total // max(int(threads_per_worker), 1), 1)
        return max(1, min(tasks, by_threads))
    if tier == "paper":
        return max(1, min(tasks, 2))
    return max(1, min(tasks, 4))


def _run_tasks_parallel(task_items: list[dict[str, Any]], *, max_workers: int) -> dict[tuple[str, int], dict[str, Any]]:
    run_results: dict[tuple[str, int], dict[str, Any]] = {}
    if max_workers <= 1:
        for task in task_items:
            result = _run_candidate_seed(task)
            run_results[(str(result["candidate"]), int(result["seed"]))] = result
        return run_results
    try:
        ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            fut_map = {ex.submit(_run_candidate_seed, task): task for task in task_items}
            for fut in cf.as_completed(fut_map):
                result = fut.result()
                run_results[(str(result["candidate"]), int(result["seed"]))] = result
        return run_results
    except (PermissionError, OSError):
        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_map = {ex.submit(_run_candidate_seed, task): task for task in task_items}
            for fut in cf.as_completed(fut_map):
                result = fut.result()
                run_results[(str(result["candidate"]), int(result["seed"]))] = result
        return run_results


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run named Phase 1 calibration candidates against the frozen SIMUS v2 acceptance gate.")
    ap.add_argument("--profile", type=str, default="ClinIntraOp-Pf-v2")
    ap.add_argument("--tier", type=str, default="paper", choices=["smoke", "paper"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", action="append", default=None, help="Seed list(s); comma-separated allowed. Overrides --seed when provided.")
    ap.add_argument("--candidate", action="append", default=None, help="Candidate name(s); comma-separated allowed.")
    ap.add_argument("--anchor-json", type=Path, default=Path("reports/simus_v2/anchors/simus_v2_anchor_envelopes.json"))
    ap.add_argument("--profile-gate", type=str, default=None)
    ap.add_argument("--anchor-preset", type=str, choices=sorted(ANCHOR_PRESETS.keys()), default=None)
    ap.add_argument("--anchor-kind", action="append", default=None, help="Anchor kind(s); comma-separated allowed.")
    ap.add_argument("--out-root", type=Path, default=Path("runs/sim"))
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--reuse-existing", action="store_true")
    ap.add_argument("--max-workers", type=int, default=0, help="Parallel candidate/seed workers (0=auto, 1=serial).")
    ap.add_argument("--threads-per-worker", type=int, default=0, help="BLAS/OpenMP threads per worker (0=leave unchanged).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    candidates = _parse_list(args.candidate) or ["base", "calA", "calB", "calC", "calD"]
    seeds = [int(s) for s in (_parse_list(args.seeds) if args.seeds else [str(args.seed)])]
    unknown = [name for name in candidates if name not in CANDIDATES]
    if unknown:
        raise SystemExit(f"unknown candidates: {', '.join(sorted(unknown))}")

    anchor_payload = json.loads(Path(args.anchor_json).read_text(encoding="utf-8"))
    anchor_kinds = _parse_list(args.anchor_kind)
    if args.profile_gate and (args.anchor_preset or anchor_kinds):
        raise SystemExit("--profile-gate is mutually exclusive with --anchor-preset/--anchor-kind")
    if args.anchor_preset:
        if anchor_kinds:
            raise SystemExit("--anchor-preset and --anchor-kind are mutually exclusive")
        anchor_kinds = list(ANCHOR_PRESETS[str(args.anchor_preset)])
    if not anchor_kinds and not args.profile_gate:
        if str(args.profile) == "ClinIntraOp-Pf-v2":
            anchor_kinds = list(ANCHOR_PRESETS["intraop_brainlike"])
        else:
            anchor_kinds = list(ANCHOR_PRESETS["pooled_iq"])
    metrics = list(DEFAULT_ACCEPTANCE_METRICS)
    acceptance_env = None
    if not args.profile_gate:
        acceptance_env = _combine_anchor_envelope(
            anchor_payload,
            anchor_kinds,
            metrics,
            lower_q=0.10,
            upper_q=0.90,
        )
    bands = BandEdges(pf_lo_hz=30.0, pf_hi_hz=250.0, pg_lo_hz=250.0, pg_hi_hz=400.0, pa_lo_hz=400.0)
    tile = TileSpec(h=8, w=8, stride=3)

    seed_tag = "_".join(f"s{seed}" for seed in seeds)
    stem = f"simus_v2_phase1_calibration_{args.profile.replace('-', '_').lower()}_{args.tier}_{seed_tag}"
    out_csv = args.out_csv or Path("reports/simus_v2/acceptance") / f"{stem}.csv"
    out_json = args.out_json or Path("reports/simus_v2/acceptance") / f"{stem}.json"
    out_runs_csv = out_csv.with_name(f"{out_csv.stem}_runs.csv")
    threads_per_worker = int(args.threads_per_worker) if int(args.threads_per_worker) > 0 else None
    _prepare_thread_env(threads_per_worker)
    task_items: list[dict[str, Any]] = []
    for candidate_name in candidates:
        for seed in seeds:
            run_root = Path(args.out_root) / f"{stem}_{candidate_name}_seed{seed}"
            task_items.append(
                {
                    "profile": args.profile,
                    "tier": args.tier,
                    "seed": int(seed),
                    "candidate": candidate_name,
                    "run_root": str(run_root),
                    "anchor_payload": anchor_payload,
                    "acceptance_env": acceptance_env,
                    "profile_gate": args.profile_gate,
                    "reuse_existing": bool(args.reuse_existing),
                    "threads_per_worker": threads_per_worker,
                }
            )
    max_workers = int(args.max_workers) if int(args.max_workers) > 0 else _default_max_workers(
        tier=str(args.tier),
        tasks=len(task_items),
        threads_per_worker=threads_per_worker,
    )

    rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    payload_runs: list[dict[str, Any]] = []
    run_results = _run_tasks_parallel(task_items, max_workers=max_workers)

    for candidate_name in candidates:
        candidate_runs: list[dict[str, Any]] = []
        seeds_passed = 0
        total_passed_metrics = 0
        total_required_metrics = 0
        total_mean_miss = 0.0
        max_norm_miss = 0.0
        worst_pass_fraction = 1.0
        for seed in seeds:
            result = dict(run_results[(candidate_name, int(seed))])
            summary = dict(result["summary"])
            metric_rows = list(result["metrics"])
            run_row = dict(result["run_row"])
            run_rows.append(run_row)
            candidate_runs.append(
                {
                    "seed": int(seed),
                    "summary": summary,
                    "metrics": metric_rows,
                }
            )
            seeds_passed += int(bool(summary["overall_pass"]))
            total_passed_metrics += int(summary["passed_metrics"])
            total_required_metrics += int(summary["required_metrics"])
            total_mean_miss += float(run_row["mean_norm_miss"])
            worst_pass_fraction = min(worst_pass_fraction, float(summary["pass_fraction"]))
            max_norm_miss = max(max_norm_miss, float(run_row["max_norm_miss"]))
        mean_norm_miss = float(total_mean_miss / max(len(seeds), 1))
        row = {
            "candidate": candidate_name,
            "description": CANDIDATES[candidate_name]["description"],
            "seeds": ",".join(str(seed) for seed in seeds),
            "seeds_total": len(seeds),
            "seeds_passed": seeds_passed,
            "all_seeds_pass": seeds_passed == len(seeds),
            "passed_metrics": total_passed_metrics,
            "required_metrics": total_required_metrics,
            "failed_metrics": total_required_metrics - total_passed_metrics,
            "pass_fraction": float(total_passed_metrics / max(total_required_metrics, 1)),
            "worst_seed_pass_fraction": worst_pass_fraction,
            "mean_norm_miss": mean_norm_miss,
            "max_norm_miss": max_norm_miss,
        }
        for seed in seeds:
            seed_run = next(run for run in candidate_runs if int(run["seed"]) == int(seed))
            row[f"seed{seed}__pass_fraction"] = float(seed_run["summary"]["pass_fraction"])
            row[f"seed{seed}__overall_pass"] = bool(seed_run["summary"]["overall_pass"])
            for metric_row in seed_run["metrics"]:
                metric = str(metric_row["metric"])
                row[f"seed{seed}__{metric}__value"] = metric_row["value"]
                row[f"seed{seed}__{metric}__status"] = metric_row["status"]
        rows.append(row)
        payload_runs.append(
            {
                "candidate": candidate_name,
                "description": CANDIDATES[candidate_name]["description"],
                "aggregate": row,
                "runs": candidate_runs,
            }
        )

    rows.sort(
        key=lambda r: (
            -int(bool(r["all_seeds_pass"])),
            -int(r["seeds_passed"]),
            -float(r["worst_seed_pass_fraction"]),
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
        "seeds": seeds,
        "anchor_json": str(args.anchor_json),
        "profile_gate": None if args.profile_gate is None else str(args.profile_gate),
        "anchor_preset": None if args.anchor_preset is None else str(args.anchor_preset),
        "anchor_kinds": anchor_kinds,
        "metrics": metrics,
        "max_workers": int(max_workers),
        "threads_per_worker": threads_per_worker,
        "best_candidate": None if best is None else best["candidate"],
        "runs": payload_runs,
    }
    _write_csv(out_csv, rows)
    _write_csv(out_runs_csv, run_rows)
    _write_json(out_json, out_payload)
    print(f"[simus-v2-phase1-calibrate] wrote {out_csv}")
    print(f"[simus-v2-phase1-calibrate] wrote {out_runs_csv}")
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
