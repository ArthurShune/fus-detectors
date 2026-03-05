from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal


SimusPreset = Literal["microvascular_like", "alias_stress"]
SimusProfile = Literal["ClinIntraOp-Pf-v1", "ClinMobile-Pf-v1"]
SimusTier = Literal["smoke", "paper"]
VesselRole = Literal["microvascular", "nuisance_pa"]


@dataclass(frozen=True)
class DopplerBandSpec:
    pf_low_hz: float = 30.0
    pf_high_hz: float = 250.0
    guard_low_hz: float = 250.0
    guard_high_hz: float = 400.0
    pa_low_hz: float = 400.0
    pa_high_hz: float = 750.0
    pf_core_low_hz: float = 60.0
    pf_core_high_hz: float = 180.0


@dataclass(frozen=True)
class VesselSpec:
    name: str
    role: VesselRole = "microvascular"
    center_x_m: float = 0.0
    radius_m: float = 0.001
    z_min_m: float | None = None
    z_max_m: float | None = None
    blood_count: int = 200
    blood_rc_scale: float = 0.25
    blood_vmax_mps: float = 0.02
    blood_profile: Literal["plug", "poiseuille"] = "poiseuille"


@dataclass(frozen=True)
class SimusConfig:
    preset: SimusPreset = "microvascular_like"
    tier: SimusTier = "smoke"
    profile: str | None = None

    seed: int = 0
    prf_hz: float = 1500.0
    T: int = 8

    H: int = 24
    W: int = 24
    x_min_m: float = -0.020
    x_max_m: float = 0.020
    z_min_m: float = 0.010
    z_max_m: float = 0.050

    probe: str = "P4-2v"
    c_mps: float = 1540.0
    fs_mult: float = 4.0
    tilt_deg: float = 0.0

    tissue_count: int = 400
    tissue_rc_scale: float = 1.0

    # Backward-compatible single-vessel fields. New code should prefer `vessels`.
    blood_count: int = 200
    blood_rc_scale: float = 0.25
    vessel_center_x_m: float = 0.0
    vessel_radius_m: float = 0.004
    blood_vmax_mps: float = 0.020
    blood_profile: Literal["plug", "poiseuille"] = "poiseuille"

    vessels: tuple[VesselSpec, ...] = ()
    bands: DopplerBandSpec = DopplerBandSpec()
    label_guard_px: int = 2
    bg_top_exclusion_frac: float = 0.10

    reservoir_scale: int = 4
    reinject_depth_span_m: float = 0.003


def _default_grid(tier: SimusTier) -> dict[str, Any]:
    if tier == "paper":
        return {
            "prf_hz": 1500.0,
            "T": 64,
            "H": 128,
            "W": 128,
            "x_min_m": -9.6e-3,
            "x_max_m": 9.6e-3,
            "z_min_m": 5.0e-3,
            "z_max_m": 24.2e-3,
            "probe": "L11-5v",
            "c_mps": 1540.0,
            "fs_mult": 4.0,
        }
    return {
        "prf_hz": 1500.0,
        "T": 8,
        "H": 32,
        "W": 32,
        "x_min_m": -0.010,
        "x_max_m": 0.010,
        "z_min_m": 0.005,
        "z_max_m": 0.025,
        "probe": "P4-2v",
        "c_mps": 1540.0,
        "fs_mult": 4.0,
    }


def _legacy_vessel(
    *,
    name: str,
    role: VesselRole,
    center_x_m: float,
    radius_m: float,
    blood_count: int,
    blood_rc_scale: float,
    blood_vmax_mps: float,
    blood_profile: Literal["plug", "poiseuille"],
    z_min_m: float | None = None,
    z_max_m: float | None = None,
) -> VesselSpec:
    return VesselSpec(
        name=name,
        role=role,
        center_x_m=float(center_x_m),
        radius_m=float(radius_m),
        z_min_m=z_min_m,
        z_max_m=z_max_m,
        blood_count=int(blood_count),
        blood_rc_scale=float(blood_rc_scale),
        blood_vmax_mps=float(blood_vmax_mps),
        blood_profile=str(blood_profile),  # type: ignore[arg-type]
    )


def default_config(*, preset: SimusPreset, tier: SimusTier, seed: int) -> SimusConfig:
    grid = _default_grid(tier)
    if tier == "paper":
        vessel_radius_m = 1.2e-3 if preset == "microvascular_like" else 1.8e-3
        blood_vmax_mps = 0.015 if preset == "microvascular_like" else 0.090
        tissue_count = 2000
        blood_count = 1000
        reinject_depth_span_m = 1.5e-3
    else:
        vessel_radius_m = 3.0e-3 if preset == "microvascular_like" else 4.0e-3
        blood_vmax_mps = 0.03 if preset == "microvascular_like" else 0.30
        tissue_count = 400
        blood_count = 200
        reinject_depth_span_m = 3.0e-3

    vessel = _legacy_vessel(
        name="main_vessel",
        role="microvascular",
        center_x_m=0.0,
        radius_m=vessel_radius_m,
        blood_count=blood_count,
        blood_rc_scale=0.25,
        blood_vmax_mps=blood_vmax_mps,
        blood_profile="poiseuille",
    )
    return SimusConfig(
        preset=str(preset),  # type: ignore[arg-type]
        tier=str(tier),  # type: ignore[arg-type]
        profile=None,
        seed=int(seed),
        tissue_count=int(tissue_count),
        tissue_rc_scale=1.0,
        blood_count=int(blood_count),
        blood_rc_scale=0.25,
        vessel_center_x_m=0.0,
        vessel_radius_m=float(vessel_radius_m),
        blood_vmax_mps=float(blood_vmax_mps),
        blood_profile="poiseuille",
        vessels=(vessel,),
        reservoir_scale=4,
        reinject_depth_span_m=float(reinject_depth_span_m),
        **grid,
    )


def default_profile_config(*, profile: SimusProfile, tier: SimusTier, seed: int) -> SimusConfig:
    grid = _default_grid(tier)
    if tier == "paper":
        tissue_count = 2600
        guard_px = 2
        reservoir_span = 1.5e-3
        micro_blood_count = 280
        nuisance_blood_count = 520
        nuisance_vmax = 0.065 if profile == "ClinIntraOp-Pf-v1" else 0.090
        micro_vessels = (
            _legacy_vessel(
                name="micro_left",
                role="microvascular",
                center_x_m=-2.6e-3,
                radius_m=4.8e-4,
                z_min_m=9.0e-3,
                z_max_m=18.0e-3,
                blood_count=micro_blood_count,
                blood_rc_scale=0.18,
                blood_vmax_mps=0.012,
                blood_profile="poiseuille",
            ),
            _legacy_vessel(
                name="micro_mid",
                role="microvascular",
                center_x_m=7.0e-4,
                radius_m=5.5e-4,
                z_min_m=11.0e-3,
                z_max_m=21.5e-3,
                blood_count=micro_blood_count,
                blood_rc_scale=0.18,
                blood_vmax_mps=0.015,
                blood_profile="poiseuille",
            ),
            _legacy_vessel(
                name="micro_right",
                role="microvascular",
                center_x_m=3.2e-3,
                radius_m=4.5e-4,
                z_min_m=13.0e-3,
                z_max_m=22.5e-3,
                blood_count=micro_blood_count,
                blood_rc_scale=0.18,
                blood_vmax_mps=0.011,
                blood_profile="poiseuille",
            ),
        )
        nuisance_vessels = (
            _legacy_vessel(
                name="nuisance_superficial",
                role="nuisance_pa",
                center_x_m=-5.0e-3,
                radius_m=1.2e-3,
                z_min_m=5.8e-3,
                z_max_m=10.8e-3,
                blood_count=nuisance_blood_count,
                blood_rc_scale=0.28,
                blood_vmax_mps=nuisance_vmax,
                blood_profile="poiseuille",
            ),
        )
    else:
        tissue_count = 720
        guard_px = 1
        reservoir_span = 2.0e-3
        micro_blood_count = 90
        nuisance_blood_count = 180
        nuisance_vmax = 0.17 if profile == "ClinIntraOp-Pf-v1" else 0.22
        micro_vessels = (
            _legacy_vessel(
                name="micro_left",
                role="microvascular",
                center_x_m=-2.5e-3,
                radius_m=7.5e-4,
                z_min_m=9.0e-3,
                z_max_m=18.0e-3,
                blood_count=micro_blood_count,
                blood_rc_scale=0.18,
                blood_vmax_mps=0.020,
                blood_profile="poiseuille",
            ),
            _legacy_vessel(
                name="micro_mid",
                role="microvascular",
                center_x_m=1.0e-3,
                radius_m=9.0e-4,
                z_min_m=11.0e-3,
                z_max_m=21.0e-3,
                blood_count=micro_blood_count,
                blood_rc_scale=0.18,
                blood_vmax_mps=0.026,
                blood_profile="poiseuille",
            ),
        )
        nuisance_vessels = (
            _legacy_vessel(
                name="nuisance_superficial",
                role="nuisance_pa",
                center_x_m=-5.5e-3,
                radius_m=1.6e-3,
                z_min_m=5.8e-3,
                z_max_m=10.2e-3,
                blood_count=nuisance_blood_count,
                blood_rc_scale=0.28,
                blood_vmax_mps=nuisance_vmax,
                blood_profile="poiseuille",
            ),
        )

    vessels = micro_vessels + nuisance_vessels
    main_micro = micro_vessels[0]
    return SimusConfig(
        preset="microvascular_like",
        tier=str(tier),  # type: ignore[arg-type]
        profile=str(profile),
        seed=int(seed),
        tissue_count=int(tissue_count),
        tissue_rc_scale=1.0,
        blood_count=int(sum(v.blood_count for v in vessels)),
        blood_rc_scale=float(main_micro.blood_rc_scale),
        vessel_center_x_m=float(main_micro.center_x_m),
        vessel_radius_m=float(main_micro.radius_m),
        blood_vmax_mps=float(main_micro.blood_vmax_mps),
        blood_profile=str(main_micro.blood_profile),  # type: ignore[arg-type]
        vessels=vessels,
        bands=DopplerBandSpec(),
        label_guard_px=int(guard_px),
        bg_top_exclusion_frac=0.10,
        reservoir_scale=4,
        reinject_depth_span_m=float(reservoir_span),
        **grid,
    )


def resolve_vessels(cfg: SimusConfig) -> tuple[VesselSpec, ...]:
    if cfg.vessels:
        return tuple(cfg.vessels)
    return (
        _legacy_vessel(
            name="main_vessel",
            role="microvascular",
            center_x_m=float(cfg.vessel_center_x_m),
            radius_m=float(cfg.vessel_radius_m),
            z_min_m=None,
            z_max_m=None,
            blood_count=int(cfg.blood_count),
            blood_rc_scale=float(cfg.blood_rc_scale),
            blood_vmax_mps=float(cfg.blood_vmax_mps),
            blood_profile=str(cfg.blood_profile),  # type: ignore[arg-type]
        ),
    )


def dataset_meta(cfg: SimusConfig) -> dict[str, Any]:
    return dataclasses.asdict(cfg)
