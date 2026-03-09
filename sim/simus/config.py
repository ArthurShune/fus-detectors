from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal


SimusPreset = Literal["microvascular_like", "alias_stress"]
SimusProfile = Literal[
    "ClinIntraOp-Pf-v1",
    "ClinMobile-Pf-v1",
    "ClinIntraOp-Pf-Struct-v2",
    "ClinIntraOp-Pf-v2",
    "ClinMobile-Pf-v2",
]
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
class MotionSpec:
    enabled: bool = False
    breathing_hz: float = 0.0
    breathing_amp_x_px: float = 0.0
    breathing_amp_z_px: float = 0.0
    cardiac_hz: float = 0.0
    cardiac_amp_x_px: float = 0.0
    cardiac_amp_z_px: float = 0.0
    random_walk_sigma_px: float = 0.0
    pulse_jitter_sigma_px: float = 0.0
    drift_x_px: float = 0.0
    drift_z_px: float = 0.0
    elastic_amp_px: float = 0.0
    elastic_sigma_px: float = 12.0
    elastic_depth_decay_frac: float = 0.35
    elastic_temporal_rho: float = 0.98
    elastic_mode_count: int = 1
    elastic_lateral_scale: float = 1.0
    elastic_axial_scale: float = 0.6


@dataclass(frozen=True)
class PhaseScreenSpec:
    enabled: bool = False
    std_rad: float = 0.0
    corr_len_elem: float = 12.0
    drift_rho: float = 1.0
    drift_sigma_rad: float = 0.0


@dataclass(frozen=True)
class StructuredClutterSpec:
    name: str
    x0_m: float
    z0_m: float
    x1_m: float
    z1_m: float
    thickness_m: float = 4.0e-4
    scatterer_count: int = 200
    rc_scale: float = 1.0


@dataclass(frozen=True)
class BackgroundCompartmentSpec:
    name: str
    center_x_m: float
    center_z_m: float
    sigma_x_m: float
    sigma_z_m: float
    scatterer_count: int = 120
    rc_scale: float = 0.4
    motion_amp_px: float = 0.0
    motion_sigma_px: float = 10.0
    motion_rho: float = 0.9
    motion_jitter_sigma_px: float = 0.0
    lateral_scale: float = 1.0
    axial_scale: float = 0.7


@dataclass(frozen=True)
class NoiseSpec:
    enabled: bool = False
    iq_rms_frac: float = 0.0


@dataclass(frozen=True)
class OrdinaryBackgroundSpec:
    mode: Literal["independent_compartments", "coupled_mass_shear_v2"] = "independent_compartments"
    global_cutoff_hz: float = 7.0
    shear1_cutoff_hz: float = 11.0
    shear2_cutoff_hz: float = 13.0
    residual_cutoff_hz: float = 9.0
    global_disp_px: float = 0.12
    shear1_disp_px: float = 0.045
    shear2_disp_px: float = 0.035
    residual_disp_px: float = 0.015
    global_dx_scale: float = 0.10
    global_dz_scale: float = 1.00
    shear_dx_scale: float = 0.24
    shear_dz_scale: float = 0.20
    deep_dz_scale: float = 0.12
    sector_mix_scale: float = 0.14
    anchor_motion_scale: float = 0.18
    structured_motion_scale: float = 1.15
    sectorize_mid: bool = False


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
    motion: MotionSpec = MotionSpec()
    phase_screen: PhaseScreenSpec = PhaseScreenSpec()
    noise: NoiseSpec = NoiseSpec()
    structured_clutter: tuple[StructuredClutterSpec, ...] = ()
    background_compartments: tuple[BackgroundCompartmentSpec, ...] = ()
    ordinary_background: OrdinaryBackgroundSpec = OrdinaryBackgroundSpec()

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
        motion=MotionSpec(enabled=False),
        phase_screen=PhaseScreenSpec(enabled=False),
        noise=NoiseSpec(enabled=False),
        ordinary_background=OrdinaryBackgroundSpec(mode="independent_compartments"),
        reservoir_scale=4,
        reinject_depth_span_m=float(reinject_depth_span_m),
        **grid,
    )


def default_profile_config(*, profile: SimusProfile, tier: SimusTier, seed: int) -> SimusConfig:
    grid = _default_grid(tier)
    structural_profile = profile == "ClinIntraOp-Pf-Struct-v2"
    intraop_v2 = profile == "ClinIntraOp-Pf-v2"
    mobile_v2 = profile == "ClinMobile-Pf-v2"
    clin_v2 = intraop_v2 or mobile_v2
    if tier == "paper":
        tissue_count = 1200 if intraop_v2 else (1000 if mobile_v2 else 2600)
        guard_px = 2
        reservoir_span = 1.5e-3
        micro_blood_count = 280 if not (intraop_v2 or mobile_v2) else (320 if intraop_v2 else 340)
        nuisance_blood_count = 520 if not (intraop_v2 or mobile_v2) else (620 if intraop_v2 else 760)
        nuisance_vmax = 0.065 if profile in ("ClinIntraOp-Pf-v1", "ClinIntraOp-Pf-Struct-v2") else 0.090
        noise = NoiseSpec(enabled=False)
        if structural_profile:
            motion = MotionSpec(enabled=False)
            phase_screen = PhaseScreenSpec(enabled=False)
            micro_blood_rc_scale = 0.35
        elif intraop_v2:
            motion = MotionSpec(
                enabled=True,
                breathing_hz=0.30,
                breathing_amp_x_px=0.35,
                breathing_amp_z_px=0.18,
                cardiac_hz=1.10,
                cardiac_amp_x_px=0.10,
                cardiac_amp_z_px=0.05,
                random_walk_sigma_px=0.04,
                pulse_jitter_sigma_px=0.06,
                drift_x_px=0.20,
                drift_z_px=0.09,
                elastic_amp_px=0.42,
                elastic_sigma_px=14.0,
                elastic_depth_decay_frac=0.32,
                elastic_temporal_rho=0.72,
                elastic_mode_count=4,
                elastic_lateral_scale=1.0,
                elastic_axial_scale=0.60,
            )
            phase_screen = PhaseScreenSpec(
                enabled=True,
                std_rad=0.42,
                corr_len_elem=8.0,
                drift_rho=0.95,
                drift_sigma_rad=0.018,
            )
            noise = NoiseSpec(enabled=True, iq_rms_frac=0.24)
            micro_blood_rc_scale = 0.22
            nuisance_vmax = 0.085
        elif mobile_v2:
            motion = MotionSpec(
                enabled=True,
                breathing_hz=0.45,
                breathing_amp_x_px=0.65,
                breathing_amp_z_px=0.32,
                cardiac_hz=1.30,
                cardiac_amp_x_px=0.14,
                cardiac_amp_z_px=0.07,
                random_walk_sigma_px=0.06,
                pulse_jitter_sigma_px=0.10,
                drift_x_px=0.42,
                drift_z_px=0.18,
                elastic_amp_px=0.62,
                elastic_sigma_px=12.0,
                elastic_depth_decay_frac=0.36,
                elastic_temporal_rho=0.66,
                elastic_mode_count=5,
                elastic_lateral_scale=1.0,
                elastic_axial_scale=0.72,
            )
            phase_screen = PhaseScreenSpec(
                enabled=True,
                std_rad=0.58,
                corr_len_elem=7.0,
                drift_rho=0.92,
                drift_sigma_rad=0.030,
            )
            noise = NoiseSpec(enabled=True, iq_rms_frac=0.28)
            micro_blood_rc_scale = 0.22
            nuisance_vmax = 0.115
        elif profile == "ClinIntraOp-Pf-v1":
            motion = MotionSpec(
                enabled=True,
                breathing_hz=0.25,
                breathing_amp_x_px=0.8,
                breathing_amp_z_px=0.35,
                cardiac_hz=1.2,
                cardiac_amp_x_px=0.28,
                cardiac_amp_z_px=0.14,
                random_walk_sigma_px=0.04,
                elastic_amp_px=0.40,
                elastic_sigma_px=20.0,
                elastic_depth_decay_frac=0.35,
                elastic_temporal_rho=0.985,
                elastic_lateral_scale=1.0,
                elastic_axial_scale=0.65,
            )
            phase_screen = PhaseScreenSpec(enabled=True, std_rad=0.60, corr_len_elem=12.0, drift_rho=1.0, drift_sigma_rad=0.0)
            noise = NoiseSpec(enabled=False)
            micro_blood_rc_scale = 0.18
        else:
            motion = MotionSpec(
                enabled=True,
                breathing_hz=1.8,
                breathing_amp_x_px=1.8,
                breathing_amp_z_px=0.9,
                cardiac_hz=0.4,
                cardiac_amp_x_px=0.35,
                cardiac_amp_z_px=0.18,
                random_walk_sigma_px=0.05,
                drift_x_px=0.8,
                drift_z_px=0.4,
                elastic_amp_px=0.80,
                elastic_sigma_px=24.0,
                elastic_depth_decay_frac=0.45,
                elastic_temporal_rho=0.992,
                elastic_lateral_scale=1.0,
                elastic_axial_scale=0.70,
            )
            phase_screen = PhaseScreenSpec(
                enabled=True,
                std_rad=0.60,
                corr_len_elem=12.0,
                drift_rho=0.995,
                drift_sigma_rad=0.015,
            )
            noise = NoiseSpec(enabled=False)
            micro_blood_rc_scale = 0.18
        micro_vessels = (
            _legacy_vessel(
                name="micro_left",
                role="microvascular",
                center_x_m=-2.6e-3,
                radius_m=4.8e-4,
                z_min_m=9.0e-3,
                z_max_m=18.0e-3,
                blood_count=micro_blood_count,
                blood_rc_scale=micro_blood_rc_scale,
                blood_vmax_mps=0.012 if not clin_v2 else 0.014,
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
                blood_rc_scale=micro_blood_rc_scale,
                blood_vmax_mps=0.015 if not clin_v2 else 0.018,
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
                blood_rc_scale=micro_blood_rc_scale,
                blood_vmax_mps=0.011 if not clin_v2 else 0.013,
                blood_profile="poiseuille",
            ),
            _legacy_vessel(
                name="micro_far_right",
                role="microvascular",
                center_x_m=5.4e-3,
                radius_m=4.0e-4,
                z_min_m=12.5e-3,
                z_max_m=20.5e-3,
                blood_count=micro_blood_count if (intraop_v2 or mobile_v2) else 0,
                blood_rc_scale=micro_blood_rc_scale,
                blood_vmax_mps=0.016,
                blood_profile="poiseuille",
            ),
        )
        nuisance_vessels = (
            _legacy_vessel(
                name="nuisance_superficial",
                role="nuisance_pa",
                center_x_m=-5.0e-3,
                radius_m=1.2e-3 if not mobile_v2 else 1.6e-3,
                z_min_m=5.8e-3,
                z_max_m=10.8e-3 if not mobile_v2 else 11.8e-3,
                blood_count=nuisance_blood_count,
                blood_rc_scale=0.28,
                blood_vmax_mps=nuisance_vmax,
                blood_profile="poiseuille",
            ),
        ) + (
            (
                _legacy_vessel(
                    name="nuisance_mid",
                    role="nuisance_pa",
                    center_x_m=4.8e-3,
                    radius_m=1.1e-3,
                    z_min_m=8.0e-3,
                    z_max_m=14.5e-3,
                    blood_count=max(1, nuisance_blood_count // 2),
                    blood_rc_scale=0.24,
                    blood_vmax_mps=0.095,
                    blood_profile="poiseuille",
                ),
            )
            if mobile_v2
            else ()
        )
        structured_clutter = (
            StructuredClutterSpec(
                name="superficial_sheet",
                x0_m=-8.5e-3,
                z0_m=6.4e-3,
                x1_m=8.5e-3,
                z1_m=6.8e-3,
                thickness_m=3.8e-4,
                scatterer_count=420 if not mobile_v2 else 520,
                rc_scale=1.4,
            ),
            StructuredClutterSpec(
                name="oblique_boundary",
                x0_m=-8.0e-3,
                z0_m=7.2e-3,
                x1_m=1.5e-3,
                z1_m=17.8e-3,
                thickness_m=5.5e-4,
                scatterer_count=340 if not mobile_v2 else 420,
                rc_scale=1.2,
            ),
        ) + (
            (
                StructuredClutterSpec(
                    name="deep_boundary",
                    x0_m=-1.0e-3,
                    z0_m=12.5e-3,
                    x1_m=8.0e-3,
                    z1_m=22.5e-3,
                    thickness_m=6.0e-4,
                    scatterer_count=360,
                    rc_scale=1.1,
                ),
            )
            if mobile_v2
            else ()
        ) if (intraop_v2 or mobile_v2) else ()
        background_compartments = (
            BackgroundCompartmentSpec(
                name="bg_superficial_left",
                center_x_m=-4.6e-3,
                center_z_m=9.0e-3,
                sigma_x_m=1.8e-3,
                sigma_z_m=1.4e-3,
                scatterer_count=280,
                rc_scale=0.55,
                motion_amp_px=0.32,
                motion_sigma_px=10.0,
                motion_rho=0.78,
                motion_jitter_sigma_px=0.03,
                lateral_scale=1.0,
                axial_scale=0.72,
            ),
            BackgroundCompartmentSpec(
                name="bg_mid_core",
                center_x_m=1.0e-3,
                center_z_m=13.0e-3,
                sigma_x_m=2.6e-3,
                sigma_z_m=2.2e-3,
                scatterer_count=320,
                rc_scale=0.50,
                motion_amp_px=0.36,
                motion_sigma_px=12.0,
                motion_rho=0.74,
                motion_jitter_sigma_px=0.025,
                lateral_scale=1.0,
                axial_scale=0.68,
            ),
            BackgroundCompartmentSpec(
                name="bg_deep_right",
                center_x_m=4.8e-3,
                center_z_m=17.2e-3,
                sigma_x_m=2.0e-3,
                sigma_z_m=2.5e-3,
                scatterer_count=280,
                rc_scale=0.52,
                motion_amp_px=0.30,
                motion_sigma_px=11.0,
                motion_rho=0.80,
                motion_jitter_sigma_px=0.025,
                lateral_scale=0.95,
                axial_scale=0.72,
            ),
        ) + (
            (
                BackgroundCompartmentSpec(
                    name="bg_deep_left",
                    center_x_m=-3.2e-3,
                    center_z_m=18.5e-3,
                    sigma_x_m=2.2e-3,
                    sigma_z_m=2.6e-3,
                    scatterer_count=260,
                    rc_scale=0.56,
                    motion_amp_px=0.34,
                    motion_sigma_px=12.0,
                    motion_rho=0.72,
                    motion_jitter_sigma_px=0.03,
                    lateral_scale=1.0,
                    axial_scale=0.78,
                ),
            )
            if mobile_v2
            else ()
        ) if (intraop_v2 or mobile_v2) else ()
    else:
        tissue_count = 340 if intraop_v2 else (280 if mobile_v2 else 720)
        guard_px = 1
        reservoir_span = 2.0e-3
        micro_blood_count = 90 if not (intraop_v2 or mobile_v2) else (120 if intraop_v2 else 130)
        nuisance_blood_count = 180 if not (intraop_v2 or mobile_v2) else (220 if intraop_v2 else 280)
        nuisance_vmax = 0.17 if profile in ("ClinIntraOp-Pf-v1", "ClinIntraOp-Pf-Struct-v2") else 0.22
        noise = NoiseSpec(enabled=False)
        if structural_profile:
            motion = MotionSpec(enabled=False)
            phase_screen = PhaseScreenSpec(enabled=False)
            micro_blood_rc_scale = 0.35
        elif intraop_v2:
            motion = MotionSpec(
                enabled=True,
                breathing_hz=0.30,
                breathing_amp_x_px=0.22,
                breathing_amp_z_px=0.10,
                cardiac_hz=1.10,
                cardiac_amp_x_px=0.05,
                cardiac_amp_z_px=0.03,
                random_walk_sigma_px=0.02,
                pulse_jitter_sigma_px=0.04,
                drift_x_px=0.10,
                drift_z_px=0.05,
                elastic_amp_px=0.20,
                elastic_sigma_px=6.0,
                elastic_depth_decay_frac=0.30,
                elastic_temporal_rho=0.75,
                elastic_mode_count=3,
                elastic_lateral_scale=1.0,
                elastic_axial_scale=0.60,
            )
            phase_screen = PhaseScreenSpec(
                enabled=True,
                std_rad=0.34,
                corr_len_elem=6.0,
                drift_rho=0.95,
                drift_sigma_rad=0.012,
            )
            noise = NoiseSpec(enabled=True, iq_rms_frac=0.16)
            micro_blood_rc_scale = 0.22
            nuisance_vmax = 0.19
        elif mobile_v2:
            motion = MotionSpec(
                enabled=True,
                breathing_hz=0.45,
                breathing_amp_x_px=0.40,
                breathing_amp_z_px=0.18,
                cardiac_hz=1.30,
                cardiac_amp_x_px=0.08,
                cardiac_amp_z_px=0.04,
                random_walk_sigma_px=0.04,
                pulse_jitter_sigma_px=0.08,
                drift_x_px=0.22,
                drift_z_px=0.10,
                elastic_amp_px=0.34,
                elastic_sigma_px=5.5,
                elastic_depth_decay_frac=0.34,
                elastic_temporal_rho=0.68,
                elastic_mode_count=4,
                elastic_lateral_scale=1.0,
                elastic_axial_scale=0.72,
            )
            phase_screen = PhaseScreenSpec(
                enabled=True,
                std_rad=0.44,
                corr_len_elem=5.0,
                drift_rho=0.92,
                drift_sigma_rad=0.022,
            )
            noise = NoiseSpec(enabled=True, iq_rms_frac=0.20)
            micro_blood_rc_scale = 0.22
            nuisance_vmax = 0.26
        elif profile == "ClinIntraOp-Pf-v1":
            motion = MotionSpec(
                enabled=True,
                breathing_hz=0.25,
                breathing_amp_x_px=0.6,
                breathing_amp_z_px=0.25,
                cardiac_hz=1.2,
                cardiac_amp_x_px=0.20,
                cardiac_amp_z_px=0.10,
                random_walk_sigma_px=0.03,
                elastic_amp_px=0.25,
                elastic_sigma_px=6.0,
                elastic_depth_decay_frac=0.35,
                elastic_temporal_rho=0.98,
                elastic_lateral_scale=1.0,
                elastic_axial_scale=0.65,
            )
            phase_screen = PhaseScreenSpec(enabled=True, std_rad=0.45, corr_len_elem=6.0, drift_rho=1.0, drift_sigma_rad=0.0)
            noise = NoiseSpec(enabled=False)
            micro_blood_rc_scale = 0.18
        else:
            motion = MotionSpec(
                enabled=True,
                breathing_hz=1.8,
                breathing_amp_x_px=1.25,
                breathing_amp_z_px=0.60,
                cardiac_hz=0.4,
                cardiac_amp_x_px=0.25,
                cardiac_amp_z_px=0.12,
                random_walk_sigma_px=0.05,
                drift_x_px=0.6,
                drift_z_px=0.25,
                elastic_amp_px=0.50,
                elastic_sigma_px=8.0,
                elastic_depth_decay_frac=0.45,
                elastic_temporal_rho=0.99,
                elastic_lateral_scale=1.0,
                elastic_axial_scale=0.70,
            )
            phase_screen = PhaseScreenSpec(
                enabled=True,
                std_rad=0.45,
                corr_len_elem=6.0,
                drift_rho=0.992,
                drift_sigma_rad=0.012,
            )
            noise = NoiseSpec(enabled=False)
            micro_blood_rc_scale = 0.18
        micro_vessels = (
            _legacy_vessel(
                name="micro_left",
                role="microvascular",
                center_x_m=-2.5e-3,
                radius_m=7.5e-4,
                z_min_m=9.0e-3,
                z_max_m=18.0e-3,
                blood_count=micro_blood_count,
                blood_rc_scale=micro_blood_rc_scale,
                blood_vmax_mps=0.020 if not clin_v2 else 0.016,
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
                blood_rc_scale=micro_blood_rc_scale,
                blood_vmax_mps=0.026 if not clin_v2 else 0.020,
                blood_profile="poiseuille",
            ),
            _legacy_vessel(
                name="micro_right",
                role="microvascular",
                center_x_m=4.2e-3,
                radius_m=7.0e-4,
                z_min_m=12.0e-3,
                z_max_m=19.0e-3,
                blood_count=micro_blood_count if (intraop_v2 or mobile_v2) else 0,
                blood_rc_scale=micro_blood_rc_scale,
                blood_vmax_mps=0.018,
                blood_profile="poiseuille",
            ),
        )
        nuisance_vessels = (
            _legacy_vessel(
                name="nuisance_superficial",
                role="nuisance_pa",
                center_x_m=-5.5e-3,
                radius_m=1.6e-3 if not mobile_v2 else 1.9e-3,
                z_min_m=5.8e-3,
                z_max_m=10.2e-3 if not mobile_v2 else 11.2e-3,
                blood_count=nuisance_blood_count,
                blood_rc_scale=0.28,
                blood_vmax_mps=nuisance_vmax,
                blood_profile="poiseuille",
            ),
        ) + (
            (
                _legacy_vessel(
                    name="nuisance_mid",
                    role="nuisance_pa",
                    center_x_m=4.6e-3,
                    radius_m=1.2e-3,
                    z_min_m=8.0e-3,
                    z_max_m=13.8e-3,
                    blood_count=max(1, nuisance_blood_count // 2),
                    blood_rc_scale=0.24,
                    blood_vmax_mps=0.22,
                    blood_profile="poiseuille",
                ),
            )
            if mobile_v2
            else ()
        )
        structured_clutter = (
            StructuredClutterSpec(
                name="superficial_sheet",
                x0_m=-9.0e-3,
                z0_m=6.2e-3,
                x1_m=9.0e-3,
                z1_m=6.6e-3,
                thickness_m=4.0e-4,
                scatterer_count=160 if not mobile_v2 else 220,
                rc_scale=1.35,
            ),
            StructuredClutterSpec(
                name="oblique_boundary",
                x0_m=-7.5e-3,
                z0_m=7.0e-3,
                x1_m=1.0e-3,
                z1_m=16.0e-3,
                thickness_m=5.0e-4,
                scatterer_count=140 if not mobile_v2 else 180,
                rc_scale=1.15,
            ),
        ) + (
            (
                StructuredClutterSpec(
                    name="deep_boundary",
                    x0_m=-1.0e-3,
                    z0_m=12.0e-3,
                    x1_m=8.0e-3,
                    z1_m=20.5e-3,
                    thickness_m=5.5e-4,
                    scatterer_count=150,
                    rc_scale=1.1,
                ),
            )
            if mobile_v2
            else ()
        ) if (intraop_v2 or mobile_v2) else ()
        background_compartments = (
            BackgroundCompartmentSpec(
                name="bg_superficial_left",
                center_x_m=-4.0e-3,
                center_z_m=8.8e-3,
                sigma_x_m=1.6e-3,
                sigma_z_m=1.2e-3,
                scatterer_count=90,
                rc_scale=0.55,
                motion_amp_px=0.22,
                motion_sigma_px=6.0,
                motion_rho=0.78,
                motion_jitter_sigma_px=0.02,
                lateral_scale=1.0,
                axial_scale=0.72,
            ),
            BackgroundCompartmentSpec(
                name="bg_mid_core",
                center_x_m=1.0e-3,
                center_z_m=13.0e-3,
                sigma_x_m=2.4e-3,
                sigma_z_m=2.0e-3,
                scatterer_count=110,
                rc_scale=0.50,
                motion_amp_px=0.26,
                motion_sigma_px=7.0,
                motion_rho=0.74,
                motion_jitter_sigma_px=0.015,
                lateral_scale=1.0,
                axial_scale=0.68,
            ),
            BackgroundCompartmentSpec(
                name="bg_deep_right",
                center_x_m=4.2e-3,
                center_z_m=17.0e-3,
                sigma_x_m=1.8e-3,
                sigma_z_m=2.2e-3,
                scatterer_count=90,
                rc_scale=0.52,
                motion_amp_px=0.22,
                motion_sigma_px=6.5,
                motion_rho=0.80,
                motion_jitter_sigma_px=0.015,
                lateral_scale=0.95,
                axial_scale=0.72,
            ),
        ) + (
            (
                BackgroundCompartmentSpec(
                    name="bg_deep_left",
                    center_x_m=-3.0e-3,
                    center_z_m=18.2e-3,
                    sigma_x_m=2.0e-3,
                    sigma_z_m=2.4e-3,
                    scatterer_count=90,
                    rc_scale=0.56,
                    motion_amp_px=0.28,
                    motion_sigma_px=7.0,
                    motion_rho=0.72,
                    motion_jitter_sigma_px=0.02,
                    lateral_scale=1.0,
                    axial_scale=0.78,
                ),
            )
            if mobile_v2
            else ()
        ) if (intraop_v2 or mobile_v2) else ()

    ordinary_background = OrdinaryBackgroundSpec(mode="independent_compartments")

    if not (intraop_v2 or mobile_v2):
        background_compartments = ()
        noise = NoiseSpec(enabled=False)

    vessels = tuple(v for v in micro_vessels + nuisance_vessels if int(v.blood_count) > 0)
    main_micro = micro_vessels[0]
    return SimusConfig(
        preset="microvascular_like",
        tier=str(tier),  # type: ignore[arg-type]
        profile=str(profile),
        seed=int(seed),
        tissue_count=int(tissue_count),
        tissue_rc_scale=0.85 if (intraop_v2 or mobile_v2) else 1.0,
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
        motion=motion,
        phase_screen=phase_screen,
        noise=noise,
        structured_clutter=structured_clutter,
        background_compartments=background_compartments,
        ordinary_background=ordinary_background,
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
