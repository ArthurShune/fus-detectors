from __future__ import annotations

from dataclasses import replace

from sim.simus.config import default_profile_config
from sim.simus.motion import build_motion_artifacts


def test_random_walk_sigma_controls_motion_amplitude():
    cfg = default_profile_config(profile="ClinIntraOp-Pf-v1", tier="paper", seed=21)
    base_motion = replace(
        cfg.motion,
        breathing_amp_x_px=0.0,
        breathing_amp_z_px=0.0,
        cardiac_amp_x_px=0.0,
        cardiac_amp_z_px=0.0,
        drift_x_px=0.0,
        drift_z_px=0.0,
        elastic_amp_px=0.0,
    )

    cfg_small = replace(cfg, motion=replace(base_motion, enabled=True, random_walk_sigma_px=0.01))
    cfg_large = replace(cfg, motion=replace(base_motion, enabled=True, random_walk_sigma_px=0.08))

    art_small = build_motion_artifacts(cfg=cfg_small, seed=21)
    art_large = build_motion_artifacts(cfg=cfg_large, seed=21)

    assert art_small.telemetry["rigid_rms_px"] > 0.0
    assert art_large.telemetry["rigid_rms_px"] > art_small.telemetry["rigid_rms_px"] * 4.0
