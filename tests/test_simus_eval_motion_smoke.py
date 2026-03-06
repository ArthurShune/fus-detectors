import dataclasses


def test_motion_ladder_scaling_helpers_disable_zero_and_scale_nonzero():
    from scripts.simus_eval_motion import make_motion_ladder_config, scale_motion_spec, scale_phase_spec
    from sim.simus.config import default_profile_config
    from sim.simus.bundle import select_simus_stap_profile

    cfg = default_profile_config(profile="ClinIntraOp-Pf-v1", tier="smoke", seed=0)
    motion_half = scale_motion_spec(cfg.motion, 0.5)
    phase_half = scale_phase_spec(cfg.phase_screen, 0.5)
    motion_zero = scale_motion_spec(cfg.motion, 0.0)
    phase_zero = scale_phase_spec(cfg.phase_screen, 0.0)

    assert motion_half.enabled is True
    assert phase_half.enabled is True
    assert motion_half.breathing_amp_x_px == cfg.motion.breathing_amp_x_px * 0.5
    assert phase_half.std_rad == cfg.phase_screen.std_rad * 0.5
    assert motion_zero.enabled is False
    assert phase_zero.enabled is False

    ladder_cfg = make_motion_ladder_config(
        profile="ClinIntraOp-Pf-v1",
        tier="smoke",
        seed=0,
        motion_scale=0.5,
        phase_scale=0.0,
    )
    ladder_cfg = dataclasses.replace(ladder_cfg, T=2, tissue_count=64)
    assert "motionx0p5" in str(ladder_cfg.profile)
    assert ladder_cfg.motion.enabled is True
    assert ladder_cfg.phase_screen.enabled is False

    low_profile, low_meta = select_simus_stap_profile(
        requested_profile="Brain-SIMUS-Clin",
        policy="Brain-SIMUS-Clin-MotionDisp-v0",
        motion_disp_rms_px=1.5,
    )
    high_profile, high_meta = select_simus_stap_profile(
        requested_profile="Brain-SIMUS-Clin",
        policy="Brain-SIMUS-Clin-MotionDisp-v0",
        motion_disp_rms_px=2.5,
    )
    assert low_profile == "Brain-SIMUS-Clin-MotionRobust-v0"
    assert high_profile == "Brain-SIMUS-Clin-MotionMidRobust-v0"
    assert low_meta["feature"] == "motion_disp_rms_px"
    assert high_meta["threshold"] == 2.05
