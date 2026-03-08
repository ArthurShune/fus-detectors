import dataclasses

import numpy as np


def test_pymust_motion_profiles_are_deterministic_and_mobile_is_harder():
    from sim.simus.config import default_profile_config
    from sim.simus.pymust_smoke import generate_icube

    intra = default_profile_config(profile="ClinIntraOp-Pf-v1", tier="smoke", seed=0)
    intra = dataclasses.replace(intra, T=3, tissue_count=96)
    intra_a = generate_icube(intra)
    intra_b = generate_icube(intra)

    assert np.allclose(np.asarray(intra_a["Icube"]), np.asarray(intra_b["Icube"]), atol=0.0, rtol=0.0)
    assert np.allclose(
        np.asarray(intra_a["debug"]["phase_screen_rad"]),
        np.asarray(intra_b["debug"]["phase_screen_rad"]),
        atol=0.0,
        rtol=0.0,
    )

    mobile = default_profile_config(profile="ClinMobile-Pf-v1", tier="smoke", seed=0)
    mobile = dataclasses.replace(mobile, T=3, tissue_count=96)
    mobile_out = generate_icube(mobile)

    intra_motion = dict(intra_a["debug"]["motion_telemetry"])
    mobile_motion = dict(mobile_out["debug"]["motion_telemetry"])
    intra_phase = dict(intra_a["debug"]["phase_screen_telemetry"])
    mobile_phase = dict(mobile_out["debug"]["phase_screen_telemetry"])

    assert intra_motion["enabled"] is True
    assert mobile_motion["enabled"] is True
    assert float(intra_motion["disp_rms_px"]) > 0.0
    assert float(mobile_motion["disp_rms_px"]) > float(intra_motion["disp_rms_px"])
    assert float(intra_phase["phase_rms_rad"]) > 0.0
    assert float(mobile_phase["phase_rms_rad"]) > 0.0
    assert float(mobile_phase["drift_sigma_rad"]) > float(intra_phase["drift_sigma_rad"])


def test_pymust_structural_profile_disables_motion_and_phase():
    from sim.simus.config import default_profile_config
    from sim.simus.pymust_smoke import generate_icube

    structural = default_profile_config(profile="ClinIntraOp-Pf-Struct-v2", tier="smoke", seed=0)
    structural = dataclasses.replace(structural, T=3, tissue_count=96)
    out = generate_icube(structural)

    motion = dict(out["debug"]["motion_telemetry"])
    phase = dict(out["debug"]["phase_screen_telemetry"])

    assert motion["enabled"] is False
    assert float(motion["disp_rms_px"]) == 0.0
    assert phase["enabled"] is False
    assert float(phase["phase_rms_rad"]) == 0.0


def test_pymust_clin_v2_profile_keeps_motion_and_phase_moderate():
    from sim.simus.config import default_profile_config
    from sim.simus.pymust_smoke import generate_icube

    cfg = default_profile_config(profile="ClinIntraOp-Pf-v2", tier="smoke", seed=0)
    cfg = dataclasses.replace(cfg, T=3, tissue_count=96)
    out = generate_icube(cfg)

    motion = dict(out["debug"]["motion_telemetry"])
    phase = dict(out["debug"]["phase_screen_telemetry"])

    assert motion["enabled"] is True
    assert 0.0 < float(motion["disp_rms_px"]) < 1.0
    assert phase["enabled"] is True
    assert 0.0 < float(phase["phase_rms_rad"]) < 1.0
