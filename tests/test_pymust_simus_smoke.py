import dataclasses
import numpy as np


def test_pymust_simus_smoke_is_deterministic_and_nontrivial():
    from sim.simus.pymust_smoke import SimusConfig, generate_icube

    cfg = SimusConfig(
        preset="microvascular_like",
        tier="smoke",
        seed=0,
        prf_hz=1500.0,
        T=2,
        H=16,
        W=16,
        x_min_m=-0.01,
        x_max_m=0.01,
        z_min_m=0.01,
        z_max_m=0.03,
        probe="P4-2v",
        tissue_count=80,
        blood_count=40,
        blood_vmax_mps=0.03,
        blood_profile="poiseuille",
        vessel_radius_m=0.002,
    )
    out1 = generate_icube(cfg)
    out2 = generate_icube(cfg)
    I1 = np.asarray(out1["Icube"])
    I2 = np.asarray(out2["Icube"])
    assert I1.shape == (2, 16, 16)
    assert I1.dtype == np.complex64
    assert np.allclose(I1, I2, atol=0.0, rtol=0.0)

    mask_flow = np.asarray(out1["mask_flow"], dtype=bool)
    assert mask_flow.shape == (16, 16)
    assert int(mask_flow.sum()) > 0

    # Motion should change at least some flow-region IQ over pulses.
    diff = np.mean(np.abs(I1[1] - I1[0])[mask_flow])
    assert float(diff) > 0.0


def test_pymust_simus_alias_preset_produces_expected_alias_mask():
    from sim.simus.pymust_smoke import default_config, generate_icube

    cfg = default_config(preset="alias_stress", tier="smoke", seed=0)
    cfg = dataclasses.replace(cfg, T=2, H=16, W=16, tissue_count=80, blood_count=40)
    out = generate_icube(cfg)
    mask_flow = np.asarray(out["mask_flow"], dtype=bool)
    mask_alias = np.asarray(out["mask_alias_expected"], dtype=bool)
    assert int(mask_flow.sum()) > 0
    assert int(mask_alias.sum()) > 0
    assert int(mask_alias.sum()) < int(mask_flow.sum())


def test_pymust_simus_clin_profile_emits_h1_h0_masks():
    from sim.simus.config import default_profile_config
    from sim.simus.pymust_smoke import generate_icube

    cfg = default_profile_config(profile="ClinIntraOp-Pf-v1", tier="smoke", seed=0)
    cfg = dataclasses.replace(cfg, T=2, tissue_count=120)
    out = generate_icube(cfg)

    mask_pf = np.asarray(out["mask_h1_pf_main"], dtype=bool)
    mask_alias_qc = np.asarray(out["mask_h1_alias_qc"], dtype=bool)
    mask_nuisance = np.asarray(out["mask_h0_nuisance_pa"], dtype=bool)
    mask_bg = np.asarray(out["mask_h0_bg"], dtype=bool)
    mask_guard = np.asarray(out["mask_guard"], dtype=bool)

    assert int(mask_pf.sum()) > 0
    assert int(mask_nuisance.sum()) > 0
    assert int(mask_bg.sum()) > 0
    assert not np.any(mask_pf & mask_nuisance)
    assert not np.any(mask_pf & mask_guard)
    assert not np.any(mask_bg & mask_guard)
    assert int(mask_alias_qc.sum()) >= 0


def test_pymust_simus_structural_profile_is_masked_and_deterministic():
    from sim.simus.config import default_profile_config
    from sim.simus.pymust_smoke import generate_icube

    cfg = default_profile_config(profile="ClinIntraOp-Pf-Struct-v2", tier="smoke", seed=0)
    cfg = dataclasses.replace(cfg, T=2, tissue_count=120)
    out_a = generate_icube(cfg)
    out_b = generate_icube(cfg)

    assert np.allclose(np.asarray(out_a["Icube"]), np.asarray(out_b["Icube"]), atol=0.0, rtol=0.0)
    assert int(np.asarray(out_a["mask_h1_pf_main"], dtype=bool).sum()) > 0
    assert int(np.asarray(out_a["mask_h0_nuisance_pa"], dtype=bool).sum()) > 0
    assert bool(out_a["debug"]["motion_telemetry"]["enabled"]) is False
    assert bool(out_a["debug"]["phase_screen_telemetry"]["enabled"]) is False
