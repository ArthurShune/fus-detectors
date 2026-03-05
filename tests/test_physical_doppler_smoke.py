import dataclasses

import numpy as np

from sim.kwave.physical_doppler import NoiseSpec, PsfSpec, default_brainlike_config, generate_icube


def test_physical_doppler_generator_is_deterministic_and_nontrivial() -> None:
    cfg = default_brainlike_config(
        preset="microvascular_like",
        seed=0,
        Nx=48,
        Ny=48,
        prf_hz=1500.0,
        pulses_per_set=8,
        ensembles=2,
    )
    # Make the test assertion robust by disabling AWGN and PSF blur, and
    # boosting blood power so flow motion dominates.
    cfg = dataclasses.replace(
        cfg,
        blood_to_tissue_power_db=0.0,
        psf=PsfSpec(sigma_x0_px=0.0, sigma_z0_px=0.0),
        noise=NoiseSpec(snr_db=None),
    )

    out1 = generate_icube(cfg)
    out2 = generate_icube(cfg)

    icube1 = out1["Icube"]
    icube2 = out2["Icube"]
    assert icube1.shape == (cfg.T, cfg.sim_geom.Ny, cfg.sim_geom.Nx)
    assert icube1.dtype == np.complex64
    assert np.array_equal(icube1, icube2)

    mask_flow = out1["mask_flow"]
    mask_bg = out1["mask_bg"]
    assert mask_flow.shape == (cfg.sim_geom.Ny, cfg.sim_geom.Nx)
    assert mask_bg.shape == mask_flow.shape
    assert int(mask_flow.sum()) > 0
    assert int(mask_bg.sum()) > 0

    # Tissue is static; background is exactly constant without AWGN/PSF. Flow
    # region must change over time due to advection + Doppler phase increments.
    delta = np.abs(icube1[1] - icube1[0])
    assert float(delta[mask_bg].max()) == 0.0
    assert float(delta[mask_flow].mean()) > 1e-4


def test_physical_doppler_alias_preset_produces_expected_alias_mask() -> None:
    cfg = default_brainlike_config(
        preset="alias_stress",
        seed=1,
        Nx=48,
        Ny=48,
        prf_hz=1500.0,
        pulses_per_set=4,
        ensembles=2,
    )
    out = generate_icube(cfg)
    mask_alias = out["mask_alias_expected"]
    assert mask_alias.shape == (cfg.sim_geom.Ny, cfg.sim_geom.Nx)
    assert int(mask_alias.sum()) > 0
