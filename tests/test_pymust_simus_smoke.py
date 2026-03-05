import numpy as np


def test_pymust_simus_smoke_is_deterministic_and_nontrivial():
    from sim.simus.pymust_smoke import SimusSmokeConfig, generate_icube

    cfg = SimusSmokeConfig(
        seed=0,
        T=2,
        H=16,
        W=16,
        x_min_m=-0.01,
        x_max_m=0.01,
        z_min_m=0.01,
        z_max_m=0.03,
        tissue_count=80,
        blood_count=40,
        blood_vz_mps=0.03,
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

