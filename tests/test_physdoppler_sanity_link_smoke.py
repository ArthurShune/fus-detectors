import numpy as np


def test_physdoppler_sanity_link_summarize_icube_smoke():
    from scripts.physical_doppler_sanity_link import BandEdges, TileSpec, summarize_icube

    rng = np.random.default_rng(0)
    T, H, W = 16, 32, 32
    # Synthetic complex IQ with a localized "flow" region.
    Icube = (rng.standard_normal((T, H, W)) + 1j * rng.standard_normal((T, H, W))).astype(np.complex64)
    mask_flow = np.zeros((H, W), dtype=bool)
    mask_flow[8:16, 10:22] = True
    mask_bg = ~mask_flow

    out = summarize_icube(
        name="unit",
        Icube=Icube,
        prf_hz=1500.0,
        tile=TileSpec(h=8, w=8, stride=4),
        bands=BandEdges(),
        mask_flow=mask_flow,
        mask_bg=mask_bg,
        derive_masks=False,
        derive_vessel_q=0.99,
        derive_bg_q=0.20,
    )
    assert out["meta"]["prf_hz"] == 1500.0
    assert out["meta"]["shape"] == [T, H, W]
    assert "summary" in out and "svd" in out

