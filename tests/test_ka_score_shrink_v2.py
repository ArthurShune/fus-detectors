import numpy as np

from pipeline.stap.ka_contract_v2 import (
    KaContractV2Config,
    derive_score_shrink_v2_tile_scales,
    evaluate_ka_contract_v2,
)


def test_score_shrink_v2_scales_are_shrink_only_for_pd_scores():
    cfg = KaContractV2Config()
    rng = np.random.default_rng(0)
    n = 6000

    # Simulate a PD-like detector score: S = -PD (higher = more flow evidence).
    pd = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    s_base = (-pd).astype(np.float32)

    # Coverage proxy and telemetry.
    c_flow = rng.uniform(0.0, 0.30, size=n).astype(np.float32)
    r_guard = rng.uniform(0.0, 0.10, size=n).astype(np.float32)
    m_alias = rng.standard_normal(size=n).astype(np.float32)
    valid = np.ones(n, dtype=bool)

    rep = evaluate_ka_contract_v2(
        s_base=s_base,
        m_alias=m_alias,
        r_guard=r_guard,
        c_flow=c_flow,
        valid_mask=valid,
        config=cfg,
    )
    assert rep["state"] == "C1_SAFETY"
    assert rep["reason"] == "ok"

    shrink = derive_score_shrink_v2_tile_scales(
        report=rep,
        s_base=s_base,
        m_alias=m_alias,
        c_flow=c_flow,
        valid_mask=valid,
        mode="safety",
    )
    assert shrink["apply"] is True
    scale = np.asarray(shrink["scale_tiles"], dtype=np.float32)
    assert scale.shape == (n,)
    assert np.all(scale >= 1.0)

    prot = np.asarray(shrink["protected_tiles"], dtype=bool)
    assert prot.shape == (n,)
    assert np.all(scale[prot] == 1.0)

    gated = np.asarray(shrink["gated_tiles"], dtype=bool)
    assert gated.shape == (n,)
    assert gated.any()
    assert float(np.max(scale[gated])) > 1.0

    # Shrink-only check in score space for S = -PD:
    pd_new = pd * scale
    s_new = -pd_new
    assert np.all(s_new <= s_base + 1e-12)

