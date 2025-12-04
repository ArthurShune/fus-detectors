from pipeline.confirm2.policy import rho_inflate_policy


def test_policy_monotone_with_harder_conditions():
    base = rho_inflate_policy(0.5, rho_groups=0.2, motion_um=80, dropout=0.0)
    harder = rho_inflate_policy(0.6, rho_groups=0.4, motion_um=150, dropout=0.12)

    assert harder.delta >= base.delta
    assert harder.rho_eff >= base.rho_eff
    assert harder.delta <= 0.10
