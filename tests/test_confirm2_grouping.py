import numpy as np

from pipeline.confirm2.grouping import spectral_bisection_groups


def test_grouping_blocks_recover_low_intergroup_corr():
    rng = np.random.default_rng(0)
    A, T = 8, 5000
    base1 = rng.normal(size=T)
    base2 = rng.normal(size=T)
    z_by_angle = []
    for k in range(A):
        if k < 4:
            z = 0.8 * base1 + 0.2 * rng.normal(size=T)
        else:
            z = 0.8 * base2 + 0.2 * rng.normal(size=T)
        z_by_angle.append(z)

    g1, g2, rho = spectral_bisection_groups(z_by_angle)
    assert len(g1) > 0 and len(g2) > 0
    assert abs(rho) < 0.2
