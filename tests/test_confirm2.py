"""Tests for confirm2 helpers."""

import numpy as np
from scipy.stats import norm

from pipeline.confirm2.bvn_tail import joint_tail, per_look_alpha_from_pair, solve_z_for_pair_alpha


def test_bvn_tail_monotone() -> None:
    tail = joint_tail(0.0, 0.5)
    assert 0.0 < tail < 1.0


def test_joint_tail_monotone():
    r = 0.5
    p1 = joint_tail(3.0, r)
    p2 = joint_tail(3.5, r)
    assert p2 < p1 and p1 > 0 and p2 > 0


def test_solver_roundtrip():
    for rho in [0.0, 0.3, 0.7]:
        alpha2 = 1e-4
        z = solve_z_for_pair_alpha(alpha2, rho)
        p = joint_tail(z, rho)
        assert abs(p - alpha2) / alpha2 < 0.05


def test_independence_check():
    # For rho=0, joint_tail ≈ (1 - Phi(z))^2
    z = 4.0
    alpha_indep = (1.0 - norm.cdf(z)) ** 2
    p = joint_tail(z, 0.0)
    assert abs(p - alpha_indep) / alpha_indep < 0.05


def test_per_look_alpha_mapping():
    rho = 0.4
    alpha2 = 1e-5
    a1 = per_look_alpha_from_pair(alpha2, rho)
    assert 0 < a1 < np.sqrt(alpha2)  # positively correlated => a1 > alpha2 but < sqrt(alpha2)
