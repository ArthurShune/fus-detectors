import numpy as np

from pipeline.stap.ka_contract_v2 import KaContractV2Config, evaluate_ka_contract_v2


def _base_inputs(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    c_flow = rng.uniform(0.0, 0.30, size=n).astype(np.float32)
    s_base = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    r_guard = rng.uniform(0.0, 0.10, size=n).astype(np.float32)
    m_alias = rng.standard_normal(size=n).astype(np.float32)
    valid = np.ones(n, dtype=bool)
    return s_base, m_alias, r_guard, c_flow, valid


def test_contract_off_when_alias_metric_flat():
    n = 2000
    s_base, m_alias, r_guard, c_flow, valid = _base_inputs(n)
    m_alias[:] = 1.0
    rep = evaluate_ka_contract_v2(
        s_base=s_base, m_alias=m_alias, r_guard=r_guard, c_flow=c_flow, valid_mask=valid
    )
    assert rep["state"] == "C0_OFF"
    assert rep["reason"] == "alias_metric_flat"


def test_contract_off_when_guard_dominates():
    n = 2000
    s_base, m_alias, r_guard, c_flow, valid = _base_inputs(n)
    r_guard[:] = 0.6
    rep = evaluate_ka_contract_v2(
        s_base=s_base, m_alias=m_alias, r_guard=r_guard, c_flow=c_flow, valid_mask=valid
    )
    assert rep["state"] == "C0_OFF"
    assert rep["reason"] == "guard_dominates_not_actionable"


def test_contract_off_when_guard_is_extreme():
    n = 2000
    s_base, m_alias, r_guard, c_flow, valid = _base_inputs(n)
    r_guard[:] = 0.95
    rep = evaluate_ka_contract_v2(
        s_base=s_base, m_alias=m_alias, r_guard=r_guard, c_flow=c_flow, valid_mask=valid
    )
    assert rep["state"] == "C0_OFF"
    assert rep["reason"] == "guard_extreme_out_of_regime"


def test_contract_c1_safety_when_actionability_is_weak():
    n = 4000
    s_base, m_alias, r_guard, c_flow, valid = _base_inputs(n, seed=1)
    # Make tail association very small by decoupling alias from the baseline score.
    # (m_alias already independent of s_base in _base_inputs.)
    rep = evaluate_ka_contract_v2(
        s_base=s_base,
        m_alias=m_alias,
        r_guard=r_guard,
        c_flow=c_flow,
        valid_mask=valid,
    )
    assert rep["state"] == "C1_SAFETY"
    assert rep["reason"] == "ok"
    metrics = rep["metrics"]
    assert metrics.get("uplift_eligible") is False


def test_contract_c2_uplift_when_actionability_is_strong():
    # Ensure the background proxy set is large enough that the 0.99-quantile
    # tail contains >= n_tail_min samples under the default c_bg=0.05.
    n = 15000
    s_base, m_alias, r_guard, c_flow, valid = _base_inputs(n, seed=2)
    # Create label-free "actionability": in bg proxy tiles (low coverage),
    # high-score tiles should also be more alias-like.
    bg = c_flow <= 0.05
    flow = c_flow >= 0.20
    # Global association between alias and score, plus a bg-specific boost so
    # that (i) bg is more alias-like than flow and (ii) within bg, high-score
    # tiles are more alias-like than mid-score tiles.
    m_alias += 1.5 * s_base
    m_alias[bg] += -0.1 + 0.7 * s_base[bg]
    m_alias[flow] -= 0.3
    # Provide Pf-peak telemetry so the C2 band-realization guard can pass.
    pf_peak = np.ones(n, dtype=bool)
    rep = evaluate_ka_contract_v2(
        s_base=s_base,
        m_alias=m_alias,
        r_guard=r_guard,
        pf_peak=pf_peak,
        c_flow=c_flow,
        valid_mask=valid,
        config=KaContractV2Config(),
    )
    assert rep["state"] == "C2_UPLIFT"
    assert rep["reason"] == "ok"
    assert rep["metrics"].get("uplift_eligible") is True


def test_contract_c1_safety_guard_when_guard_is_actionable():
    n = 20000
    s_base, m_alias, r_guard, c_flow, valid = _base_inputs(n, seed=3)
    bg = c_flow <= 0.05
    # Make guard fraction dominate (uplift veto), but remain below the
    # safety ceiling and correlate with background tail "false-alarm-like"
    # tiles (high s_base within bg proxy).
    rng = np.random.default_rng(4)
    r_guard = 0.30 + 0.22 * s_base + 0.02 * rng.standard_normal(size=n).astype(np.float32)
    r_guard[bg] += 0.12 * s_base[bg]
    r_guard = np.clip(r_guard, 0.0, 0.69).astype(np.float32)

    rep = evaluate_ka_contract_v2(
        s_base=s_base,
        m_alias=m_alias,
        r_guard=r_guard,
        c_flow=c_flow,
        valid_mask=valid,
    )
    assert rep["state"] == "C1_SAFETY"
    assert rep["reason"] == "ok"
    assert rep["metrics"].get("risk_mode") == "guard"
    assert rep["metrics"].get("uplift_vetoed_by_guard") is True
