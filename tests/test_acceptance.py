# tests/test_acceptance.py

import numpy as np

from eval.acceptance import AcceptanceTargets, DetectorDataset, Masks, acceptance_report


def test_acceptance_synthetic_pass_like():
    rng = np.random.default_rng(0)
    # Synthetic PD maps
    H, W = 32, 32
    mask_flow = np.zeros((H, W), dtype=bool)
    mask_flow[8:16, 8:16] = True
    mask_bg = ~mask_flow
    pd_base = rng.exponential(scale=1.0, size=(H, W)).astype(np.float32)
    pd_stap = pd_base.copy()
    pd_stap[mask_flow] += 3.0  # better SNR

    # Scores: baseline worse separation than stap
    n_pos, n_neg = 20000, 20000
    base_pos = rng.normal(loc=1.5, scale=1.0, size=n_pos)
    base_neg = rng.normal(loc=0.0, scale=1.0, size=n_neg)
    stap_pos = rng.normal(loc=2.2, scale=1.0, size=n_pos)
    stap_neg = rng.normal(loc=0.0, scale=1.0, size=n_neg)

    base = DetectorDataset(scores_pos=base_pos, scores_null=base_neg, pd_map=pd_base)
    stap = DetectorDataset(scores_pos=stap_pos, scores_null=stap_neg, pd_map=pd_stap)
    masks = Masks(mask_flow=mask_flow, mask_bg=mask_bg)

    targets = AcceptanceTargets(
        delta_pdsnrdB_min=3.0,
        delta_tpr_at_fpr_min=0.05,
        fpr_target=1e-3,
        alpha_for_calibration=1e-3,
    )
    # Use milder (1e-3) targets to keep test fast/stable
    rep = acceptance_report(base, stap, masks, targets, seed=42, evd_mode="gpd")

    assert rep["gates"]["gate_delta_pd_snr"] is True
    assert rep["gates"]["gate_delta_tpr_at_fpr"] is True
    assert rep["gates"]["gate_calibration_ci"] is True
    assert rep["overall_pass"] is True
