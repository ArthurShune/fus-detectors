# tests/test_metrics.py

import numpy as np

from eval.metrics import (
    clopper_pearson_ci,
    partial_auc,
    pd_snr_db,
    roc_curve,
    tpr_at_fpr_target,
)


def test_pd_snr_db_sanity():
    rng = np.random.default_rng(0)
    H, W = 32, 32
    # Construct PD map with brighter vessel region
    pd = rng.exponential(scale=1.0, size=(H, W))
    mask_flow = np.zeros((H, W), dtype=bool)
    mask_flow[8:16, 8:16] = True
    pd[mask_flow] += 3.0
    mask_bg = ~mask_flow
    snr = pd_snr_db(pd, mask_flow, mask_bg)
    assert snr > 0.0


def test_roc_and_tpr_at_fpr():
    rng = np.random.default_rng(1)
    n_pos = 20000
    n_neg = 20000
    # Positives shifted to the right
    s_pos = rng.normal(loc=2.0, scale=1.0, size=n_pos)
    s_neg = rng.normal(loc=0.0, scale=1.0, size=n_neg)
    fpr, tpr, thr = roc_curve(s_pos, s_neg, num_thresh=2048)
    t_small = tpr_at_fpr_target(fpr, tpr, target_fpr=1e-3)
    assert 0.05 < t_small < 0.2
    auc_p = partial_auc(fpr, tpr, fpr_max=1e-2)
    assert 0.0 < auc_p <= 1e-2  # bounded by width


def test_clopper_pearson_edges():
    lo, hi = clopper_pearson_ci(0, 1000)
    assert lo == 0.0 and 0.0 < hi < 0.01
    lo2, hi2 = clopper_pearson_ci(1000, 1000)
    assert hi2 == 1.0 and 0.99 < lo2 <= 1.0
