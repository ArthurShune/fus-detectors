import numpy as np


def _roc_tail_resolution(stap_neg: np.ndarray, target_fpr: float) -> tuple[float, bool]:
    n_neg = stap_neg.size
    fpr_min = 1.0 / max(n_neg, 1)
    return fpr_min, target_fpr >= fpr_min


def test_tail_resolution_flag_behavior():
    neg = np.random.randn(60000).astype(np.float32)
    fpr_min, resolvable = _roc_tail_resolution(neg, target_fpr=1e-5)
    assert np.isclose(fpr_min, 1.0 / 60000.0)
    assert resolvable is False

    fpr_min2, resolvable2 = _roc_tail_resolution(neg, target_fpr=5e-5)
    assert resolvable2 is True
