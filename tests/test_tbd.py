import numpy as np

from pipeline.detect.tbd import HMMParams, viterbi_binary


def test_tbd_improves_recall_on_bursty_series():
    rng = np.random.default_rng(0)
    T = 300
    scores = rng.standard_normal(T) * 0.8
    scores[120:150] += 3.0
    thr = 2.0

    raw = scores > thr
    smoothed = viterbi_binary(scores, thr, HMMParams(p11=0.98, p00=0.995))

    rec_raw = raw[120:150].mean()
    rec_sm = smoothed[120:150].mean()
    assert rec_sm >= rec_raw

    assert smoothed[:100].mean() <= 0.05 + 1e-6
