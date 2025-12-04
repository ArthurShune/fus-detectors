import numpy as np

from sim.kwave.common import _rank_consistency_same_class


def test_rank_consistency_perfect_alignment() -> None:
    base = np.arange(10, dtype=np.float32)
    stap = base * 2.0
    score = _rank_consistency_same_class(base, stap)
    assert score is not None and abs(score - 1.0) < 1e-6


def test_rank_consistency_opposite_order() -> None:
    base = np.arange(10, dtype=np.float32)
    stap = -base
    score = _rank_consistency_same_class(base, stap)
    assert score is not None and score == 0.0


def test_rank_consistency_sampling_handles_large() -> None:
    base = np.random.randn(1000)
    stap = base + 0.1 * np.random.randn(1000)
    score = _rank_consistency_same_class(base, stap, max_pairs=1000)
    assert score is not None
    assert 0.0 <= score <= 1.0
