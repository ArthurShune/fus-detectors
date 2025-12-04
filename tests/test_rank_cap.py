import numpy as np
import torch

from sim.kwave.common import _cap_kc_from_rank


def test_cap_kc_respects_rank_and_bounds():
    evals = torch.tensor([5.0, 4.0, 1.0, 0.2], dtype=torch.float32)
    R = torch.diag(evals).to(torch.complex64)
    kc = _cap_kc_from_rank(R, min_pts=5, max_pts=13)
    assert kc % 2 == 1
    # Limited by Lt=4 (=> max odd 3)
    assert 1 <= kc <= 3
    assert kc == 1


def test_cap_kc_handles_degenerate_trace():
    R = torch.zeros((4, 4), dtype=torch.complex64)
    kc = _cap_kc_from_rank(R, min_pts=3, max_pts=9)
    assert kc == 3
