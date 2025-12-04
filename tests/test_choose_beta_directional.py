import torch

from pipeline.stap.temporal import choose_beta_directional


def _proj(rank: int, Lt: int) -> torch.Tensor:
    Pf = torch.zeros((Lt, Lt), dtype=torch.complex64)
    for i in range(rank):
        Pf[i, i] = 1.0
    return Pf


def _diag(vals):
    arr = torch.tensor(vals, dtype=torch.float32)
    return torch.diag(arr).to(torch.complex64)


def test_choose_beta_directional_prefers_positive_beta_when_safe() -> None:
    Lt = 4
    Pf = _proj(rank=2, Lt=Lt)
    Rhat = _diag([4.0, 3.0, 1.0, 1.0])
    # Prior matches Pf energy but is significantly lighter off-band
    R0 = _diag([4.0, 3.0, 0.2, 0.2])

    beta, info = choose_beta_directional(Rhat, R0, Pf, beta_max=0.5)

    assert beta > 0.0
    assert info["pf_min_ratio"] is None or info["pf_min_ratio"] >= 0.9
    assert info["pf_max_ratio"] is None or info["pf_max_ratio"] <= 1.1
    assert info["perp_max_ratio"] is None or info["perp_max_ratio"] <= 1.2


def test_choose_beta_directional_returns_zero_when_prior_inflates_perp() -> None:
    Lt = 4
    Pf = _proj(rank=2, Lt=Lt)
    Rhat = _diag([4.0, 3.0, 1.0, 1.0])
    # Prior is heavier off-band (no shrink benefit)
    R0 = _diag([4.0, 3.0, 2.0, 2.0])

    beta, info = choose_beta_directional(Rhat, R0, Pf, beta_max=0.5)

    assert beta == 0.0
    # shrink_perp should reflect the heavy prior
    assert info["shrink_perp"] >= 1.0
