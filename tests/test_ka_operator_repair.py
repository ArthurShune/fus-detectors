import torch

from pipeline.stap.temporal import _generalized_band_metrics, _repair_band_metrics


def _proj(rank: int, size: int) -> torch.Tensor:
    Pf = torch.zeros((size, size), dtype=torch.complex64)
    for i in range(rank):
        Pf[i, i] = 1.0
    return Pf


def test_repair_band_metrics_enforces_targets():
    Lt = 4
    Pf = _proj(2, Lt)
    R_sample = torch.diag(torch.tensor([1.0, 0.9, 0.7, 0.6], dtype=torch.complex64))
    R_beta = torch.diag(torch.tensor([4.0, 3.5, 0.05, 0.05], dtype=torch.complex64))

    repaired, _, stats = _repair_band_metrics(
        R_sample,
        R_beta,
        Pf,
        beta_init=0.2,
        max_passes=4,
        pf_min_target=0.95,
        perp_max_target=1.10,
    )
    metrics = _generalized_band_metrics(R_sample, repaired, Pf)
    assert metrics["pf_min"] is not None and metrics["pf_min"] >= 0.95 - 1e-3
    assert metrics["perp_max"] is not None and metrics["perp_max"] <= 1.10 + 1e-3
    assert not stats.get("repair_failed")


def test_repair_band_metrics_flags_mixing_failure():
    Lt = 4
    Pf = _proj(2, Lt)
    R_sample = torch.eye(Lt, dtype=torch.complex64)
    R_beta = torch.eye(Lt, dtype=torch.complex64)
    R_beta[0, 2] = 0.4 + 0.0j
    R_beta[2, 0] = 0.4 + 0.0j

    _, _, stats = _repair_band_metrics(
        R_sample,
        R_beta,
        Pf,
        beta_init=0.5,
        max_passes=1,
        mix_target=1e-3,
    )
    assert stats.get("repair_failed")
