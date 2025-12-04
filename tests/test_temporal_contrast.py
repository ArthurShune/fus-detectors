import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pipeline.stap.temporal import (
    build_temporal_hankels_and_cov,
    msd_contrast_score_batched,
)


def test_contrast_flow_exceeds_motion_on_injected_tone():
    """Injected high-frequency tone should yield higher flow fraction and score."""
    rng = np.random.default_rng(1)
    T, h, w, Lt = 64, 3, 3, 4
    prf_hz = 3000.0

    cube = (0.05 * (rng.standard_normal((T, h, w)) + 1j * rng.standard_normal((T, h, w)))).astype(
        np.complex64
    )
    t = np.arange(T)
    fd_hz = 600.0
    tone = np.exp(1j * 2 * np.pi * fd_hz * t / prf_hz).astype(np.complex64)
    cube[:, 1, 1] += 0.3 * tone

    S, Rt, _ = build_temporal_hankels_and_cov(cube, Lt=Lt, device="cpu")
    fd_grid = np.linspace(-900.0, 900.0, 9, dtype=np.float32)
    motion_half_span = 150.0

    score, r_flow, r_motion = msd_contrast_score_batched(
        Rt,
        S,
        prf_hz=prf_hz,
        fd_grid_hz=fd_grid,
        motion_half_span_hz=motion_half_span,
        lam_abs=3e-2,
        ridge=0.1,
        agg="median",
        contrast_alpha=0.7,
        device="cpu",
    )

    tone_idx = (1, 1)
    flow_fraction = float(r_flow[tone_idx].item())
    motion_fraction = float(r_motion[tone_idx].item())
    background_idx = (0, 0)
    assert flow_fraction >= motion_fraction - 1e-4
    assert float(score[tone_idx].item()) >= float(score[background_idx].item()) - 1e-4
    assert motion_fraction <= float(r_motion[background_idx].item()) + 1e-4
    assert float(score[tone_idx].item()) >= float(torch.quantile(score, 0.75).item()) - 1e-4


def test_contrast_fallback_routes_to_ratio_when_flow_is_removed():
    rng = np.random.default_rng(2)
    T, h, w, Lt = 64, 2, 2, 4
    prf_hz = 3000.0

    cube = (0.05 * (rng.standard_normal((T, h, w)) + 1j * rng.standard_normal((T, h, w)))).astype(
        np.complex64
    )
    # Inject only low-frequency (motion) content
    cube += 0.4 + 0j

    S, Rt, _ = build_temporal_hankels_and_cov(cube, Lt=Lt, device="cpu")
    fd_grid = np.linspace(-900.0, 900.0, 9, dtype=np.float32)
    motion_half_span = 200.0

    score, r_flow, r_motion, details = msd_contrast_score_batched(
        Rt,
        S,
        prf_hz=prf_hz,
        fd_grid_hz=fd_grid,
        motion_half_span_hz=motion_half_span,
        lam_abs=3e-2,
        ridge=0.1,
        agg="median",
        contrast_alpha=0.7,
        ratio_rho=0.05,
        return_details=True,
        device="cpu",
    )

    assert details["score_mode"] in {"msd", "msd_contrast_mix"}
    assert torch.all(torch.isfinite(score))
    assert float(details["beta_est"]) >= float(details["zeta_est"])
