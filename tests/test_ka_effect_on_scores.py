import numpy as np
import torch

from pipeline.gpu.linalg import cholesky_solve_hermitian
from pipeline.stap.temporal import (
    bandpass_constraints_temporal,
    build_temporal_hankels_and_cov,
    ka_blend_covariance_temporal,
    ka_prior_temporal_from_psd,
)


def _print_metrics(tag: str, **metrics: float) -> None:
    parts: list[str] = []
    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            parts.append(f"{key}={float(value):.6f}")
        elif isinstance(value, (int, np.integer)):
            parts.append(f"{key}={int(value)}")
        else:
            parts.append(f"{key}={value}")
    print(f"[{tag}] " + " ".join(parts))


def _tpr_at_fpr(pos: np.ndarray, neg: np.ndarray, fpr: float) -> float:
    if neg.size == 0:
        return 0.0
    thresh = np.quantile(neg, 1.0 - fpr, method="nearest")
    return float(np.mean(pos >= thresh))


def _mean_msd_ratio(
    Rt: torch.Tensor,
    S: torch.Tensor,
    Cf: torch.Tensor,
    lam_abs: float,
    ridge: float,
) -> torch.Tensor:
    Lt = Rt.shape[0]
    N, h, w = S.shape[1], S.shape[2], S.shape[3]
    eye = torch.eye(Lt, dtype=Rt.dtype, device=Rt.device)
    lam_eff = lam_abs if lam_abs > 0.0 else 4e-2
    R_loaded = Rt + lam_eff * eye
    L = torch.linalg.cholesky(R_loaded)
    Sw = torch.linalg.solve_triangular(L, S.reshape(Lt, N * h * w), upper=False)
    Cw = torch.linalg.solve_triangular(L, Cf, upper=False)
    z = Cw.conj().transpose(-2, -1) @ Sw
    gram = Cw.conj().transpose(-2, -1) @ Cw
    gram = gram + ridge * torch.eye(gram.shape[-1], dtype=Rt.dtype, device=Rt.device)
    proj, _ = cholesky_solve_hermitian(gram, z, jitter_init=1e-10, max_tries=3)
    t_band = torch.sum(z.conj() * proj, dim=0).real
    sw_pow = torch.sum(Sw.conj() * Sw, dim=0).real
    ratio = t_band / torch.clamp(sw_pow - t_band, min=1e-10)
    return ratio.reshape(N, h, w).mean(dim=0)


def test_ka_tightens_null_tail_and_preserves_flow_score() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    T, h, w, Lt = 48, 20, 20, 4
    prf = 3000.0
    t = torch.arange(T, dtype=torch.float32) / prf

    noise = 0.1 * (
        torch.randn(T, h, w, dtype=torch.float32) + 1j * torch.randn(T, h, w, dtype=torch.float32)
    )
    f_motion = 75.0
    drift = 0.05 * torch.exp(1j * 2 * np.pi * f_motion * t)[:, None, None]
    cube = (noise + drift).to(torch.complex64)
    y0, x0 = 7, 8
    f_flow = 600.0
    cube[:, y0, x0] = cube[:, y0, x0] + 0.12 * torch.exp(1j * 2 * np.pi * f_flow * t)

    S, Rt, _ = build_temporal_hankels_and_cov(
        cube,
        Lt=Lt,
        center=True,
        estimator="huber",
        huber_c=5.0,
        device="cpu",
        dtype=torch.complex64,
    )
    Rt = Rt.to(torch.complex64)
    Cf = bandpass_constraints_temporal(
        Lt=Lt,
        prf_hz=prf,
        fd_grid_hz=np.array([-600.0, -300.0, 0.0, 300.0, 600.0], dtype=np.float64),
        device="cpu",
    )

    msd_no = _mean_msd_ratio(Rt, S, Cf, lam_abs=2e-2, ridge=0.20).cpu().numpy()

    R0 = ka_prior_temporal_from_psd(
        Lt=Lt,
        prf_hz=prf,
        f_peaks_hz=(0.0,),
        width_bins=1,
        add_deriv=True,
        device="cpu",
    )
    R_lam, _ = ka_blend_covariance_temporal(
        R_sample=Rt,
        R0_prior=R0,
        Cf_flow=Cf,
        beta=None,
        kappa_target=40.0,
        device="cpu",
    )
    msd_ka = _mean_msd_ratio(R_lam, S, Cf, lam_abs=0.0, ridge=0.20).cpu().numpy()

    mask_flow = np.zeros((h, w), dtype=bool)
    mask_flow[y0, x0] = True
    mask_bg = ~mask_flow

    neg_no = msd_no[mask_bg]
    neg_ka = msd_ka[mask_bg]
    pos_no = msd_no[mask_flow]
    pos_ka = msd_ka[mask_flow]

    q99_no = float(np.quantile(neg_no, 0.99))
    q99_ka = float(np.quantile(neg_ka, 0.99))
    _print_metrics(
        "ka_tail_stats",
        q99_no=q99_no,
        q99_ka=q99_ka,
        neg_mean_no=float(np.mean(neg_no)),
        neg_mean_ka=float(np.mean(neg_ka)),
        neg_std_no=float(np.std(neg_no)),
        neg_std_ka=float(np.std(neg_ka)),
    )
    assert q99_ka <= q99_no * 1.01

    pos_mean_ka = float(pos_ka.mean())
    neg_mean_ka = float(neg_ka.mean())
    pos_mean_no = float(pos_no.mean())
    pos_mean_no_bg_ratio = pos_mean_no / (neg_mean_ka + 1e-12)
    pos_mean_ka_bg_ratio = pos_mean_ka / (neg_mean_ka + 1e-12)
    _print_metrics(
        "ka_flow_stats",
        pos_mean_no=pos_mean_no,
        pos_mean_ka=pos_mean_ka,
        neg_mean_ka=neg_mean_ka,
        pos_mean_no_bg_ratio=pos_mean_no_bg_ratio,
        pos_mean_ka_bg_ratio=pos_mean_ka_bg_ratio,
        pos_max_no=float(pos_no.max()),
        pos_max_ka=float(pos_ka.max()),
    )
    assert pos_mean_ka >= neg_mean_ka * 1.05
    assert pos_mean_ka >= pos_mean_no * 0.8

    tpr_no = _tpr_at_fpr(pos_no, neg_no, 1e-2)
    tpr_ka = _tpr_at_fpr(pos_ka, neg_ka, 1e-2)
    _print_metrics("ka_detection_metrics", tpr_no=tpr_no, tpr_ka=tpr_ka)
    assert tpr_ka >= tpr_no
