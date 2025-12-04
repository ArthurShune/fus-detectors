import numpy as np
import torch

from pipeline.stap.temporal import (
    aggregate_over_snapshots,
    bandpass_constraints_temporal,
    build_temporal_hankels_and_cov,
    ka_blend_covariance_temporal,
    ka_prior_temporal_from_psd,
    msd_snapshot_energies_batched,
)


def _ratio_map(
    R_t,
    S,
    Cf,
    *,
    lam_abs,
    ridge,
    ratio_rho=0.0,
    R0_prior=None,
    Cf_flow=None,
    ka_opts=None,
    device="cpu",
):
    opts = ka_opts
    if opts is None and R0_prior is not None:
        opts = {"kappa_target": 40.0}
    T_band, sw_pow = msd_snapshot_energies_batched(
        R_t,
        S,
        Cf,
        lam_abs=lam_abs,
        ridge=ridge,
        ratio_rho=ratio_rho,
        R0_prior=R0_prior,
        Cf_flow=Cf_flow,
        ka_opts=opts,
        device=device,
    )
    ratio_snap = T_band / torch.clamp(sw_pow - T_band + ratio_rho * sw_pow, min=1e-10)
    return aggregate_over_snapshots(ratio_snap, mode="mean").cpu().numpy()


def _pd_snr_db(flow_vals, bg_vals):
    mu_flow = float(np.mean(flow_vals))
    var_bg = float(np.var(bg_vals))
    return 10.0 * np.log10((mu_flow**2) / max(var_bg, 1e-18))


def test_ka_band_limited_pd_snr_not_lower() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    T, H, W, Lt, prf = 64, 22, 22, 4, 3000.0
    y0, x0 = 9, 10
    t = torch.arange(T, dtype=torch.float32) / prf
    cube = (
        0.10 * (torch.randn(T, H, W) + 1j * torch.randn(T, H, W))
        + 0.03 * torch.exp(1j * 2 * np.pi * 50.0 * t)[:, None, None]
    ).to(torch.complex64)
    cube[:, y0, x0] += 0.18 * torch.exp(1j * 2 * np.pi * 600.0 * t)

    S_np, R_np, _ = build_temporal_hankels_and_cov(
        cube.numpy(),
        Lt=Lt,
        estimator="huber",
        huber_c=5.0,
        device="cpu",
        dtype=torch.complex64,
    )
    R_t = torch.as_tensor(R_np, dtype=torch.complex64)
    S = torch.as_tensor(S_np, dtype=torch.complex64)
    Cf = bandpass_constraints_temporal(
        Lt, prf, np.array([-600.0, -300.0, 0.0, 300.0, 600.0], dtype=np.float64), device="cpu"
    )

    ratio_no = _ratio_map(R_t, S, Cf, lam_abs=0.0, ridge=0.20, ratio_rho=0.05)

    R0 = ka_prior_temporal_from_psd(
        Lt=Lt,
        prf_hz=prf,
        f_peaks_hz=(0.0,),
        width_bins=1,
        add_deriv=True,
        device="cpu",
    )
    ratio_ka = _ratio_map(
        R_t,
        S,
        Cf,
        lam_abs=0.0,
        ridge=0.20,
        ratio_rho=0.05,
        R0_prior=R0,
        Cf_flow=Cf,
        ka_opts={"beta": 0.3, "kappa_target": 40.0, "lambda_override": 0.0},
    )

    mask_flow = np.zeros((H, W), dtype=bool)
    mask_flow[y0 - 1 : y0 + 2, x0 - 1 : x0 + 2] = True
    mask_bg = ~mask_flow

    pd_no = _pd_snr_db(ratio_no[mask_flow], ratio_no[mask_bg])
    pd_ka = _pd_snr_db(ratio_ka[mask_flow], ratio_ka[mask_bg])
    assert pd_ka >= pd_no - 0.1
