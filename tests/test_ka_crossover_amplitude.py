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


def _make_base_scene(T=64, H=24, W=24, prf=3000.0):
    torch.manual_seed(0)
    np.random.seed(0)
    t = torch.arange(T, dtype=torch.float32) / prf
    noise = 0.10 * (torch.randn(T, H, W) + 1j * torch.randn(T, H, W))
    drift = 0.05 * torch.exp(1j * 2 * np.pi * 75.0 * t)[:, None, None]
    cube = (noise + drift).to(torch.complex64)
    mask_flow = np.zeros((H, W), dtype=bool)
    mask_flow[H // 2 - 1 : H // 2 + 2, W // 2 - 1 : W // 2 + 2] = True
    mask_bg = ~mask_flow
    return cube, mask_flow, mask_bg


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
):
    opts = {"kappa_target": 40.0}
    if ka_opts:
        opts.update(ka_opts)
    T_band, sw_pow = msd_snapshot_energies_batched(
        R_t,
        S,
        Cf,
        lam_abs=lam_abs,
        ridge=ridge,
        ratio_rho=ratio_rho,
        R0_prior=R0_prior,
        Cf_flow=Cf_flow,
        ka_opts=opts if R0_prior is not None else None,
        device="cpu",
    )
    ratio_snap = T_band / torch.clamp(sw_pow - T_band + ratio_rho * sw_pow, min=1e-10)
    ratio_map = aggregate_over_snapshots(ratio_snap, mode="mean")
    return ratio_map.cpu().numpy()


def _tpr_at_fpr(pos, neg, fpr):
    if pos.size == 0 or neg.size == 0:
        return 0.0
    thr = np.quantile(neg, 1.0 - fpr, method="linear")
    return float((pos >= thr).mean())


def test_ka_improves_tpr_at_fixed_amplitude() -> None:
    cube_base, mask_flow, mask_bg = _make_base_scene()
    T, H, W = cube_base.shape
    prf = 3000.0
    Lt = 4
    y0, x0 = H // 2, W // 2
    t = torch.arange(T, dtype=torch.float32) / prf
    fd = np.array([-600.0, -300.0, 0.0, 300.0, 600.0], dtype=np.float64)
    Cf = bandpass_constraints_temporal(Lt, prf, fd, device="cpu")
    R0 = ka_prior_temporal_from_psd(
        Lt=Lt, prf_hz=prf, f_peaks_hz=(0.0,), width_bins=1, add_deriv=True, device="cpu"
    )

    amps = np.linspace(0.10, 0.32, 8)
    target_fpr = 1e-2
    target_tpr = 0.1
    hit_no = None
    hit_ka = None

    for amp in amps:
        cube = cube_base.clone()
        cube[:, y0, x0] = cube[:, y0, x0] + amp * torch.exp(1j * 2 * np.pi * 600.0 * t)

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

        score_no = _ratio_map(R_t, S, Cf, lam_abs=0.02, ridge=0.20, ratio_rho=0.05)
        score_ka = _ratio_map(
            R_t,
            S,
            Cf,
            lam_abs=0.0,
            ridge=0.20,
            ratio_rho=0.05,
            R0_prior=R0,
            Cf_flow=Cf,
            ka_opts={
                "lambda_override_split": 0.02,
                "beta_directional": True,
                "ridge_split": True,
                "target_retain_f": 0.95,
                "target_shrink_perp": 0.95,
                "beta_bounds": (0.05, 0.18),
            },
        )

        pos_no = score_no[mask_flow]
        neg_no = score_no[mask_bg]
        pos_ka = score_ka[mask_flow]
        neg_ka = score_ka[mask_bg]

        tpr_no = _tpr_at_fpr(pos_no, neg_no, target_fpr)
        tpr_ka = _tpr_at_fpr(pos_ka, neg_ka, target_fpr)

        if hit_no is None and tpr_no >= target_tpr:
            hit_no = amp
        if hit_ka is None and tpr_ka >= target_tpr:
            hit_ka = amp

    assert hit_ka is not None, "KA never achieved the target TPR; ensure flow is on-grid."
    if hit_no is not None:
        assert (
            hit_ka <= hit_no + 1e-9
        ), f"KA should not require higher amplitude (KA={hit_ka:.3f}, noKA={hit_no:.3f})"
