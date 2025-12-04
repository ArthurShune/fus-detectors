import numpy as np
import torch

from pipeline.stap.temporal import (
    aggregate_over_snapshots,
    bandpass_constraints_temporal,
    build_temporal_hankels_and_cov,
    ka_prior_temporal_from_psd,
    msd_snapshot_energies_batched,
)
from sim.kwave.common import _stap_pd


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


def _make_synthetic_cube(
    flow_amp: float = 1.0,
    *,
    T: int = 48,
    H: int = 12,
    W: int = 12,
    prf_hz: float = 3000.0,
    motion_freq: float = 75.0,
    flow_freq: float = 600.0,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    torch.manual_seed(0)
    np.random.seed(0)
    t = torch.arange(T, dtype=torch.float32) / prf_hz
    noise = 0.1 * (torch.randn(T, H, W) + 1j * torch.randn(T, H, W))
    noise = noise.to(torch.complex64)
    drift = 0.05 * torch.exp(1j * 2 * np.pi * motion_freq * t)[:, None, None]
    cube = (noise + drift).to(torch.complex64)
    cube[:, H // 2, W // 2] += flow_amp * torch.exp(1j * 2 * np.pi * flow_freq * t)
    mask_flow = np.zeros((H, W), dtype=bool)
    mask_flow[H // 2, W // 2] = True
    mask_bg = ~mask_flow
    return cube, mask_flow, mask_bg


def _ratio_snapshots(
    R_t: torch.Tensor,
    S: torch.Tensor,
    Cf: torch.Tensor,
    *,
    lam_abs: float,
    ridge: float,
    ratio_rho: float = 0.0,
    R0_prior: torch.Tensor | None = None,
    Cf_flow: torch.Tensor | None = None,
    device: str = "cpu",
    ka_opts: dict | None = None,
) -> torch.Tensor:
    opts = ka_opts
    if opts is None and R0_prior is not None:
        opts = {"kappa_target": 40.0, "lambda_override": 0.0}
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
    denom = torch.clamp((sw_pow - T_band) * (1.0 + ratio_rho), min=1e-10)
    ratio = torch.clamp(T_band / denom, min=0.0)
    return ratio


def _ratio_map(*args, agg: str = "mean", ka_opts: dict | None = None, **kwargs) -> torch.Tensor:
    ratio_snap = _ratio_snapshots(*args, ka_opts=ka_opts, **kwargs)
    return aggregate_over_snapshots(ratio_snap, mode=agg)


def test_msd_ratio_synthetic_flow_beats_background() -> None:
    cube, mask_flow_np, mask_bg_np = _make_synthetic_cube(flow_amp=1.0)
    S, R_t, _ = build_temporal_hankels_and_cov(
        cube,
        Lt=4,
        center=True,
        estimator="huber",
        huber_c=5.0,
        device="cpu",
        dtype=torch.complex64,
    )
    Cf = bandpass_constraints_temporal(
        Lt=4,
        prf_hz=3000.0,
        fd_grid_hz=np.array([-600.0, -300.0, 0.0, 300.0, 600.0], dtype=np.float64),
        device="cpu",
    )
    ratio_map = _ratio_map(R_t, S, Cf, lam_abs=0.02, ridge=0.20, ratio_rho=0.0, device="cpu")
    mask_flow = torch.from_numpy(mask_flow_np)
    mask_bg = torch.from_numpy(mask_bg_np)
    flow_mean = float(ratio_map[mask_flow].mean().item())
    bg_q99 = float(torch.quantile(ratio_map[mask_bg], 0.99).item())
    bg_q999 = float(torch.quantile(ratio_map[mask_bg], 0.999).item())
    flow_max = float(ratio_map[mask_flow].max().item())
    bg_mean = float(ratio_map[mask_bg].mean().item())
    _print_metrics(
        "temporal_flow_vs_bg",
        flow_mean=flow_mean,
        flow_max=flow_max,
        bg_q99=bg_q99,
        bg_q999=bg_q999,
        bg_mean=bg_mean,
    )
    assert flow_mean > bg_q99 * 1.5
    assert float(ratio_map[mask_flow].max().item()) > bg_q999


def test_ka_blend_tightens_null_tail_without_crushing_flow() -> None:
    cube, mask_flow_np, mask_bg_np = _make_synthetic_cube(flow_amp=1.0)
    S, R_t, _ = build_temporal_hankels_and_cov(
        cube,
        Lt=4,
        center=True,
        estimator="huber",
        huber_c=5.0,
        device="cpu",
        dtype=torch.complex64,
    )
    Cf = bandpass_constraints_temporal(
        Lt=4,
        prf_hz=3000.0,
        fd_grid_hz=np.array([-600.0, -300.0, 0.0, 300.0, 600.0], dtype=np.float64),
        device="cpu",
    )
    ratio_no = _ratio_map(R_t, S, Cf, lam_abs=0.02, ridge=0.20, ratio_rho=0.05, device="cpu")
    R0 = ka_prior_temporal_from_psd(
        Lt=4,
        prf_hz=3000.0,
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
        device="cpu",
        ka_opts={"kappa_target": 40.0, "lambda_override": 0.06},
    )
    mask_flow = torch.from_numpy(mask_flow_np)
    mask_bg = torch.from_numpy(mask_bg_np)
    q999_no = float(torch.quantile(ratio_no[mask_bg], 0.999).item())
    q999_ka = float(torch.quantile(ratio_ka[mask_bg], 0.999).item())
    flow_mean_no = float(ratio_no[mask_flow].mean().item())
    flow_mean_ka = float(ratio_ka[mask_flow].mean().item())
    flow_max_no = float(ratio_no[mask_flow].max().item())
    flow_max_ka = float(ratio_ka[mask_flow].max().item())
    bg_mean_no = float(ratio_no[mask_bg].mean().item())
    bg_mean_ka = float(ratio_ka[mask_bg].mean().item())
    _print_metrics(
        "temporal_guardrails",
        q999_no=q999_no,
        q999_ka=q999_ka,
        flow_mean_no=flow_mean_no,
        flow_mean_ka=flow_mean_ka,
        flow_max_no=flow_max_no,
        flow_max_ka=flow_max_ka,
        bg_mean_no=bg_mean_no,
        bg_mean_ka=bg_mean_ka,
    )
    assert q999_ka <= q999_no + 1e-4
    assert flow_mean_ka >= 0.8 * flow_mean_no


def test_median_aggregation_retains_flow_peak() -> None:
    cube, mask_flow_np, mask_bg_np = _make_synthetic_cube(flow_amp=1.0)
    S, R_t, _ = build_temporal_hankels_and_cov(
        cube,
        Lt=4,
        center=True,
        estimator="huber",
        huber_c=5.0,
        device="cpu",
        dtype=torch.complex64,
    )
    Cf = bandpass_constraints_temporal(
        Lt=4,
        prf_hz=3000.0,
        fd_grid_hz=np.array([-600.0, -300.0, 0.0, 300.0, 600.0], dtype=np.float64),
        device="cpu",
    )
    ratio_snap = _ratio_snapshots(R_t, S, Cf, lam_abs=0.02, ridge=0.20, device="cpu")
    ratio_mean = aggregate_over_snapshots(ratio_snap, mode="mean")
    ratio_median = aggregate_over_snapshots(ratio_snap, mode="median")
    mask_flow = torch.from_numpy(mask_flow_np)
    mask_bg = torch.from_numpy(mask_bg_np)
    flow_mean = float(ratio_mean[mask_flow].mean().item())
    flow_median = float(ratio_median[mask_flow].mean().item())
    bg_q95_median = float(torch.quantile(ratio_median[mask_bg], 0.95).item())
    bg_mean_median = float(ratio_median[mask_bg].mean().item())
    _print_metrics(
        "temporal_snapshot_agg",
        flow_mean=flow_mean,
        flow_median=flow_median,
        flow_max=float(ratio_median[mask_flow].max().item()),
        bg_q95=bg_q95_median,
        bg_mean=bg_mean_median,
    )
    assert flow_median >= 0.6 * flow_mean
    assert flow_median > bg_q95_median


def test_ratio_shrinkage_reduces_background_tail() -> None:
    cube, mask_flow_np, mask_bg_np = _make_synthetic_cube(flow_amp=1.0)
    S, R_t, _ = build_temporal_hankels_and_cov(
        cube,
        Lt=4,
        center=True,
        estimator="huber",
        huber_c=5.0,
        device="cpu",
        dtype=torch.complex64,
    )
    Cf = bandpass_constraints_temporal(
        Lt=4,
        prf_hz=3000.0,
        fd_grid_hz=np.array([-600.0, -300.0, 0.0, 300.0, 600.0], dtype=np.float64),
        device="cpu",
    )
    ratio_lo = _ratio_map(R_t, S, Cf, lam_abs=0.02, ridge=0.20, ratio_rho=0.0, device="cpu")
    ratio_hi = _ratio_map(R_t, S, Cf, lam_abs=0.02, ridge=0.20, ratio_rho=0.2, device="cpu")
    mask_flow = torch.from_numpy(mask_flow_np)
    mask_bg = torch.from_numpy(mask_bg_np)
    q99_lo = float(torch.quantile(ratio_lo[mask_bg], 0.99).item())
    q99_hi = float(torch.quantile(ratio_hi[mask_bg], 0.99).item())
    flow_lo = float(ratio_lo[mask_flow].mean().item())
    flow_hi = float(ratio_hi[mask_flow].mean().item())
    flow_hi_max = float(ratio_hi[mask_flow].max().item())
    flow_lo_max = float(ratio_lo[mask_flow].max().item())
    _print_metrics(
        "temporal_ratio_shrink",
        q99_lo=q99_lo,
        q99_hi=q99_hi,
        flow_lo=flow_lo,
        flow_hi=flow_hi,
        flow_lo_max=flow_lo_max,
        flow_hi_max=flow_hi_max,
    )
    assert q99_hi < q99_lo
    assert flow_hi >= 0.6 * flow_lo


def test_stap_pd_reports_healthy_flow_ratio() -> None:
    cube, mask_flow_np, mask_bg_np = _make_synthetic_cube(flow_amp=1.2, H=16, W=16)
    Icube = cube.cpu().numpy()
    pd_base = np.mean(np.abs(Icube) ** 2, axis=0).astype(np.float32)
    pd_map, score_map, info = _stap_pd(
        Icube,
        tile_hw=(8, 8),
        stride=4,
        Lt=4,
        prf_hz=3000.0,
        diag_load=1e-2,
        estimator="tyler_pca",
        huber_c=5.0,
        mvdr_load_mode="auto",
        mvdr_auto_kappa=40.0,
        constraint_ridge=0.12,
        fd_span_mode="psd",
        fd_span_rel=(0.30, 1.10),
        fd_fixed_span_hz=None,
        grid_step_rel=0.10,
        min_pts=3,
        max_pts=7,
        msd_lambda=0.02,
        msd_ridge=0.20,
        msd_agg_mode="median",
        msd_ratio_rho=0.07,
        motion_half_span_rel=None,
        msd_contrast_alpha=None,
        debug_max_samples=0,
        stap_device="cpu",
        pd_base_full=pd_base,
        mask_flow=mask_flow_np,
        mask_bg=mask_bg_np,
        ka_mode="analytic",
    )
    flow_mu = float(np.mean(pd_map[mask_flow_np]))
    bg_mu = float(np.mean(pd_map[mask_bg_np]))
    flow_median = float(np.median(pd_map[mask_flow_np]))
    bg_median = float(np.median(pd_map[mask_bg_np]))
    _print_metrics(
        "stap_pd_summary",
        total_tiles=int(info["total_tiles"]),
        flow_mu_ratio=float(info["median_flow_mu_ratio"]),
        flow_mu=flow_mu,
        flow_median=flow_median,
        bg_mu=bg_mu,
        bg_median=bg_median,
        ka_mode=info.get("ka_mode"),
        ka_retain_total=float(info.get("ka_median_retain_f_total") or 0.0),
        ka_shrink_total=float(info.get("ka_median_shrink_perp_total") or 0.0),
    )
    assert pd_map.shape == (16, 16)
    assert score_map.shape == (16, 16)
    assert info["score_mode_histogram"].get("msd", 0) == info["total_tiles"]
    assert info["median_flow_mu_ratio"] > 0.45
    assert info["ka_mode"] != "none"
    # Gate on TOTAL directional effects (β + λ-split)
    assert (
        info["ka_median_retain_f_total"] is not None and info["ka_median_retain_f_total"] >= 0.90
    )
    # allow gentle shrink for small Lt, but it should not explode
    assert (
        info["ka_median_shrink_perp_total"] is not None
        and info["ka_median_shrink_perp_total"] <= 1.0
    )
    assert info["ka_ridge_split_fraction"] is not None and info["ka_ridge_split_fraction"] >= 0.9
    strat_hist = info.get("ka_lambda_strategy_hist", {})
    if strat_hist:
        split_frac = strat_hist.get("split", 0) / max(sum(strat_hist.values()), 1)
        assert split_frac >= 0.8
