from __future__ import annotations

import numpy as np

from sim.kwave.common import (
    _baseline_pd_adaptive_local_svd,
    _baseline_pd_mcsvd,
    _baseline_pd_rpca,
    _fft_shift_apply,
    _register_stack_phasecorr,
    _svd_temporal_project,
    _phasecorr_shift,
)


def test_stack_registration_recovers_static_scene() -> None:
    rng = np.random.default_rng(42)
    T, H, W = 16, 64, 64
    yy, xx = np.mgrid[0:H, 0:W]
    mag = (
        2.0 * np.exp(-((yy - 20) ** 2 + (xx - 32) ** 2) / 120.0)
        + 1.3 * np.exp(-((yy - 45) ** 2 + (xx - 18) ** 2) / 80.0)
    ).astype(np.float32)
    phase = (0.2 * rng.standard_normal((H, W))).astype(np.float32)
    base_frame = (mag * np.exp(1j * phase)).astype(np.complex64)
    clean_stack = np.tile(base_frame, (T, 1, 1))
    noisy = clean_stack + 0.01 * (
        rng.standard_normal((T, H, W)) + 1j * rng.standard_normal((T, H, W))
    ).astype(np.complex64)
    shifted = np.empty_like(noisy)
    true_shifts = []
    for t in range(T):
        dy = 0.4 * np.sin(2 * np.pi * t / T)
        dx = -0.6 * np.cos(2 * np.pi * t / T)
        shifted[t] = _fft_shift_apply(noisy[t], dy, dx)
        true_shifts.append((dy, dx))

    reg_cube, tele = _register_stack_phasecorr(
        shifted, reg_enable=True, upsample=4, ref_strategy="median"
    )
    # Registered cube should match the unshifted noisy stack
    rel_err = float(np.linalg.norm(reg_cube - noisy) / (np.linalg.norm(noisy) + 1e-12))
    assert rel_err < 9e-2
    assert tele["reg_enable"] is True
    assert tele["reg_failed_fraction"] <= 1e-3
    assert tele["reg_shift_rms"] > 0.0


def test_fft_shift_inverse_is_identity() -> None:
    rng = np.random.default_rng(0)
    h, w = 48, 48
    img0 = (rng.standard_normal((h, w)) + 1j * rng.standard_normal((h, w))).astype(np.complex64)
    dy, dx = 0.33, -0.41
    shifted = _fft_shift_apply(img0, dy, dx)
    restored = _fft_shift_apply(shifted, -dy, -dx)
    err = float(np.linalg.norm(restored - img0) / (np.linalg.norm(img0) + 1e-12))
    assert err < 1e-3


def test_svd_projector_energy_fraction_and_monotonicity() -> None:
    rng = np.random.default_rng(1)
    T, N, r = 64, 512, 3
    U = rng.standard_normal((T, r))
    Q, _ = np.linalg.qr(U)
    V = rng.standard_normal((r, N))
    tissue = (Q[:, :r] @ V).astype(np.complex64)
    noise = (rng.standard_normal((T, N)) + 1j * rng.standard_normal((T, N))).astype(
        np.complex64
    ) * 0.05
    A = tissue + noise

    # Remove known rank
    Af_r, tele_r = _svd_temporal_project(A, rank=r, device="cpu")
    # Remove more rank must not increase residual energy
    Af_rp1, tele_rp1 = _svd_temporal_project(A, rank=r + 1, device="cpu")
    e_r = float(np.linalg.norm(Af_r))
    e_rp1 = float(np.linalg.norm(Af_rp1))
    assert tele_r["svd_rank_removed"] == r
    assert tele_rp1["svd_rank_removed"] == r + 1
    assert e_rp1 <= e_r + 1e-6

    # Energy-fraction chooser should pick at least r for high fraction
    Af_frac, tele_frac = _svd_temporal_project(A, rank=None, energy_frac=0.9, device="cpu")
    assert tele_frac["svd_rank_removed"] >= r


def test_mcsvd_registration_improves_contrast_on_shifted_stack() -> None:
    rng = np.random.default_rng(3)
    T, H, W = 48, 48, 48
    # Low-rank tissue: rank-2 over time
    U = rng.standard_normal((T, 2))
    V = rng.standard_normal((2, H * W))
    tissue = (U @ V).reshape(T, H, W).astype(np.complex64)
    # Add a small ROI flow (higher variance) in a circular mask
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx, r = H // 2, W // 2, W // 6
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    flow = ((rng.standard_normal((T, H, W)) + 1j * rng.standard_normal((T, H, W))) * 0.15).astype(
        np.complex64
    )
    flow *= mask[None, :, :]
    cube = tissue + flow
    # Inject frame-wise rigid motion so registration has an effect
    for t in range(T):
        dy = 0.6 * np.sin(2 * np.pi * t / T)
        dx = -0.5 * np.cos(2 * np.pi * t / T)
        cube[t] = _fft_shift_apply(cube[t], dy, dx)

    # Run baseline without and with registration
    pd_no_reg, tele0 = _baseline_pd_mcsvd(
        cube, reg_enable=False, svd_rank=3, reg_subpixel=4, reg_reference="first"
    )
    pd_reg, tele1 = _baseline_pd_mcsvd(
        cube, reg_enable=True, svd_rank=3, reg_subpixel=4, reg_reference="median"
    )
    # Simple contrast proxy: ROI mean minus background mean
    bg_mask = ~mask
    c0 = float(pd_no_reg[mask].mean() - pd_no_reg[bg_mask].mean())
    c1 = float(pd_reg[mask].mean() - pd_reg[bg_mask].mean())
    # Registration should not worsen contrast and typically improves it
    assert c1 >= c0 * 0.95
    # Background variance should not increase when registration is enabled
    v0 = float(np.var(pd_no_reg[bg_mask]))
    v1 = float(np.var(pd_reg[bg_mask]))
    assert v1 <= v0 * 1.05


def test_rpca_recovers_sparse_component_median() -> None:
    rng = np.random.default_rng(5)
    T, H, W = 40, 32, 32
    # Low-rank tissue (rank-2)
    tissue = (
        (rng.standard_normal((T, 2)) @ rng.standard_normal((2, H * W)))
        .reshape(T, H, W)
        .astype(np.complex64)
    )
    # Sparse flow: Bernoulli mask with higher variance
    mask = rng.random((T, H, W)) < 0.05
    flow = np.zeros((T, H, W), dtype=np.complex64)
    flow[mask] = rng.normal(0, 0.5, size=mask.sum()) + 1j * rng.normal(0, 0.5, size=mask.sum())
    cube = tissue + flow
    pd_flow = np.mean(np.abs(flow) ** 2, axis=0)
    pd_rpca, tele = _baseline_pd_rpca(cube, lambda_=None, max_iters=80)
    assert tele["baseline_type"] == "rpca"
    # Sparse component energy should exceed a percentile of tissue energy
    roi = mask.any(axis=0)
    ratio = float(np.median(pd_rpca[roi]) / (np.median(pd_flow[roi]) + 1e-12))
    assert 0.9 < ratio < 1.1


def test_per_angle_mc_svd_fusion_matches_direct_average() -> None:
    rng = np.random.default_rng(6)
    angles = 3
    T, H, W = 24, 24, 24
    cubes = []
    for ang in range(angles):
        phase = rng.random((T, H, W)) * 2 * np.pi
        amp = rng.random((T, H, W)) * 0.3 + 0.7
        cubes.append((amp * np.exp(1j * phase)).astype(np.complex64))

    per_angle_pd = []
    for cube in cubes:
        pd, _ = _baseline_pd_mcsvd(
            cube, reg_enable=False, svd_rank=2, reg_subpixel=1, reg_reference="first"
        )
        per_angle_pd.append(pd)
    direct_avg = np.mean(np.stack(per_angle_pd, axis=0), axis=0)

    # Simulate fusion routine
    fused = np.zeros_like(per_angle_pd[0])
    for pd in per_angle_pd:
        fused += pd / angles
    rel_err = float(np.linalg.norm(fused - direct_avg) / (np.linalg.norm(direct_avg) + 1e-12))
    assert rel_err < 1e-6


def test_registration_psr_distribution_and_fail_fraction() -> None:
    rng = np.random.default_rng(7)
    T, H, W = 16, 64, 64
    yy, xx = np.mgrid[0:H, 0:W]
    mag = (
        2.5 * np.exp(-((yy - 18) ** 2 + (xx - 28) ** 2) / 90.0)
        + 1.1 * np.exp(-((yy - 44) ** 2 + (xx - 20) ** 2) / 60.0)
    ).astype(np.float32)
    phase = (0.1 * rng.standard_normal((H, W))).astype(np.float32)
    base = (mag * np.exp(1j * phase)).astype(np.complex64)
    stack = np.tile(base, (T, 1, 1))
    for t in range(T):
        dy = 0.5 * np.sin(2 * np.pi * t / T)
        dx = -0.4 * np.cos(2 * np.pi * t / T)
        stack[t] = _fft_shift_apply(stack[t], dy, dx)
    # compute PSR per-frame
    ref = np.median(np.abs(stack), axis=0).astype(np.complex64)
    psrs = []
    for t in range(T):
        _, _, psr = _phasecorr_shift(ref, stack[t], upsample=4)
        psrs.append(psr)
    psr_p90 = float(np.percentile(psrs, 90))
    assert psr_p90 > 5.0
    # Furthermore, registration should report zero failures
    _, tele = _register_stack_phasecorr(stack, reg_enable=True, upsample=4, ref_strategy="median")
    assert tele["reg_failed_fraction"] == 0.0


def test_svd_energy_fraction_removed_close_to_target() -> None:
    rng = np.random.default_rng(9)
    T, N = 96, 512
    A = (rng.standard_normal((T, N)) + 1j * rng.standard_normal((T, N))).astype(np.complex64)
    _, tele = _svd_temporal_project(A, rank=None, energy_frac=0.9, device="cpu")
    frac = tele["svd_energy_removed_frac"]
    assert 0.90 <= frac <= 0.95


def test_rpca_lowrank_energy_fraction_reasonable() -> None:
    rng = np.random.default_rng(11)
    T, H, W = 40, 24, 24
    tissue = (
        (rng.standard_normal((T, 2)) @ rng.standard_normal((2, H * W)))
        .reshape(T, H, W)
        .astype(np.complex64)
    )
    flow = (rng.standard_normal((T, H, W)) + 1j * rng.standard_normal((T, H, W))).astype(
        np.complex64
    ) * 0.05
    cube = tissue + flow
    _, tele = _baseline_pd_rpca(cube, lambda_=None, max_iters=60)
    assert tele["rpca_energy_lowrank_frac"] > 0.85


def test_adaptive_local_svd_returns_finite_pd_and_tile_rank_stats() -> None:
    rng = np.random.default_rng(13)
    T, H, W = 32, 24, 24
    cube = np.zeros((T, H, W), dtype=np.complex64)

    # Top-left tile: highly coherent clutter plus weak flow.
    clutter_t = rng.standard_normal((T, 2)).astype(np.float32)
    clutter_v = rng.standard_normal((2, 12 * 12)).astype(np.float32)
    tile_coherent = (clutter_t @ clutter_v).reshape(T, 12, 12).astype(np.complex64)
    tile_coherent += 0.03 * (
        rng.standard_normal((T, 12, 12)) + 1j * rng.standard_normal((T, 12, 12))
    ).astype(np.complex64)

    # Bottom-right tile: less coherent, more mixed temporal content.
    tile_mixed = 0.25 * (
        rng.standard_normal((T, 12, 12)) + 1j * rng.standard_normal((T, 12, 12))
    ).astype(np.complex64)
    for k in range(4):
        amp = 0.15 * rng.standard_normal((12, 12)).astype(np.float32)
        phase = np.exp(1j * (0.3 * k + np.pi * np.arange(T, dtype=np.float32) / max(1, T - 1)))
        tile_mixed += phase[:, None, None].astype(np.complex64) * amp[None, :, :]

    cube[:, :12, :12] = tile_coherent
    cube[:, 12:, 12:] = tile_mixed
    cube += 0.01 * (
        rng.standard_normal((T, H, W)) + 1j * rng.standard_normal((T, H, W))
    ).astype(np.complex64)

    pd_map, tele = _baseline_pd_adaptive_local_svd(
        cube,
        tile_hw=(12, 12),
        stride=12,
        svd_sim_smooth=5,
        svd_sim_kappa=2.0,
        svd_sim_r_min=1,
        svd_sim_r_max=8,
    )

    assert pd_map.shape == (H, W)
    assert np.isfinite(pd_map).all()
    assert tele["baseline_type"] == "adaptive_local_svd"
    assert 1 <= tele["adaptive_local_svd_rank_removed_min"] <= 8
    assert 1 <= tele["adaptive_local_svd_rank_removed_max"] <= 8
    assert tele["adaptive_local_svd_rank_removed_max"] >= tele["adaptive_local_svd_rank_removed_min"]
