import json
from pathlib import Path

import numpy as np

from sim.kwave.common import (
    AngleData,
    SimGeom,
    build_grid_and_time,
    build_linear_masks,
    build_source_p,
    plane_wave_sample_shifts,
    write_acceptance_bundle,
)


def test_plane_wave_shifts_sign_and_center_zero():
    g = SimGeom(Nx=32, Ny=24, dx=100e-6, dy=100e-6, c0=1540.0, rho0=1000.0, ncycles=2)
    kgrid = build_grid_and_time(g)
    shifts_pos = plane_wave_sample_shifts(kgrid, g, theta_rad=np.deg2rad(10.0))
    shifts_neg = plane_wave_sample_shifts(kgrid, g, theta_rad=np.deg2rad(-10.0))

    # Center should be ~0 shift
    assert abs(int(round(shifts_pos[g.Nx // 2]))) <= 1
    assert abs(int(round(shifts_neg[g.Nx // 2]))) <= 1

    # Sign pattern: sign(theta)*sign(x)
    # Right side (x>0): positive shift for theta>0; negative for theta<0
    assert np.all(shifts_pos[g.Nx // 2 + 1 :] >= 0)
    assert np.all(shifts_neg[g.Nx // 2 + 1 :] <= 0)


def test_build_source_p_aligns_with_shifts():
    g = SimGeom(Nx=32, Ny=24, dx=100e-6, dy=100e-6, c0=1540.0, rho0=1000.0, ncycles=2)
    kgrid = build_grid_and_time(g)
    tx_mask, _ = build_linear_masks(g)
    theta = np.deg2rad(12.0)
    shifts = plane_wave_sample_shifts(kgrid, g, theta)
    p = build_source_p(g, kgrid, tx_mask, theta)

    # Build per-row expected first nonzero indices and compare with shifts for a few samples
    # Map mask row indices to x positions
    src_idx = np.argwhere(tx_mask)[:, 0]
    first_idx = []
    for row in range(p.shape[0]):
        nonzero = np.flatnonzero(np.abs(p[row]) > 0)
        first_idx.append(int(nonzero[0]) if nonzero.size else 0)
    first_idx = np.array(first_idx)

    # Compare differences between two positions (robust to truncation at edges)
    left = 2
    right = len(src_idx) - 3
    xi_l = src_idx[left]
    xi_r = src_idx[right]
    # Account for negative shifts being truncated to index 0 in build_source_p
    exp_diff = int(max(0, shifts[xi_r]) - max(0, shifts[xi_l]))
    got_diff = int(first_idx[right] - first_idx[left])
    # Allow small off-by-one due to burst envelope start
    assert abs(got_diff - exp_diff) <= 2


def test_write_acceptance_bundle_contract(tmp_path):
    # Use synthetic AngleData to avoid heavy k-Wave invocation
    g = SimGeom(Nx=32, Ny=32, dx=90e-6, dy=90e-6, c0=1540.0, rho0=1000.0, ncycles=1, f0=7.0e6)
    prf = 3000.0
    rng = np.random.default_rng(0)
    Nt = 96
    dt = g.cfl * min(g.dx, g.dy) / g.c0
    angles = [0.0, 8.0]
    angle_sets = []
    for i, ang in enumerate(angles):
        rf = rng.standard_normal((Nt, g.Nx)).astype(np.float32)
        # inject a small low-frequency tone so PSD span selection has something to see
        t = np.arange(Nt, dtype=np.float32) * dt
        rf += (0.05 * np.sin(2.0 * np.pi * (0.05 + 0.01 * i) * t))[:, None].astype(np.float32)
        angle_sets.append(AngleData(angle_deg=ang, rf=rf, dt=float(dt)))

    paths = write_acceptance_bundle(
        out_root=Path(tmp_path),
        g=g,
        angle_sets=[angle_sets],
        pulses_per_set=4,
        prf_hz=prf,
        seed=42,
        tile_hw=(8, 8),
        tile_stride=4,
        Lt=3,
        diag_load=1e-2,
        cov_estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="absolute",
        mvdr_auto_kappa=40.0,
        constraint_ridge=0.12,
        fd_span_mode="psd",
        fd_span_rel=(0.30, 1.10),
        fd_fixed_span_hz=None,
        grid_step_rel=0.08,
        fd_min_pts=5,
        fd_max_pts=7,
        msd_lambda=2e-2,
        msd_ridge=0.10,
        msd_agg_mode="trim10",
        stap_debug_samples=1,
        stap_device="cpu",
        dataset_name="test_bundle",
        meta_extra={"source": "unit_test"},
    )

    # Files exist
    required = [
        "base_pos",
        "base_neg",
        "stap_pos",
        "stap_neg",
        "pd_base",
        "pd_stap",
        "base_score_map",
        "stap_score_map",
        "stap_score_pool_map",
        "mask_flow",
        "mask_bg",
        "meta",
    ]
    for k in required:
        assert k in paths and Path(paths[k]).exists()

    # Confirm-2 pairs count matches stap_neg length // 2
    stap_neg = np.load(paths["stap_neg"])
    meta = json.loads(Path(paths["meta"]).read_text())
    assert meta["confirm2_pairs"]["n_pairs"] == (stap_neg.size // 2)
    assert meta["stap_device"] == "cpu"

    assert meta.get("score_pool_default") == "msd"
    score_pool_files = meta.get("score_pool_files")
    assert score_pool_files and set(score_pool_files.keys()) == {"msd", "pd", "band_ratio"}
    for mode, mapping in score_pool_files.items():
        assert set(mapping.keys()) == {"base_pos", "base_neg", "stap_pos", "stap_neg"}
        for rel in mapping.values():
            assert rel.endswith(".npy")

    bundle_files = meta.get("bundle_files")
    assert bundle_files and bundle_files["pd_base"].endswith("pd_base.npy")
    score_stats = meta.get("score_stats", {})
    assert score_stats.get("mode") == "msd"
    assert "map_stats" in score_stats and "pd" in score_stats["map_stats"]
    pool_stats = score_stats.get("pool_stats", {})
    assert pool_stats.get("pd", {}).get("npos")

    # Telemetry sanity
    stap_tel = meta.get("stap_fallback_telemetry", {})
    med_grid = stap_tel.get("median_grid_step_hz")
    assert med_grid is not None and med_grid > 0.0 and med_grid < prf / 2.0
    pd_stats = meta.get("pd_stats", {})
    # PD medians should be finite if masks are non-empty
    for k in ["baseline_flow_median", "baseline_bg_median", "stap_flow_median", "stap_bg_median"]:
        assert k in pd_stats and pd_stats[k] is not None and np.isfinite(pd_stats[k])
