import json
from pathlib import Path

import numpy as np

from sim.kwave.common import (
    AngleData,
    SimGeom,
    _inject_temporal_clutter,
    _sample_phase_screen_vector,
    write_acceptance_bundle,
)


def test_phase_screen_vector_stats():
    n = 64
    std = 0.8
    corr = 12.0
    seed = 123
    v = _sample_phase_screen_vector(n, std, corr, seed)
    assert v is not None
    # RMS close to requested std (within 15%)
    rms = float(np.sqrt(np.mean(v**2)))
    assert 0.85 * std <= rms <= 1.15 * std
    # Autocorrelation decays with lag
    ac1 = float(np.corrcoef(v[:-1], v[1:])[0, 1])
    ac8 = float(np.corrcoef(v[:-8], v[8:])[0, 1])
    assert ac1 > ac8


def test_inject_temporal_clutter_snr():
    T, H, W = 64, 24, 24
    rng = np.random.default_rng(0)
    cube = (rng.standard_normal((T, H, W)) + 1j * rng.standard_normal((T, H, W))).astype(
        np.complex64
    ) * 0.1
    mask_bg = np.ones((H, W), dtype=bool)
    beta = 1.0
    snr_db = -6.0
    cube2, meta = _inject_temporal_clutter(
        cube,
        mask_bg,
        beta=beta,
        snr_db=snr_db,
        depth_min_frac=0.2,
        depth_max_frac=0.9,
        seed=42,
    )
    assert meta is not None
    # Actual clutter SNR should be within ~3 dB of target
    assert abs(float(meta["snr_db_actual"]) - snr_db) <= 3.5
    assert int(meta["n_pixels"]) > 0


def test_write_acceptance_bundle_phase_clutter_alias(tmp_path):
    # Minimal synthetic bundle with phase screen + clutter + alias flags
    g = SimGeom(Nx=32, Ny=32, dx=90e-6, dy=90e-6, c0=1540.0, rho0=1000.0, ncycles=1, f0=7.5e6)
    prf = 1500.0
    rng = np.random.default_rng(1)
    Nt = 96
    dt = g.cfl * min(g.dx, g.dy) / g.c0
    angles = [0.0, 10.0]
    angle_sets = []
    for i, ang in enumerate(angles):
        rf = rng.standard_normal((Nt, g.Nx)).astype(np.float32)
        angle_sets.append(AngleData(angle_deg=ang, rf=rf, dt=float(dt)))

    paths = write_acceptance_bundle(
        out_root=Path(tmp_path),
        g=g,
        angle_sets=[angle_sets],
        pulses_per_set=3,
        prf_hz=prf,
        seed=7,
        tile_hw=(8, 8),
        tile_stride=3,
        Lt=3,
        diag_load=1e-2,
        cov_estimator="scm",
        huber_c=5.0,
        msd_lambda=2e-2,
        msd_ridge=0.10,
        msd_agg_mode="median",
        stap_debug_samples=0,
        stap_device="cpu",
        flow_alias_hz=900.0,
        flow_alias_fraction=0.55,
        flow_alias_depth_min_frac=0.2,
        flow_alias_depth_max_frac=0.7,
        flow_alias_jitter_hz=25.0,
        aperture_phase_std=0.8,
        aperture_phase_corr_len=14.0,
        aperture_phase_seed=111,
        clutter_beta=1.0,
        clutter_snr_db=-6.0,
        clutter_depth_min_frac=0.2,
        clutter_depth_max_frac=0.9,
        dataset_name="test_r4c",
    )
    meta = json.loads(Path(paths["meta"]).read_text())
    # Check new meta sections exist and have expected keys
    assert "flow_alias" in meta and meta["flow_alias"].get("flow_alias_hz") is not None
    assert meta["flow_alias"].get("flow_alias_depth_min_frac") == 0.2
    assert meta["flow_alias"].get("flow_alias_depth_max_frac") == 0.7
    assert "phase_screen" in meta and meta["phase_screen"].get("phase_std") == 0.8
    assert "temporal_clutter" in meta and meta["temporal_clutter"].get("beta") == 1.0
    # Telemetry should echo some fields
    tel = meta.get("stap_fallback_telemetry", {})
    assert "bg_var_ratio_actual" in tel
    assert tel.get("psd_alias_ratio_count", 0) >= 0
    assert "psd_alias_ratio_median" in tel
    assert "alias_flag_ratio_thresh" in tel
    # Band-ratio maps should be written
    for k in ("base_band_ratio_map", "stap_band_ratio_map"):
        assert k in paths and Path(paths[k]).exists()
