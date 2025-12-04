import json
from pathlib import Path

import numpy as np
import pytest

from eval import acceptance_cli
from sim.kwave.common import AngleData, SimGeom, slice_angle_data, write_acceptance_bundle


def _make_bundle(
    root: Path,
    *,
    score_mode: str = "msd",
    dataset_name: str = "bundle_case",
    dataset_suffix: str | None = None,
) -> Path:
    g = SimGeom(Nx=40, Ny=40, dx=90e-6, dy=90e-6, c0=1540.0, rho0=1000.0, ncycles=1, f0=7.0e6)
    prf = 2800.0
    rng = np.random.default_rng(1)
    Nt = 64
    dt = g.cfl * min(g.dx, g.dy) / g.c0
    angles = [0.0, 6.0]
    angle_sets = []
    for i, ang in enumerate(angles):
        rf = rng.standard_normal((Nt, g.Nx)).astype(np.float32)
        t = np.arange(Nt, dtype=np.float32) * dt
        rf += (0.03 * np.sin(2.0 * np.pi * (0.04 + 0.01 * i) * t))[:, None].astype(np.float32)
        angle_sets.append(AngleData(angle_deg=ang, rf=rf, dt=float(dt)))

    write_acceptance_bundle(
        out_root=root,
        g=g,
        angle_sets=[angle_sets],
        pulses_per_set=4,
        prf_hz=prf,
        seed=7,
        tile_hw=(6, 6),
        tile_stride=3,
        Lt=3,
        diag_load=5e-3,
        cov_estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="absolute",
        mvdr_auto_kappa=30.0,
        constraint_ridge=0.1,
        fd_span_mode="psd",
        fd_span_rel=(0.3, 1.0),
        fd_fixed_span_hz=None,
        grid_step_rel=0.08,
        fd_min_pts=5,
        fd_max_pts=7,
        msd_lambda=1e-2,
        msd_ridge=0.05,
        msd_agg_mode="trim10",
        stap_debug_samples=0,
        stap_device="cpu",
        dataset_name=dataset_name,
        dataset_suffix=dataset_suffix,
        score_mode=score_mode,
    )
    suffix = f"_{dataset_suffix}" if dataset_suffix else ""
    bundle_dir = root / f"{dataset_name}{suffix}"
    for npy in bundle_dir.glob("stap_neg*.npy"):
        arr = np.load(npy)
        if arr.size:
            hi = min(5000, arr.size)
            third = hi // 3 or 1
            arr[:third] = 0.5
            arr[third : 2 * third] = 0.4
            arr[2 * third : hi] = 0.35
            np.save(npy, arr)
    return bundle_dir


def test_acceptance_cli_bundle_pd(tmp_path):
    bundle_dir = _make_bundle(tmp_path)
    args = acceptance_cli.build_parser().parse_args(
        [
            "--bundle",
            str(bundle_dir),
            "--score-mode",
            "pd",
        ]
    )
    meta, resolved = acceptance_cli._apply_bundle_defaults(args)
    assert resolved == "pd"
    assert meta is not None
    pool_files = meta["score_pool_files"]["pd"]
    for key in ("base_pos", "base_neg", "stap_pos", "stap_neg"):
        path = Path(getattr(args, key))
        assert path.name == pool_files[key]
        assert path.exists()


def test_acceptance_cli_bundle_band_ratio(tmp_path):
    bundle_dir = _make_bundle(tmp_path / "band_ratio_case", score_mode="band_ratio")
    args = acceptance_cli.build_parser().parse_args(
        [
            "--bundle",
            str(bundle_dir),
            "--score-mode",
            "band_ratio",
        ]
    )
    meta, resolved = acceptance_cli._apply_bundle_defaults(args)
    assert resolved == "band_ratio"
    assert meta is not None
    assert meta["score_pool_default"] == "band_ratio"
    assert "band_ratio" in meta["score_pool_files"]
    ratio_file = meta["bundle_files"].get("stap_band_ratio_map")
    assert ratio_file is not None
    assert (bundle_dir / ratio_file).exists()


def test_slice_angle_data_window():
    dt = 1e-6
    base_rf = np.arange(20 * 3, dtype=np.float32).reshape(20, 3)
    angle_sets = [
        [
            AngleData(angle_deg=0.0, rf=base_rf, dt=dt),
            AngleData(angle_deg=5.0, rf=base_rf + 1.0, dt=dt),
        ]
    ]
    sliced = slice_angle_data(angle_sets, offset=4, length=8)
    assert len(sliced) == 1
    assert len(sliced[0]) == 2
    assert sliced[0][0].rf.shape == (8, 3)
    np.testing.assert_allclose(sliced[0][0].rf, base_rf[4:12])
    np.testing.assert_allclose(sliced[0][1].rf, (base_rf + 1.0)[4:12])


def test_slice_angle_data_window_invalid():
    dt = 1e-6
    rf = np.zeros((10, 2), dtype=np.float32)
    angle_sets = [[AngleData(angle_deg=0.0, rf=rf, dt=dt)]]
    with pytest.raises(ValueError):
        slice_angle_data(angle_sets, offset=8, length=5)


def test_write_acceptance_bundle_dataset_suffix(tmp_path):
    bundle_dir = _make_bundle(
        tmp_path / "suffix_case",
        dataset_name="basecase",
        dataset_suffix="win0",
    )
    assert bundle_dir.name == "basecase_win0"
    assert bundle_dir.exists()


def test_rank_consistency_metadata(tmp_path):
    bundle_dir = _make_bundle(tmp_path / "rank_case")
    meta_path = bundle_dir / "meta.json"
    assert meta_path.exists()
    meta = json.load(meta_path.open("r"))
    rc = meta.get("rank_consistency")
    assert rc is not None
    for cls in ("pos", "neg"):
        assert cls in rc
        entry = rc[cls]
        assert "concordance" in entry
        assert "kendall_tau" in entry
        prob = entry["concordance"]
        tau = entry["kendall_tau"]
        if prob is not None:
            assert tau is not None
            assert abs(tau - (2.0 * prob - 1.0)) < 1e-6

    rc_gated = meta.get("rank_consistency_gated")
    assert rc_gated is not None
    for cls in ("flow", "bg"):
        assert cls in rc_gated
        entry = rc_gated[cls]
        assert "pixels" in entry
        assert "concordance" in entry
        assert "kendall_tau" in entry

    score_transform = meta.get("score_transform_gated")
    assert score_transform is not None
    for cls in ("flow", "bg"):
        assert cls in score_transform
        entry = score_transform[cls]
        assert "pixels" in entry
        assert "p10_ratio" in entry
        assert "p90_ratio" in entry

    telemetry = meta.get("stap_fallback_telemetry", {})
    flow_align = telemetry.get("flow_band_alignment_stats")
    assert isinstance(flow_align, dict)
    assert "median" in flow_align
    assert "flow_band_alignment_stats_flow" in telemetry
    assert "flow_band_alignment_stats_bg" in telemetry
    flow_motion = telemetry.get("flow_motion_angle_stats")
    assert isinstance(flow_motion, dict)
    assert "median" in flow_motion
    assert (flow_motion.get("count") or 0) > 0
    assert "flow_motion_angle_stats_flow" in telemetry
    assert "flow_motion_angle_stats_bg" in telemetry
    depth_stats = telemetry.get("flow_motion_angle_stats_depth")
    assert isinstance(depth_stats, dict)
    psd_align = telemetry.get("psd_peak_alignment")
    assert isinstance(psd_align, dict)
    assert "flow_fraction_in_band" in psd_align
    assert "alias_fraction_in_band" in psd_align
    gate_feature_stats = telemetry.get("ka_gate_feature_stats")
    assert isinstance(gate_feature_stats, dict)
    assert "flow" in gate_feature_stats
    assert "bg" in gate_feature_stats
    assert telemetry.get("median_motion_half_span_rel_used") is not None
    assert telemetry.get("median_fd_motion_freqs") is not None
    assert telemetry.get("median_fd_flow_freqs_after_split") is not None
