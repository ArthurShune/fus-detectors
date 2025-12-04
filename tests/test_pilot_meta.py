import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def _run_pilot(script: str, out_dir: Path, extra_args: list[str]) -> Path:
    cmd = [
        sys.executable,
        script,
        "--out",
        str(out_dir),
        "--Nx",
        "32",
        "--Ny",
        "32",
        "--pulses",
        "12",
        "--angles",
        "0",
        "--prf",
        "2500",
        "--tile-h",
        "8",
        "--tile-w",
        "8",
        "--tile-stride",
        "4",
        "--diag-load",
        "1e-2",
        "--lt",
        "4",
        "--grid-step-rel",
        "0.12",
        "--max-pts",
        "5",
        "--stap-debug-samples",
        "2",
        "--synthetic",
        "--force-cpu",
        "--stap-device",
        "cpu",
    ] + extra_args
    subprocess.check_call(cmd)
    bundle = next(out_dir.glob("pw_*"))
    return bundle


def _median(masked: np.ndarray, mask: np.ndarray) -> float:
    vals = masked[mask]
    return float(np.median(vals)) if vals.size else 0.0


def test_pilot_meta_contains_expected_stats(tmp_path):
    out_dir = tmp_path / "r1"
    out_dir.mkdir()
    bundle = _run_pilot(
        "sim/kwave/pilot_r1.py",
        out_dir,
        [
            "--msd-lambda",
            "5e-2",
            "--msd-ridge",
            "0.10",
            "--constraint-ridge",
            "0.12",
        ],
    )

    meta_path = bundle / "meta.json"
    meta = json.loads(meta_path.read_text())
    assert meta["stap_device"] == "cpu"

    pd_base = np.load(bundle / "pd_base.npy")
    pd_stap = np.load(bundle / "pd_stap.npy")
    mask_flow = np.load(bundle / "mask_flow.npy")
    mask_bg = np.load(bundle / "mask_bg.npy")

    assert np.isclose(
        _median(pd_base, mask_flow),
        meta["pd_stats"]["baseline_flow_median"],
        rtol=0.1,
        atol=1e-6,
    )
    assert np.isclose(
        _median(pd_stap, mask_bg),
        meta["pd_stats"]["stap_bg_median"],
        rtol=0.1,
        atol=1e-6,
    )

    score_map = np.load(bundle / "stap_score_map.npy")
    assert score_map.shape == pd_base.shape

    debug_dir = bundle / "stap_debug"
    assert debug_dir.exists()
    sample = next(debug_dir.glob("*.npz"))
    data = np.load(sample, allow_pickle=False)
    assert "score_tile" in data and "band_fraction_tile" in data
    assert "band_fraction_quantiles" in data and "score_quantiles" in data
    tele = meta["stap_fallback_telemetry"]
    assert "median_band_fraction_q50" in tele
    assert "median_score_q50" in tele
    coverage_keys = [
        "tile_flow_coverage_p50",
        "tile_flow_coverage_p90",
        "flow_cov_ge_20_fraction",
        "flow_cov_ge_50_fraction",
        "flow_cov_ge_80_fraction",
    ]
    for key in coverage_keys:
        assert key in tele
    assert meta["msd_config"]["agg"] in {"trim10", "median", "mean"}
    assert "ratio_rho" in meta["msd_config"]

    # Telemetry guardrails reflecting projector/PD behavior
    tele = meta["stap_fallback_telemetry"]
    Lt = meta["Lt"]
    if tele.get("median_band_Kc") is not None:
        assert tele["median_band_Kc"] <= Lt
    if tele.get("median_band_fraction_q50") is not None:
        assert tele["median_band_fraction_q50"] < 0.95
    if tele.get("median_grid_step_hz") is not None:
        step_exp = 0.12 * (meta["prf_hz"] / float(Lt))
        assert abs(tele["median_grid_step_hz"] - step_exp) <= 0.3 * step_exp
    assert tele.get("stap_device") == "cpu"

    from eval.acceptance_cli import build_parser, run

    run_out = tmp_path / "accept_runs"
    report_dir = tmp_path / "accept_reports"
    figs_dir = tmp_path / "accept_figs"
    run_out.mkdir()
    report_dir.mkdir()
    figs_dir.mkdir()

    args = build_parser().parse_args(
        [
            "--base_pos",
            str(bundle / "base_pos.npy"),
            "--base_neg",
            str(bundle / "base_neg.npy"),
            "--stap_pos",
            str(bundle / "stap_pos.npy"),
            "--stap_neg",
            str(bundle / "stap_neg.npy"),
            "--base_pd",
            str(bundle / "pd_base.npy"),
            "--stap_pd",
            str(bundle / "pd_stap.npy"),
            "--mask_flow",
            str(bundle / "mask_flow.npy"),
            "--mask_bg",
            str(bundle / "mask_bg.npy"),
            "--out_dir",
            str(run_out),
            "--report_dir",
            str(report_dir),
            "--fig_dir",
            str(figs_dir),
            "--delta_snr_min",
            "0.0",
            "--delta_tpr_min",
            "0.0",
            "--fpr_target",
            "1e-4",
            "--alpha",
            "1e-4",
            "--evd-mode",
            "gpd",
        ]
    )
    with pytest.raises(ValueError, match="Too few exceedances"):
        run(args)


def test_pilot_debug_records_fallback_fields(tmp_path):
    out_dir = tmp_path / "r2"
    out_dir.mkdir()
    bundle = _run_pilot(
        "sim/kwave/pilot_motion.py",
        out_dir,
        [
            "--ensembles",
            "2",
            "--pulses",
            "16",
            "--msd-lambda",
            "6e-2",
            "--msd-ridge",
            "0.15",
            "--constraint-ridge",
            "0.15",
            "--max-pts",
            "5",
        ],
    )
    meta = json.loads((bundle / "meta.json").read_text())
    telemetry = meta["stap_fallback_telemetry"]
    assert telemetry["score_mode_histogram"]
    debug_dir = bundle / "stap_debug"
    sample = np.load(next(debug_dir.glob("*.npz")), allow_pickle=False)
    assert "score_tile" in sample and "diag_load" in sample
