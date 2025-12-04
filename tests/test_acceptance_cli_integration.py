import json
import pytest
from pathlib import Path

import numpy as np

from eval.acceptance_cli import build_parser, run


def _write(path: Path, arr: np.ndarray) -> str:
    np.save(path, arr, allow_pickle=False)
    return str(path)


def _make_masks(shape=(32, 32)):
    H, W = shape
    mask_flow = np.zeros((H, W), dtype=bool)
    y0 = max(H // 2 - 2, 0)
    y1 = min(H // 2 + 2, H)
    x0 = max(W // 2 - 2, 0)
    x1 = min(W // 2 + 2, W)
    mask_flow[y0:y1, x0:x1] = True
    mask_bg = ~mask_flow
    return mask_flow, mask_bg


def _find_latest_json(out_dir: Path) -> Path:
    return max(out_dir.glob("acceptance_*.json"), key=lambda p: p.stat().st_mtime)


def test_acceptance_positive_improvement(tmp_path):
    rng = np.random.default_rng(0)
    base_pos = rng.exponential(scale=1.0, size=6000).astype(np.float32)
    base_neg = rng.exponential(scale=1.0, size=60000).astype(np.float32)
    stap_pos = base_pos + 1.8
    stap_neg = base_neg + 0.4
    pd_base = rng.random((32, 32)).astype(np.float32)
    mask_flow, mask_bg = _make_masks()
    band_frac = (
        0.35 + 0.55 * mask_flow.astype(np.float32) + 0.05 * rng.random((32, 32)).astype(np.float32)
    )
    band_frac = np.clip(band_frac, 0.1, 0.95).astype(np.float32)
    pd_stap = pd_base * band_frac

    data_dir = tmp_path / "data"
    report_dir = tmp_path / "reports"
    figs_dir = tmp_path / "figs"
    data_dir.mkdir()
    report_dir.mkdir()
    figs_dir.mkdir()

    args = build_parser().parse_args(
        [
            "--base_pos",
            _write(data_dir / "base_pos.npy", base_pos),
            "--base_neg",
            _write(data_dir / "base_neg.npy", base_neg),
            "--stap_pos",
            _write(data_dir / "stap_pos.npy", stap_pos),
            "--stap_neg",
            _write(data_dir / "stap_neg.npy", stap_neg),
            "--base_pd",
            _write(data_dir / "pd_base.npy", pd_base),
            "--stap_pd",
            _write(data_dir / "pd_stap.npy", pd_stap),
            "--mask_flow",
            _write(data_dir / "mask_flow.npy", mask_flow),
            "--mask_bg",
            _write(data_dir / "mask_bg.npy", mask_bg),
            "--out_dir",
            str(tmp_path / "runs"),
            "--report_dir",
            str(report_dir),
            "--fig_dir",
            str(figs_dir),
            "--delta_snr_min",
            "0.0",
            "--delta_tpr_min",
            "0.0",
            "--fpr_target",
            "1e-3",
            "--alpha",
            "1e-3",
            "--evd-mode",
            "gpd",
        ]
    )
    run(args)
    payload = json.loads(_find_latest_json(tmp_path / "runs").read_text())
    assert payload["overall_pass"] is True
    assert payload["performance"]["tpr_at_fpr_delta"] > 0.0
    assert payload["performance"]["pd_snr_delta_db"] > -1.0  # raw PD may shrink but bounded
    assert payload["performance"]["pd_snr_band_delta_db"] >= -1e-6
    assert payload["gates"]["gate_delta_pd_snr_band"] is True
    assert payload["evt_diagnostics"]["status"] == "ok"
    assert payload["gates"]["gate_evt_diagnostics"] is True
    roc_summary = payload["roc_summary"]
    assert roc_summary["target_resolvable"] is True
    assert roc_summary["fpr_min"] == pytest.approx(1.0 / len(stap_neg))
    assert "null_quantiles" in roc_summary


def test_acceptance_negative_identical_scores(tmp_path):
    rng = np.random.default_rng(1)
    base_pos = rng.normal(size=6000).astype(np.float32)
    base_neg = rng.normal(size=6000).astype(np.float32)
    pd_map = rng.random((16, 16)).astype(np.float32)
    mask_flow, mask_bg = _make_masks((16, 16))

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    args = build_parser().parse_args(
        [
            "--base_pos",
            _write(data_dir / "base_pos.npy", base_pos),
            "--base_neg",
            _write(data_dir / "base_neg.npy", base_neg),
            "--stap_pos",
            _write(data_dir / "stap_pos.npy", base_pos.copy()),
            "--stap_neg",
            _write(data_dir / "stap_neg.npy", base_neg.copy()),
            "--base_pd",
            _write(data_dir / "pd_base.npy", pd_map),
            "--stap_pd",
            _write(data_dir / "pd_stap.npy", pd_map.copy()),
            "--mask_flow",
            _write(data_dir / "mask_flow.npy", mask_flow),
            "--mask_bg",
            _write(data_dir / "mask_bg.npy", mask_bg),
            "--out_dir",
            str(tmp_path / "runs"),
            "--report_dir",
            str(tmp_path / "reports"),
            "--fig_dir",
            str(tmp_path / "figs"),
            "--fpr_target",
            "1e-4",
            "--alpha",
            "1e-3",
            "--evd-mode",
            "gpd",
        ]
    )
    run(args)
    payload = json.loads(_find_latest_json(tmp_path / "runs").read_text())
    assert payload["overall_pass"] is False
    assert payload["evt_diagnostics"]["status"] in {
        "ok",
        "fallback_r2",
        "fallback_count",
        "fallback_default",
    }
    assert "gate_evt_diagnostics" in payload["gates"]
    roc_summary = payload["roc_summary"]
    assert roc_summary["target_resolvable"] is False
