import json
import sys
from pathlib import Path

import numpy as np


def test_simus_v2_acceptance_smoke(tmp_path, monkeypatch):
    from scripts import simus_v2_acceptance as mod

    anchor = {
        "rows": [
            {
                "anchor_kind": "shin",
                "flow_fpeak_q50": 100.0,
                "bg_fpeak_q50": 30.0,
                "flow_coh1_q50": 0.9,
                "bg_coh1_q50": 0.4,
                "flow_malias_q50": 0.1,
                "bg_malias_q50": -0.2,
                "svd_flow_cum_r1": 0.5,
                "svd_flow_cum_r2": 0.7,
                "svd_bg_cum_r1": 0.6,
                "svd_bg_cum_r2": 0.8,
                "reg_shift_rms": 0.1,
                "reg_shift_p90": 0.2,
                "reg_psr_median": 8.0,
            },
            {
                "anchor_kind": "shin",
                "flow_fpeak_q50": 110.0,
                "bg_fpeak_q50": 35.0,
                "flow_coh1_q50": 0.95,
                "bg_coh1_q50": 0.35,
                "flow_malias_q50": 0.2,
                "bg_malias_q50": -0.1,
                "svd_flow_cum_r1": 0.55,
                "svd_flow_cum_r2": 0.75,
                "svd_bg_cum_r1": 0.65,
                "svd_bg_cum_r2": 0.82,
                "reg_shift_rms": 0.2,
                "reg_shift_p90": 0.3,
                "reg_psr_median": 9.0,
            },
        ]
    }
    anchor_json = tmp_path / "anchor.json"
    anchor_json.write_text(json.dumps(anchor), encoding="utf-8")

    run_dir = tmp_path / "run"
    ds = run_dir / "dataset"
    ds.mkdir(parents=True)
    np.save(ds / "icube.npy", np.zeros((8, 4, 4), dtype=np.complex64), allow_pickle=False)
    np.save(ds / "mask_flow.npy", np.zeros((4, 4), dtype=bool), allow_pickle=False)
    np.save(ds / "mask_bg.npy", np.ones((4, 4), dtype=bool), allow_pickle=False)
    meta = {
        "acquisition": {"prf_hz": 1500.0},
        "bundle_policy_features": {
            "reg_shift_rms": 0.15,
            "reg_shift_p90": 0.25,
            "reg_psr_median": 8.5,
        },
    }
    (ds / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    fake_report = {
        "summary": {
            "flow": {
                "malias": {"q50": 0.15},
                "fpeak_hz": {"q50": 105.0},
                "coh1": {"q50": 0.92},
            },
            "bg": {
                "malias": {"q50": -0.15},
                "fpeak_hz": {"q50": 33.0},
                "coh1": {"q50": 0.38},
            },
        },
        "svd": {
            "flow": {"cum_r1": 0.53, "cum_r2": 0.72},
            "bg": {"cum_r1": 0.62, "cum_r2": 0.81},
        },
    }

    monkeypatch.setattr(mod, "summarize_icube", lambda **kwargs: fake_report)

    out_csv = tmp_path / "accept.csv"
    out_json = tmp_path / "accept.json"
    argv_prev = sys.argv[:]
    sys.argv = [
        "simus_v2_acceptance.py",
        "--run",
        str(run_dir),
        "--anchor-json",
        str(anchor_json),
        "--anchor-kind",
        "shin",
        "--out-csv",
        str(out_csv),
        "--out-json",
        str(out_json),
    ]
    try:
        mod.main()
    finally:
        sys.argv = argv_prev

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["runs"][0]["overall_pass"] is True


def test_simus_v2_acceptance_anchor_preset_smoke(tmp_path, monkeypatch):
    from scripts import simus_v2_acceptance as mod

    anchor = {
        "rows": [
            {
                "anchor_kind": "shin",
                "flow_fpeak_q50": 100.0,
                "bg_fpeak_q50": 30.0,
                "flow_coh1_q50": 0.9,
                "bg_coh1_q50": 0.4,
                "flow_malias_q50": 0.1,
                "bg_malias_q50": -0.2,
                "svd_flow_cum_r1": 0.5,
                "svd_flow_cum_r2": 0.7,
                "svd_bg_cum_r1": 0.6,
                "svd_bg_cum_r2": 0.8,
                "reg_shift_rms": 0.1,
                "reg_shift_p90": 0.2,
                "reg_psr_median": 8.0,
            },
            {
                "anchor_kind": "ulm_7883227",
                "flow_fpeak_q50": 100.0,
                "bg_fpeak_q50": 30.0,
                "flow_coh1_q50": 0.9,
                "bg_coh1_q50": 0.4,
                "flow_malias_q50": 0.1,
                "bg_malias_q50": -0.2,
                "svd_flow_cum_r1": 0.5,
                "svd_flow_cum_r2": 0.7,
                "svd_bg_cum_r1": 0.6,
                "svd_bg_cum_r2": 0.8,
                "reg_shift_rms": 0.1,
                "reg_shift_p90": 0.2,
                "reg_psr_median": 8.0,
            },
        ]
    }
    anchor_json = tmp_path / "anchor.json"
    anchor_json.write_text(json.dumps(anchor), encoding="utf-8")

    run_dir = tmp_path / "run"
    ds = run_dir / "dataset"
    ds.mkdir(parents=True)
    np.save(ds / "icube.npy", np.zeros((8, 4, 4), dtype=np.complex64), allow_pickle=False)
    np.save(ds / "mask_flow.npy", np.zeros((4, 4), dtype=bool), allow_pickle=False)
    np.save(ds / "mask_bg.npy", np.ones((4, 4), dtype=bool), allow_pickle=False)
    meta = {
        "acquisition": {"prf_hz": 1500.0},
        "bundle_policy_features": {
            "reg_shift_rms": 0.1,
            "reg_shift_p90": 0.2,
            "reg_psr_median": 8.0,
        },
    }
    (ds / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    fake_report = {
        "summary": {
            "flow": {
                "malias": {"q50": 0.1},
                "fpeak_hz": {"q50": 100.0},
                "coh1": {"q50": 0.9},
            },
            "bg": {
                "malias": {"q50": -0.2},
                "fpeak_hz": {"q50": 30.0},
                "coh1": {"q50": 0.4},
            },
        },
        "svd": {
            "flow": {"cum_r1": 0.5, "cum_r2": 0.7},
            "bg": {"cum_r1": 0.6, "cum_r2": 0.8},
        },
    }
    monkeypatch.setattr(mod, "summarize_icube", lambda **kwargs: fake_report)

    out_csv = tmp_path / "accept.csv"
    out_json = tmp_path / "accept.json"
    argv_prev = sys.argv[:]
    sys.argv = [
        "simus_v2_acceptance.py",
        "--run",
        str(run_dir),
        "--anchor-json",
        str(anchor_json),
        "--anchor-preset",
        "intraop_brainlike",
        "--out-csv",
        str(out_csv),
        "--out-json",
        str(out_json),
    ]
    try:
        mod.main()
    finally:
        sys.argv = argv_prev

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["anchor_preset"] == "intraop_brainlike"
    assert payload["anchor_kinds"] == ["shin", "ulm_7883227"]


def test_simus_v2_acceptance_profile_gate_smoke(tmp_path, monkeypatch):
    from scripts import simus_v2_acceptance as mod

    anchor = {
        "rows": [
            {
                "anchor_kind": "shin",
                "flow_fpeak_q50": 12.0,
                "bg_fpeak_q50": 24.0,
                "flow_coh1_q50": 0.99,
                "bg_coh1_q50": 0.50,
                "flow_malias_q50": -7.3,
                "bg_malias_q50": -1.5,
                "svd_flow_cum_r1": 0.97,
                "svd_flow_cum_r2": 0.99,
                "svd_bg_cum_r1": 0.30,
                "svd_bg_cum_r2": 0.40,
                "reg_shift_rms": 0.004,
                "reg_shift_p90": 0.006,
                "reg_psr_median": 250.0,
            },
            {
                "anchor_kind": "ulm_7883227",
                "flow_fpeak_q50": 10.0,
                "bg_fpeak_q50": 22.0,
                "flow_coh1_q50": 0.985,
                "bg_coh1_q50": 0.55,
                "flow_malias_q50": -7.2,
                "bg_malias_q50": -1.6,
                "svd_flow_cum_r1": 0.96,
                "svd_flow_cum_r2": 0.985,
                "svd_bg_cum_r1": 0.28,
                "svd_bg_cum_r2": 0.38,
                "reg_shift_rms": 0.005,
                "reg_shift_p90": 0.007,
                "reg_psr_median": 260.0,
            },
            {
                "anchor_kind": "gammex_along",
                "flow_fpeak_q50": 300.0,
                "bg_fpeak_q50": 300.0,
                "flow_coh1_q50": 0.4,
                "bg_coh1_q50": 0.4,
                "flow_malias_q50": 2.0,
                "bg_malias_q50": -0.5,
                "svd_flow_cum_r1": 0.8,
                "svd_flow_cum_r2": 0.9,
                "svd_bg_cum_r1": 0.4,
                "svd_bg_cum_r2": 0.6,
                "reg_shift_rms": 1.0,
                "reg_shift_p90": 1.3,
                "reg_psr_median": 20.0,
            },
            {
                "anchor_kind": "gammex_across",
                "flow_fpeak_q50": 440.0,
                "bg_fpeak_q50": 440.0,
                "flow_coh1_q50": 0.45,
                "bg_coh1_q50": 0.45,
                "flow_malias_q50": 2.5,
                "bg_malias_q50": -0.4,
                "svd_flow_cum_r1": 0.8,
                "svd_flow_cum_r2": 0.9,
                "svd_bg_cum_r1": 0.4,
                "svd_bg_cum_r2": 0.6,
                "reg_shift_rms": 0.5,
                "reg_shift_p90": 0.8,
                "reg_psr_median": 15.0,
            },
        ]
    }
    anchor_json = tmp_path / "anchor.json"
    anchor_json.write_text(json.dumps(anchor), encoding="utf-8")

    run_dir = tmp_path / "run"
    ds = run_dir / "dataset"
    ds.mkdir(parents=True)
    np.save(ds / "icube.npy", np.zeros((8, 4, 4), dtype=np.complex64), allow_pickle=False)
    np.save(ds / "mask_flow.npy", np.zeros((4, 4), dtype=bool), allow_pickle=False)
    np.save(ds / "mask_bg.npy", np.ones((4, 4), dtype=bool), allow_pickle=False)
    meta = {
        "acquisition": {"prf_hz": 1500.0},
        "bundle_policy_features": {
            "reg_shift_rms": 0.1,
            "reg_shift_p90": 0.2,
            "reg_psr_median": 8.0,
        },
        "scene": {
            "expected_fd_sampled_q50_hz": 120.0,
            "h1_alias_qc_fraction": 0.05,
            "h0_nuisance_fraction": 0.03,
        },
    }
    (ds / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    fake_report = {
        "summary": {
            "flow": {
                "malias": {"q50": -2.0},
                "fpeak_hz": {"q50": 80.0},
                "coh1": {"q50": 0.7},
            },
            "bg": {
                "malias": {"q50": -0.45},
                "fpeak_hz": {"q50": 23.0},
                "coh1": {"q50": 0.52},
            },
        },
        "svd": {
            "flow": {"cum_r1": 0.7, "cum_r2": 0.8},
            "bg": {"cum_r1": 0.29, "cum_r2": 0.39},
        },
    }
    monkeypatch.setattr(mod, "summarize_icube", lambda **kwargs: fake_report)

    out_csv = tmp_path / "accept.csv"
    out_json = tmp_path / "accept.json"
    argv_prev = sys.argv[:]
    sys.argv = [
        "simus_v2_acceptance.py",
        "--run",
        str(run_dir),
        "--anchor-json",
        str(anchor_json),
        "--profile-gate",
        "ClinIntraOp-Pf-v2",
        "--out-csv",
        str(out_csv),
        "--out-json",
        str(out_json),
    ]
    try:
        mod.main()
    finally:
        sys.argv = argv_prev

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["profile_gate"] == "ClinIntraOp-Pf-v2"
    assert payload["runs"][0]["overall_pass"] is True
