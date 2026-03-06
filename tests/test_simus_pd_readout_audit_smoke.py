import json
import sys
from pathlib import Path

import numpy as np
import pytest


def test_simus_pd_readout_audit_reports_background_identity(tmp_path):
    from scripts import simus_pd_readout_audit as audit

    run_dir = tmp_path / "sim_run"
    ds = run_dir / "dataset"
    ds.mkdir(parents=True)

    icube = np.zeros((8, 2, 2), dtype=np.complex64)
    np.save(ds / "icube.npy", icube, allow_pickle=False)
    np.save(ds / "mask_flow.npy", np.array([[1, 0], [0, 1]], dtype=bool), allow_pickle=False)
    np.save(ds / "mask_bg.npy", np.array([[0, 1], [0, 0]], dtype=bool), allow_pickle=False)
    np.save(ds / "mask_h1_pf_main.npy", np.array([[1, 0], [0, 0]], dtype=bool), allow_pickle=False)
    np.save(ds / "mask_h0_bg.npy", np.array([[0, 1], [0, 0]], dtype=bool), allow_pickle=False)
    np.save(ds / "mask_h0_nuisance_pa.npy", np.array([[0, 0], [0, 1]], dtype=bool), allow_pickle=False)
    np.save(ds / "mask_h1_alias_qc.npy", np.array([[0, 0], [1, 0]], dtype=bool), allow_pickle=False)
    meta = {
        "simus": {"profile": "ClinIntraOp-Pf-v1", "tier": "smoke"},
        "motion": {"telemetry": {"disp_rms_px": 1.0}},
        "phase_screen": {"telemetry": {"phase_rms_rad": 0.2}},
    }
    (ds / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    pd_base = np.array([[10.0, 5.0], [8.0, 7.0]], dtype=np.float32)
    pd_stap = np.array([[6.0, 5.0], [8.0, 4.0]], dtype=np.float32)
    score = np.array([[1.8, 1.0], [1.1, 1.2]], dtype=np.float32)
    np.save(bundle_dir / "pd_base.npy", pd_base, allow_pickle=False)
    np.save(bundle_dir / "pd_stap.npy", pd_stap, allow_pickle=False)
    np.save(bundle_dir / "score_stap_preka.npy", score, allow_pickle=False)
    (bundle_dir / "meta.json").write_text(
        json.dumps({"source_run": str(run_dir)}), encoding="utf-8"
    )

    out_csv = tmp_path / "audit.csv"
    out_json = tmp_path / "audit.json"
    argv_prev = sys.argv[:]
    sys.argv = [
        "simus_pd_readout_audit.py",
        "--bundle",
        str(bundle_dir),
        "--out-csv",
        str(out_csv),
        "--out-json",
        str(out_json),
        "--fprs",
        "1e-3",
        "--match-tprs",
        "0.5",
    ]
    try:
        audit.main()
    finally:
        sys.argv = argv_prev

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    bundle_key = next(iter(payload["details"]["bundles"].keys()))
    detail = payload["details"]["bundles"][bundle_key]
    assert detail["bg_identity_fraction_h0_bg"] == 1.0
    assert detail["band_fraction_eq1_fraction_h0_bg"] == 1.0
    candidates = {row["candidate"]: row for row in payload["rows"]}
    assert candidates["pd_stap"]["bg_median"] == pytest.approx(
        candidates["pd_stap"]["h1_median"] / 1.2
    )
    assert detail["corr_pd_stap_vs_score"] is not None
