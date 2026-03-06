import json
import sys
from pathlib import Path

import numpy as np


def test_simus_stap_compromise_search_smoke(tmp_path, monkeypatch):
    from scripts import simus_stap_compromise_search as search

    def fake_write_simus_run(*, out_root, cfg, skip_bundle):
        run_dir = Path(out_root)
        ds = run_dir / "dataset"
        ds.mkdir(parents=True, exist_ok=True)
        np.save(ds / "icube.npy", np.zeros((8, 2, 2), dtype=np.complex64), allow_pickle=False)
        np.save(ds / "mask_h1_pf_main.npy", np.array([[1, 0], [0, 0]], dtype=bool), allow_pickle=False)
        np.save(ds / "mask_h0_bg.npy", np.array([[0, 1], [1, 0]], dtype=bool), allow_pickle=False)
        np.save(ds / "mask_h0_nuisance_pa.npy", np.array([[0, 0], [0, 1]], dtype=bool), allow_pickle=False)
        np.save(ds / "mask_h1_alias_qc.npy", np.array([[0, 0], [1, 0]], dtype=bool), allow_pickle=False)
        (ds / "meta.json").write_text(
            json.dumps(
                {
                    "simus": {"profile": str(cfg.profile), "tier": str(cfg.tier)},
                    "motion": {"telemetry": {"disp_rms_px": 1.25}},
                    "phase_screen": {"telemetry": {"phase_rms_rad": 0.15}},
                }
            ),
            encoding="utf-8",
        )
        return {"dataset_dir": ds}

    def fake_derive_bundle_from_run(**kwargs):
        out_root = Path(kwargs["out_root"])
        dataset_name = str(kwargs["dataset_name"])
        stap_profile = str(kwargs["stap_profile"])
        bundle_dir = out_root / dataset_name
        bundle_dir.mkdir(parents=True, exist_ok=True)
        np.save(bundle_dir / "score_base.npy", np.array([[10.0, 4.0], [3.0, 6.0]], dtype=np.float32), allow_pickle=False)
        peak = {
            "Brain-SIMUS-Clin": 1.8,
            "Brain-SIMUS-Clin-MotionShort-v0": 2.3,
            "Brain-SIMUS-Clin-MotionMid-v0": 2.1,
        }.get(stap_profile, 1.7)
        np.save(bundle_dir / "score_stap_preka.npy", np.array([[peak, 1.0], [1.2, 1.4]], dtype=np.float32), allow_pickle=False)
        meta = {
            "stap_fallback_telemetry": {
                "reg_shift_rms": 1.25,
                "reg_shift_p90": 1.8,
                "reg_psr_median": 14.0,
                "flow_cov_ge_50_fraction": 0.4,
            }
        }
        (bundle_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return bundle_dir

    monkeypatch.setattr(search, "write_simus_run", fake_write_simus_run)
    monkeypatch.setattr(search, "derive_bundle_from_run", fake_derive_bundle_from_run)

    out_csv = tmp_path / "compromise.csv"
    out_json = tmp_path / "compromise.json"
    argv_prev = sys.argv[:]
    sys.argv = [
        "simus_stap_compromise_search.py",
        "--simus-profiles",
        "ClinIntraOp-Pf-v1",
        "--seeds",
        "1",
        "--motion-scales",
        "0.25",
        "--stap-profiles",
        "Brain-SIMUS-Clin,Brain-SIMUS-Clin-MotionShort-v0,Brain-SIMUS-Clin-MotionMid-v0",
        "--out-root",
        str(tmp_path / "out"),
        "--out-csv",
        str(out_csv),
        "--out-json",
        str(out_json),
        "--no-reuse-existing",
        "--no-reuse-bundles",
    ]
    try:
        search.main()
    finally:
        sys.argv = argv_prev

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["details"]["summary_by_profile"]["Brain-SIMUS-Clin-MotionShort-v0"]["count"] == 1
    assert any(row["role"] == "baseline" for row in payload["rows"])
    assert any(row["stap_profile"] == "Brain-SIMUS-Clin-MotionMid-v0" for row in payload["rows"])
