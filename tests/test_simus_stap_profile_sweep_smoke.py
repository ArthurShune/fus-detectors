import json
import sys
from pathlib import Path

import numpy as np


def test_simus_stap_profile_sweep_smoke(tmp_path, monkeypatch):
    from scripts import simus_stap_profile_sweep as sweep

    run_dir = tmp_path / "run"
    ds = run_dir / "dataset"
    ds.mkdir(parents=True)
    np.save(ds / "icube.npy", np.zeros((8, 2, 2), dtype=np.complex64), allow_pickle=False)
    np.save(ds / "mask_h1_pf_main.npy", np.array([[1, 0], [0, 0]], dtype=bool), allow_pickle=False)
    np.save(ds / "mask_h0_bg.npy", np.array([[0, 1], [1, 0]], dtype=bool), allow_pickle=False)
    np.save(ds / "mask_h0_nuisance_pa.npy", np.array([[0, 0], [0, 1]], dtype=bool), allow_pickle=False)
    np.save(ds / "mask_h1_alias_qc.npy", np.array([[0, 0], [1, 0]], dtype=bool), allow_pickle=False)
    (ds / "meta.json").write_text(
        json.dumps(
            {
                "simus": {"profile": "ClinIntraOp-Pf-v1", "tier": "smoke"},
                "motion": {"telemetry": {"disp_rms_px": 1.0}},
                "phase_screen": {"telemetry": {"phase_rms_rad": 0.1}},
            }
        ),
        encoding="utf-8",
    )

    def fake_derive_bundle_from_run(**kwargs):
        out_root = Path(kwargs["out_root"])
        dataset_name = str(kwargs["dataset_name"])
        stap_profile = str(kwargs["stap_profile"])
        bundle_dir = out_root / dataset_name
        bundle_dir.mkdir(parents=True, exist_ok=True)
        np.save(bundle_dir / "score_base.npy", np.array([[10.0, 4.0], [3.0, 6.0]], dtype=np.float32), allow_pickle=False)
        peak = 2.5 if stap_profile.endswith("MotionWide-v0") else 2.0
        np.save(
            bundle_dir / "score_stap_preka.npy",
            np.array([[peak, 1.0], [1.2, 1.8]], dtype=np.float32),
            allow_pickle=False,
        )
        (bundle_dir / "meta.json").write_text("{}", encoding="utf-8")
        return bundle_dir

    monkeypatch.setattr(sweep, "derive_bundle_from_run", fake_derive_bundle_from_run)

    out_csv = tmp_path / "profile_sweep.csv"
    out_json = tmp_path / "profile_sweep.json"
    argv_prev = sys.argv[:]
    sys.argv = [
        "simus_stap_profile_sweep.py",
        "--case",
        f"demo::{run_dir}",
        "--stap-profiles",
        "Brain-SIMUS-Clin,Brain-SIMUS-Clin-MotionWide-v0",
        "--out-root",
        str(tmp_path / "bundles"),
        "--out-csv",
        str(out_csv),
        "--out-json",
        str(out_json),
        "--fprs",
        "1e-3",
        "--match-tprs",
        "0.5",
        "--no-reuse-bundles",
    ]
    try:
        sweep.main()
    finally:
        sys.argv = argv_prev

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    rows = payload["rows"]
    assert any(row["role"] == "baseline" for row in rows)
    assert any(row["stap_profile"] == "Brain-SIMUS-Clin" for row in rows)
    assert any(row["stap_profile"] == "Brain-SIMUS-Clin-MotionWide-v0" for row in rows)
    assert payload["details"]["cases"]["demo"]["best_auc_main_vs_nuisance"]["stap_profile"] in {
        "Brain-SIMUS-Clin",
        "Brain-SIMUS-Clin-MotionWide-v0",
    }
