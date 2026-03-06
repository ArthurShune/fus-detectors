import json
import sys
from pathlib import Path

import numpy as np


def test_simus_fusion_readout_bench_emits_expected_candidates(tmp_path):
    from scripts import simus_fusion_readout_bench as bench

    run_dir = tmp_path / "run"
    ds = run_dir / "dataset"
    ds.mkdir(parents=True)
    np.save(ds / "icube.npy", np.zeros((8, 2, 2), dtype=np.complex64), allow_pickle=False)
    np.save(ds / "mask_h1_pf_main.npy", np.array([[1, 0], [0, 0]], dtype=bool), allow_pickle=False)
    np.save(ds / "mask_h0_bg.npy", np.array([[0, 1], [1, 0]], dtype=bool), allow_pickle=False)
    np.save(ds / "mask_h0_nuisance_pa.npy", np.array([[0, 0], [0, 1]], dtype=bool), allow_pickle=False)
    np.save(ds / "mask_h1_alias_qc.npy", np.array([[0, 0], [1, 0]], dtype=bool), allow_pickle=False)
    (ds / "meta.json").write_text(
        json.dumps({"simus": {"profile": "TestProfile", "tier": "smoke"}}), encoding="utf-8"
    )

    base_bundle = tmp_path / "base_bundle"
    stap_bundle = tmp_path / "stap_bundle"
    base_bundle.mkdir()
    stap_bundle.mkdir()
    np.save(base_bundle / "score_base.npy", np.array([[10.0, 4.0], [3.0, 6.0]], dtype=np.float32), allow_pickle=False)
    np.save(stap_bundle / "score_stap_preka.npy", np.array([[2.0, 1.0], [1.2, 1.8]], dtype=np.float32), allow_pickle=False)

    out_csv = tmp_path / "fusion.csv"
    out_json = tmp_path / "fusion.json"
    argv_prev = sys.argv[:]
    sys.argv = [
        "simus_fusion_readout_bench.py",
        "--case",
        f"demo::{run_dir}::{base_bundle}::{stap_bundle}",
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
        bench.main()
    finally:
        sys.argv = argv_prev

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    names = {row["candidate"] for row in payload["rows"]}
    assert "baseline_score" in names
    assert "stap_detector" in names
    assert "fusion_tail_fisher" in names
    assert "fusion_base_x_rankstap" in names
    assert payload["details"]["cases"]["demo"]["best_auc_main_vs_nuisance"]["candidate"] in names
