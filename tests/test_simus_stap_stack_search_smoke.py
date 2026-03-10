from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def test_simus_stap_stack_search_writes_valid_json(tmp_path, monkeypatch):
    from scripts import simus_stap_stack_search as search

    def fake_load_search_payload(_path):
        return {
            "details": {
                "dev_cases": [{"name": "dev", "run_dir": str(tmp_path / "dev_run")}],
                "eval_cases": [{"name": "eval", "run_dir": str(tmp_path / "eval_run")}],
            }
        }

    def fake_selected_residual_specs(_payload):
        return [
            search.ResidualSpec(
                method_family="mc_svd",
                config_name="rank6",
                baseline_type="mc_svd",
                override_builder_name="build_rank6",
            ),
            search.ResidualSpec(
                method_family="rpca",
                config_name="lam1",
                baseline_type="rpca",
                override_builder_name="build_lam1",
            ),
        ]

    def fake_load_canonical_run(_run_dir):
        icube = np.zeros((8, 3, 3), dtype=np.complex64)
        masks = {
            "mask_h1_pf_main": np.array(
                [[1, 1, 0], [0, 0, 0], [0, 0, 0]],
                dtype=bool,
            ),
            "mask_h0_bg": np.array(
                [[0, 0, 1], [1, 1, 1], [0, 1, 0]],
                dtype=bool,
            ),
            "mask_h0_nuisance_pa": np.array(
                [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
                dtype=bool,
            ),
            "mask_h1_alias_qc": np.array(
                [[0, 0, 0], [0, 0, 0], [1, 0, 0]],
                dtype=bool,
            ),
        }
        meta = {"acquisition": {"prf_hz": 1500.0}}
        return icube, masks, meta

    def fake_derive_bundle_from_run(**kwargs):
        bundle_dir = Path(kwargs["out_root"]) / str(kwargs["dataset_name"])
        bundle_dir.mkdir(parents=True, exist_ok=True)
        residual_type = str(kwargs["baseline_type"])
        stap_profile = str(kwargs["stap_profile"])
        if residual_type == "rpca" and stap_profile.endswith("MotionRobust-v0"):
            score = np.array(
                [[5.0, 4.8, 1.0], [0.9, 1.0, 0.8], [4.6, 0.9, 0.1]],
                dtype=np.float32,
            )
        elif residual_type == "mc_svd" and stap_profile.endswith("MotionRobust-v0"):
            score = np.array(
                [[4.0, 3.8, 1.2], [1.1, 1.2, 1.0], [3.6, 1.1, 0.3]],
                dtype=np.float32,
            )
        else:
            score = np.array(
                [[2.5, 2.2, 1.3], [1.2, 1.3, 1.1], [2.0, 1.2, 0.6]],
                dtype=np.float32,
            )
        np.save(bundle_dir / "score_stap_preka.npy", score, allow_pickle=False)
        (bundle_dir / "meta.json").write_text("{}", encoding="utf-8")
        return bundle_dir

    monkeypatch.setattr(search, "_load_search_payload", fake_load_search_payload)
    monkeypatch.setattr(search, "_selected_residual_specs", fake_selected_residual_specs)
    monkeypatch.setattr(
        search,
        "_override_builder_lookup",
        lambda: {
            ("mc_svd", "rank6"): (lambda _shape: {"svd_rank": 6}),
            ("rpca", "lam1"): (lambda _shape: {"rpca_lambda": 0.01}),
        },
    )
    monkeypatch.setattr(search, "load_canonical_run", fake_load_canonical_run)
    monkeypatch.setattr(search, "derive_bundle_from_run", fake_derive_bundle_from_run)

    out_csv = tmp_path / "stack.csv"
    out_json = tmp_path / "stack.json"
    argv_prev = sys.argv[:]
    sys.argv = [
        "simus_stap_stack_search.py",
        "--search-json",
        str(tmp_path / "dummy.json"),
        "--stap-profiles",
        "Brain-SIMUS-Clin,Brain-SIMUS-Clin-MotionRobust-v0",
        "--out-root",
        str(tmp_path / "out"),
        "--out-csv",
        str(out_csv),
        "--out-json",
        str(out_json),
        "--stap-device",
        "cpu",
        "--no-reuse-bundles",
    ]
    try:
        search.main()
    finally:
        sys.argv = argv_prev

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    best_dev = max(payload["dev_summary"], key=lambda r: float(r["selection_score"]))
    assert payload["selected_stack"]["residual_family"] == best_dev["residual_family"]
    assert payload["selected_stack"]["stap_profile"] == best_dev["stap_profile"]
    assert payload["selected_stack_key"] == {
        "residual_family": best_dev["residual_family"],
        "stap_profile": best_dev["stap_profile"],
    }
    assert len(payload["eval_summary"]) == 1
    assert payload["eval_summary"][0]["residual_family"] == best_dev["residual_family"]
