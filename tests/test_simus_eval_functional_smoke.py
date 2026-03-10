from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def test_simus_eval_functional_writes_valid_json(tmp_path, monkeypatch):
    from scripts import simus_eval_functional as functional_eval

    def fake_write_functional_case(*, out_root, base_profile, tier, seed, null_run, design_spec, max_workers, threads_per_worker, reuse_existing):
        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        E = int(design_spec.ensemble_count)
        reg = np.linspace(0.0, 1.0, E, dtype=np.float32)
        np.save(out_root / "task_regressor.npy", (reg > 0.5).astype(np.float32), allow_pickle=False)
        np.save(out_root / "hemo_regressor.npy", reg, allow_pickle=False)
        mask_act = np.zeros((3, 3), dtype=bool)
        mask_act[0, 0] = True
        mask_act[0, 1] = True
        mask_bg = np.ones((3, 3), dtype=bool)
        mask_bg[0, 0] = False
        mask_bg[0, 1] = False
        mask_nuis = np.zeros((3, 3), dtype=bool)
        mask_nuis[2, 2] = True
        mask_spec = np.zeros((3, 3), dtype=bool)
        np.save(out_root / "mask_activation_roi.npy", mask_act, allow_pickle=False)
        np.save(out_root / "mask_h0_bg.npy", mask_bg, allow_pickle=False)
        np.save(out_root / "mask_h0_nuisance_pa.npy", mask_nuis, allow_pickle=False)
        np.save(out_root / "mask_h0_specular_struct.npy", mask_spec, allow_pickle=False)
        rows = []
        for idx in range(E):
            ensemble_dir = out_root / f"ensemble_{idx:03d}"
            dataset_dir = ensemble_dir / "dataset"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            np.save(dataset_dir / "icube.npy", np.zeros((8, 3, 3), dtype=np.complex64), allow_pickle=False)
            (dataset_dir / "meta.json").write_text("{}", encoding="utf-8")
            rows.append(
                {
                    "ensemble_index": idx,
                    "ensemble_dir": str(ensemble_dir),
                    "dataset_dir": str(dataset_dir),
                    "scene_seed": seed,
                    "realization_seed": seed * 1000 + idx,
                    "activation_gain": 0.0 if bool(null_run) else float(reg[idx]),
                }
            )
        (out_root / "design.json").write_text(json.dumps({"base_profile": base_profile, "null_run": bool(null_run)}), encoding="utf-8")
        (out_root / "ensemble_table.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")
        (out_root / "ensemble_table.csv").write_text("ensemble_index\n0\n", encoding="utf-8")
        return {"case_root": out_root, "rows": rows}

    def fake_derive_bundle_from_run(**kwargs):
        bundle_dir = Path(kwargs["out_root"]) / str(kwargs["dataset_name"])
        bundle_dir.mkdir(parents=True, exist_ok=True)
        run_dir = Path(kwargs["run_dir"])
        idx = int(run_dir.name.split("_")[-1])
        amp = float(idx + 1)
        base = np.array([[amp, amp * 0.9, 0.3], [0.3, 0.2, 0.2], [0.2, 0.2, 0.05]], dtype=np.float32)
        if kwargs["baseline_type"] == "rpca":
            base += 0.3
        if kwargs["run_stap"]:
            stap = np.array([[amp + 1.5, amp + 1.3, 0.2], [0.2, 0.2, 0.15], [0.15, 0.15, 0.02]], dtype=np.float32)
            np.save(bundle_dir / "score_stap_preka.npy", stap, allow_pickle=False)
            np.save(bundle_dir / "score_pd_stap.npy", stap * 0.9, allow_pickle=False)
            np.save(bundle_dir / "score_base.npy", base, allow_pickle=False)
            np.save(bundle_dir / "score_base_kasai.npy", base * 0.8, allow_pickle=False)
        else:
            np.save(bundle_dir / "score_base.npy", base, allow_pickle=False)
            np.save(bundle_dir / "score_base_kasai.npy", base * 0.8, allow_pickle=False)
        (bundle_dir / "meta.json").write_text("{}", encoding="utf-8")
        return bundle_dir

    monkeypatch.setattr(functional_eval, "write_functional_case", fake_write_functional_case)
    monkeypatch.setattr(functional_eval, "derive_bundle_from_run", fake_derive_bundle_from_run)

    search_json = tmp_path / "search.json"
    stack_json = tmp_path / "stack.json"
    search_json.write_text(
        json.dumps(
            {
                "details": {
                    "selected_configs": {
                        "stap": {"config_name": "Brain-SIMUS-Clin-MotionRobust-v0"},
                        "mc_svd": {"config_name": "rank6"},
                        "rpca": {"config_name": "lam1_it250_ds2_t32_r4"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    stack_json.write_text(
        json.dumps({"selected_stack": {"residual_family": "rpca", "stap_profile": "Brain-SIMUS-Clin-MotionRobust-v0"}}),
        encoding="utf-8",
    )

    out_csv = tmp_path / "functional.csv"
    out_json = tmp_path / "functional.json"
    out_headline_csv = tmp_path / "functional_headline.csv"
    out_headline_json = tmp_path / "functional_headline.json"
    argv_prev = sys.argv[:]
    sys.argv = [
        "simus_eval_functional.py",
        "--search-json",
        str(search_json),
        "--stack-json",
        str(stack_json),
        "--profiles",
        "ClinMobile-Pf-v2",
        "--tier",
        "smoke",
        "--dev-seeds",
        "1",
        "--eval-seeds",
        "2",
        "--ensemble-count",
        "6",
        "--max-workers",
        "1",
        "--threads-per-worker",
        "1",
        "--stap-device",
        "cpu",
        "--out-root",
        str(tmp_path / "out"),
        "--out-csv",
        str(out_csv),
        "--out-json",
        str(out_json),
        "--out-headline-csv",
        str(out_headline_csv),
        "--out-headline-json",
        str(out_headline_json),
        "--no-reuse-cases",
        "--no-reuse-bundles",
    ]
    try:
        functional_eval.main()
    finally:
        sys.argv = argv_prev

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["stap_profile"] == "Brain-SIMUS-Clin-MotionRobust-v0"
    assert len(payload["rows"]) > 0
    assert len(payload["selected_native_simple_by_family"]) > 0
