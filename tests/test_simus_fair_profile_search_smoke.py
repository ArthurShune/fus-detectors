import json
import sys
from pathlib import Path

import numpy as np


def test_simus_fair_profile_search_selects_frozen_configs(tmp_path, monkeypatch):
    from scripts import simus_fair_profile_search as search

    def fake_load_canonical_run(run_dir):
        icube = np.zeros((8, 3, 3), dtype=np.complex64)
        masks = {
            "mask_h1_pf_main": np.array(
                [[1, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=bool
            ),
            "mask_h0_bg": np.array(
                [[0, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=bool
            ),
            "mask_h0_nuisance_pa": np.array(
                [[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=bool
            ),
            "mask_h1_alias_qc": np.array(
                [[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=bool
            ),
        }
        meta = {
            "acquisition": {"prf_hz": 1500.0},
            "motion": {"telemetry": {"disp_rms_px": 0.5}},
            "phase_screen": {"telemetry": {"phase_rms_rad": 0.1}},
        }
        return icube, masks, meta

    def fake_derive_bundle_from_run(**kwargs):
        out_root = Path(kwargs["out_root"])
        dataset_name = str(kwargs["dataset_name"])
        bundle_dir = out_root / dataset_name
        bundle_dir.mkdir(parents=True, exist_ok=True)
        baseline_type = str(kwargs["baseline_type"])
        run_stap = bool(kwargs["run_stap"])
        overrides = dict(kwargs.get("bundle_overrides") or {})
        stap_profile = str(kwargs.get("stap_profile"))
        pos = 2.0
        pos2 = 1.8
        bg = 1.5
        bg2 = 1.4
        nuis = 1.3
        if run_stap:
            if stap_profile.endswith("MotionRobust-v0"):
                pos, pos2, bg, bg2, nuis = 5.0, 4.6, 1.0, 0.9, 0.2
            elif stap_profile.endswith("MotionMidRobust-v0"):
                pos, pos2, bg, bg2, nuis = 2.6, 1.5, 1.2, 1.1, 0.3
            elif stap_profile.endswith("MotionShort-v0"):
                pos, pos2, bg, bg2, nuis = 2.2, 1.35, 1.3, 1.2, 0.4
            else:
                pos, pos2, bg, bg2, nuis = 1.15, 1.05, 1.1, 1.0, 0.5
        elif baseline_type == "mc_svd":
            rank = overrides.get("svd_rank", None)
            if rank is not None:
                if int(rank) <= 2:
                    pos, pos2, bg, bg2, nuis = 3.6, 3.0, 1.25, 1.15, 0.75
                elif int(rank) <= 4:
                    pos, pos2, bg, bg2, nuis = 3.8, 3.2, 1.22, 1.12, 0.72
                else:
                    pos, pos2, bg, bg2, nuis = 3.4, 2.9, 1.28, 1.18, 0.78
            else:
                ef = float(overrides.get("svd_energy_frac", 0.90))
                if ef >= 0.95:
                    pos, pos2, bg, bg2, nuis = 4.0, 3.5, 1.2, 1.1, 0.7
                elif ef <= 0.85:
                    pos, pos2, bg, bg2, nuis = 2.5, 1.7, 1.5, 1.4, 1.0
                else:
                    pos, pos2, bg, bg2, nuis = 3.0, 2.0, 1.4, 1.3, 0.9
        elif baseline_type == "svd_similarity":
            kappa = float(overrides.get("svd_sim_kappa", 2.5))
            if kappa >= 3.0:
                pos, pos2, bg, bg2, nuis = 3.6, 2.8, 1.2, 1.1, 0.8
            elif kappa <= 2.0:
                pos, pos2, bg, bg2, nuis = 3.0, 1.8, 1.4, 1.3, 1.0
            else:
                pos, pos2, bg, bg2, nuis = 3.2, 2.2, 1.3, 1.2, 0.9
        elif baseline_type == "local_svd":
            tile_hw = tuple(overrides.get("tile_hw", (8, 8)))
            if tile_hw == (12, 12):
                pos, pos2, bg, bg2, nuis = 3.4, 2.4, 1.3, 1.2, 0.9
            elif tile_hw == (6, 6):
                pos, pos2, bg, bg2, nuis = 3.0, 1.9, 1.5, 1.4, 1.1
            else:
                pos, pos2, bg, bg2, nuis = 3.2, 2.1, 1.4, 1.3, 1.0
        elif baseline_type == "rpca":
            lam = float(overrides.get("rpca_lambda", 0.01))
            if lam > 0.01:
                pos, pos2, bg, bg2, nuis = 2.8, 1.9, 1.3, 1.2, 0.95
            else:
                pos, pos2, bg, bg2, nuis = 2.6, 1.8, 1.4, 1.3, 1.0
        elif baseline_type == "hosvd":
            ef = tuple(overrides.get("hosvd_energy_fracs", (0.99, 0.99, 0.99)))
            if ef[0] <= 0.95:
                pos, pos2, bg, bg2, nuis = 3.3, 2.5, 1.2, 1.1, 0.85
            else:
                pos, pos2, bg, bg2, nuis = 3.1, 2.2, 1.3, 1.2, 0.95
        score = np.array(
            [
                [pos, pos2, bg],
                [bg2, bg, bg2],
                [pos2, bg2, nuis],
            ],
            dtype=np.float32,
        )
        np.save(bundle_dir / "score_base.npy", score, allow_pickle=False)
        np.save(bundle_dir / "score_stap_preka.npy", score, allow_pickle=False)
        (bundle_dir / "meta.json").write_text("{}", encoding="utf-8")
        return bundle_dir

    monkeypatch.setattr(search, "load_canonical_run", fake_load_canonical_run)
    monkeypatch.setattr(search, "derive_bundle_from_run", fake_derive_bundle_from_run)

    out_csv = tmp_path / "fair.csv"
    out_json = tmp_path / "fair.json"
    argv_prev = sys.argv[:]
    sys.argv = [
        "simus_fair_profile_search.py",
        "--dev-case",
        f"dev::{tmp_path / 'dev_run'}",
        "--eval-case",
        f"eval::{tmp_path / 'eval_run'}",
        "--out-root",
        str(tmp_path / "out"),
        "--out-csv",
        str(out_csv),
        "--out-json",
        str(out_json),
        "--no-reuse-bundles",
    ]
    try:
        search.main()
    finally:
        sys.argv = argv_prev

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    selected = payload["details"]["selected_configs"]
    assert selected["stap"]["config_name"] in {
        "Brain-SIMUS-Clin-MotionShort-v0",
        "Brain-SIMUS-Clin-MotionRobust-v0",
        "Brain-SIMUS-Clin-MotionMidRobust-v0",
    }
    assert "mc_svd" in selected
    assert "svd_similarity" in selected
    eval_rows = [row for row in payload["rows"] if row["split"] == "eval"]
    assert all(row["config_name"] == selected[row["method_family"]]["config_name"] for row in eval_rows)
