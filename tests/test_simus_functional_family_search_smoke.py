from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _make_case(root: Path, profile: str, seed: int, null_run: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    reg = np.linspace(0.0, 1.0, 6, dtype=np.float32)
    np.save(root / "hemo_regressor.npy", reg, allow_pickle=False)
    mask_act = np.zeros((3, 3), dtype=bool)
    mask_act[0, 0] = True
    mask_bg = np.ones((3, 3), dtype=bool)
    mask_bg[0, 0] = False
    mask_nuis = np.zeros((3, 3), dtype=bool)
    mask_nuis[2, 2] = True
    mask_spec = np.zeros((3, 3), dtype=bool)
    np.save(root / "mask_activation_roi.npy", mask_act, allow_pickle=False)
    np.save(root / "mask_h0_bg.npy", mask_bg, allow_pickle=False)
    np.save(root / "mask_h0_nuisance_pa.npy", mask_nuis, allow_pickle=False)
    np.save(root / "mask_h0_specular_struct.npy", mask_spec, allow_pickle=False)
    rows = []
    for idx in range(6):
        ensemble_dir = root / f"ensemble_{idx:03d}"
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
                "activation_gain": 0.0 if null_run else float(reg[idx]),
            }
        )
    (root / "ensemble_table.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")


def test_functional_family_search_pair_filter(tmp_path: Path, monkeypatch) -> None:
    from scripts import simus_functional_family_search as family_search

    cases_root = tmp_path / "cases"
    for split, seeds in (("dev", [1]), ("eval", [2])):
        for profile in ("ClinMobile-Pf-v2", "ClinIntraOpParenchyma-Pf-v3"):
            for seed in seeds:
                for null_run in (False, True):
                    tag = "null" if null_run else "task"
                    _make_case(cases_root / split / f"{profile}_seed{seed}_{tag}", profile, seed, null_run)

    monkeypatch.setattr(
        family_search,
        "_candidate_grid",
        lambda families: [
            {
                "method_family": "local_svd",
                "config_name": "tile16_s4_ef95_rect",
                "baseline_type": "local_svd",
                "override_builder": lambda _shape: {},
            },
            {
                "method_family": "adaptive_local_svd",
                "config_name": "tile12_s4_bal_r8",
                "baseline_type": "adaptive_local_svd",
                "override_builder": lambda _shape: {},
            },
        ],
    )

    def fake_derive_bundle_timed(**kwargs):
        bundle_dir = Path(kwargs["out_root"]) / str(kwargs["dataset_name"])
        bundle_dir.mkdir(parents=True, exist_ok=True)
        run_dir = Path(kwargs["run_dir"])
        idx = int(run_dir.name.split("_")[-1])
        base_amp = float(idx + 1)
        fam = str(kwargs["baseline_type"])
        if fam == "local_svd":
            base = np.array([[base_amp, 0.8, 0.2], [0.2, 0.2, 0.1], [0.1, 0.1, 0.05]], dtype=np.float32)
        else:
            base = np.array([[0.8, 0.7, 0.2], [0.2, 0.2, 0.1], [0.1, 0.1, 0.05]], dtype=np.float32)
        np.save(bundle_dir / "score_base.npy", base, allow_pickle=False)
        np.save(bundle_dir / "score_base_kasai.npy", base * 0.9, allow_pickle=False)
        if kwargs["run_stap"]:
            stap = base + (0.4 if fam == "local_svd" else 0.1)
            np.save(bundle_dir / "score_stap_preka.npy", stap, allow_pickle=False)
            np.save(bundle_dir / "score_pd_stap.npy", stap * 0.95, allow_pickle=False)
        (bundle_dir / "meta.json").write_text("{}", encoding="utf-8")
        return bundle_dir, 1.0

    monkeypatch.setattr(family_search, "_derive_bundle_timed", fake_derive_bundle_timed)

    out_csv = tmp_path / "rows.csv"
    out_json = tmp_path / "rows.json"
    out_headline_csv = tmp_path / "headline.csv"
    out_headline_json = tmp_path / "headline.json"
    argv_prev = sys.argv[:]
    sys.argv = [
        "simus_functional_family_search.py",
        "--cases-root",
        str(cases_root),
        "--profiles",
        "ClinMobile-Pf-v2,ClinIntraOpParenchyma-Pf-v3",
        "--dev-seeds",
        "1",
        "--eval-seeds",
        "2",
        "--families",
        "local_svd,adaptive_local_svd",
        "--profile-family-pairs",
        "ClinMobile-Pf-v2:local_svd",
        "--readout-mode",
        "basic",
        "--stap-profiles",
        "Brain-SIMUS-Clin",
        "--stap-device",
        "cpu",
        "--no-reuse-bundles",
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
    ]
    try:
        family_search.main()
    finally:
        sys.argv = argv_prev

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["profile_family_pairs"] == [
        {"base_profile": "ClinMobile-Pf-v2", "method_family": "local_svd"}
    ]
    assert len(payload["comparison"]) == 1
    row = payload["comparison"][0]
    assert row["base_profile"] == "ClinMobile-Pf-v2"
    assert row["method_family"] == "local_svd"
