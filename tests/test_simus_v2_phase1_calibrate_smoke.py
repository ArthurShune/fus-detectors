from __future__ import annotations

import json
from pathlib import Path
import subprocess


def test_simus_v2_phase1_calibrate_smoke(tmp_path: Path) -> None:
    out_csv = tmp_path / "phase1.csv"
    out_json = tmp_path / "phase1.json"
    out_root = tmp_path / "runs"
    cmd = [
        "python",
        "scripts/simus_v2_phase1_calibrate.py",
        "--profile",
        "ClinIntraOp-Pf-v2",
        "--tier",
        "smoke",
        "--seed",
        "0",
        "--candidate",
        "base",
        "--out-root",
        str(out_root),
        "--out-csv",
        str(out_csv),
        "--out-json",
        str(out_json),
    ]
    subprocess.run(cmd, check=True)
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["best_candidate"] == "base"
    rows = out_csv.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) >= 2


def test_simus_v2_phase1_calibrate_multiseed_smoke(tmp_path: Path) -> None:
    out_csv = tmp_path / "phase1_multi.csv"
    out_json = tmp_path / "phase1_multi.json"
    out_root = tmp_path / "runs_multi"
    cmd = [
        "python",
        "scripts/simus_v2_phase1_calibrate.py",
        "--profile",
        "ClinIntraOp-Pf-v2",
        "--tier",
        "smoke",
        "--seeds",
        "0,1",
        "--candidate",
        "base",
        "--profile-gate",
        "ClinIntraOp-Pf-v2",
        "--out-root",
        str(out_root),
        "--out-csv",
        str(out_csv),
        "--out-json",
        str(out_json),
    ]
    subprocess.run(cmd, check=True)
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["best_candidate"] == "base"
    assert payload["seeds"] == [0, 1]
    rows = out_csv.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) >= 2
