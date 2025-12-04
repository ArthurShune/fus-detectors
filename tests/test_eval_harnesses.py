import csv
import json
from pathlib import Path

from eval import aggregate as agg
from eval.stress_suite import build_parser as build_stress_parser
from eval.stress_suite import run_stress
from eval.sweep_mc import build_parser as build_mc_parser
from eval.sweep_mc import run_sweep


def _read_csv(path: Path):
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def test_sweep_mc_smoke(tmp_path, monkeypatch):
    # Disable tqdm output for test runs
    monkeypatch.setenv("TQDM_DISABLE", "1")
    out_csv = tmp_path / "mc.csv"
    json_dir = tmp_path / "json"
    args = build_mc_parser().parse_args(
        [
            "--fprs",
            "1e-3",
            "5e-4",
            "--seeds",
            "1",
            "--npos",
            "200",
            "--nneg",
            "200",
            "--height",
            "32",
            "--width",
            "32",
            "--roc-thresholds",
            "32",
            "--device",
            "none",
            "--out",
            str(out_csv),
            "--json-dir",
            str(json_dir),
            "--no-progress",
        ]
    )
    run_sweep(args)

    rows = _read_csv(out_csv)
    assert len(rows) == 2
    # Ensure core metrics are present
    for row in rows:
        assert "pd_snr_delta_db" in row
        assert "tpr_at_fpr_delta" in row
        assert "timing_sec" in row
        assert float(row["fpr_target"]) in (1e-3, 5e-4)

    payload = json.loads((json_dir / "mc_seed_0.json").read_text())
    assert "compute" in payload
    assert "performance" in payload


def test_stress_suite_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("TQDM_DISABLE", "1")
    outdir = tmp_path / "stress"
    args = build_stress_parser().parse_args(
        [
            "--grid",
            "motion_um=10",
            "motion_freq_hz=0.5",
            "heterogeneity=low",
            "prf=2000",
            "prf_jitter_pct=5",
            "sensor_dropout=0.05",
            "scatter_density=1.1",
            "skull_db=3",
            "K=5",
            "--tiles",
            "8x8:4",
            "--seeds",
            "1",
            "--npos",
            "200",
            "--nneg",
            "200",
            "--height",
            "32",
            "--width",
            "32",
            "--roc-thresholds",
            "32",
            "--device",
            "none",
            "--outdir",
            str(outdir),
            "--no-progress",
        ]
    )
    run_stress(args)

    csv_rows = _read_csv(outdir / "stress_results.csv")
    assert len(csv_rows) == 1
    row = csv_rows[0]
    assert row["tile"] == "8x8:4"
    assert float(row["K"]) == 5.0
    assert "timing_sec" in row
    assert row["steer_fuse"] == "max"
    assert row["motion_comp"] in {"0", "1"}
    assert "steer_grid" in row
    assert row["heterogeneity"] == "low"
    assert abs(float(row["sensor_dropout"]) - 0.05) < 1e-6
    assert float(row["motion_freq_hz"]) == 0.5
    assert float(row["skull_db"]) == 3.0

    json_files = list(outdir.glob("stress_seed*.json"))
    assert len(json_files) == 1
    payload = json.loads(json_files[0].read_text())
    assert "compute" in payload
    assert payload["stress"]["tile"] == "8x8:4"
    assert payload["stress"]["heterogeneity"] == "low"
    assert payload["stress"]["sensor_dropout"] == 0.05
    assert "steer_grid" in payload["stress"]


def test_aggregate_tables(tmp_path):
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    rows = [
        {"fpr_target": 1e-3, "pd_snr_delta_db": 1.0, "tpr_at_fpr_delta": 0.05},
        {"fpr_target": 1e-3, "pd_snr_delta_db": 2.0, "tpr_at_fpr_delta": 0.10},
        {"fpr_target": 5e-4, "pd_snr_delta_db": 0.5, "tpr_at_fpr_delta": 0.02},
    ]
    with csv_a.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerow(rows[0])
        writer.writerow(rows[1])
    with csv_b.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerow(rows[2])

    out_csv = tmp_path / "summary.csv"
    out_md = tmp_path / "summary.md"
    agg.aggregate_tables([str(csv_a), str(csv_b)], ["fpr_target"], str(out_csv), str(out_md))

    summary_rows = _read_csv(out_csv)
    assert len(summary_rows) == 2
    fpr_values = {float(r["fpr_target"]) for r in summary_rows}
    assert fpr_values == {1e-3, 5e-4}

    md_text = out_md.read_text()
    assert "| fpr_target" in md_text
