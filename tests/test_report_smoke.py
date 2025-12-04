# tests/test_report_smoke.py

from eval.acceptance_cli import main as acceptance_main


def test_acceptance_and_pdf_smoke(tmp_path, monkeypatch):
    # Fast simulation to keep CI snappy
    monkeypatch.setenv("FAST_ACCEPT", "1")
    runs = tmp_path / "runs"
    reports = tmp_path / "reports"
    runs.mkdir()
    reports.mkdir()
    # Run acceptance in simulate mode, writing into temp dirs
    acceptance_main(
        [
            "--simulate",
            "--out_dir",
            str(runs),
            "--report_dir",
            str(reports),
            "--fig_dir",
            "figs/outputs",
            "--device",
            "none",
            "--seed",
            "123",
            "--alpha",
            "1e-3",
            "--fpr_target",
            "1e-3",
        ]
    )
    # Verify artifacts exist
    jsons = list(runs.glob("acceptance_*.json"))
    pdfs = list(reports.glob("acceptance_summary_*.pdf"))
    assert len(jsons) >= 1
    assert len(pdfs) >= 1
    # Non-empty files
    assert jsons[0].stat().st_size > 100
    assert pdfs[0].stat().st_size > 5000
