import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from eval.acceptance_cli import build_parser, run
from pipeline.confirm2.bvn_tail import joint_tail
from pipeline.confirm2.validator import calibrate_confirm2, evaluate_confirm2


def _latest(path: Path) -> Path:
    return max(path.glob("acceptance_*.json"), key=lambda p: p.stat().st_mtime)


def test_acceptance_simulate_confirm2(tmp_path):
    runs = tmp_path / "runs"
    reports = tmp_path / "reports"
    figs = tmp_path / "figs"
    runs.mkdir()
    reports.mkdir()
    figs.mkdir()

    args = build_parser().parse_args(
        [
            "--simulate",
            "--confirm2",
            "--confirm2-alpha2",
            "1e-4",
            "--confirm2-ci-alpha",
            "0.1",
            "--confirm2-rho",
            "0.25",
            "--height",
            "48",
            "--width",
            "48",
            "--seed",
            "7",
            "--out_dir",
            str(runs),
            "--report_dir",
            str(reports),
            "--fig_dir",
            str(figs),
            "--delta_snr_min",
            "0.0",
            "--delta_tpr_min",
            "0.0",
            "--fpr_target",
            "1e-4",
        ]
    )
    run(args)
    payload = json.loads(_latest(runs).read_text())
    assert payload["overall_pass"] in {True, False}
    assert "roc_summary" in payload
    confirm = payload["confirm2"]
    assert 0.0 <= confirm["rho_hat"] <= 1.0
    assert confirm["alpha1_per_look"] > 0.0
    assert confirm["empirical_pair_pfa"] >= 0.0
    assert confirm["copula_mode"] in {"gaussian", "t", "empirical"}
    assert confirm["lambda_u_emp"] >= 0.0
    assert confirm["lambda_u_gauss"] >= 0.0
    assert abs(confirm["empirical_pair_pfa"] - confirm["predicted_pair_pfa"]) < max(
        5e-3, confirm["predicted_pair_pfa"] * 5
    )


def test_confirm2_tail_dependence_fallback():
    rng = np.random.default_rng(3)
    n = 60000
    df = 4.0
    rho = 0.65
    cov = np.array([[1.0, rho], [rho, 1.0]], dtype=np.float64)
    L = np.linalg.cholesky(cov)
    g = rng.standard_normal((n, 2))
    y = g @ L.T
    chi = rng.chisquare(df, size=n)
    t_samples = y / np.sqrt(chi[:, None] / df)

    s1 = t_samples[:, 0]
    s2 = t_samples[:, 1]
    half = n // 2
    calib = calibrate_confirm2(
        s1[:half],
        s2[:half],
        alpha2_target=1e-4,
        seed=12,
        evd_mode="gpd",
    )
    assert calib.copula_mode in {"t", "empirical"}
    ev = evaluate_confirm2(calib, s1[half:], s2[half:], alpha=0.1)
    gauss_pred = float(joint_tail(norm.isf(calib.alpha1), calib.rho_hat))
    tail_pred = calib.lambda_u_emp * calib.alpha1
    assert ev.predicted_pair_pfa >= gauss_pred - 1e-12
    assert ev.predicted_pair_pfa >= tail_pred - 1e-12
