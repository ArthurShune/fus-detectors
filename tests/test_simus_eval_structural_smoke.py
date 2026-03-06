import numpy as np


def test_simus_structural_metrics_report_nuisance_fpr_and_alias_qc():
    from scripts.simus_eval_structural import evaluate_structural_metrics

    score = np.array(
        [
            [0.9, 0.8, 0.2],
            [0.7, 0.3, 0.1],
            [0.95, 0.6, 0.05],
        ],
        dtype=np.float32,
    )
    mask_h1_pf_main = np.array(
        [
            [True, True, False],
            [False, False, False],
            [False, False, False],
        ]
    )
    mask_h0_bg = np.array(
        [
            [False, False, True],
            [False, True, True],
            [False, False, True],
        ]
    )
    mask_h0_nuisance_pa = np.array(
        [
            [False, False, False],
            [True, False, False],
            [False, True, False],
        ]
    )
    mask_h1_alias_qc = np.array(
        [
            [False, False, False],
            [False, False, False],
            [True, False, False],
        ]
    )

    out = evaluate_structural_metrics(
        score=score,
        mask_h1_pf_main=mask_h1_pf_main,
        mask_h0_bg=mask_h0_bg,
        mask_h0_nuisance_pa=mask_h0_nuisance_pa,
        mask_h1_alias_qc=mask_h1_alias_qc,
        fprs=[0.5],
        match_tprs=[0.5],
    )

    assert out["n_h1_pf_main"] == 2
    assert out["n_h0_bg"] == 4
    assert out["n_h0_nuisance_pa"] == 2
    assert out["auc_main_vs_bg"] is not None
    assert out["auc_main_vs_nuisance"] is not None
    assert out["thr@5e-01"] is not None
    assert out["tpr_main@5e-01"] >= 0.5
    assert out["fpr_nuisance@5e-01"] >= 0.0
    assert out["tpr_alias_qc@5e-01"] >= 0.0
    assert out["thr_match_tpr@0p5"] is not None
    assert out["tpr_main_match@0p5"] >= 0.5
    assert out["fpr_nuisance_match@0p5"] >= 0.0
