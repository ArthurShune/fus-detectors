from __future__ import annotations

from scripts.simus_motion_proxy_bundle_bench import summarize_rows


def test_summarize_rows_ignores_baseline_and_averages_stap_rows():
    rows = [
        {"role": "baseline", "method_label": "MC-SVD", "auc_main_vs_bg": 0.4, "auc_main_vs_nuisance": 0.3, "fpr_nuisance_match@0p5": 0.8},
        {
            "role": "stap",
            "method_label": "MC-SVD -> STAP (A)",
            "auc_main_vs_bg": 0.6,
            "auc_main_vs_nuisance": 0.7,
            "fpr_nuisance_match@0p5": 0.2,
            "delta_auc_main_vs_bg": 0.2,
            "delta_auc_main_vs_nuisance": 0.4,
            "delta_fpr_nuisance_match@0p5": -0.6,
        },
        {
            "role": "stap",
            "method_label": "MC-SVD -> STAP (A)",
            "auc_main_vs_bg": 0.8,
            "auc_main_vs_nuisance": 0.9,
            "fpr_nuisance_match@0p5": 0.1,
            "delta_auc_main_vs_bg": 0.3,
            "delta_auc_main_vs_nuisance": 0.5,
            "delta_fpr_nuisance_match@0p5": -0.7,
        },
    ]
    out = summarize_rows(rows)
    assert out["MC-SVD -> STAP (A)"]["count"] == 2
    assert out["MC-SVD -> STAP (A)"]["mean_auc_main_vs_bg"] == 0.7
    assert out["MC-SVD -> STAP (A)"]["mean_auc_main_vs_nuisance"] == 0.8
    assert out["MC-SVD -> STAP (A)"]["mean_fpr_nuisance_match@0p5"] == 0.15000000000000002
