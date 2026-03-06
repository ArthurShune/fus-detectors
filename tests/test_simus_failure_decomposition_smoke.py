import pytest


def test_simus_failure_decomposition_infers_upstream_mcsvd_when_raw_stap_is_better():
    from scripts.simus_failure_decomposition import infer_primary_cause

    rows = [
        {"variant": "mcsvd_regon_base", "eval_score": "pd", "auc_main_vs_nuisance": 0.10},
        {"variant": "mcsvd_regon_stap", "eval_score": "pd", "auc_main_vs_nuisance": 0.05},
        {"variant": "mcsvd_regon_stap", "eval_score": "vnext", "auc_main_vs_nuisance": 0.07},
        {"variant": "mcsvd_regoff_stap", "eval_score": "pd", "auc_main_vs_nuisance": 0.06},
        {"variant": "raw_regon_stap", "eval_score": "pd", "auc_main_vs_nuisance": 0.20},
    ]

    out = infer_primary_cause(rows)
    assert out["primary_cause"] == "upstream_mcsvd"
    assert out["delta_raw_regon_stap_minus_mcsvd_regon_stap_auc_nuis_pd"] == pytest.approx(0.15)
