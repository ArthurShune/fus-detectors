import numpy as np

from sim.kwave.common import _operator_metric_stats


def test_operator_metric_stats_returns_medians():
    tile_infos = [
        {
            "ka_pf_lambda_min": 0.92,
            "ka_pf_lambda_max": 1.05,
            "ka_pf_lambda_mean": 0.99,
            "ka_perp_lambda_min": 0.40,
            "ka_perp_lambda_max": 1.08,
            "ka_perp_lambda_mean": 0.75,
            "ka_alias_lambda_min": 1.40,
            "ka_alias_lambda_max": 1.80,
            "ka_alias_lambda_mean": 1.60,
            "ka_noise_lambda_min": 0.95,
            "ka_noise_lambda_max": 1.05,
            "ka_noise_lambda_mean": 1.00,
            "ka_operator_mixing_epsilon": 0.04,
            "ka_score_scale_ratio": 1.2,
            "sample_pf_lambda_min": 0.8,
            "sample_pf_lambda_max": 1.1,
            "sample_pf_lambda_mean": 0.95,
            "sample_perp_lambda_min": 0.3,
            "sample_perp_lambda_max": 0.9,
            "sample_perp_lambda_mean": 0.6,
            "sample_alias_lambda_min": 1.2,
            "sample_alias_lambda_max": 1.6,
            "sample_alias_lambda_mean": 1.4,
            "sample_noise_lambda_min": 0.9,
            "sample_noise_lambda_max": 1.0,
            "sample_noise_lambda_mean": 0.95,
            "sample_po_noise_floor": 0.15,
            "prior_po_noise_floor": 0.12,
        },
        {
            "ka_pf_lambda_min": 0.98,
            "ka_pf_lambda_max": 1.08,
            "ka_pf_lambda_mean": 1.01,
            "ka_perp_lambda_min": 0.45,
            "ka_perp_lambda_max": 1.20,
            "ka_perp_lambda_mean": 0.80,
            "ka_alias_lambda_min": 1.30,
            "ka_alias_lambda_max": 1.70,
            "ka_alias_lambda_mean": 1.50,
            "ka_noise_lambda_min": 0.90,
            "ka_noise_lambda_max": 1.00,
            "ka_noise_lambda_mean": 0.95,
            "ka_operator_mixing_epsilon": 0.06,
            "ka_score_scale_ratio": 0.8,
            "sample_pf_lambda_min": 0.9,
            "sample_pf_lambda_max": 1.2,
            "sample_pf_lambda_mean": 1.0,
            "sample_perp_lambda_min": 0.35,
            "sample_perp_lambda_max": 1.0,
            "sample_perp_lambda_mean": 0.7,
            "sample_alias_lambda_min": 1.1,
            "sample_alias_lambda_max": 1.5,
            "sample_alias_lambda_mean": 1.3,
            "sample_noise_lambda_min": 0.85,
            "sample_noise_lambda_max": 0.95,
            "sample_noise_lambda_mean": 0.9,
            "sample_po_noise_floor": 0.2,
            "prior_po_noise_floor": 0.14,
        },
    ]
    stats = _operator_metric_stats(tile_infos)
    assert np.isclose(stats["ka_pf_lambda_min"], 0.95)
    assert np.isclose(stats["ka_perp_lambda_max"], 1.14)
    assert np.isclose(stats["ka_operator_mixing_epsilon"], 0.05)
    assert np.isclose(stats["ka_score_scale_ratio"], 1.0)
    assert np.isclose(stats["ka_alias_lambda_mean"], 1.55)
    assert np.isclose(stats["ka_noise_lambda_mean"], 0.975)
    assert stats["ka_pf_lambda_min_p10"] is not None
    assert stats["ka_perp_lambda_max_p90"] is not None
    assert stats["ka_alias_lambda_mean_p90"] is not None
    assert stats["ka_operator_mixing_epsilon_p10"] is not None
    assert np.isclose(stats["sample_pf_lambda_min"], 0.85)
    assert np.isclose(stats["sample_perp_lambda_max"], 0.95)
    assert np.isclose(stats["sample_alias_lambda_mean"], 1.35)
    assert np.isclose(stats["sample_noise_lambda_mean"], 0.925)
    assert np.isclose(stats["sample_po_noise_floor"], 0.175)
    assert stats["sample_po_noise_floor_p10"] is not None
    assert stats["prior_po_noise_floor"] is not None


def test_operator_metric_stats_handles_missing_values():
    stats = _operator_metric_stats([{}])
    assert stats["ka_pf_lambda_min"] is None
    assert stats["ka_operator_mixing_epsilon"] is None
    assert stats["ka_score_scale_ratio"] is None
    assert stats["ka_alias_lambda_mean"] is None
    assert stats["sample_alias_lambda_min"] is None
    assert stats["sample_po_noise_floor"] is None
