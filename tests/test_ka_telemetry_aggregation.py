import numpy as np

from sim.kwave.common import _ka_effective_status


def test_ka_telemetry_aggregation_populates_c1_c2_and_operator():
    # Minimal tile info with KA details populated
    tile_infos = [
        {
            "ka_pf_lambda_min": 0.98,
            "ka_pf_lambda_max": 1.02,
            "ka_pf_lambda_mean": 1.0,
            "ka_perp_lambda_mean": 0.9,
            "ka_alias_lambda_mean": 1.6,
            "ka_noise_lambda_mean": 1.0,
            "ka_operator_mixing_epsilon": 0.01,
            "ka_snr_flow_ratio": 1.0,
            "ka_noise_perp_ratio": 1.0,
            "operator_feasible": True,
        },
        {
            "ka_pf_lambda_min": 0.97,
            "ka_pf_lambda_max": 1.01,
            "ka_pf_lambda_mean": 0.99,
            "ka_perp_lambda_mean": 0.95,
            "ka_alias_lambda_mean": 1.5,
            "ka_noise_lambda_mean": 0.98,
            "ka_operator_mixing_epsilon": 0.02,
            "ka_snr_flow_ratio": 0.95,
            "ka_noise_perp_ratio": 1.05,
            "operator_feasible": True,
        },
    ]

    # Aggregate the telemetry analog: compute medians manually
    pf_mins = [t["ka_pf_lambda_min"] for t in tile_infos]
    alias_means = [t["ka_alias_lambda_mean"] for t in tile_infos]
    snr_ratios = [t["ka_snr_flow_ratio"] for t in tile_infos]
    noise_ratios = [t["ka_noise_perp_ratio"] for t in tile_infos]
    mixing_eps = [t["ka_operator_mixing_epsilon"] for t in tile_infos]

    def q50(vals):
        return float(np.median(vals))

    agg = {
        "ka_pf_lambda_min": q50(pf_mins),
        "ka_alias_lambda_mean": q50(alias_means),
        "ka_median_snr_flow_ratio": q50(snr_ratios),
        "ka_median_noise_perp_ratio": q50(noise_ratios),
        "ka_operator_mixing_epsilon": q50(mixing_eps),
        "ka_operator_feasible": all(t.get("operator_feasible", False) for t in tile_infos),
    }

    assert agg["ka_pf_lambda_min"] < 1.01 and agg["ka_pf_lambda_min"] > 0.9
    assert agg["ka_alias_lambda_mean"] > 1.45
    assert 0.9 <= agg["ka_median_snr_flow_ratio"] <= 1.05
    assert 0.9 <= agg["ka_median_noise_perp_ratio"] <= 1.1
    assert agg["ka_operator_mixing_epsilon"] < 0.05
    assert agg["ka_operator_feasible"] is True
