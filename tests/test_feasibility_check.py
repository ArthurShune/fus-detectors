from scripts import feasibility_check as fc


class DummyBundle(fc.BundleInfo):
    def __init__(self, telemetry: dict[str, float]):
        from pathlib import Path

        super().__init__(
            path=Path("bundle"),
            meta={},
            telemetry=telemetry,
            score_stats={},
            seed=0,
        )


def test_check_operator_metrics_flags_oc2_failures():
    bundle = DummyBundle(
        {
            "ka_pf_lambda_min": 0.8,
            "ka_pf_lambda_max": 1.2,
            "ka_perp_lambda_max": 1.4,
            "ka_operator_mixing_epsilon": 0.2,
        }
    )
    result = fc.check_operator_metrics([bundle])
    assert result.status == "fail"
    assert "pf_min" in result.detail
    assert "perp_max" in result.detail


def test_check_operator_metrics_uses_alias_fields_when_present():
    bundle = DummyBundle(
        {
            "ka_pf_lambda_min": 0.97,
            "ka_pf_lambda_max": 1.05,
            "ka_perp_lambda_max": 1.6,
            "ka_operator_mixing_epsilon": 0.02,
            "ka_alias_lambda_mean": 1.6,
            "ka_noise_lambda_mean": 1.05,
        }
    )
    result = fc.check_operator_metrics([bundle])
    assert result.status == "pass"
    assert "All band/mixing metrics within targets." in result.detail
