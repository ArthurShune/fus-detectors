from __future__ import annotations

from scripts.simus_symmetric_pipeline_compare import (
    _select_best_detector_any,
    _select_best_simple_detector,
    _selected_residual_specs,
)


def test_selected_residual_specs_extracts_non_stap_families() -> None:
    payload = {
        "details": {
            "selected_configs": {
                "stap": {
                    "method_family": "stap",
                    "config_name": "Brain-SIMUS-Clin-MotionMidRobust-v0",
                },
                "mc_svd": {
                    "method_family": "mc_svd",
                    "config_name": "ef95",
                },
                "local_svd": {
                    "method_family": "local_svd",
                    "config_name": "tile16_s4_ef95",
                },
            }
        }
    }
    specs, stap_profile = _selected_residual_specs(payload)
    assert stap_profile == "Brain-SIMUS-Clin-MotionMidRobust-v0"
    assert [s.method_family for s in specs] == ["local_svd", "mc_svd"]
    assert [s.config_name for s in specs] == ["tile16_s4_ef95", "ef95"]


def test_best_simple_detector_selection_ignores_stap() -> None:
    dev_summary = [
        {
            "method_family": "mc_svd",
            "config_name": "ef95",
            "detector_head": "pd",
            "selection_score": 0.2,
        },
        {
            "method_family": "mc_svd",
            "config_name": "ef95",
            "detector_head": "kasai",
            "selection_score": 0.5,
        },
        {
            "method_family": "mc_svd",
            "config_name": "ef95",
            "detector_head": "stap",
            "selection_score": 1.9,
        },
    ]
    best_simple = _select_best_simple_detector(dev_summary)
    best_any = _select_best_detector_any(dev_summary)
    assert best_simple["mc_svd"]["detector_head"] == "kasai"
    assert best_any["mc_svd"]["detector_head"] == "stap"
