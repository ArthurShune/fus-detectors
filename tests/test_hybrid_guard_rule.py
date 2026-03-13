from __future__ import annotations

import numpy as np

from sim.kwave.common import (
    _apply_hybrid_rescue_score_map,
    _hybrid_choose_advanced_tile_mask,
    _normalize_hybrid_rescue_rule,
)


def test_guard_promote_rule_has_frozen_threshold() -> None:
    rule = _normalize_hybrid_rescue_rule("guard_promote_v1")
    assert rule["feature"] == "base_guard_frac_map"
    assert rule["direction"] == ">="
    assert float(rule["threshold"]) == 0.09700579196214676


def test_adaptive_guard_rule_has_frozen_tile_threshold() -> None:
    rule = _normalize_hybrid_rescue_rule("guard_promote_tile_v1")
    assert rule["feature"] == "base_guard_frac_map"
    assert rule["direction"] == ">="
    assert rule["aggregation"] == "tile_mean"
    assert bool(rule["prefer_advanced_on_invalid"]) is False
    assert float(rule["threshold"]) == 0.1453727245330811


def test_guard_promote_rule_uses_unwhitened_default_and_whitened_promotion() -> None:
    advanced = np.array([[10.0, 10.0], [10.0, 10.0]], dtype=np.float32)
    rescue = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    guard = np.array([[0.02, 0.10], [0.20, 0.05]], dtype=np.float32)
    rule = _normalize_hybrid_rescue_rule("guard_promote_v1")

    hybrid, rescue_mask, stats = _apply_hybrid_rescue_score_map(
        advanced,
        rescue,
        feature_name=str(rule["feature"]),
        feature_map=guard,
        direction=str(rule["direction"]),
        threshold=float(rule["threshold"]),
    )

    expected = np.array([[1.0, 10.0], [10.0, 1.0]], dtype=np.float32)
    expected_rescue = np.array([[True, False], [False, True]])
    np.testing.assert_allclose(hybrid, expected)
    np.testing.assert_array_equal(rescue_mask, expected_rescue)
    assert stats["advanced_pixels"] == 2
    assert stats["rescue_pixels"] == 2


def test_tile_guard_promote_rule_promotes_whole_tiles() -> None:
    guard = np.array(
        [
            [0.00, 0.12, 0.00, 0.00],
            [0.00, 0.12, 0.00, 0.00],
            [0.00, 0.00, 0.20, 0.20],
            [0.00, 0.00, 0.20, 0.20],
        ],
        dtype=np.float32,
    )
    choose_advanced, promote_tiles = _hybrid_choose_advanced_tile_mask(
        guard,
        tile_hw=(2, 2),
        stride=2,
        direction=">=",
        threshold=0.1453727245330811,
        reduction="mean",
        prefer_advanced_on_invalid=False,
    )
    expected_tiles = np.array([False, False, False, True], dtype=bool)
    expected_map = np.array(
        [
            [False, False, False, False],
            [False, False, False, False],
            [False, False, True, True],
            [False, False, True, True],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(promote_tiles, expected_tiles)
    np.testing.assert_array_equal(choose_advanced, expected_map)
