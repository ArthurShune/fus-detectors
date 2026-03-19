from __future__ import annotations

import argparse

from fus_detectors import DetectorConfig
from fus_detectors.defaults import (
    ADAPTIVE_GUARD_DEFAULTS,
    CLINICAL_REPLAY_DEFAULTS,
    PUBLIC_DETECTOR_DEFAULTS,
)
from scripts.refactor.replay_profiles import apply_stap_profile_defaults


def test_public_detector_config_tracks_shared_defaults():
    config = DetectorConfig()

    assert config.tile_shape == PUBLIC_DETECTOR_DEFAULTS.tile_shape
    assert config.tile_stride == PUBLIC_DETECTOR_DEFAULTS.tile_stride
    assert config.temporal_support == PUBLIC_DETECTOR_DEFAULTS.temporal_support
    assert config.diag_load == PUBLIC_DETECTOR_DEFAULTS.diag_load
    assert config.covariance_estimator == PUBLIC_DETECTOR_DEFAULTS.covariance_estimator
    assert config.huber_c == PUBLIC_DETECTOR_DEFAULTS.huber_c
    assert config.grid_step_rel == PUBLIC_DETECTOR_DEFAULTS.grid_step_rel
    assert config.fd_span_rel == PUBLIC_DETECTOR_DEFAULTS.fd_span_rel
    assert config.min_frequency_bins == PUBLIC_DETECTOR_DEFAULTS.min_frequency_bins
    assert config.max_frequency_bins == PUBLIC_DETECTOR_DEFAULTS.max_frequency_bins
    assert config.msd_lambda == PUBLIC_DETECTOR_DEFAULTS.msd_lambda
    assert config.msd_ridge == PUBLIC_DETECTOR_DEFAULTS.msd_ridge
    assert config.msd_aggregation == PUBLIC_DETECTOR_DEFAULTS.msd_aggregation
    assert config.msd_ratio_rho == PUBLIC_DETECTOR_DEFAULTS.msd_ratio_rho
    assert config.adaptive_guard_flow_band_hz == ADAPTIVE_GUARD_DEFAULTS.flow_band_hz
    assert config.adaptive_guard_alias_center_hz == ADAPTIVE_GUARD_DEFAULTS.alias_center_hz
    assert config.adaptive_guard_alias_width_hz == ADAPTIVE_GUARD_DEFAULTS.alias_width_hz
    assert config.adaptive_guard_promote_threshold == ADAPTIVE_GUARD_DEFAULTS.promote_threshold


def test_replay_clinical_profile_tracks_shared_defaults():
    args = argparse.Namespace(
        stap_profile="clinical",
        tile_h=12,
        tile_w=12,
        tile_stride=6,
        lt=4,
        diag_load=1e-2,
        baseline="mc_svd",
        score_mode="pd",
        time_window_length=None,
    )

    apply_stap_profile_defaults(args)

    assert (args.tile_h, args.tile_w) == PUBLIC_DETECTOR_DEFAULTS.tile_shape
    assert args.tile_stride == PUBLIC_DETECTOR_DEFAULTS.tile_stride
    assert args.lt == PUBLIC_DETECTOR_DEFAULTS.temporal_support
    assert args.diag_load == PUBLIC_DETECTOR_DEFAULTS.diag_load
    assert args.cov_estimator == PUBLIC_DETECTOR_DEFAULTS.covariance_estimator
    assert args.huber_c == PUBLIC_DETECTOR_DEFAULTS.huber_c
    assert args.grid_step_rel == PUBLIC_DETECTOR_DEFAULTS.grid_step_rel
    assert args.fd_span_rel == "0.30,1.10"
    assert args.fd_fixed_span_hz == CLINICAL_REPLAY_DEFAULTS.fd_fixed_span_hz
    assert args.max_pts == PUBLIC_DETECTOR_DEFAULTS.max_frequency_bins
    assert args.fd_min_pts == PUBLIC_DETECTOR_DEFAULTS.min_frequency_bins
    assert args.constraint_mode == CLINICAL_REPLAY_DEFAULTS.constraint_mode
    assert args.constraint_ridge == CLINICAL_REPLAY_DEFAULTS.constraint_ridge
    assert args.mvdr_load_mode == CLINICAL_REPLAY_DEFAULTS.mvdr_load_mode
    assert args.mvdr_auto_kappa == CLINICAL_REPLAY_DEFAULTS.mvdr_auto_kappa
    assert args.msd_lambda == PUBLIC_DETECTOR_DEFAULTS.msd_lambda
    assert args.msd_ridge == PUBLIC_DETECTOR_DEFAULTS.msd_ridge
    assert args.msd_agg == PUBLIC_DETECTOR_DEFAULTS.msd_aggregation
    assert args.msd_ratio_rho == PUBLIC_DETECTOR_DEFAULTS.msd_ratio_rho
    assert args.band_ratio_mode == CLINICAL_REPLAY_DEFAULTS.band_ratio_mode
    assert args.psd_br_flow_low == ADAPTIVE_GUARD_DEFAULTS.flow_band_hz[0]
    assert args.psd_br_flow_high == ADAPTIVE_GUARD_DEFAULTS.flow_band_hz[1]
    assert args.psd_br_alias_center == ADAPTIVE_GUARD_DEFAULTS.alias_center_hz
    assert args.psd_br_alias_width == ADAPTIVE_GUARD_DEFAULTS.alias_width_hz
    assert args.time_window_length == CLINICAL_REPLAY_DEFAULTS.time_window_length
