"""Stable install-facing API for fus-detectors."""

from .api import (
    DetectorConfig,
    DetectorResult,
    DetectorSummary,
    score_residual_batch,
    score_residual_cube,
)

__all__ = [
    "DetectorConfig",
    "DetectorResult",
    "DetectorSummary",
    "score_residual_cube",
    "score_residual_batch",
]
