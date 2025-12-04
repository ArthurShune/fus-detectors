"""Calibration thresholds stubs."""

from __future__ import annotations

import numpy as np


def quantile(values: np.ndarray, q: float) -> float:
    """Return q-quantile using numpy."""
    return float(np.quantile(values, q))
