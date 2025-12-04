"""Scalar threshold helpers."""

from __future__ import annotations

import numpy as np


def otsu_level(values: np.ndarray) -> float:
    """Return Otsu threshold for a flattened array."""
    hist, bin_edges = np.histogram(values.ravel(), bins=256)
    hist = hist.astype(float)
    prob = hist.cumsum() / hist.sum()
    mean = (hist * bin_edges[:-1]).cumsum() / hist.sum()
    total_mean = mean[-1]
    between = (total_mean * prob - mean) ** 2 / (prob * (1 - prob) + 1e-12)
    return bin_edges[np.nanargmax(between)]
