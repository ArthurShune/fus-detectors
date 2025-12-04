"""Simple coherent compounding helpers."""

from __future__ import annotations

import numpy as np


def coherent(images: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Return a coherent compound of the provided images."""
    if weights is None:
        weights = np.ones(images.shape[0], dtype=images.dtype)
    weights = weights[:, None, None]
    compound = (weights * images).sum(axis=0)
    denom = np.abs(weights).sum()
    return compound / max(denom, 1e-12)
