"""Random helper utilities."""

from __future__ import annotations

import numpy as np


def seed_all(seed: int) -> None:
    """Seed numpy's RNG."""
    np.random.seed(seed)
