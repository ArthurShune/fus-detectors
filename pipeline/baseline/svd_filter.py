"""Global spatiotemporal SVD filter placeholder."""

from __future__ import annotations

import numpy as np


def svd_project(data: np.ndarray, rank: int) -> np.ndarray:
    """Subtract the leading ``rank`` singular components."""
    u, _, vh = np.linalg.svd(data, full_matrices=False)
    basis = u[:, :rank]
    return data - basis @ (basis.conj().T @ data)
