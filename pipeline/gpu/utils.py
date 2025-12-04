"""GPU utility functions."""

from __future__ import annotations

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None


def asarray(x):
    """Convert numpy array to CuPy array when possible."""
    if cp is None:
        raise RuntimeError("CuPy not available")
    return cp.asarray(x)
