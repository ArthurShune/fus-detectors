# pipeline/confirm2/grouping.py
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def spectral_bisection_groups(z_by_angle: List[np.ndarray]) -> Tuple[List[int], List[int], float]:
    """
    Partition angles into two groups that minimise cross-group correlation.

    Parameters
    ----------
    z_by_angle : list of 1D arrays
        Each entry contains z-scores (or any zero-mean proxy) for one angle.

    Returns
    -------
    group1, group2 : list[int]
        Angle indices assigned to each Confirm-2 look.
    rho_groups : float
        Estimated correlation between the mean-z traces of the two groups.
    """
    if len(z_by_angle) < 2:
        raise ValueError("Need at least two angles for grouping.")

    Z = [np.asarray(z, dtype=np.float64).ravel() for z in z_by_angle]
    T = min(len(z) for z in Z)
    if T < 10:
        raise ValueError("Insufficient samples per angle for grouping.")
    Z = [z[:T] for z in Z]
    A = len(Z)

    # Normalise each angle
    normed = []
    for z in Z:
        mu = np.mean(z)
        sig = np.std(z) + 1e-12
        normed.append((z - mu) / sig)

    # Absolute correlation matrix
    C = np.eye(A, dtype=np.float64)
    for i in range(A):
        for j in range(i + 1, A):
            r = float(np.corrcoef(normed[i], normed[j])[0, 1])
            C[i, j] = C[j, i] = abs(r)

    # Graph Laplacian and Fiedler vector
    W = C.copy()
    np.fill_diagonal(W, 0.0)
    degrees = np.sum(W, axis=1)
    L = np.diag(degrees) - W
    eigvals, eigvecs = np.linalg.eigh(L)
    idx = np.argsort(eigvals)
    if len(idx) < 2:
        # Degenerate fallback
        g1 = list(range(0, A, 2))
        g2 = [i for i in range(A) if i not in g1]
    else:
        fiedler = eigvecs[:, idx[1]]
        g1 = [int(i) for i, v in enumerate(fiedler) if v >= 0.0]
        g2 = [int(i) for i, v in enumerate(fiedler) if v < 0.0]
        if len(g1) == 0 or len(g2) == 0:
            # Greedy min-cut fallback
            order = np.argsort(-degrees)
            g1 = [int(order[0])]
            g2: List[int] = []
            for i in order[1:]:
                cut1 = np.sum(W[i, g2]) if g2 else 0.0
                cut2 = np.sum(W[i, g1])
                if cut1 <= cut2:
                    g1.append(int(i))
                else:
                    g2.append(int(i))

    z1 = np.mean(np.stack([normed[i] for i in g1], axis=0), axis=0)
    z2 = np.mean(np.stack([normed[i] for i in g2], axis=0), axis=0)
    rho = float(np.corrcoef(z1, z2)[0, 1])

    return g1, g2, rho
