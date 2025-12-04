from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class RhoInflateDecision:
    rho_hat: float
    rho_groups: Optional[float]
    motion_um: Optional[float]
    dropout: Optional[float]
    delta: float
    rho_eff: float


def rho_inflate_policy(
    rho_hat: float,
    rho_groups: Optional[float] = None,
    motion_um: Optional[float] = None,
    dropout: Optional[float] = None,
    base: float = 0.03,
    cap: float = 0.10,
) -> RhoInflateDecision:
    """
    Conservative heuristic for Confirm-2 rho inflation.

    Increments:
      +0.02 if rho_hat > 0.55
      +0.02 if rho_groups > 0.35
      +0.02 if motion_um >= 120
      +0.01 if dropout >= 0.10
    """
    delta = float(base)
    if rho_hat > 0.55:
        delta += 0.02
    if rho_groups is not None and rho_groups > 0.35:
        delta += 0.02
    if motion_um is not None and motion_um >= 120.0:
        delta += 0.02
    if dropout is not None and dropout >= 0.10:
        delta += 0.01

    delta = float(np.clip(delta, 0.0, cap))
    rho_eff = float(np.clip(rho_hat + delta, -0.999, 0.999))
    return RhoInflateDecision(
        rho_hat=float(rho_hat),
        rho_groups=None if rho_groups is None else float(rho_groups),
        motion_um=None if motion_um is None else float(motion_um),
        dropout=None if dropout is None else float(dropout),
        delta=delta,
        rho_eff=rho_eff,
    )
