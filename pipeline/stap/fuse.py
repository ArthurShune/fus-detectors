from __future__ import annotations

import torch


def lse_fuse(S: torch.Tensor, tau: float = 2.0) -> torch.Tensor:
    """
    Log-sum-exp fusion (soft max) across steering dimension.
    S: (F, N) non-negative energies.
    tau: temperature. Lower tau -> sharper max, higher -> averaging.
    """
    if S.dim() != 2:
        raise ValueError("S must be (F, N)")
    if tau <= 0:
        raise ValueError("tau must be positive")
    Z = S / tau
    m = torch.amax(Z, dim=0, keepdim=True)
    out = torch.sum(torch.exp(Z - m), dim=0)
    return out * torch.exp(m.squeeze(0))
