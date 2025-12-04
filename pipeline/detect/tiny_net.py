"""Tiny torch model placeholder."""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    """One hidden-layer CNN for smoke tests."""

    def __init__(self, in_ch: int = 1, hidden: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
