"""
Unified 1D-Conv embedding shared by all 7 models.
All baselines must use this module to ensure fair comparison.
"""

import torch
import torch.nn as nn


class ConvEmbedding(nn.Module):
    """
    Shared input embedding: Conv1d(kernel=7, stride=2) + BN + GELU.
    Input:  [B, C_in, L]    (raw vibration, C_in channels)
    Output: [B, d_model, L//2]
    """

    def __init__(self, in_channels: int = 2, d_model: int = 64):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, d_model,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    @property
    def out_channels(self) -> int:
        return self.conv.out_channels
