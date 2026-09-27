"""
Baseline: DRSN — Deep Residual Shrinkage Network (Zhao et al., TII 2020).
Ref: Zhao M, et al. IEEE Trans. Ind. Inform. 16(7):4681-4690, 2020.
"""

import torch
import torch.nn as nn
from .common import ConvEmbedding


class ShrinkageBlock(nn.Module):
    """
    Residual block with channel-wise soft-thresholding.
    Threshold is learned adaptively via a sub-network (attention over absolute value).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)

        # Threshold sub-network: global avg → FC → sigmoid → scale by mean(|x|)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels // 4)
        self.fc2 = nn.Linear(channels // 4, channels)
        self.act = nn.GELU()

    def _soft_threshold(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        abs_mean = self.gap(x.abs()).squeeze(-1)          # [B, C]
        t = torch.sigmoid(self.fc2(self.act(self.fc1(abs_mean))))  # [B, C]
        threshold = t * abs_mean                           # [B, C]
        threshold = threshold.unsqueeze(-1)                # [B, C, 1]
        return torch.sign(x) * torch.relu(x.abs() - threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(self.act(self.bn1(x)))
        out = self.conv2(self.act(self.bn2(out)))
        out = self._soft_threshold(out)
        return out + residual


class DRSN(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        d_model: int = 64,
        num_blocks: int = 4,
        num_classes: int = 4,
    ):
        super().__init__()
        self.embed = ConvEmbedding(in_channels, d_model)
        self.blocks = nn.Sequential(*[ShrinkageBlock(d_model) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embed(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)
