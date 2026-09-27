"""Baseline: 1D-CNN with unified embedding."""

import torch.nn as nn
from .common import ConvEmbedding


class CNN1D(nn.Module):
    def __init__(self, in_channels: int = 2, d_model: int = 64, num_classes: int = 4):
        super().__init__()
        self.embed = ConvEmbedding(in_channels, d_model)
        self.backbone = nn.Sequential(
            nn.Conv1d(d_model, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.GELU(),
            nn.Conv1d(256, 256, 3, padding=1), nn.BatchNorm1d(256), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.embed(x)
        x = self.backbone(x).squeeze(-1)
        return self.classifier(x)
