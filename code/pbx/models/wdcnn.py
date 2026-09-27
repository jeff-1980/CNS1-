"""Baseline: WDCNN (Zhang et al., Sensors 2017) with unified embedding."""

import torch.nn as nn
from .common import ConvEmbedding


class WDCNN(nn.Module):
    """
    Wide-kernel first layer for noise suppression (original: kernel=64, stride=16).
    We keep the wide first conv *after* the shared embedding so the embedding remains
    the single fair entry point — the wide conv is the WDCNN-specific feature.
    """

    def __init__(self, in_channels: int = 2, d_model: int = 64, num_classes: int = 4):
        super().__init__()
        self.embed = ConvEmbedding(in_channels, d_model)
        self.wide = nn.Sequential(
            nn.Conv1d(d_model, 32, kernel_size=32, stride=4, padding=14), nn.BatchNorm1d(32), nn.GELU(),
            nn.MaxPool1d(2),
        )
        self.deep = nn.Sequential(
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embed(x)
        x = self.wide(x)
        x = self.deep(x).squeeze(-1)
        return self.classifier(x)
