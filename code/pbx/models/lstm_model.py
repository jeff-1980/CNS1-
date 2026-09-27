"""Baseline: Bi-LSTM with unified embedding."""

import torch.nn as nn
from .common import ConvEmbedding


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        d_model: int = 64,
        hidden: int = 128,
        num_layers: int = 2,
        num_classes: int = 4,
    ):
        super().__init__()
        self.embed = ConvEmbedding(in_channels, d_model)
        self.lstm = nn.LSTM(
            d_model, hidden, num_layers,
            batch_first=True, bidirectional=True, dropout=0.1
        )
        self.classifier = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        x = self.embed(x).transpose(1, 2)      # [B, L/2, d_model]
        out, _ = self.lstm(x)
        out = out.mean(dim=1)
        return self.classifier(out)
