"""
Unified PyTorch Dataset wrapper.
Supports online noise injection at configurable SNR.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .noise_augment import add_awgn_tensor, add_impulsive_tensor


class BearingDataset(Dataset):
    """
    Wraps pre-loaded (X, y) arrays and optionally adds noise online.

    X : float32 [N, C, L]
    y : int64   [N]

    noise_type : None | 'awgn' | 'impulsive'
    snr_db     : fixed SNR (float) or list of SNR values to sample from
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        noise_type: Optional[str] = None,
        snr_db: float | List[float] = 0.0,
    ):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.noise_type = noise_type
        self.snr_db = snr_db if isinstance(snr_db, list) else [snr_db]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].clone()

        if self.noise_type is not None:
            snr = float(np.random.choice(self.snr_db))
            if self.noise_type == "awgn":
                x = add_awgn_tensor(x, snr)
            elif self.noise_type == "impulsive":
                x = add_impulsive_tensor(x, snr)

        return x, self.y[idx]


def make_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 64,
    noise_type: Optional[str] = None,
    snr_db: float | List[float] = 0.0,
    num_workers: int = 4,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build train and validation DataLoaders with optional online noise."""
    train_ds = BearingDataset(X_train, y_train, noise_type=noise_type, snr_db=snr_db)
    val_ds = BearingDataset(X_val, y_val, noise_type=None)   # no noise at val time

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_dl, val_dl
