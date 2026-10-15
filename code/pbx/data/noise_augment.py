"""
Noise augmentation for vibration signals.
Supports AWGN and Impulsive noise (Laplacian-based).
"""

import numpy as np
import torch


def add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add Additive White Gaussian Noise at the given SNR (dB).

    signal: [..., L] float array
    """
    sig_power = np.mean(signal ** 2, axis=-1, keepdims=True)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / (snr_linear + 1e-12)
    noise = np.random.randn(*signal.shape) * np.sqrt(noise_power)
    return signal + noise


def add_impulsive(signal: np.ndarray, snr_db: float, p: float = 0.05) -> np.ndarray:
    """
    Add impulsive (Laplacian-based) noise — most relevant for industrial environments.

    p: fraction of samples that receive an impulse (default 5%).
    The impulse amplitude is calibrated to achieve the target SNR.
    """
    sig_power = np.mean(signal ** 2, axis=-1, keepdims=True)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / (snr_linear + 1e-12)

    # Laplace scale parameter b = sqrt(noise_power / 2)
    b = np.sqrt(noise_power / 2.0)

    # Only a fraction p of samples get impulsive noise
    noise = np.zeros_like(signal)
    mask = np.random.rand(*signal.shape) < p
    impulse_vals = np.random.laplace(0, b, size=signal.shape)
    # Scale impulse so that the average affected power matches target
    # (mask fraction p → scale by 1/p to keep overall noise power correct)
    noise = mask * impulse_vals / (p + 1e-6)
    return signal + noise


def add_noise(signal: np.ndarray, snr_db: float, noise_type: str = "awgn") -> np.ndarray:
    """Unified noise addition interface."""
    if noise_type == "awgn":
        return add_awgn(signal, snr_db)
    elif noise_type == "impulsive":
        return add_impulsive(signal, snr_db)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type!r}. Choose 'awgn' or 'impulsive'.")


# ------------------------------------------------------------------ #
# PyTorch-compatible versions (operate on tensors, for online augment) #
# ------------------------------------------------------------------ #

def add_awgn_tensor(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    sig_power = x.pow(2).mean(dim=-1, keepdim=True)
    snr_lin = 10 ** (snr_db / 10.0)
    noise_std = (sig_power / (snr_lin + 1e-12)).sqrt()
    return x + torch.randn_like(x) * noise_std


def add_impulsive_tensor(x: torch.Tensor, snr_db: float, p: float = 0.05) -> torch.Tensor:
    sig_power = x.pow(2).mean(dim=-1, keepdim=True)
    snr_lin = 10 ** (snr_db / 10.0)
    b = ((sig_power / (snr_lin + 1e-12)) / 2.0).sqrt()
    u = torch.rand_like(x)
    # Laplace sample via inverse CDF
    laplace = -b * torch.sign(u - 0.5) * torch.log(1 - 2 * (u - 0.5).abs() + 1e-12)
    mask = (torch.rand_like(x) < p).float()
    return x + mask * laplace / (p + 1e-6)
