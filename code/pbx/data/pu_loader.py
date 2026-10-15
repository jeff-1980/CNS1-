"""
Paderborn University (PU) Bearing Dataset Loader — flat file format.

File naming: {CONDITION}_{BEARING_ID}_{IDX}.mat
  e.g.  N15_M07_F10_K001_1.mat

Mat structure:
  rec = mat[stem]  (struct with Info, X, Y, Description)
  rec['X'] = x-axis (time vectors) — NOT the vibration
  rec['Y'] = measurement channels:
      Y[0] force        (Mech_4kHz, 16001 pts)
      Y[1] phase_current_1  (HostService, 256001 pts)
      Y[2] phase_current_2  (HostService, 256001 pts)
      Y[3] speed        (Mech_4kHz, 16001 pts)
      Y[4] temp         (Temp_1Hz, 5 pts)
      Y[5] torque       (Mech_4kHz, 16001 pts)
      Y[6] vibration_1  (HostService, ~256001 pts at ~64kHz) ← USE THIS

Conditions:
  N09_M07_F10  — 900 rpm, 0.7 Nm
  N15_M07_F10  — 1500 rpm, 0.7 Nm  (primary condition)
  N15_M07_F04  — 1500 rpm, 0.7 Nm, reduced radial force
  N15_M01_F10  — 1500 rpm, 0.1 Nm

Bearing IDs → 4 fault classes:
  K001, K002        → 0  Healthy
  KI01, KI03, KI05  → 1  Inner Race
  KA04, KA15        → 2  Outer Race
  KB23, KB24, KB27  → 3  Ball
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio


_LABEL_MAP: Dict[str, int] = {
    "K001": 0, "K002": 0,
    "KI01": 1, "KI03": 1, "KI05": 1,
    "KA04": 2, "KA15": 2,
    "KB23": 3, "KB24": 3, "KB27": 3,
}

_CONDITION_SPEED: Dict[str, int] = {
    "N09_M07_F10": 900,
    "N15_M07_F10": 1500,
    "N15_M07_F04": 1500,
    "N15_M01_F10": 1500,
}

_FILE_RE = re.compile(r'^(N\d+_M\d+_F\d+)_(K[A-Z]?\d+)_(\d+)$')


def _parse_filename(stem: str) -> Optional[Tuple[str, str, int]]:
    m = _FILE_RE.match(stem)
    if m is None:
        return None
    return m.group(1), m.group(2), int(m.group(3))


def _load_vibration(fpath: Path) -> Optional[np.ndarray]:
    """
    Load vibration_1 channel from a PU flat .mat file.
    Returns 1D float32 array (~256001 samples at ~64 kHz).
    """
    try:
        mat = sio.loadmat(str(fpath))
    except Exception:
        return None

    stem = fpath.stem
    if stem not in mat:
        return None

    try:
        rec = mat[stem]
        Y = rec["Y"][0, 0]                   # measurement channels struct
        for i in range(Y.shape[1]):
            ch = Y[0, i]
            name = ch["Name"]
            if name.size > 0 and "vibration" in str(name[0]).lower():
                data = ch["Data"].squeeze()
                return data.astype(np.float32)
    except Exception:
        return None

    return None


def load_pu_flat(
    root: str | Path,
    conditions: Optional[List[str]] = None,
    segment_len: int = 1024,
    overlap: float = 0.5,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load PU flat dataset using vibration_1 channel (~64 kHz).

    Parameters
    ----------
    root       : directory containing *.mat files (flat)
    conditions : list of condition strings; default = all N15_* conditions
    segment_len: samples per segment (1024 @ 64kHz = 16 ms)
    overlap    : segment overlap fraction
    normalize  : per-segment zero-mean / unit-variance

    Returns
    -------
    X : float32 [N, 1, segment_len]
    y : int64   [N]
    """
    root = Path(root)
    if conditions is None:
        conditions = [c for c in _CONDITION_SPEED if _CONDITION_SPEED[c] == 1500]

    step = max(1, int(segment_len * (1 - overlap)))
    all_X, all_y = [], []

    for fpath in sorted(root.glob("*.mat")):
        parsed = _parse_filename(fpath.stem)
        if parsed is None:
            continue
        cond, bid, _ = parsed
        if cond not in conditions:
            continue
        label = _LABEL_MAP.get(bid)
        if label is None:
            continue

        signal = _load_vibration(fpath)
        if signal is None:
            continue

        for s in range(0, len(signal) - segment_len + 1, step):
            seg = signal[s: s + segment_len]
            if normalize:
                seg = (seg - seg.mean()) / (seg.std() + 1e-8)
            all_X.append(seg)
            all_y.append(label)

    if not all_X:
        raise RuntimeError(f"No data loaded from {root}. Check path and conditions.")

    X = np.stack(all_X)[:, np.newaxis, :]     # [N, 1, L]
    y = np.array(all_y, dtype=np.int64)
    return X, y


def load_pu_cross_speed(
    root: str | Path,
    source_condition: str = "N15_M07_F10",
    target_condition: str = "N09_M07_F10",
    **kwargs,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Cross-speed split: train on source (1500 rpm), test on target (900 rpm).
    """
    train = load_pu_flat(root, conditions=[source_condition], **kwargs)
    test  = load_pu_flat(root, conditions=[target_condition], **kwargs)
    return train, test


def load_pu_cross_load(
    root: str | Path,
    source_condition: str = "N15_M07_F10",
    target_condition: str = "N15_M01_F10",
    **kwargs,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Cross-load split: same speed (1500 rpm), different load torque.
    Source: N15_M07_F10 (0.7 Nm) — train + val
    Target: N15_M01_F10 (0.1 Nm) — test only
    """
    train = load_pu_flat(root, conditions=[source_condition], **kwargs)
    test  = load_pu_flat(root, conditions=[target_condition], **kwargs)
    return train, test
