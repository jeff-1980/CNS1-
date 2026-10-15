"""
JNU (Jiangnan University) Bearing Dataset Loader.

12 CSV files, single-column float, no header.
Naming convention:
  n{rpm}_3_2.csv   → Normal (class 0)
  ib{rpm}_2.csv    → Inner Race (class 1)
  ob{rpm}_2.csv    → Outer Race (class 2)
  tb{rpm}_2.csv    → Ball / Rolling Element (class 3)

Specs:
  PCB MA352A60, vertical, 50 kHz, ~10 s per fault / ~30 s per Normal condition.
  File lengths: Normal ≈ 1,501,500 samples; Fault ≈ 500,500 samples.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


CLASS_NAMES = ["Normal", "InnerRace", "OuterRace", "Ball"]

# prefix → label
_PREFIX_LABEL: Dict[str, int] = {
    "n":  0,   # Normal
    "ib": 1,   # Inner Race
    "ob": 2,   # Outer Race
    "tb": 3,   # Ball
}

# Regex: optional leading alpha chars, then rpm digits, optional extra tags, _2.csv
_FILE_RE = re.compile(r'^([a-z]+?)(\d+)(?:_3)?_2$')


def _parse_stem(stem: str) -> Optional[Tuple[str, int]]:
    """Return (prefix, rpm) or None if stem doesn't match JNU naming."""
    m = _FILE_RE.match(stem)
    if m is None:
        return None
    return m.group(1), int(m.group(2))


def _load_signal(fpath: Path) -> np.ndarray:
    """Load single-column float32 CSV (no header)."""
    arr = pd.read_csv(str(fpath), header=None).values.squeeze()
    return arr.astype(np.float32)


def load_jnu(
    root: str | Path,
    speeds: List[int],
    segment_len: int = 1024,
    overlap: float = 0.5,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load JNU segments for specified RPM speeds.

    Parameters
    ----------
    root        : directory containing *.csv files
    speeds      : list of rpm values to include, e.g. [600] or [600, 1000]
    segment_len : samples per segment
    overlap     : stride fraction overlap [0, 1)
    normalize   : z-score per segment

    Returns
    -------
    X : float32 [N, 1, segment_len]
    y : int64   [N]
    """
    root = Path(root)
    stride = max(1, int(segment_len * (1 - overlap)))
    all_X, all_y = [], []

    for fpath in sorted(root.glob("*.csv")):
        parsed = _parse_stem(fpath.stem)
        if parsed is None:
            continue
        prefix, rpm = parsed
        if rpm not in speeds:
            continue
        label = _PREFIX_LABEL.get(prefix)
        if label is None:
            continue

        signal = _load_signal(fpath)
        for s in range(0, len(signal) - segment_len + 1, stride):
            seg = signal[s: s + segment_len]
            if normalize:
                seg = (seg - seg.mean()) / (seg.std() + 1e-8)
            all_X.append(seg)
            all_y.append(label)

    if not all_X:
        raise RuntimeError(
            f"No data loaded from {root}. Check path and speed values {speeds}.")

    X = np.stack(all_X)[:, np.newaxis, :]
    y = np.array(all_y, dtype=np.int64)
    return X, y


def load_jnu_mode_a(
    root: str | Path,
    train_rpm: int = 600,
    segment_len: int = 1024,
    overlap: float = 0.5,
    seed: int = 0,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """
    Mode A — in-domain noise test.
    80/10/10 stratified split on train_rpm segments.

    Returns (X_tr, y_tr), (X_val, y_val), (X_te, y_te)
    """
    X, y = load_jnu(root, speeds=[train_rpm],
                    segment_len=segment_len, overlap=overlap)

    # First split off 10% test
    idx = np.arange(len(y))
    idx_tv, idx_te = train_test_split(idx, test_size=0.10,
                                       stratify=y, random_state=seed)
    # Then split remaining 90% into 80% train + 10% val (i.e. 8/9 vs 1/9 of tv)
    idx_tr, idx_val = train_test_split(idx_tv, test_size=1/9,
                                        stratify=y[idx_tv], random_state=seed)
    return (X[idx_tr], y[idx_tr]), (X[idx_val], y[idx_val]), (X[idx_te], y[idx_te])


def load_jnu_mode_b(
    root: str | Path,
    train_rpm: int = 600,
    test_rpm:  int = 1000,
    segment_len: int = 1024,
    overlap: float = 0.5,
    seed: int = 0,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """
    Mode B — cross-speed (67% speed shift: 600→1000 rpm).
    Train/val: all train_rpm segments (90/10 split).
    Test: all test_rpm segments.

    Returns (X_tr, y_tr), (X_val, y_val), (X_te, y_te)
    """
    X_src, y_src = load_jnu(root, speeds=[train_rpm],
                              segment_len=segment_len, overlap=overlap)
    X_te,  y_te  = load_jnu(root, speeds=[test_rpm],
                              segment_len=segment_len, overlap=overlap)

    idx = np.arange(len(y_src))
    idx_tr, idx_val = train_test_split(idx, test_size=0.10,
                                        stratify=y_src, random_state=seed)
    return (X_src[idx_tr], y_src[idx_tr]), (X_src[idx_val], y_src[idx_val]), (X_te, y_te)
