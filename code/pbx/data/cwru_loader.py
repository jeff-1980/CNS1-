"""
CWRU (Case Western Reserve University) Bearing Dataset Loader.

Expected directory layout (download from CWRU website):
    <root>/
        Normal/           97.mat, 98.mat, 99.mat, 100.mat
        IR007/            105.mat, 106.mat, 107.mat, 108.mat   (Inner race 0.007")
        IR014/            169.mat, 170.mat, 171.mat, 172.mat
        IR021/            209.mat, 210.mat, 211.mat, 212.mat
        B007/             118.mat, 119.mat, 120.mat, 121.mat   (Ball 0.007")
        B014/             185.mat, 186.mat, 187.mat, 188.mat
        B021/             222.mat, 223.mat, 224.mat, 225.mat
        OR007@6/          130.mat, 131.mat, 132.mat, 133.mat   (Outer race @6 o'clock)
        OR014@6/          197.mat, 198.mat, 199.mat, 200.mat
        OR021@6/          234.mat, 235.mat, 236.mat, 237.mat

4 classes: 0=Normal, 1=Inner Race, 2=Outer Race, 3=Ball
Load condition: 0hp (idx 0), 1hp (1), 2hp (2), 3hp (3)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio


# Canonical 4-class mapping
LABEL_MAP = {
    "Normal": 0,
    "IR": 1,     # Inner Race
    "OR": 2,     # Outer Race
    "Ball": 3,
}

# File numbers per fault type and load (0-3 hp)
_FILE_TABLE: Dict[str, List[int]] = {
    "Normal": [97, 98, 99, 100],
    "IR007":  [105, 106, 107, 108],
    "IR014":  [169, 170, 171, 172],
    "IR021":  [209, 210, 211, 212],
    "B007":   [118, 119, 120, 121],
    "B014":   [185, 186, 187, 188],
    "B021":   [222, 223, 224, 225],
    "OR007@6": [130, 131, 132, 133],
    "OR014@6": [197, 198, 199, 200],
    "OR021@6": [234, 235, 236, 237],
}

# Fault-type → class label
_FAULT_TO_LABEL = {
    "Normal": 0,
    "IR007": 1, "IR014": 1, "IR021": 1,
    "B007":  3, "B014":  3, "B021":  3,
    "OR007@6": 2, "OR014@6": 2, "OR021@6": 2,
}


def _extract_de_signal(mat: dict) -> Optional[np.ndarray]:
    """Extract the Drive End accelerometer time-series from a .mat dict."""
    for key in mat:
        if re.search(r'DE_time', key, re.IGNORECASE):
            arr = mat[key].squeeze()
            if arr.ndim == 1:
                return arr.astype(np.float32)
    return None


def _load_mat_signal(path: Path) -> Optional[np.ndarray]:
    """Load the drive-end signal of one CWRU file.

    2026-10-13 fix: CWRU 99.mat stores two drive-end arrays, X098_DE_time (a copy
    of 98.mat) and X099_DE_time; the old behaviour took the first match, so the
    training "99" was byte-identical to the test file 98. The key whose number
    matches the file name is now preferred. Set CWRU_LEGACY_FIRST_KEY=1 only to
    reproduce results archived before the fix (release v1.0).
    """
    try:
        mat = sio.loadmat(str(path))
    except Exception:
        return None
    if os.environ.get("CWRU_LEGACY_FIRST_KEY") != "1":
        stem = Path(path).stem
        if stem.isdigit():
            own = f"X{int(stem):03d}_DE_time"
            if own in mat and mat[own].squeeze().ndim == 1:
                return mat[own].squeeze().astype(np.float32)
    return _extract_de_signal(mat)


def load_cwru(
    root: str | Path,
    loads: List[int] = (0, 1, 2, 3),
    fault_types: Optional[List[str]] = None,
    segment_len: int = 1024,
    overlap: float = 0.5,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and segment CWRU vibration data.

    Parameters
    ----------
    root : path to the CWRU root directory (contains Normal/, IR007/, …)
    loads : which load conditions to include (0=0hp … 3=3hp)
    fault_types : subset of fault types (default: all)
    segment_len : samples per segment
    overlap : overlap fraction between consecutive segments [0, 1)
    normalize : z-score normalise each segment independently

    Returns
    -------
    X : float32 [N, 1, segment_len]
    y : int64 [N]
    """
    root = Path(root)
    fault_types = fault_types or list(_FILE_TABLE.keys())

    step = max(1, int(segment_len * (1 - overlap)))

    all_X, all_y = [], []

    for fault in fault_types:
        if fault not in _FILE_TABLE:
            continue
        label = _FAULT_TO_LABEL[fault]
        folder = root / fault

        for load_idx in loads:
            fnum = _FILE_TABLE[fault][load_idx]
            mat_path = folder / f"{fnum}.mat"
            if not mat_path.exists():
                continue

            signal = _load_mat_signal(mat_path)
            if signal is None:
                continue

            # Segment with overlap
            starts = range(0, len(signal) - segment_len + 1, step)
            for s in starts:
                seg = signal[s: s + segment_len]
                if normalize:
                    mu, std = seg.mean(), seg.std()
                    seg = (seg - mu) / (std + 1e-8)
                all_X.append(seg)
                all_y.append(label)

    X = np.stack(all_X, axis=0)[:, np.newaxis, :]   # [N, 1, L]
    y = np.array(all_y, dtype=np.int64)
    return X, y


# ------------------------------------------------------------------ #
# Flat layout loader (all .mat files directly in one directory)        #
# ------------------------------------------------------------------ #

# Maps file number → (fault_type, load_idx)
_NUM_TO_FAULT: Dict[int, Tuple[str, int]] = {}
for _ft, _nums in _FILE_TABLE.items():
    for _li, _fn in enumerate(_nums):
        _NUM_TO_FAULT[_fn] = (_ft, _li)


def load_cwru_flat(
    root: str | Path,
    loads: List[int] = (0, 1, 2, 3),
    fault_types: Optional[List[str]] = None,
    segment_len: int = 1024,
    overlap: float = 0.5,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CWRU data from a flat directory (all .mat files in one folder).
    Auto-detects class from filename number (e.g. 97.mat → Normal).

    root : directory containing *.mat files directly (no subfolders needed)
    """
    root = Path(root)
    fault_types = fault_types or list(_FILE_TABLE.keys())
    step = max(1, int(segment_len * (1 - overlap)))
    all_X, all_y = [], []

    for mat_path in sorted(root.glob("*.mat")):
        try:
            fnum = int(mat_path.stem)
        except ValueError:
            continue
        if fnum not in _NUM_TO_FAULT:
            continue
        fault, load_idx = _NUM_TO_FAULT[fnum]
        if fault not in fault_types or load_idx not in loads:
            continue

        signal = _load_mat_signal(mat_path)
        if signal is None:
            continue

        label = _FAULT_TO_LABEL[fault]
        starts = range(0, len(signal) - segment_len + 1, step)
        for s in starts:
            seg = signal[s: s + segment_len]
            if normalize:
                seg = (seg - seg.mean()) / (seg.std() + 1e-8)
            all_X.append(seg)
            all_y.append(label)

    if not all_X:
        raise RuntimeError(
            f"No data loaded from {root}. "
            "Check that .mat files are named by CWRU file number (e.g. 97.mat, 105.mat)."
        )

    X = np.stack(all_X, axis=0)[:, np.newaxis, :]
    y = np.array(all_y, dtype=np.int64)
    return X, y


def load_cwru_flat_cross_load(
    root: str | Path,
    source_load: int = 0,
    target_load: int = 3,
    **kwargs,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    train = load_cwru_flat(root, loads=[source_load], **kwargs)
    test  = load_cwru_flat(root, loads=[target_load], **kwargs)
    return train, test


def load_cwru_cross_load(
    root: str | Path,
    source_load: int = 0,
    target_load: int = 3,
    **kwargs,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Cross-load split: train on source_load, test on target_load.
    Returns (X_train, y_train), (X_test, y_test).
    """
    train = load_cwru(root, loads=[source_load], **kwargs)
    test = load_cwru(root, loads=[target_load], **kwargs)
    return train, test
