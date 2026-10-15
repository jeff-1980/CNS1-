"""
Leak-free train/val/test splitting for CWRU and JNU Mode A.

This module is NEW code that sits alongside (does not modify) the existing
`cwru_loader.py` / `jnu_loader.py`. It fixes the window-adjacency leakage
documented in results/leakage_audit/report.md:

  - CWRU main pipeline: previously windowed the whole pooled signal first,
    then randomly split individual (overlapping) windows into train/val,
    with no held-out test set at all.
  - JNU Mode A: previously windowed the whole single-rpm recording first,
    then randomly split individual (overlapping) windows into train/val/test,
    so test windows could be the direct 512-sample overlapping neighbor of a
    train window.

Fix strategy:
  - CWRU: split at the *source .mat file* level (one file == one
    (fault_type, load) combination, which already encodes fault type, fault
    size, and load). All windows from a given file go to exactly one split.
    Windowing happens independently within each split's set of files, so no
    window can share samples with a window in another split.
  - JNU Mode A: split each (class, rpm) file along the time axis into
    contiguous 60/20/20 train/val/test blocks *before* windowing, with a
    ``GAP_SAMPLES``-sample gap dropped at each boundary. Windowing then
    happens independently within each block.

Both pipelines record window provenance (file id + [start, end) sample
range) so `verify_no_leakage()` can assert, from first principles, that no
window in one split shares a single sample with a window in another split.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .cwru_loader import _FAULT_TO_LABEL, _load_mat_signal, _NUM_TO_FAULT
from .jnu_loader import _parse_stem, _load_signal, _PREFIX_LABEL

SPLIT_SEED_DEFAULT = 42       # decoupled from the training seed on purpose
GAP_SAMPLES = 1024            # JNU inter-segment gap (== segment_len)

SplitName = str  # 'train' | 'val' | 'test'


@dataclass
class WindowRecord:
    """Provenance for a single window, used only for leakage verification."""
    file_id: str
    split: SplitName
    start: int
    end: int


@dataclass
class SplitBundle:
    X: Dict[SplitName, np.ndarray] = field(default_factory=dict)
    y: Dict[SplitName, np.ndarray] = field(default_factory=dict)
    windows: List[WindowRecord] = field(default_factory=list)


def _window_starts(n_samples: int, segment_len: int, step: int) -> List[int]:
    return list(range(0, n_samples - segment_len + 1, step))


def _segment_and_window(
    signal: np.ndarray,
    file_id: str,
    split: SplitName,
    label: int,
    segment_len: int,
    step: int,
    normalize: bool,
    base_offset: int,
    X_out: List[np.ndarray],
    y_out: List[int],
    windows_out: List[WindowRecord],
) -> None:
    """Window one contiguous raw-signal block and append results in place.

    `base_offset` is this block's absolute start index within the original
    file, so provenance ranges are always expressed in the file's own
    coordinate system (comparable across splits for leakage checking).
    """
    for s in _window_starts(len(signal), segment_len, step):
        seg = signal[s: s + segment_len]
        if normalize:
            mu, sd = seg.mean(), seg.std()
            seg = (seg - mu) / (sd + 1e-8)
        X_out.append(seg)
        y_out.append(label)
        windows_out.append(WindowRecord(
            file_id=file_id, split=split,
            start=base_offset + s, end=base_offset + s + segment_len,
        ))


# --------------------------------------------------------------------- #
# CWRU: group split by source .mat file (fault_type, fault_size, load)  #
# --------------------------------------------------------------------- #

def _stratified_group_split(
    groups_by_label: Dict[int, List[str]],
    train_frac: float,
    val_frac: float,
    rng: np.random.RandomState,
) -> Tuple[set, set, set]:
    """Split group ids (not windows) per label into train/val/test sets.

    Guarantees at least one group per split whenever the class has >=3
    distinct source files, so every split sees every class.
    """
    train_ids, val_ids, test_ids = set(), set(), set()
    for label, gids in groups_by_label.items():
        gids = list(gids)
        rng.shuffle(gids)
        n = len(gids)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        n_test = n - n_train - n_val
        # Guarantee non-empty val/test when there is enough material to do so.
        if n_test == 0 and n - n_train >= 1:
            n_val = max(0, n_val - 1)
            n_test = 1
        if n_val == 0 and n - n_train - n_test >= 1:
            n_val = 1
            n_test = n - n_train - n_val
        train_ids.update(gids[:n_train])
        val_ids.update(gids[n_train:n_train + n_val])
        test_ids.update(gids[n_train + n_val:n_train + n_val + n_test])
    return train_ids, val_ids, test_ids


def build_cwru_splits(
    root: str | Path,
    loads: List[int] = (0, 1, 2, 3),
    fault_types: Optional[List[str]] = None,
    segment_len: int = 1024,
    overlap: float = 0.5,
    normalize: bool = True,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    split_seed: int = SPLIT_SEED_DEFAULT,
) -> SplitBundle:
    """Leak-free CWRU split: group = one source .mat file (fault_type,
    fault_size, load). All windows from a file land in exactly one split;
    windowing happens after the group assignment, never before.
    """
    root = Path(root)
    fault_types = set(fault_types) if fault_types else None
    step = max(1, int(segment_len * (1 - overlap)))

    # Discover files present on disk (flat layout: <fnum>.mat), same
    # convention as cwru_loader.load_cwru_flat.
    files_by_group: Dict[str, Path] = {}
    label_of_group: Dict[str, int] = {}
    for mat_path in sorted(root.glob("*.mat")):
        try:
            fnum = int(mat_path.stem)
        except ValueError:
            continue
        if fnum not in _NUM_TO_FAULT:
            continue
        fault, load_idx = _NUM_TO_FAULT[fnum]
        if fault_types is not None and fault not in fault_types:
            continue
        if load_idx not in loads:
            continue
        group_id = f"{fault}_load{load_idx}"
        files_by_group[group_id] = mat_path
        label_of_group[group_id] = _FAULT_TO_LABEL[fault]

    if not files_by_group:
        raise RuntimeError(f"No CWRU files found under {root} for loads={loads}")

    groups_by_label: Dict[int, List[str]] = {}
    for gid, label in label_of_group.items():
        groups_by_label.setdefault(label, []).append(gid)

    rng = np.random.RandomState(split_seed)
    train_ids, val_ids, test_ids = _stratified_group_split(
        groups_by_label, train_frac, val_frac, rng
    )
    assign = {}
    for gid in train_ids:
        assign[gid] = "train"
    for gid in val_ids:
        assign[gid] = "val"
    for gid in test_ids:
        assign[gid] = "test"

    bundle = SplitBundle()
    buf_X = {"train": [], "val": [], "test": []}
    buf_y = {"train": [], "val": [], "test": []}

    for gid, mat_path in files_by_group.items():
        split = assign.get(gid)
        if split is None:
            continue  # group excluded (shouldn't happen given full assignment above)
        signal = _load_mat_signal(mat_path)
        if signal is None:
            continue
        label = label_of_group[gid]
        _segment_and_window(
            signal, file_id=gid, split=split, label=label,
            segment_len=segment_len, step=step, normalize=normalize,
            base_offset=0,
            X_out=buf_X[split], y_out=buf_y[split], windows_out=bundle.windows,
        )

    for split in ("train", "val", "test"):
        if not buf_X[split]:
            raise RuntimeError(f"CWRU split '{split}' is empty — check train/val_frac and file counts")
        bundle.X[split] = np.stack(buf_X[split], axis=0)[:, np.newaxis, :]
        bundle.y[split] = np.array(buf_y[split], dtype=np.int64)

    return bundle


# --------------------------------------------------------------------- #
# JNU Mode A: contiguous time-axis split within each (class, rpm) file  #
# --------------------------------------------------------------------- #

def build_jnu_modeA_splits(
    root: str | Path,
    rpm: int,
    segment_len: int = 1024,
    overlap: float = 0.5,
    normalize: bool = True,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    gap_samples: int = GAP_SAMPLES,
) -> SplitBundle:
    """Leak-free JNU Mode A split: for each (class, rpm) file, cut the raw
    signal into contiguous [0, tr_end) / [tr_end+gap, val_end) /
    [val_end+gap, len) train/val/test blocks *before* windowing, with a
    `gap_samples`-sample gap dropped at each boundary. Each block is then
    windowed independently, so no window can straddle a split boundary or
    share samples with a window from another split.
    """
    root = Path(root)
    step = max(1, int(segment_len * (1 - overlap)))

    bundle = SplitBundle()
    buf_X = {"train": [], "val": [], "test": []}
    buf_y = {"train": [], "val": [], "test": []}

    n_files = 0
    for fpath in sorted(root.glob("*.csv")):
        parsed = _parse_stem(fpath.stem)
        if parsed is None:
            continue
        prefix, file_rpm = parsed
        if file_rpm != rpm:
            continue
        label = _PREFIX_LABEL.get(prefix)
        if label is None:
            continue

        signal = _load_signal(fpath)
        n = len(signal)
        tr_end = int(n * train_frac)
        val_start = tr_end + gap_samples
        val_len = int(n * val_frac)
        val_end = val_start + val_len
        test_start = val_end + gap_samples
        if test_start >= n:
            raise RuntimeError(
                f"{fpath}: signal too short ({n} samples) for 60/20/20 split "
                f"with {gap_samples}-sample gaps"
            )

        file_id = fpath.stem
        _segment_and_window(
            signal[0:tr_end], file_id, "train", label,
            segment_len, step, normalize, base_offset=0,
            X_out=buf_X["train"], y_out=buf_y["train"], windows_out=bundle.windows,
        )
        _segment_and_window(
            signal[val_start:val_end], file_id, "val", label,
            segment_len, step, normalize, base_offset=val_start,
            X_out=buf_X["val"], y_out=buf_y["val"], windows_out=bundle.windows,
        )
        _segment_and_window(
            signal[test_start:n], file_id, "test", label,
            segment_len, step, normalize, base_offset=test_start,
            X_out=buf_X["test"], y_out=buf_y["test"], windows_out=bundle.windows,
        )
        n_files += 1

    if n_files == 0:
        raise RuntimeError(f"No JNU files found under {root} for rpm={rpm}")

    for split in ("train", "val", "test"):
        if not buf_X[split]:
            raise RuntimeError(f"JNU split '{split}' is empty for rpm={rpm}")
        bundle.X[split] = np.stack(buf_X[split], axis=0)[:, np.newaxis, :]
        bundle.y[split] = np.array(buf_y[split], dtype=np.int64)

    return bundle


# --------------------------------------------------------------------- #
# Leakage verification                                                  #
# --------------------------------------------------------------------- #

def verify_no_leakage(bundle: SplitBundle, name: str = "") -> None:
    """Assert that no two windows from different splits share a single
    sample of the same source file. Raises AssertionError on any overlap;
    prints a summary log on success.

    Method: group window records by file_id, sort by start within each
    file, sweep to find overlapping [start, end) intervals, and check that
    any overlapping pair belongs to the *same* split. Any overlapping pair
    from two different splits is a leak.
    """
    by_file: Dict[str, List[WindowRecord]] = {}
    for w in bundle.windows:
        by_file.setdefault(w.file_id, []).append(w)

    n_checked_pairs = 0
    leaks = []
    for file_id, recs in by_file.items():
        recs_sorted = sorted(recs, key=lambda r: r.start)
        for i in range(len(recs_sorted)):
            for j in range(i + 1, len(recs_sorted)):
                a, b = recs_sorted[i], recs_sorted[j]
                if b.start >= a.end:
                    break  # sorted by start; no further j can overlap a
                n_checked_pairs += 1
                if a.split != b.split:
                    leaks.append((file_id, a, b))

    split_counts = {s: sum(1 for w in bundle.windows if w.split == s) for s in ("train", "val", "test")}
    file_split_counts = {
        s: len({w.file_id for w in bundle.windows if w.split == s}) for s in ("train", "val", "test")
    }

    if leaks:
        detail = "\n".join(
            f"  file={fid} split_a={a.split}[{a.start},{a.end}) split_b={b.split}[{b.start},{b.end})"
            for fid, a, b in leaks[:10]
        )
        raise AssertionError(
            f"[verify_no_leakage:{name}] FOUND {len(leaks)} cross-split overlapping window "
            f"pairs (showing up to 10):\n{detail}"
        )

    print(
        f"[verify_no_leakage:{name}] PASSED — {len(bundle.windows)} windows, "
        f"{n_checked_pairs} overlapping-interval pairs checked, 0 cross-split leaks. "
        f"windows(train/val/test)={split_counts} files(train/val/test)={file_split_counts}"
    )
