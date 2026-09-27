"""Paderborn (PU) leak-free splits — 阶段 1′ 的第二个数据集。

设计约束（实测确认，非假定）：
  - 27 个损伤代码，每个 80 个 .mat（4 工况 × 20 次重复）。
  - 每个 .mat 的 `vibration_1` 通道 = 256000 样本（64 kHz × 4 s）。
  - **PU 没有 Ball 类**：K00x=健康，KAxx=外圈，KIxx=内圈，KBxx=复合（排除）。
    所以 PU 是 3 类，JNU 是 4 类 —— 这是数据集事实，不是选择。

与 JNU 的关键差别（更严，不是更松）：
  JNU Mode A 每类只有 1 个文件，只能沿时间轴切 60/20/20。
  PU 每类有多个**轴承个体**，因此按 code（个体）分组切分：
  train/val/test 用不同的轴承。窗口不可能跨 split 共享样本，
  且模型必须泛化到没见过的个体。

类平衡：每类取相同数量的代码 × 相同数量的文件 → 严格平衡。
这是 VERDICT_cns_value §5 风险 2 要求的（JNU-a 不平衡比 3.005）。
"""
from __future__ import annotations
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PAPERB_CODE = _os.environ.get("PAPERB_CODE", _os.path.join(_ROOT, "external/paperB_code"))
_PAPERB_RESULTS = _os.environ.get("PAPERB_RESULTS", _os.path.join(_ROOT, "external/paperB_results"))
_CWRU40 = _os.environ.get("CWRU40_DIR", _os.path.join(_ROOT, "data/cwru_12k_de"))
_CWRU16 = _os.environ.get("CWRU16_DIR", _os.path.join(_ROOT, "data/cwru_16"))
_PU = _os.environ.get("PU_DIR", _os.path.join(_ROOT, "data/paderborn"))
_JNU = _os.environ.get("JNU_DIR", _os.path.join(_ROOT, "data/jnu"))
_JNU_RAW = _os.environ.get("JNU_RAW_DIR", _JNU)

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import scipy.io as sio

import sys
sys.path.insert(0, str(Path(__file__).parent / "pbx"))
from data.splits_v2 import (SplitBundle, WindowRecord, _segment_and_window,
                            GAP_SAMPLES)

PU_ROOT = _PU

# 真实加速寿命损伤（PU 官方文档表 4）。KB* = 复合损伤，排除：
# 它同时含 IR+OR，标签语义与其他类不可比。
HEALTHY = ["K001", "K002", "K003", "K004", "K005"]
OUTER   = ["KA04", "KA15", "KA16", "KA22", "KA30"]   # real OR damage
INNER   = ["KI04", "KI14", "KI16", "KI17", "KI18", "KI21"]  # real IR damage

CLASS_NAMES_PU = {0: "Healthy", 1: "OuterRace", 2: "InnerRace"}

_PAT = re.compile(r"_(K[AIB]?\d{2,3})_(\d+)\.mat$")


def _load_pu_signal(fpath: Path) -> np.ndarray:
    """取 vibration_1 通道。结构：m[<stem>].Y[i].Name == 'vibration_1'。"""
    m = sio.loadmat(str(fpath), squeeze_me=True, struct_as_record=False)
    key = [k for k in m if not k.startswith("__")]
    if len(key) != 1:
        raise RuntimeError(f"{fpath}: unexpected keys {key}")
    for y in m[key[0]].Y:
        if str(y.Name) == "vibration_1":
            return np.asarray(y.Data, dtype=np.float64).ravel()
    raise RuntimeError(f"{fpath}: no vibration_1 channel")


def _codes_present(root: Path) -> Dict[str, List[Path]]:
    """code -> 文件列表（去重：同名 stem 只留一份，KI01 有重复副本）。"""
    out: Dict[str, Dict[str, Path]] = {}
    for f in root.rglob("*.mat"):
        m = _PAT.search(f.name)
        if m:
            out.setdefault(m.group(1), {}).setdefault(f.stem, f)
    return {c: sorted(d.values()) for c, d in out.items()}


def build_pu_time_splits(
    root: str | Path = PU_ROOT,
    op: str = "N15_M07_F10",
    segment_len: int = 1024,
    overlap: float = 0.5,
    normalize: bool = True,
    codes_per_class: int = 3,
    files_per_code: int = 6,
    balanced: bool = True,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    gap_samples: int = GAP_SAMPLES,
) -> SplitBundle:
    """**与 JNU Mode A 同口径**：每个文件沿时间轴切 60/20/20，块间留 gap，
    再各自开窗。所有轴承个体都出现在三个 split 里。

    为什么用这个而非 `build_pu_splits`（个体分组）：
      个体分组下模型训不起来（wdcnn 40 epoch val=0.243 < 随机 0.333；
      手工特征 train 0.982 → 跨个体 val 0.495）。PU 的个体间差异大于
      类间差异，文献里的 PU 结果普遍也用同个体切分。
      **阶段 1′ 要测的是方差份额，前提是模型先能学会任务** ——
      个体泛化是另一个问题，不在本轮命题内。
    """
    root = Path(root)
    step = max(1, int(segment_len * (1 - overlap)))
    avail = _codes_present(root)
    groups = {0: HEALTHY, 1: OUTER, 2: INNER}

    bundle = SplitBundle()
    buf_X: Dict[str, List[np.ndarray]] = {"train": [], "val": [], "test": []}
    buf_y: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    used: Dict[int, List[str]] = {}

    for label, codes in groups.items():
        usable = [c for c in codes if c in avail][:codes_per_class]
        if len(usable) < codes_per_class:
            raise RuntimeError(f"class {label}: only {usable}")
        used[label] = usable
        for code in usable:
            files = [f for f in avail[code] if f.name.startswith(op)]
            for f in files[:files_per_code]:
                sig = _load_pu_signal(f)
                n = len(sig)
                tr_end = int(n * train_frac)
                val_start = tr_end + gap_samples
                val_end = val_start + int(n * val_frac)
                test_start = val_end + gap_samples
                if test_start >= n:
                    raise RuntimeError(f"{f}: too short ({n})")
                for split, (a, z) in {
                        "train": (0, tr_end),
                        "val": (val_start, val_end),
                        "test": (test_start, n)}.items():
                    _segment_and_window(
                        sig[a:z], f.stem, split, label, segment_len, step,
                        normalize, base_offset=a, X_out=buf_X[split],
                        y_out=buf_y[split], windows_out=bundle.windows)

    for split in ("train", "val", "test"):
        if not buf_X[split]:
            raise RuntimeError(f"PU split '{split}' empty")
        bundle.X[split] = np.stack(buf_X[split], axis=0
                                   )[:, np.newaxis, :].astype(np.float32)
        bundle.y[split] = np.array(buf_y[split], dtype=np.int64)

    if balanced:
        rng = np.random.default_rng(42)
        for split in ("train", "val", "test"):
            y = bundle.y[split]
            per = np.bincount(y).min()
            keep = np.concatenate([rng.choice(np.where(y == c)[0], per,
                                              replace=False) for c in range(3)])
            keep.sort()
            bundle.X[split] = bundle.X[split][keep]
            bundle.y[split] = bundle.y[split][keep]

    bundle.pu_assign = used            # type: ignore[attr-defined]
    return bundle


def build_pu_splits(
    root: str | Path = PU_ROOT,
    op: str = "N15_M07_F10",
    segment_len: int = 1024,
    overlap: float = 0.5,
    normalize: bool = True,
    codes_per_class: int = 3,
    files_per_code: int = 6,
    balanced: bool = True,
) -> SplitBundle:
    """按轴承个体（code）分组切分：每类 codes_per_class 个个体，
    第 1 个 → train，第 2 个 → val，第 3 个 → test。

    `op` 选单一工况（默认 N15_M07_F10 = 1500 rpm / 0.7 Nm / 1000 N，
    PU 文档里的标准工况），避免工况混入成为额外方差源。
    """
    root = Path(root)
    step = max(1, int(segment_len * (1 - overlap)))
    avail = _codes_present(root)

    groups = {0: HEALTHY, 1: OUTER, 2: INNER}
    bundle = SplitBundle()
    buf_X: Dict[str, List[np.ndarray]] = {"train": [], "val": [], "test": []}
    buf_y: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    assign: Dict[int, Dict[str, str]] = {}

    for label, codes in groups.items():
        usable = [c for c in codes if c in avail]
        if len(usable) < codes_per_class:
            raise RuntimeError(
                f"class {label}: only {len(usable)} codes present "
                f"({usable}), need {codes_per_class}")
        picked = usable[:codes_per_class]
        assign[label] = dict(zip(("train", "val", "test"), picked))

        for split, code in assign[label].items():
            files = [f for f in avail[code] if f.name.startswith(op)]
            if not files:
                raise RuntimeError(f"{code}: no files for op={op}")
            for f in files[:files_per_code]:
                sig = _load_pu_signal(f)
                _segment_and_window(
                    sig, f.stem, split, label, segment_len, step, normalize,
                    base_offset=0, X_out=buf_X[split], y_out=buf_y[split],
                    windows_out=bundle.windows)

    for split in ("train", "val", "test"):
        if not buf_X[split]:
            raise RuntimeError(f"PU split '{split}' is empty")
        # PU 的 .mat 是 float64；JNU/CWRU 加载器给 float32。统一为 float32，
        # 否则 conv1d 报 DoubleTensor/FloatTensor 不匹配。
        bundle.X[split] = np.stack(buf_X[split], axis=0
                                   )[:, np.newaxis, :].astype(np.float32)
        bundle.y[split] = np.array(buf_y[split], dtype=np.int64)

    if balanced:
        rng = np.random.default_rng(42)
        for split in ("train", "val", "test"):
            y = bundle.y[split]
            per = np.bincount(y).min()
            keep = np.concatenate([
                rng.choice(np.where(y == c)[0], per, replace=False)
                for c in range(len(groups))])
            keep.sort()
            bundle.X[split] = bundle.X[split][keep]
            bundle.y[split] = bundle.y[split][keep]

    bundle.pu_assign = assign          # type: ignore[attr-defined]
    return bundle


def verify_pu_group_split(bundle: SplitBundle) -> None:
    """比 verify_no_leakage 更强：断言没有任何 code 出现在两个 split 里。"""
    by_split: Dict[str, set] = {}
    for w in bundle.windows:
        m = _PAT.search(w.file_id + ".mat")
        code = m.group(1) if m else w.file_id
        by_split.setdefault(w.split, set()).add(code)
    splits = sorted(by_split)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            shared = by_split[a] & by_split[b]
            if shared:
                raise AssertionError(
                    f"PU group leakage: codes {sorted(shared)} in both "
                    f"'{a}' and '{b}'")
    print(f"[PU] group split OK — " +
          " | ".join(f"{s}:{sorted(by_split[s])}" for s in splits))


if __name__ == "__main__":
    b = build_pu_splits()
    verify_pu_group_split(b)
    for s in ("train", "val", "test"):
        y = b.y[s]
        print(f"  {s:5s} X={b.X[s].shape}  per-class={np.bincount(y).tolist()}")
    print("  assign:", b.pu_assign)   # type: ignore[attr-defined]
