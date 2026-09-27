"""JNU-a 类平衡干预：测「类不平衡 → 种子方差份额」这条因果。

## 为什么需要三臂而不是两臂

JNU-a 的 Normal 恰好是其他三类的 3 倍（3 个文件 vs 各 1 个），
train = {0:1758, 1:585, 2:585, 3:585}，不平衡比 3.005。

把 Normal 下采样到 585 可得严格平衡（4×585），但训练集同时
从 3513 掉到 2340（−33%）。**样本量本身会影响方差**，所以
两臂设计（原始 vs 平衡）无法区分「平衡起了作用」与
「样本变少起了作用」。

三臂：

| 臂 | train n | 不平衡比 | 作用 |
|---|---|---|---|
| `orig`     | 3513 | 3.005 | 基线（已跑，种子份额 48.1%）|
| `balanced` | 2340 | 1.000 | 干预臂 |
| `subsample`| 2340 | 3.005 | **样本量对照**：等量下采样但保持比例 |

若种子份额在 `balanced` 掉下来而 `subsample` 不掉 → 平衡是原因。
若两臂都掉 → 是样本量，不是平衡。
若都不掉 → 不平衡与方差份额无关，JNU/PU 的差异另有来源。

注意：`prior_screen.py` 的损失已是 class-weighted
（`cnt.sum()/(len(cnt)*cnt)`），所以不平衡的影响路径**不是**梯度权重
（那已被补偿），而是样本量结构本身。平衡后权重自动变为全 1。
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

import collections
from typing import Literal

import numpy as np

ARMS = ("orig", "balanced", "subsample")
Arm = Literal["orig", "balanced", "subsample"]


def _idx_by_class(y: np.ndarray) -> dict[int, np.ndarray]:
    return {int(c): np.flatnonzero(y == c) for c in np.unique(y)}


def rebalance_indices(y: np.ndarray, arm: Arm, seed: int) -> np.ndarray:
    """返回训练集的保留下标。切分本身不动，只挑样本。

    seed 与 run 的种子绑定，使不同 run 的下采样也不同 —— 否则下采样
    会变成一个固定的额外设计因子，污染方差归属。
    """
    if arm == "orig":
        return np.arange(len(y))

    by = _idx_by_class(y)
    rng = np.random.default_rng(10_000 + seed)

    if arm == "balanced":
        k = min(len(v) for v in by.values())
        keep = [rng.choice(v, k, replace=False) for v in by.values()]

    elif arm == "subsample":
        # 保持原比例，总量对齐 balanced 臂
        n_bal = min(len(v) for v in by.values()) * len(by)
        frac = n_bal / len(y)
        keep = []
        for v in by.values():
            k = max(1, int(round(len(v) * frac)))
            keep.append(rng.choice(v, k, replace=False))
    else:
        raise ValueError(arm)

    return np.sort(np.concatenate(keep))


def describe(y: np.ndarray, idx: np.ndarray) -> dict:
    c = collections.Counter(y[idx].tolist())
    return {"n": int(len(idx)),
            "counts": {int(k): int(v) for k, v in sorted(c.items())},
            "ratio": round(max(c.values()) / min(c.values()), 3)}


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, "pbx"); sys.path.insert(0, os.getcwd())
    from data.splits_v2 import build_jnu_modeA_splits
    b = build_jnu_modeA_splits(_JNU, rpm=600,
                               segment_len=1024, overlap=0.5)
    y = b.y["train"]
    for arm in ARMS:
        for sd in (0, 1):
            d = describe(y, rebalance_indices(y, arm, sd))
            print(f"  {arm:10s} seed{sd}  n={d['n']:5d}  ratio={d['ratio']:.3f}  "
                  f"{d['counts']}")
