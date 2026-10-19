"""Verify the loader fix: no two CWRU files load the same signal; no Normal test window has a byte copy in train; splits unchanged."""
import sys, os, hashlib, collections, numpy as np
sys.path.insert(0, "../code"); sys.path.insert(0, "../code/pbx")
from pathlib import Path
from data.cwru_loader import _load_mat_signal
from data.splits_v2 import build_cwru_splits
H = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
import os as _o
_R = _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))
for root in (_o.environ.get("CWRU40_DIR", _o.path.join(_R, "data/cwru_12k_de")), _o.environ.get("CWRU16_DIR", _o.path.join(_R, "data/cwru_16"))):
    sig = {p.stem: H(_load_mat_signal(p)) for p in Path(root).glob("*.mat")}
    dup = [v for v in collections.defaultdict(list, {}).values()]
    by = collections.defaultdict(list); [by[h].append(k) for k, h in sig.items()]
    b = build_cwru_splits(root, loads=(0, 1, 2, 3), segment_len=1024, overlap=0.5, split_seed=42)
    tr = {H(x) for x in b.X["train"]}; va = {H(x) for x in b.X["val"]}
    te = [H(x) for x in b.X["test"]]
    split = {w.file_id: w.split for w in b.windows}
    print(Path(root).name, "| duplicate signals:", [v for v in by.values() if len(v) > 1], "| 99.mat sha:", sig["99"][:12], "98.mat sha:", sig["98"][:12],
          "| test windows with copy in train/val:", sum(h in tr for h in te), sum(h in va for h in te),
          "| class counts", {k: np.bincount(v, minlength=4).tolist() for k, v in b.y.items()},
          "| Normal splits", {k: v for k, v in split.items() if k.startswith("Normal")})
