"""事后（未预注册）推理检查：把 Test_40 的 Normal 测试窗口（98.mat，训练池中有逐字节副本）换成训练从未见过的 Normal 记录，
重新评估已有 2×2 / E3 / C-INIT 2×2 权重在 −8 dB 下的主导类别。不训练、不改动存档结果。
测试变体：
  orig   — 原 Test_40（作对照，应逐位复现存档）
  N100   — Normal 换为 100.mat（val 划分；2×2 只用 train 划分训练，未见过）
  N099   — Normal 换为 99.mat 的真实信号 X099_DE_time（训练池里的“99”实为 X098，故 X099 从未见过）
故障类测试窗口不变。窗口化、逐窗口 z-score、噪声种子与 evaluate() 完全相同。
"""
import sys, os, json, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.join(HERE, "../code"); R = os.path.join(HERE, "../results")
sys.path.insert(0, CODE); sys.path.insert(0, os.path.join(CODE, "pbx"))
import numpy as np, torch, scipy.io as sio
from data.splits_v2 import build_cwru_splits, _segment_and_window
from arm2x2 import ROOT40, evaluate
from prior_screen import REGISTRY
dev = torch.device("cuda"); torch.backends.cudnn.benchmark = False

b = build_cwru_splits(ROOT40, loads=(0, 1, 2, 3), segment_len=1024, overlap=0.5, split_seed=42)
Xte, yte = b.X["test"], b.y["test"]; F = Xte[yte != 0]; Fy = yte[yte != 0]
def windows_of(sig):
    X, y, w = [], [], []; _segment_and_window(sig.astype(np.float32), "x", "test", 0, 1024, 512, True, 0, X, y, w); return np.stack(X).reshape((-1,) + F.shape[1:])
N100 = b.X["val"][(b.y["val"] == 0)]
x099 = sio.loadmat(os.path.join(ROOT40, "99.mat"))["X099_DE_time"].squeeze(); N099 = windows_of(x099)
x098 = sio.loadmat(os.path.join(ROOT40, "98.mat"))["X098_DE_time"].squeeze()
assert not np.array_equal(x099.astype(np.float32)[:len(x098)], x098.astype(np.float32)[:len(x099)])
H = lambda a: hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()
trainN = {H(x) for x, y in zip(b.X["train"], b.y["train"]) if y == 0}
assert not any(H(x) in trainN for x in N100) and not any(H(x) in trainN for x in N099), "replacement Normal windows must be unseen"
TESTS = {"orig": (Xte, yte), "N100": (np.concatenate([N100, F]), np.concatenate([np.zeros(len(N100), int), Fy])),
         "N099": (np.concatenate([N099, F]), np.concatenate([np.zeros(len(N099), int), Fy]))}
TT = {k: (torch.from_numpy(np.asarray(x, np.float32)), torch.from_numpy(np.asarray(y, np.int64))) for k, (x, y) in TESTS.items()}
print("test sizes", {k: (len(y), int((y == 0).sum())) for k, (x, y) in TESTS.items()}, flush=True)

OUT = os.path.join(HERE, "rescore_leakfree.json")
out = json.load(open(OUT)) if os.path.exists(OUT) else []
done = {(o["source"], o["arm"], o["model"], o["seed"]) for o in out}
jobs = [(src, r) for src in ("arm2x2.json", "arm2x2_lt.json", "cinit_2x2.json") for r in json.load(open(os.path.join(R, src)))]
jobs.sort(key=lambda j: j[1]["model"] in ("lstm", "transformer"))          # fast architectures first
for src, r in jobs:
        if (src, r["arm"], r["model"], r["seed"]) in done: continue
        ck = r["ckpt"] if os.path.isabs(r["ckpt"]) else os.path.normpath(os.path.join(CODE, r["ckpt"]))
        net = REGISTRY[r["model"]](4); net.load_state_dict(torch.load(ck, map_location="cpu")); net = net.to(dev)
        rec = dict(source=src, arm=r["arm"], model=r["model"], seed=r["seed"])
        with torch.no_grad():
            for k, (x, y) in TT.items():
                ev = evaluate(net, x, y, dev, r["seed"])
                rec[k] = {s: dict(top=ev[s]["top_mode"], conc=ev[s]["conc_mean"], acc=ev[s]["acc_mean"], dist=ev[s]["dist_rep0"]) for s in ("10.0", "-8.0")}
        o = r["observed"]["-8.0"]; rec["orig_reproduces_archive"] = (rec["orig"]["-8.0"]["top"] == o["top_mode"] and abs(rec["orig"]["-8.0"]["conc"] - o["conc_mean"]) < 1e-9)
        out.append(rec); tmp = OUT + ".tmp"; json.dump(out, open(tmp, "w")); os.replace(tmp, OUT)
        print("done", len(out), r["model"], flush=True)
print("=== ALL DONE", len(out), flush=True)
