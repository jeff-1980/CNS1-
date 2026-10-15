"""【事后 / 探索性，不训练，不改变任何预注册判定】T2、T3 的 CWRU 权重在无泄漏测试集上的干净准确率。
T2、T3 的 CWRU 格用的是修正前的加载器（训练池中的 "99" 实为 98 号文件的副本，Test_40 的 944 个 Normal 测试窗口在训练池中有逐字节副本；
稿件 §3.8）。本脚本按 audit/rescore_leakfree.py 的做法，把 Normal 测试窗口换成训练从未见过的 Normal 窗口，重算干净准确率：
  orig — 原 Test_40（应复现 t2_cwru.json / t3_main.json 中记录的 clean_acc）
  N100 — Normal 换为 100.mat（val 划分）
  N099 — Normal 换为 99.mat 的真实信号 X099_DE_time
故障类测试窗口不变。必须以 CWRU_LEGACY_FIRST_KEY=1 运行，使训练池与 T2 / T3 实际所见一致。
从 code/ 目录运行：CWRU_LEGACY_FIRST_KEY=1 python posthoc_t23_cwru_leakfree.py"""
from __future__ import annotations
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, os, json, hashlib, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "pbx")
assert os.environ.get("CWRU_LEGACY_FIRST_KEY") == "1", "须以 CWRU_LEGACY_FIRST_KEY=1 运行（复现 T2 / T3 训练时的加载器）"
import numpy as np, torch, scipy.io as sio
from data.splits_v2 import build_cwru_splits, _segment_and_window
import t2, t3

R = os.path.join(_ROOT, "results"); ROOT40 = t2.CWRU40
b = build_cwru_splits(ROOT40, loads=(0, 1, 2, 3), segment_len=1024, overlap=0.5, split_seed=42)
Xte, yte = b.X["test"], b.y["test"]; F, Fy = Xte[yte != 0], yte[yte != 0]


def windows_of(sig):
    X, y, w = [], [], []; _segment_and_window(sig.astype(np.float32), "x", "test", 0, 1024, 512, True, 0, X, y, w)
    return np.stack(X).reshape((-1,) + F.shape[1:])


N100 = b.X["val"][b.y["val"] == 0]
x099 = sio.loadmat(os.path.join(ROOT40, "99.mat"))["X099_DE_time"].squeeze(); N099 = windows_of(x099)
H = lambda a: hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()
trainN = {H(x) for x, y in zip(b.X["train"], b.y["train"]) if y == 0}
testN = {H(x) for x, y in zip(Xte, yte) if y == 0}
n_copies = sum(h in trainN for h in testN)
assert not any(H(x) in trainN for x in N100) and not any(H(x) in trainN for x in N099), "替换用的 Normal 窗口必须未见过"
TESTS = {"orig": (Xte, yte), "N100": (np.concatenate([N100, F]), np.concatenate([np.zeros(len(N100), int), Fy])),
         "N099": (np.concatenate([N099, F]), np.concatenate([np.zeros(len(N099), int), Fy]))}
TT = {k: (torch.from_numpy(np.asarray(x, np.float32)), torch.from_numpy(np.asarray(y, np.int64))) for k, (x, y) in TESTS.items()}
print(f"Test_40 Normal 窗口在（旧加载器的）训练池中有副本：{n_copies}/{len(testN)}；测试集大小 " +
      str({k: (len(y), int((y == 0).sum())) for k, (x, y) in TESTS.items()}), flush=True)


@torch.no_grad()
def scores(net, X, y):
    p = t2.predict(net, X).argmax(1)
    return dict(acc=float((p == y).float().mean()), normal_recall=float((p[y == 0] == 0).float().mean()))


out = []
for exp, f, mk in (("T2", "t2_cwru.json", lambda r: t2.make_model(r["model"], 1, r["seed"])[0]),
                   ("T3", "t3_main.json", lambda r: t3.make_model(r["model"], 1, 4, r["seed"])[0])):
    for r in json.load(open(os.path.join(R, f))):
        if r["domain"] != "cwru": continue
        net = mk(r); net.load_state_dict(torch.load(os.path.join(_ROOT, r["ckpt"]), map_location="cpu")); net = net.cuda()
        rec = dict(exp=exp, tier=r["tier"], model=r["model"], seed=r["seed"], recorded_clean_acc=r["clean_acc"])
        for k, (x, y) in TT.items(): rec[k] = scores(net, x, y)
        rec["orig_reproduces"] = abs(rec["orig"]["acc"] - r["clean_acc"]) < 1e-6
        out.append(rec); net.cpu()
json.dump(dict(note="事后 / 探索性，不训练，不改变任何预注册判定", normal_test_copies_in_train=[n_copies, len(testN)], records=out),
          open(os.path.join(R, "posthoc_t23_cwru_leakfree.json"), "w"), ensure_ascii=False, indent=1)
print(f"orig 复现记录的 clean_acc：{sum(o['orig_reproduces'] for o in out)}/{len(out)}")
g = collections.defaultdict(list)
for o in out: g[(o["exp"], o["tier"], o["model"])].append(o)
for k, rs in g.items():
    m = lambda t, q: np.mean([o[t][q] for o in rs])
    print(f"{k[0]} {k[1]:10s} {k[2]:6s} n={len(rs):2d} 干净准确率 orig {m('orig','acc'):.3f}  N100 {m('N100','acc'):.3f}  N099 {m('N099','acc'):.3f}   "
          f"Normal 召回 orig {m('orig','normal_recall'):.3f}  N100 {m('N100','normal_recall'):.3f}  N099 {m('N099','normal_recall'):.3f}")
