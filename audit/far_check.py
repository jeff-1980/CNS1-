"""事后（未预注册）推理检查：§4.5 的“健康窗口被报为故障 / 故障窗口被报为健康”（B0，不拒识）在三种情形下的数值。
  C-INIT 权重（训练集含 98.mat 副本）× 原 Test_40 的 Normal       → 'leaky_orig'
  C-INIT 权重 × Normal 换成未见过的 100.mat（val 划分）           → 'leaky_N100'
  C-INIT-FIX 权重（修复后加载器重训）× Test_40                    → 'fixed'
只用 cnn1d / wdcnn / drsn（lstm 过慢）。噪声种子沿用 WP5 的 5000 + 17·seed + r（r = 0..4）。不训练、不改动存档。
"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.join(HERE, "../code"); R = os.path.join(HERE, "../results")
sys.path.insert(0, CODE); sys.path.insert(0, os.path.join(CODE, "pbx"))
import numpy as np, torch
from data.splits_v2 import build_cwru_splits
from data.noise_augment import add_awgn_tensor
from arm2x2 import ROOT40
from prior_screen import REGISTRY
dev = torch.device("cuda")
assert os.environ.get("CWRU_LEGACY_FIRST_KEY") != "1"
b = build_cwru_splits(ROOT40, loads=(0, 1, 2, 3), segment_len=1024, overlap=0.5, split_seed=42)
Xte, yte = b.X["test"], b.y["test"]; F = Xte[yte != 0]; Fy = yte[yte != 0]; N100 = b.X["val"][b.y["val"] == 0]
T = {"orig": (Xte, yte), "N100": (np.concatenate([N100, F]), np.concatenate([np.zeros(len(N100), int), Fy]))}
T = {k: (torch.from_numpy(np.asarray(x, np.float32)), np.asarray(y)) for k, (x, y) in T.items()}

def rates(net, X, y, seed, snr):
    out = []
    for r in range(5):
        torch.manual_seed(5000 + 17 * seed + r); preds = []
        with torch.no_grad():
            for i in range(0, len(X), 256): preds.append(net(add_awgn_tensor(X[i:i + 256].to(dev), snr)).argmax(1).cpu())
        p = torch.cat(preds).numpy(); h = y == 0
        out.append((float((p[h] != 0).mean()), float((p[~h] == 0).mean()), float((p == y).mean())))
    return np.mean(out, 0).tolist()

res = []
jobs = [("cinit_2x2.json", ("orig", "N100"), "leaky"), ("cinit_fix_2x2.json", ("orig",), "fixed")]
for src, tests, tag in jobs:
    for r in json.load(open(os.path.join(R, src))):
        if r["model"] not in ("cnn1d", "wdcnn", "drsn"): continue
        ck = r["ckpt"] if os.path.isabs(r["ckpt"]) else os.path.normpath(os.path.join(CODE, r["ckpt"]))
        net = REGISTRY[r["model"]](4); net.load_state_dict(torch.load(ck, map_location="cpu")); net = net.to(dev).eval()
        for t in tests:
            X, y = T[t]
            for snr in (-8.0, 10.0):
                far, mfr, acc = rates(net, X, y, r["seed"], snr)
                res.append(dict(group=f"{tag}_{t}", model=r["model"], arm=r["arm"], seed=r["seed"], snr=snr, far=far, mfr=mfr, acc=acc))
    print("done", src, len(res), flush=True)
json.dump(res, open(os.path.join(HERE, "far_check.json"), "w"))
print("=== ALL DONE", flush=True)
