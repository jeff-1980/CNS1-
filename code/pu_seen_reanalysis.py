"""审稿意见 4 / 5：WP2 已有权重的事后重推理——测试集按"训练中见过 / 未见过的轴承"拆分，报告完整类别分布（不只 IR 占比）。
不改变 WP2 预注册判定（NO_SHIFT）。噪声种子规则与 WP2 相同（1000 + 17·seed + r），但作用于子集，因此噪声实现与原评估不同。"""
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PAPERB_CODE = _os.environ.get("PAPERB_CODE", _os.path.join(_ROOT, "external/paperB_code"))
_PAPERB_RESULTS = _os.environ.get("PAPERB_RESULTS", _os.path.join(_ROOT, "external/paperB_results"))
_CWRU40 = _os.environ.get("CWRU40_DIR", _os.path.join(_ROOT, "data/cwru_12k_de"))
_CWRU16 = _os.environ.get("CWRU16_DIR", _os.path.join(_ROOT, "data/cwru_16"))
_PU = _os.environ.get("PU_DIR", _os.path.join(_ROOT, "data/paderborn"))
_JNU = _os.environ.get("JNU_DIR", _os.path.join(_ROOT, "data/jnu"))
_JNU_RAW = _os.environ.get("JNU_RAW_DIR", _JNU)
import sys, os, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "pbx")
import numpy as np, torch
from pu_wp2 import ARMS, H, BLOCK, block_codes, label_of, load_pools, TEST_H, TEST_FAULT, CLS
from data.noise_augment import add_awgn_tensor
from prior_screen import REGISTRY
R = (_ROOT + "/results"); dev = torch.device("cuda")

def test_by_bearing(pools, b):
    rng = np.random.default_rng(42); out = {}
    for code in block_codes(b):
        te = pools[code][1]; k = TEST_H if code in H else TEST_FAULT
        out[code] = te[rng.choice(len(te), k, replace=False)][:, None, :]
    return out

@torch.no_grad()
def dist(net, X, y, seed, snr, reps=5):
    D, acc = [], []
    for r in range(reps):
        torch.manual_seed(1000 + 17 * seed + r); Xt = torch.from_numpy(X)
        p = torch.cat([net(add_awgn_tensor(Xt[i:i + 256].to(dev), snr) if snr is not None else Xt[i:i + 256].to(dev)).argmax(1).cpu() for i in range(0, len(Xt), 256)]).numpy()
        D.append(np.bincount(p, minlength=3) / len(p)); acc.append(float((p == y).mean()))
        if snr is None: break
    D = np.array(D); tops = D.argmax(1)
    return dict(mean_dist=D.mean(0).round(4).tolist(), top_mode=int(np.bincount(tops, minlength=3).argmax()), conc_mean=float(D.max(1).mean()), acc=float(np.mean(acc)))

res = []
for blk, files in (("R", ["wp2_R.json", "wp2_R_single.json"]), ("A", ["wp2_A.json"])):
    pools = load_pools(block_codes(blk)); T = test_by_bearing(pools, blk)
    for f in files:
        for r in json.load(open(f"{R}/{f}")):
            OR, IR = ARMS[r["arm"]]; seen = H + OR + IR; unseen = [c for c in T if c not in seen]
            net = REGISTRY[r["model"]](3).to(dev); net.load_state_dict(torch.load(r["ckpt"], map_location=dev)); net.eval()
            rec = dict(block=blk, arm=r["arm"], model=r["model"], seed=r["seed"], seen=seen, unseen=unseen, n_seen=int(sum(len(T[c]) for c in seen)), n_unseen=int(sum(len(T[c]) for c in unseen)))
            for tag, codes in (("seen", seen), ("unseen_fault", unseen), ("all", list(T))):
                X = np.concatenate([T[c] for c in codes]); y = np.concatenate([[label_of(c)] * len(T[c]) for c in codes])
                rec[tag] = {"-8.0": dist(net, X, y, r["seed"], -8.0), "clean": dist(net, X, y, r["seed"], None)}
            res.append(rec)
    print(blk, "done", len(res), flush=True)
json.dump(res, open(f"{R}/pu_seen_reanalysis.json", "w"), indent=1)
print("=== ALL DONE ===")
