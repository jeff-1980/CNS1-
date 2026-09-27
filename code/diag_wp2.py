"""WP2 事后诊断（未预注册，只描述）：可学性失败的来源 —— 无噪声准确率、见过 / 未见过轴承的逐轴承准确率。"""
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PAPERB_CODE = _os.environ.get("PAPERB_CODE", _os.path.join(_ROOT, "external/paperB_code"))
_PAPERB_RESULTS = _os.environ.get("PAPERB_RESULTS", _os.path.join(_ROOT, "external/paperB_results"))
_CWRU40 = _os.environ.get("CWRU40_DIR", _os.path.join(_ROOT, "data/cwru_12k_de"))
_CWRU16 = _os.environ.get("CWRU16_DIR", _os.path.join(_ROOT, "data/cwru_16"))
_PU = _os.environ.get("PU_DIR", _os.path.join(_ROOT, "data/paderborn"))
_JNU = _os.environ.get("JNU_DIR", _os.path.join(_ROOT, "data/jnu"))
_JNU_RAW = _os.environ.get("JNU_RAW_DIR", _JNU)
import sys, os, json, collections, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "pbx")
import pu_wp2 as W
from prior_screen import REGISTRY
from data.noise_augment import add_awgn_tensor
R = (_ROOT + "/results"); dev = "cuda"
recs = json.load(open(f"{R}/wp2_R.json")) + json.load(open(f"{R}/wp2_A.json")) + json.load(open(f"{R}/wp2_R_single.json"))
out = []
for b in ("R", "A"):
    pools = W.load_pools(W.block_codes(b)); rng = np.random.default_rng(42); X, y, code = [], [], []
    for c in W.block_codes(b):   # 与 build_test 相同的抽样顺序与种子
        te = pools[c][1]; k = W.TEST_H if c in W.H else W.TEST_FAULT
        X.append(te[rng.choice(len(te), k, replace=False)]); y += [W.label_of(c)] * k; code += [c] * k
    X = torch.from_numpy(np.concatenate(X)[:, None, :]); y = np.array(y); code = np.array(code)
    Xt, yt = W.build_test(pools, b); assert np.array_equal(Xt, X.numpy()) and np.array_equal(yt, y)
    # 训练集自身准确率所需
    for r in [r for r in recs if r["block"] == b]:
        net = REGISTRY[r["model"]](3).to(dev); net.load_state_dict(torch.load(r["ckpt"], map_location=dev)); net.eval()
        seen = set(W.H) | set(r["OR"]) | set(r["IR"])
        with torch.no_grad():
            def pred(x, snr=None):
                ps = []
                for i in range(0, len(x), 512):
                    xb = x[i:i + 512].to(dev); torch.manual_seed(7)
                    ps.append(net(xb if snr is None else add_awgn_tensor(xb, snr)).argmax(1).cpu())
                return torch.cat(ps).numpy()
            p0 = pred(X); p10 = pred(X, 10.0)
            Xtr, ytr, _ = W.build_train(pools, r["arm"], r["seed"]); ptr = pred(torch.from_numpy(Xtr))
        pb = {c: dict(acc_clean=float((p0[code == c] == y[code == c]).mean()), acc10=float((p10[code == c] == y[code == c]).mean()),
                      pred_clean=np.bincount(p0[code == c], minlength=3).tolist(), seen=c in seen) for c in W.block_codes(b)}
        sm = lambda key, s: float(np.mean([v[key] for c, v in pb.items() if v["seen"] == s and c not in W.H])) if any(v["seen"] == s and c not in W.H for c, v in pb.items()) else None
        out.append(dict(block=b, arm=r["arm"], model=r["model"], seed=r["seed"], train_acc=float((ptr == ytr).mean()),
                        test_clean=float((p0 == y).mean()), test10=float((p10 == y).mean()),
                        seen_fault_clean=sm("acc_clean", True), unseen_fault_clean=sm("acc_clean", False),
                        healthy_clean=float(np.mean([pb[c]["acc_clean"] for c in W.H])), per_bearing=pb))
json.dump(out, open(f"{R}/wp2_posthoc_learnability.json", "w"), indent=1)
agg = collections.defaultdict(list)
for o in out: agg[(o["arm"], o["model"])].append(o)
print(f"{'arm':7s} {'model':11s} train  clean  +10dB | seenF  unseenF healthy (clean)")
for (arm, m), os_ in sorted(agg.items()):
    f = lambda k: np.mean([o[k] for o in os_ if o[k] is not None]) if any(o[k] is not None for o in os_) else float("nan")
    print(f"{arm:7s} {m:11s} {f('train_acc'):.3f}  {f('test_clean'):.3f}  {f('test10'):.3f} | {f('seen_fault_clean'):.3f}  {f('unseen_fault_clean'):.3f}  {f('healthy_clean'):.3f}")
