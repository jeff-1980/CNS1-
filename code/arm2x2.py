"""2×2 配对干预：覆盖 {007, ALL} × 比例 {1:1, 3:1}。严格按 results/PREREG_2x2_coverage_ratio.md。"""
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
import sys, os, re, json, math, argparse, time, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "pbx")
import numpy as np, torch, torch.nn as nn
from data.splits_v2 import build_cwru_splits
from data.noise_augment import add_awgn_tensor
from prior_screen import REGISTRY

ROOT40 = _CWRU40
BUDGET = 1900
ARMS = {"C007_R11": ("007", "R11"), "C007_R31": ("007", "R31"),
        "CALL_R11": ("ALL", "R11"), "CALL_R31": ("ALL", "R31")}
COV_CODE = {"007": 0, "ALL": 1}; RATIO_CODE = {"R11": 0, "R31": 1}
SNRS = [10.0, 0.0, -4.0, -6.0, -8.0]
CLS = {0: "Normal", 1: "IR", 2: "OR", 3: "Ball"}

def parse(fid):
    if fid.startswith("Normal"): return 0, None
    m = re.match(r"(B|IR|OR)(\d{3})", fid); c = {"IR": 1, "OR": 2, "B": 3}[m.group(1)]
    return c, m.group(2)

def quotas(ratio):
    """返回 (n_normal, {class: n})，总和 = BUDGET。"""
    if ratio == "R11": return 475, {1: 475, 2: 475, 3: 475}
    return 950, {1: 316, 2: 318, 3: 316}   # B/IR 316 + OR 318 = 950

def size_split(n, sizes):
    """n 个窗口在尺寸间按 158/158/159 式轮转分配（余数给靠后的尺寸）。"""
    base, rem = divmod(n, len(sizes)); return {s: base + (1 if i >= len(sizes) - rem else 0) for i, s in enumerate(sizes)}

def build_arm(pool, arm, seed):
    cov, ratio = ARMS[arm]; fids, X, y = pool
    info = [parse(f) for f in fids]
    nN, qf = quotas(ratio)
    idx = []
    rngN = np.random.default_rng([seed, RATIO_CODE[ratio]])
    normal_idx = np.array([i for i, (c, _) in enumerate(info) if c == 0])
    idx += list(rngN.choice(normal_idx, nN, replace=False))
    alloc = {}
    for c, n in qf.items():
        rng = np.random.default_rng([seed, RATIO_CODE[ratio], COV_CODE[cov], c])
        sizes = ["007"] if cov == "007" else ["007", "014", "021"]
        per = size_split(n, sizes)
        for s, k in per.items():
            cand = np.array([i for i, (cc, ss) in enumerate(info) if cc == c and ss == s])
            idx += list(rng.choice(cand, k, replace=False)); alloc[(c, s)] = k
    idx = np.array(idx)
    return X[idx], y[idx], alloc, idx

def train_last(model, Xtr, ytr, device, seed, epochs=100, bs=64, lr=1e-3, wd=1e-4, log=print):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    warm = 5
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda ep: (ep + 1) / warm if ep < warm
                                              else 0.5 * (1 + math.cos(math.pi * (ep - warm) / max(1, epochs - warm))))
    cnt = torch.bincount(ytr, minlength=4).float()
    cw = (cnt.sum() / (4 * cnt.clamp(min=1))).to(device)            # 按重采样后训练集重算
    lossf = nn.CrossEntropyLoss(weight=cw); n = len(Xtr); steps = 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]; xb, yb = Xtr[b].to(device), ytr[b].to(device)
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); steps += 1
        sched.step()
        if (ep + 1) % 25 == 0: log(f"      ep{ep+1:3d} loss={loss.item():.4f}")
    return model, steps, cw.cpu().tolist()

@torch.no_grad()
def evaluate(model, Xte, yte, device, seed, n_reps=5):
    model.eval(); out = {}
    for snr in SNRS:
        reps = []
        for r in range(n_reps):
            torch.manual_seed(1000 + 17 * seed + r); preds = []
            for i in range(0, len(Xte), 256):
                preds.append(model(add_awgn_tensor(Xte[i:i + 256].to(device), snr)).argmax(1).cpu())
            p = torch.cat(preds).numpy(); c = np.bincount(p, minlength=4)
            reps.append(dict(top=int(c.argmax()), conc=float(c.max() / len(p)), acc=float((p == yte.numpy()).mean()), dist=c.tolist()))
        tops = [r["top"] for r in reps]
        out[f"{snr:.1f}"] = dict(top_mode=int(np.bincount(tops, minlength=4).argmax()), tops=tops,
                                 conc_mean=float(np.mean([r["conc"] for r in reps])),
                                 acc_mean=float(np.mean([r["acc"] for r in reps])), dist_rep0=reps[0]["dist"])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--models", nargs="+", default=["wdcnn", "cnn1d", "drsn"]); ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--out", default="../results/arm2x2.json"); ap.add_argument("--ckpt-dir", default="../results/ckpt_2x2")
    a = ap.parse_args()
    b = build_cwru_splits(ROOT40, loads=(0, 1, 2, 3), segment_len=1024, overlap=0.5, split_seed=42)
    fids = [w.file_id for w in b.windows if w.split == "train"]
    pool = (fids, b.X["train"], b.y["train"])
    if a.smoke:
        for arm in a.arms:
            for s in a.seeds[:2]:
                X, y, alloc, idx = build_arm(pool, arm, s)
                print(arm, f"s{s}", "n", len(y), "cls", np.bincount(y, minlength=4).tolist(),
                      "alloc", {f"{CLS[c]}{sz}": k for (c, sz), k in sorted(alloc.items())},
                      "uniq", len(set(idx.tolist())), "hash", hashlib.md5(idx.tobytes()).hexdigest()[:8])
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xte, yte = torch.from_numpy(b.X["test"]), torch.from_numpy(b.y["test"])
    os.makedirs(a.ckpt_dir, exist_ok=True)
    res = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["arm"], r["model"], r["seed"]) for r in res}
    for s in a.seeds:
        for m in a.models:
            for arm in a.arms:
                if (arm, m, s) in done: continue
                X, y, alloc, idx = build_arm(pool, arm, s)
                print(f"--- {arm} {m} seed{s} --- n={len(y)} cls={np.bincount(y, minlength=4).tolist()}", flush=True)
                t0 = time.time()
                net, steps, cw = train_last(REGISTRY[m](4), torch.from_numpy(X), torch.from_numpy(y), dev, s,
                                            log=lambda x: print(x, flush=True))
                ck = os.path.join(a.ckpt_dir, f"{arm}_{m}_s{s}.pt"); torch.save(net.state_dict(), ck)
                ev = evaluate(net, Xte, yte, dev, s)
                res.append(dict(arm=arm, coverage=ARMS[arm][0], ratio=ARMS[arm][1], model=m, seed=s, n_train=int(len(y)),
                                class_counts=np.bincount(y, minlength=4).tolist(), steps=steps, class_weights=cw,
                                idx_md5=hashlib.md5(idx.tobytes()).hexdigest(), ckpt=ck,
                                ckpt_sha256=hashlib.sha256(open(ck, "rb").read()).hexdigest(),
                                wall_s=round(time.time() - t0, 1), observed=ev))
                tmp = a.out + ".tmp"; json.dump(res, open(tmp, "w"), indent=1); os.replace(tmp, a.out)
                print(f"    done {len(res)}/60 wall={time.time()-t0:.0f}s", flush=True)   # 不打印靶标：避免中途看结果
    print("=== ALL DONE ===", flush=True)

if __name__ == "__main__":
    main()
