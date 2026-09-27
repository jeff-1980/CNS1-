"""WP2：PU 损伤程度构成复验。严格按 results/PREREG_WP2_pu_extent.md（SHA256 ba370e92…）。"""
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
import sys, os, json, math, argparse, time, hashlib, zlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "pbx")
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from pu_splits import _codes_present, _load_pu_signal
from data.splits_v2 import _segment_and_window, GAP_SAMPLES
from data.noise_augment import add_awgn_tensor
from prior_screen import REGISTRY

PU_ROOT, OP = Path(_PU), "N15_M07_F10"
SEG, STEP = 1024, 512
H = ["K001", "K002"]
ARMS = {"R-L1": (["KA04", "KA22"], ["KI21", "KI14"]), "R-L12a": (["KA22", "KA16"], ["KI21", "KI18"]), "R-L12b": (["KA04", "KA16"], ["KI14", "KI18"]),
        "A-L1": (["KA05", "KA07"], ["KI03", "KI05"]), "A-L12a": (["KA05", "KA08"], ["KI03", "KI08"]), "A-L12b": (["KA07", "KA03"], ["KI05", "KI08"])}
BLOCK = {"R": ["R-L1", "R-L12a", "R-L12b"], "A": ["A-L1", "A-L12a", "A-L12b"]}
PER_BEARING, TEST_FAULT, TEST_H = 600, 300, 450
STEPS, BS, LR, WD = 3000, 64, 1e-3, 1e-4
SNRS = [10.0, 0.0, -4.0, -6.0, -8.0, -10.0, -12.0]
CLS = {0: "Healthy", 1: "OR", 2: "IR"}

def block_codes(b):
    s = set(H)
    for arm in BLOCK[b]: s |= set(ARMS[arm][0]) | set(ARMS[arm][1])
    return sorted(s)

def label_of(code): return 0 if code in H else (1 if code.startswith("KA") else 2)

def load_pools(codes):
    avail = _codes_present(PU_ROOT); pools = {}
    for c in codes:
        files = [f for f in avail[c] if f.name.startswith(OP)]
        assert len(files) == 20, (c, len(files))
        Xtr, Xte = [], []
        for f in files:
            sig = _load_pu_signal(f); n = len(sig)
            tr_end = int(n * 0.6); te_start = tr_end + GAP_SAMPLES + int(n * 0.2) + GAP_SAMPLES
            for (a, z, out) in ((0, tr_end, Xtr), (te_start, n, Xte)):
                y, w = [], []
                _segment_and_window(sig[a:z], f.stem, "train", 0, SEG, STEP, True, a, out, y, w)
        pools[c] = (np.stack(Xtr).astype(np.float32), np.stack(Xte).astype(np.float32))
    return pools

def rng_for(seed, code): return np.random.default_rng([seed, zlib.crc32(code.encode())])

def build_train(pools, arm, seed):
    OR, IR = ARMS[arm]; X, y, idx_log = [], [], {}
    for code in H + OR + IR:
        tr = pools[code][0]; idx = rng_for(seed, code).choice(len(tr), PER_BEARING, replace=False)
        X.append(tr[idx]); y += [label_of(code)] * PER_BEARING; idx_log[code] = hashlib.md5(np.sort(idx).tobytes()).hexdigest()[:10]
    return np.concatenate(X)[:, None, :], np.array(y, dtype=np.int64), idx_log

def build_test(pools, b):
    rng = np.random.default_rng(42); X, y = [], []
    for code in block_codes(b):
        te = pools[code][1]; k = TEST_H if code in H else TEST_FAULT
        X.append(te[rng.choice(len(te), k, replace=False)]); y += [label_of(code)] * k
    return np.concatenate(X)[:, None, :], np.array(y, dtype=np.int64)

def train_fixed(model, Xtr, ytr, device, seed, log=print):
    torch.manual_seed(seed); np.random.seed(seed); model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD); warm = int(0.05 * STEPS)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (STEPS - warm))))
    lossf = nn.CrossEntropyLoss(); n = len(Xtr); step = 0; g = torch.Generator().manual_seed(seed)
    while step < STEPS:
        model.train(); perm = torch.randperm(n, generator=g)
        for i in range(0, n - BS + 1, BS):
            b = perm[i:i + BS]; xb, yb = Xtr[b].to(device), ytr[b].to(device)
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
            if step % 1000 == 0: log(f"      step{step} loss={loss.item():.4f}")
            if step >= STEPS: break
    return model, step

@torch.no_grad()
def evaluate(model, Xte, yte, device, seed, n_reps=5):
    model.eval(); out = {}
    for snr in SNRS:
        reps = []
        for r in range(n_reps):
            torch.manual_seed(1000 + 17 * seed + r); preds = []
            for i in range(0, len(Xte), 256): preds.append(model(add_awgn_tensor(Xte[i:i + 256].to(device), snr)).argmax(1).cpu())
            p = torch.cat(preds).numpy(); c = np.bincount(p, minlength=3)
            reps.append(dict(top=int(c.argmax()), conc=float(c.max() / len(p)), acc=float((p == yte.numpy()).mean()), dist=c.tolist()))
        tops = [r["top"] for r in reps]
        out[f"{snr:.1f}"] = dict(top_mode=int(np.bincount(tops, minlength=3).argmax()), tops=tops, conc_mean=float(np.mean([r["conc"] for r in reps])),
                                 acc_mean=float(np.mean([r["acc"] for r in reps])), dist_rep0=reps[0]["dist"])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True, choices=["R", "A"]); ap.add_argument("--models", required=True)
    ap.add_argument("--arms", default=None); ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out", required=True); ap.add_argument("--ckpt-dir", required=True); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    arms = a.arms.split(",") if a.arms else BLOCK[a.block]; assert all(x in BLOCK[a.block] for x in arms)
    models = a.models.split(","); seeds = [int(s) for s in a.seeds.split(",")]
    t0 = time.time(); pools = load_pools(block_codes(a.block))
    Xte, yte = build_test(pools, a.block)
    print(f"pools loaded {time.time()-t0:.0f}s:", {c: (len(p[0]), len(p[1])) for c, p in pools.items()}, "| test", np.bincount(yte).tolist(), flush=True)
    if a.smoke:
        for arm in arms:
            for s in seeds[:2]:
                X, y, il = build_train(pools, arm, s); print(" ", arm, f"s{s}", X.shape, np.bincount(y).tolist(), il)
        dev = "cuda"
        for m in models:
            net = REGISTRY[m](3).to(dev); print(" ", m, sum(p.numel() for p in net.parameters()), tuple(net(torch.from_numpy(Xte[:4]).to(dev)).shape))
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"; os.makedirs(a.ckpt_dir, exist_ok=True)
    Xte_t, yte_t = torch.from_numpy(Xte), torch.from_numpy(yte)
    res = json.load(open(a.out)) if os.path.exists(a.out) else []; done = {(r["arm"], r["model"], r["seed"]) for r in res}
    for s in seeds:
        for m in models:
            for arm in arms:
                if (arm, m, s) in done: continue
                X, y, il = build_train(pools, arm, s)
                print(f"--- {arm} {m} seed{s} ---", flush=True); t1 = time.time()
                net, steps = train_fixed(REGISTRY[m](3), torch.from_numpy(X), torch.from_numpy(y), dev, s, log=lambda x: print(x, flush=True))
                ck = os.path.join(a.ckpt_dir, f"{arm}_{m}_s{s}.pt"); torch.save(net.state_dict(), ck)
                res.append(dict(block=a.block, arm=arm, model=m, seed=s, OR=ARMS[arm][0], IR=ARMS[arm][1], n_train=int(len(y)), class_counts=np.bincount(y).tolist(),
                                idx_md5=il, steps=steps, ckpt=ck, ckpt_sha256=hashlib.sha256(open(ck, "rb").read()).hexdigest(),
                                wall_s=round(time.time() - t1, 1), observed=evaluate(net, Xte_t, yte_t, dev, s)))
                tmp = a.out + ".tmp"; json.dump(res, open(tmp, "w")); os.replace(tmp, a.out)
                print(f"    done {len(res)} wall={time.time()-t1:.0f}s", flush=True)   # 不打印靶标
    print("=== ALL DONE ===", flush=True)

if __name__ == "__main__":
    main()
