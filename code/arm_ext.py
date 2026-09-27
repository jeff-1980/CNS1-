"""E1（cnn1d 来源，2³ 析因）与 E2（尺寸 vs 文件数）。严格按 results/PREREG_ext_cnn1d_size_arch.md。
E3 直接复用 arm2x2.py（--models lstm transformer）。"""
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
from arm2x2 import build_arm, parse, evaluate, size_split, CLS

ROOT40 = _CWRU40; ROOT16 = _CWRU16
FRACS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]
E1_EPOCHS = {(0, 0): 100, (0, 1): 150, (1, 0): 67, (1, 1): 100}      # (N, S) → epochs

def load(fid):
    m = re.search(r"_load(\d)", fid); return int(m.group(1))

def pool_of(b, split="train"):
    ws = [w for w in b.windows if w.split == split]
    return dict(fids=[w.file_id for w in ws], starts=[w.start for w in ws], X=b.X[split], y=b.y[split])

# ------------------------- E1 -------------------------
def e1_indices(P40, P16, ID, N, seed):
    if ID == 0 and N == 0:
        _, _, _, idx = build_arm((P40["fids"], P40["X"], P40["y"]), "C007_R31", seed); return "P40", idx
    if ID == 0 and N == 1:
        info = [parse(f) for f in P40["fids"]]
        idx = [i for i, (c, _) in enumerate(info) if c == 0]
        for c in (1, 2, 3):
            cand = np.array([i for i, (cc, s) in enumerate(info) if cc == c and s == "007"])
            idx += list(np.random.default_rng([seed, 9, c]).choice(cand, 475, replace=False))
        return "P40", np.array(idx)
    if ID == 1 and N == 1:
        return "P16", np.arange(len(P16["y"]))
    # ID1 N0：Normal 与 (ID0,N0) 按 (file_id,start) 精确配对
    _, _, _, idx40 = build_arm((P40["fids"], P40["X"], P40["y"]), "C007_R31", seed)
    key16 = {(f, s): i for i, (f, s) in enumerate(zip(P16["fids"], P16["starts"]))}
    idx = [key16[(P40["fids"][i], P40["starts"][i])] for i in idx40 if P40["y"][i] == 0]
    info = [parse(f) for f in P16["fids"]]
    for c, k in ((1, 316), (2, 318), (3, 316)):
        cand = np.array([i for i, (cc, _) in enumerate(info) if cc == c])
        idx += list(np.random.default_rng([seed, 1, 5, c]).choice(cand, k, replace=False))
    return "P16", np.array(idx)

# ------------------------- E2 -------------------------
E2_CODE = {"S1L3": 0, "S3L1": 1, "S1L1": 2}
def e2_indices(P40, arm, seed):
    info = [parse(f) for f in P40["fids"]]; fids = P40["fids"]
    normal = np.array(sorted([i for i, (c, _) in enumerate(info) if c == 0], key=lambda i: (fids[i], P40["starts"][i])))
    idx = list(np.random.default_rng([seed, 7]).choice(normal, 230, replace=False)); used = {}
    for c in (1, 2, 3):
        files_by_size = {}
        for i, (cc, s) in enumerate(info):
            if cc == c: files_by_size.setdefault(s, set()).add(fids[i])
        fs = lambda s: sorted(files_by_size[s], key=load)
        if arm == "S1L3": chosen = fs("007")
        elif arm == "S1L1": chosen = [fs("007")[seed % 3]]
        else: chosen = [fs(s)[seed % len(fs(s))] for s in ("007", "014", "021")]
        alloc = size_split(230, chosen)
        for f, k in alloc.items():
            cand = np.array([i for i, ff in enumerate(fids) if ff == f])
            idx += list(np.random.default_rng([seed, 11, E2_CODE[arm], c, hash_str(f)]).choice(cand, k, replace=False))
        used[CLS[c]] = alloc
    return np.array(idx), used

def hash_str(s): return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)

# ------------------------- 训练 -------------------------
@torch.no_grad()
def target_at(model, Xte, yte, dev, seed, reps=2):
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if dev == "cuda" else []):
        model.eval(); tops, concs = [], []
        for r in range(reps):
            torch.manual_seed(1000 + 17 * seed + r); p = []
            for i in range(0, len(Xte), 256):
                p.append(model(add_awgn_tensor(Xte[i:i + 256].to(dev), -8.0)).argmax(1).cpu())
            c = np.bincount(torch.cat(p).numpy(), minlength=4); tops.append(int(c.argmax())); concs.append(float(c.max() / c.sum()))
    return dict(tops=tops, conc=float(np.mean(concs)))

def train_traj(model, X, y, dev, seed, epochs, Xte, yte, val=None, bs=64, log=print):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    warm = round(0.05 * epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda ep: (ep + 1) / warm if ep < warm
                                              else 0.5 * (1 + math.cos(math.pi * (ep - warm) / max(1, epochs - warm))))
    cnt = torch.bincount(y, minlength=4).float(); cw = (cnt.sum() / (4 * cnt.clamp(min=1))).to(dev)
    lossf = nn.CrossEntropyLoss(weight=cw); n = len(X); steps = 0
    marks = sorted({math.ceil(f * epochs) for f in FRACS}); traj = {}
    best_va, best_state, best_ep, val_curve = -1.0, None, None, []
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]; xb, yb = X[b].to(dev), y[b].to(dev)
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); steps += 1
        sched.step()
        if val is not None:
            model.eval()
            with torch.no_grad():
                pv = torch.cat([model(val[0][i:i + 256].to(dev)).argmax(1).cpu() for i in range(0, len(val[0]), 256)])
            va = (pv == val[1]).float().mean().item(); val_curve.append(round(va, 4))
            if va > best_va:
                best_va, best_ep = va, ep + 1; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep + 1 in marks: traj[str(ep + 1)] = target_at(model, Xte, yte, dev, seed)
        if (ep + 1) % 50 == 0: log(f"      ep{ep+1:3d} loss={loss.item():.4f}")
    return model, dict(steps=steps, warmup=warm, class_weights=cw.cpu().tolist(), traj=traj,
                       best_val=best_va if val is not None else None, best_epoch=best_ep, val_curve=val_curve), best_state

def run(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    b40 = build_cwru_splits(ROOT40, loads=(0, 1, 2, 3), segment_len=1024, overlap=0.5, split_seed=42)
    b16 = build_cwru_splits(ROOT16, loads=(0, 1, 2, 3), segment_len=1024, overlap=0.5, split_seed=42)
    P40, P16 = pool_of(b40), pool_of(b16)
    V16 = (torch.from_numpy(b16.X["val"]), torch.from_numpy(b16.y["val"]))
    Xte, yte = torch.from_numpy(b40.X["test"]), torch.from_numpy(b40.y["test"])
    jobs = []
    if a.exp == "e1":
        for s in a.seeds:
            for ID in (0, 1):
                for N in (0, 1):
                    for S in (0, 1): jobs.append(dict(exp="e1", cell=f"ID{ID}N{N}S{S}", ID=ID, N=N, S=S, model="cnn1d", seed=s))
    else:
        for s in a.seeds:
            for m in ("wdcnn", "drsn", "cnn1d"):
                for arm in ("S1L3", "S3L1", "S1L1"): jobs.append(dict(exp="e2", cell=arm, model=m, seed=s))
    if a.smoke:
        for j in jobs:
            if j["seed"] > 1 or (j["exp"] == "e2" and j["model"] != "wdcnn"): continue
            if j["exp"] == "e1":
                pool, idx = e1_indices(P40, P16, j["ID"], j["N"], j["seed"]); P = P40 if pool == "P40" else P16
                nrm = sorted((P["fids"][i], P["starts"][i]) for i in idx if P["y"][i] == 0)
                sizes = sorted({parse(P["fids"][i])[1] for i in idx if P["y"][i] != 0})
                ep = E1_EPOCHS[(j["N"], j["S"])]; spe = math.ceil(len(idx) / 64)
                print(j["cell"], f"s{j['seed']}", pool, "n", len(idx), "cls", np.bincount(P["y"][idx], minlength=4).tolist(),
                      "uniq", len(set(idx.tolist())), "sizes", sizes, "files/class",
                      {CLS[c]: len({P['fids'][i] for i in idx if P['y'][i] == c}) for c in (1, 2, 3)},
                      "ep", ep, "steps", spe * ep, "warm", round(0.05 * ep), "normal_md5", hashlib.md5(str(nrm).encode()).hexdigest()[:8])
            else:
                idx, used = e2_indices(P40, j["cell"], j["seed"])
                print(j["cell"], f"s{j['seed']}", "n", len(idx), "cls", np.bincount(P40["y"][idx], minlength=4).tolist(),
                      "uniq", len(set(idx.tolist())), "files", used, "steps", math.ceil(len(idx) / 64) * 200)
        return
    out = a.out; os.makedirs(a.ckpt_dir, exist_ok=True)
    res = json.load(open(out)) if os.path.exists(out) else []
    done = {(r["cell"], r["model"], r["seed"]) for r in res}
    for j in jobs:
        if (j["cell"], j["model"], j["seed"]) in done: continue
        if j["exp"] == "e1":
            pool, idx = e1_indices(P40, P16, j["ID"], j["N"], j["seed"]); P = P40 if pool == "P40" else P16
            epochs = E1_EPOCHS[(j["N"], j["S"])]; val = V16 if pool == "P16" else None; extra = {}
        else:
            idx, used = e2_indices(P40, j["cell"], j["seed"]); P = P40; epochs = 200; val = None; extra = dict(files=used)
        X, y = torch.from_numpy(P["X"][idx]), torch.from_numpy(P["y"][idx])
        print(f"--- {j['cell']} {j['model']} seed{j['seed']} --- n={len(y)} ep={epochs}", flush=True); t0 = time.time()
        net, info, best = train_traj(REGISTRY[j["model"]](4), X, y, dev, j["seed"], epochs, Xte, yte, val=val,
                                     log=lambda x: print(x, flush=True))
        ck = os.path.join(a.ckpt_dir, f"{j['cell']}_{j['model']}_s{j['seed']}.pt"); torch.save(net.state_dict(), ck)
        rec = dict(j, n_train=int(len(y)), class_counts=np.bincount(y.numpy(), minlength=4).tolist(), epochs=epochs,
                   idx_md5=hashlib.md5(np.asarray(idx).tobytes()).hexdigest(), ckpt=ck,
                   ckpt_sha256=hashlib.sha256(open(ck, "rb").read()).hexdigest(), **{k: v for k, v in info.items()}, **extra)
        rec["final"] = evaluate(net, Xte, yte, dev, j["seed"])
        if best is not None:
            net.load_state_dict(best); rec["valbest"] = evaluate(net, Xte, yte, dev, j["seed"])
            ckb = ck.replace(".pt", "_valbest.pt"); torch.save(best, ckb); rec["ckpt_valbest"] = ckb
        rec["wall_s"] = round(time.time() - t0, 1); res.append(rec)
        tmp = out + ".tmp"; json.dump(res, open(tmp, "w"), indent=1); os.replace(tmp, out)
        print(f"    done {len(res)}/{len(jobs)} wall={rec['wall_s']:.0f}s", flush=True)
    print("=== ALL DONE ===", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["e1", "e2"], required=True); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out"); ap.add_argument("--ckpt-dir")
    a = ap.parse_args(); a.out = a.out or f"../results/ext_{a.exp}.json"; a.ckpt_dir = a.ckpt_dir or f"../results/ckpt_ext_{a.exp}"
    run(a)
