"""E5：尺寸 × 文件抽取识别实验。严格按 results/PREREG_E5_size_identification.md（SHA256 431231ae…）。"""
from __future__ import annotations
import sys, os, json, time, hashlib, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "pbx")
import numpy as np, torch
from data.splits_v2 import build_cwru_splits
from prior_screen import REGISTRY
from arm2x2 import ROOT40, parse, train_last, evaluate, size_split

SETS = {"S007": ["007"], "S014": ["014"], "S021": ["021"], "S007+014": ["007", "014"], "S007+021": ["007", "021"],
        "S014+021": ["014", "021"], "S_ALL": ["007", "014", "021"]}
SIZE_CODE = {"007": 0, "014": 1, "021": 2}
PER_CLASS, EPOCHS, DRAWS = 230, 200, [0, 1, 2]

def pool_index(b):
    """文件池 = train ∪ val；返回 {file_id: (X, y)}。"""
    out = {}
    for sp in ("train", "val"):
        fids = [w.file_id for w in b.windows if w.split == sp]
        for f in sorted(set(fids)):
            idx = np.array([i for i, g in enumerate(fids) if g == f]); out[f] = (b.X[sp][idx], b.y[sp][idx])
    return out

def draw_files(pool, draw):
    """(类别, 尺寸) → 文件；只依赖 (draw, class, size)，在所有集合间配对。"""
    by = {}
    for f in pool:
        c, s = parse(f); by.setdefault((c, s), []).append(f)
    pick = {}
    for (c, s), fs in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        fs = sorted(fs); code = [draw, c] + ([SIZE_CODE[s]] if s else [])
        pick[(c, s)] = fs[int(np.random.default_rng(code).integers(len(fs)))]
    return pick

def build(pool, pick, sset, seed, draw):
    Xs, ys, used = [], [], {}
    fN = pick[(0, None)]; X, y = pool[fN]; i = np.random.default_rng([seed, draw, 0]).choice(len(y), PER_CLASS, replace=False)
    Xs.append(X[i]); ys.append(y[i]); used["Normal"] = [fN, PER_CLASS]
    for c in (1, 2, 3):
        for s, k in size_split(PER_CLASS, SETS[sset]).items():
            f = pick[(c, s)]; X, y = pool[f]; i = np.random.default_rng([seed, draw, c, SIZE_CODE[s]]).choice(len(y), k, replace=False)
            Xs.append(X[i]); ys.append(y[i]); used[f"{c}_{s}"] = [f, k]
    return np.concatenate(Xs), np.concatenate(ys), used

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--models", default="wdcnn,drsn,lstm")
    ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--out", default="../results/e5_size_id.json"); ap.add_argument("--ckpt-dir", default="../results/ckpt_e5")
    a = ap.parse_args(); models = a.models.split(","); seeds = [int(x) for x in a.seeds.split(",")]
    b = build_cwru_splits(ROOT40, loads=(0, 1, 2, 3), segment_len=1024, overlap=0.5, split_seed=42)
    pool = pool_index(b); assert len(pool) == 30, len(pool)
    test_f = {w.file_id for w in b.windows if w.split == "test"}; assert not (test_f & set(pool)), "pool/test overlap"
    picks = {d: draw_files(pool, d) for d in DRAWS}
    if a.smoke:
        for d in DRAWS: print("draw", d, {f"{k[0]}_{k[1]}": v for k, v in picks[d].items()})
        for ss in SETS:
            X, y, used = build(pool, picks[0], ss, 0, 0); print(ss, X.shape, np.bincount(y, minlength=4).tolist(), {k: v[1] for k, v in used.items()}, "steps/epoch", -(-len(y) // 64))
        return
    dev = torch.device("cuda"); Xte, yte = torch.from_numpy(b.X["test"]), torch.from_numpy(b.y["test"])
    os.makedirs(a.ckpt_dir, exist_ok=True)
    res = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["set"], r["draw"], r["model"], r["seed"]) for r in res}; total = len(SETS) * len(DRAWS) * len(models) * len(seeds)
    for s in seeds:
        for d in DRAWS:
            for m in models:
                for ss in SETS:
                    if (ss, d, m, s) in done: continue
                    X, y, used = build(pool, picks[d], ss, s, d); t0 = time.time()
                    print(f"--- {ss} d{d} {m} s{s} n={len(y)}", flush=True)
                    net, steps, cw = train_last(REGISTRY[m](4), torch.from_numpy(X), torch.from_numpy(y), dev, s, epochs=EPOCHS, log=lambda x: None)
                    ck = os.path.join(a.ckpt_dir, f"{ss}_d{d}_{m}_s{s}.pt"); torch.save(net.state_dict(), ck)
                    res.append(dict(set=ss, sizes=SETS[ss], draw=d, model=m, seed=s, files=used, n_train=int(len(y)), class_counts=np.bincount(y, minlength=4).tolist(),
                                    steps=steps, class_weights=cw, ckpt=ck, ckpt_sha256=hashlib.sha256(open(ck, "rb").read()).hexdigest(),
                                    wall_s=round(time.time() - t0, 1), observed=evaluate(net, Xte, yte, dev, s)))
                    tmp = a.out + ".tmp"; json.dump(res, open(tmp, "w")); os.replace(tmp, a.out)
                    print(f"    done {len(res)}/{total} wall={time.time()-t0:.0f}s", flush=True)   # 不打印靶标
    print("=== ALL DONE ===", flush=True)

if __name__ == "__main__":
    main()
