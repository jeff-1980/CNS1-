"""C-INIT：固定初始化的确认实验。严格按 results/PREREG_CINIT_confirmation.md（SHA256 a6617899…）。"""
import sys, os, json, time, hashlib, argparse, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "pbx")
import numpy as np, torch
from data.splits_v2 import build_cwru_splits
from prior_screen import REGISTRY
from arm2x2 import ROOT40, ARMS, build_arm, train_last, evaluate
import e5_size_id as E
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results")
tsha = lambda sd: hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in sd.values())).hexdigest()

def make_model(m, ncls, seed):
    torch.manual_seed(20000 + seed); net = REGISTRY[m](ncls); return net, tsha(net.state_dict())

def smoke(b, pool40):
    """初始化与构造顺序、评估历史、新进程无关。"""
    ref = {(m, s): make_model(m, 4, s)[1] for m in ("cnn1d", "wdcnn", "drsn", "lstm") for s in range(5)}
    Xte, yte = torch.from_numpy(b.X["test"][:256]), torch.from_numpy(b.y["test"][:256])
    evaluate(REGISTRY["wdcnn"](4).cuda(), Xte, yte, torch.device("cuda"), 3)          # 模拟"刚跑完一次评估"
    keys = list(ref); random.Random(7).shuffle(keys)
    after = {k: make_model(k[0], 4, k[1])[1] for k in keys}
    assert after == ref, "init depends on history"
    json.dump({f"{m}|s{s}": h for (m, s), h in ref.items()}, open(os.path.join(R, "cinit_smoke_inits.json"), "w"), indent=1)
    print("smoke ok: init independent of order and eval history;", len(ref), "keys")

def run(job, res_path, ckdir, X, y, Xte, yte, dev, extra):
    res = json.load(open(res_path)) if os.path.exists(res_path) else []
    net, ish = make_model(job["model"], 4, job["seed"]); t0 = time.time()
    net, steps, cw = train_last(net, torch.from_numpy(X), torch.from_numpy(y), dev, job["seed"], epochs=job["epochs"], log=lambda x: None)
    ck = os.path.join(ckdir, job["name"] + ".pt"); torch.save(net.state_dict(), ck)
    res.append(dict(**extra, model=job["model"], seed=job["seed"], init_sha256=ish, n_train=int(len(y)), class_counts=np.bincount(y, minlength=4).tolist(),
                    steps=steps, class_weights=cw, ckpt=ck, ckpt_sha256=hashlib.sha256(open(ck, "rb").read()).hexdigest(), wall_s=round(time.time() - t0, 1),
                    observed=evaluate(net, Xte, yte, dev, job["seed"])))
    tmp = res_path + ".tmp"; json.dump(res, open(tmp, "w")); os.replace(tmp, res_path); return len(res)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--models", default="wdcnn,cnn1d,drsn,lstm"); a = ap.parse_args()
    b = build_cwru_splits(ROOT40, loads=(0, 1, 2, 3), segment_len=1024, overlap=0.5, split_seed=42)
    fids = [w.file_id for w in b.windows if w.split == "train"]; pool40 = (fids, b.X["train"], b.y["train"])
    pool = E.pool_index(b); picks = {d: E.draw_files(pool, d) for d in E.DRAWS}
    if a.smoke: smoke(b, pool40); return
    dev = torch.device("cuda"); Xte, yte = torch.from_numpy(b.X["test"]), torch.from_numpy(b.y["test"])
    ck = os.path.join(R, "ckpt_cinit"); os.makedirs(ck, exist_ok=True)
    p2, p5 = os.path.join(R, "cinit_2x2.json"), os.path.join(R, "cinit_e5.json")
    done2 = {(r["arm"], r["model"], r["seed"]) for r in (json.load(open(p2)) if os.path.exists(p2) else [])}
    done5 = {(r["set"], r["draw"], r["model"], r["seed"]) for r in (json.load(open(p5)) if os.path.exists(p5) else [])}
    for m in a.models.split(","):
        for s in range(5):
            for arm in ARMS:
                if (arm, m, s) in done2: continue
                X, y, _, _ = build_arm(pool40, arm, s)
                n = run(dict(model=m, seed=s, epochs=100, name=f"2x2_{arm}_{m}_s{s}"), p2, ck, X, y, Xte, yte, dev, dict(exp="C-2x2", arm=arm, coverage=ARMS[arm][0], ratio=ARMS[arm][1]))
                print(f"    done 2x2 {n} {m}", flush=True)
        if m == "cnn1d": continue
        for s in (0, 1, 2):
            for d in E.DRAWS:
                for ss in E.SETS:
                    if (ss, d, m, s) in done5: continue
                    X, y, used = E.build(pool, picks[d], ss, s, d)
                    n = run(dict(model=m, seed=s, epochs=E.EPOCHS, name=f"e5_{ss}_d{d}_{m}_s{s}"), p5, ck, X, y, Xte, yte, dev, dict(exp="C-E5", set=ss, sizes=E.SETS[ss], draw=d, files=used))
                    print(f"    done e5 {n} {m}", flush=True)     # 不打印靶标
    print("=== ALL DONE ===", flush=True)

if __name__ == "__main__":
    main()
