"""T2：吸引子跨种子确定性——领域 × 学习强度。严格按 results/PREREG_T2_domain_strength.md（已冻结，SHA256 a1c5d277…）。
从 code/ 目录运行：python t2.py --stage smoke | calib | main [--domain cwru|ecg]
- smoke（§8）：只构造数据，不训练
- calib（§4）：校准种子 900–902，只计算干净测试准确率；结果写 results/t2_calibration.json
- main（§2、§3、§5）：180 个实例；运行中不打印任何主导类别或预测分布（§11）"""
from __future__ import annotations
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, os, ast, json, math, time, hashlib, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "pbx")
import numpy as np, pandas as pd, torch, torch.nn as nn
from data.noise_augment import add_awgn_tensor
from data.splits_v2 import build_cwru_splits
from models.wdcnn import WDCNN
from models.drsn import DRSN
from models.lstm_model import LSTMClassifier
import arm2x2
from posthoc_t1_margin import add_colored

R = os.path.join(_ROOT, "results"); PREREG = os.path.join(R, "PREREG_T2_domain_strength.md")
CWRU40 = os.environ.get("CWRU40_DIR", os.path.expanduser("~/论文CNS/data/cwru_12k_de"))
PTBXL = os.environ.get("PTBXL_DIR", os.path.expanduser("~/论文CNS/data/ptbxl_raw/ptb-xl-1.0.3"))
CACHE = os.path.expanduser("~/论文CNS/data/ptbxl_cache_t2")
ARCHS = ("wdcnn", "drsn", "lstm"); SEEDS = range(10); CAL_SEEDS = (900, 901, 902); K = 4
SNRS = [10.0, 0.0, -4.0, -6.0, -8.0, -10.0, -12.0]
CWRU_STRONG = 3000; ECG_STRONG_GRID = [1500, 3000, 6000, 12000]
WEAK_STEPS_GRID = [25, 50, 100, 200, 400, 800, 1500]; RHO_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
PROBE_SEED, N_PROBE, N_COLOR = 424242, 1000, 512
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
ECG_CLASSES = ["NORM", "MI", "STTC", "CD"]; CWRU_CLASSES = ["Normal", "IR", "OR", "Ball"]
tsha = lambda sd: hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in sd.values())).hexdigest()


def check_prereg():
    reg = open(os.path.join(R, "registry/prereg_t2_sha256.txt")).read().split()[0]
    assert hashlib.sha256(open(PREREG, "rb").read()).hexdigest() == reg, "预注册哈希不符，拒绝运行"


def make_model(m, cin, seed):
    torch.manual_seed(20000 + seed)
    net = {"wdcnn": lambda: WDCNN(in_channels=cin, d_model=64, num_classes=K),
           "drsn": lambda: DRSN(in_channels=cin, d_model=64, num_classes=K, num_blocks=4),
           "lstm": lambda: LSTMClassifier(in_channels=cin, d_model=64, num_classes=K, hidden=128)}[m]()
    return net, tsha(net.state_dict())


# ---------------- 数据 ----------------
def cwru_data():
    b = build_cwru_splits(CWRU40, loads=(0, 1, 2, 3), segment_len=1024, overlap=0.5, split_seed=42)
    fids = [w.file_id for w in b.windows if w.split == "train"]
    return (fids, b.X["train"], b.y["train"]), torch.from_numpy(b.X["test"]).float(), torch.from_numpy(b.y["test"]).long()


def cwru_train(pool, seed):
    X, y, alloc, idx = arm2x2.build_arm(pool, "CALL_R11", seed)
    return torch.from_numpy(X).float(), torch.from_numpy(y).long()


def read_12(rel):
    hea = open(os.path.join(PTBXL, rel + ".hea")).read().split("\n")
    n_sig, fs, n = (int(v) for v in hea[0].split()[1:4]); sig = [l.split() for l in hea[1:1 + n_sig]]
    assert n_sig == 12 and fs == 100 and n == 1000 and [s[-1].upper() for s in sig] == LEADS and all(s[1] == "16" for s in sig), rel
    gain = np.array([float(s[2].split("/")[0].split("(")[0]) for s in sig], np.float32)
    base = np.array([int(s[2].split("/")[0].split("(")[1].rstrip(")")) if "(" in s[2] else int(s[4]) for s in sig], np.float32)
    x = ((np.fromfile(os.path.join(PTBXL, rel + ".dat"), dtype="<i2").reshape(n, n_sig).astype(np.float32) - base) / gain).T
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)          # 逐导联 z-score


def ecg_data():
    """folds 1–8 全部单一超类记录（训练），folds 9–10（测试）；12 导联。缓存在仓库外。"""
    f = os.path.join(CACHE, "ecg12.npz")
    if os.path.exists(f):
        z = np.load(f); return (torch.from_numpy(z["Xtr"]), torch.from_numpy(z["ytr"]).long(),
                                torch.from_numpy(z["Xte"]), torch.from_numpy(z["yte"]).long(), z["ids_tr"], z["ids_te"])
    db = pd.read_csv(os.path.join(PTBXL, "ptbxl_database.csv"), index_col="ecg_id")
    st = pd.read_csv(os.path.join(PTBXL, "scp_statements.csv"), index_col=0); st = st[st.diagnostic == 1]
    sup = db.scp_codes.apply(lambda s: sorted({st.loc[k, "diagnostic_class"] for k in ast.literal_eval(s) if k in st.index}))
    db = db[sup.apply(len) == 1].copy(); db["sup"] = sup[db.index].str[0]; db = db[db.sup.isin(ECG_CLASSES)]
    out = {}
    for nm, sel in (("tr", db.strat_fold <= 8), ("te", db.strat_fold >= 9)):
        d = db[sel].sort_index()
        out["X" + nm] = np.stack([read_12(r) for r in d.filename_lr]).astype(np.float32)
        out["y" + nm] = np.array([ECG_CLASSES.index(s) for s in d.sup]); out["ids_" + nm] = d.index.values
    os.makedirs(CACHE, exist_ok=True); np.savez(f, **out)
    return ecg_data()


def label_noise(y, rho, seed):
    """对称标签噪声：用 rng([seed, 7]) 无放回选 round(ρ·n) 个样本，均匀改成其余 3 类之一。"""
    rng = np.random.default_rng([seed, 7]); y = y.clone(); n = len(y)
    sel = rng.choice(n, int(round(rho * n)), replace=False); shift = rng.integers(1, K, size=len(sel))
    y[torch.from_numpy(sel)] = (y[torch.from_numpy(sel)] + torch.from_numpy(shift)) % K
    return y


# ---------------- 训练（§3：按步调度） ----------------
def lr_factor(s, T):
    warm = max(1, round(0.05 * T))
    return (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, T - warm)))


def train_steps(net, X, y, seed, T, bs=64):
    torch.manual_seed(seed); np.random.seed(seed)
    net = net.cuda(); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, T))
    cnt = torch.bincount(y, minlength=K).float(); cw = (cnt.sum() / (K * cnt.clamp(min=1))).cuda()
    lossf = nn.CrossEntropyLoss(weight=cw); n = len(X); s = 0; Xg, yg = X.cuda(), y.cuda()
    while s < T:
        net.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs].cuda(); opt.zero_grad(); loss = lossf(net(Xg[b]), yg[b]); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); s += 1
            if s >= T: break
    return net, s, cw.cpu().tolist()


@torch.no_grad()
def predict(net, X, bs=256):
    net.eval(); return torch.cat([net(X[i:i + bs].cuda()).float().cpu() for i in range(0, len(X), bs)])


def clean_acc(net, Xte, yte): return float((predict(net, Xte).argmax(1) == yte).float().mean())


@torch.no_grad()
def evaluate(net, Xte, yte, seed, n_reps=5):
    """与 arm2x2.evaluate 同口径（SNR 网格按 §3 的预注册）。"""
    net.eval(); out = {}
    for snr in SNRS:
        reps = []
        for r in range(n_reps):
            torch.manual_seed(1000 + 17 * seed + r)
            p = torch.cat([net(add_awgn_tensor(Xte[i:i + 256].cuda(), snr)).argmax(1).cpu() for i in range(0, len(Xte), 256)]).numpy()
            c = np.bincount(p, minlength=K); reps.append(dict(top=int(c.argmax()), conc=float(c.max() / len(p)), acc=float((p == yte.numpy()).mean()), dist=c.tolist()))
        tops = [r["top"] for r in reps]
        out[f"{snr:.1f}"] = dict(top_mode=int(np.bincount(tops, minlength=K).argmax()), tops=tops, conc_mean=float(np.mean([r["conc"] for r in reps])),
                                 acc_mean=float(np.mean([r["acc"] for r in reps])), dists=[r["dist"] for r in reps])
    return out


@torch.no_grad()
def probe_and_color(net, Xte, Z):
    z = predict(net, Z); t = z.topk(2, 1).values; p = z.argmax(1).numpy()
    Xp = Xte[:N_COLOR]; torch.manual_seed(31337)
    white = int(np.bincount(predict(net, add_awgn_tensor(Xp.cuda(), -8.0).cpu()).argmax(1).numpy(), minlength=K).argmax())
    col = {}
    for kind in ("low", "high"):
        g = torch.Generator().manual_seed(4040); col[kind] = int(np.bincount(predict(net, add_colored(Xp, -8.0, kind, g, False)).argmax(1).numpy(), minlength=K).argmax())
    return dict(preds=p.astype(int).tolist(), mode=int(np.bincount(p, minlength=K).argmax()), dist=np.bincount(p, minlength=K).tolist(),
                margin_median=float((t[:, 0] - t[:, 1]).median())), dict(white=white, low=col["low"], high=col["high"])


# ---------------- 阶段 ----------------
def smoke():
    print("=== §8 冒烟：只构造数据，不训练 ===")
    pool, Xc, yc = cwru_data(); X, y = cwru_train(pool, 0)
    cnt = torch.bincount(y, minlength=K); print(f"[CWRU] 训练 {tuple(X.shape)} 类计数 {cnt.tolist()} 类权重 {(cnt.sum() / (K * cnt)).tolist()}；测试 {tuple(Xc.shape)} {torch.bincount(yc, minlength=K).tolist()}")
    yn = label_noise(y, 0.4, 0); print(f"[CWRU] ρ=0.4 实际翻转比例 {(yn != y).float().mean():.4f}，翻转后类计数 {torch.bincount(yn, minlength=K).tolist()}")
    Xtr, ytr, Xte, yte, _, _ = ecg_data(); cnt = torch.bincount(ytr, minlength=K)
    print(f"[ECG] 训练 {tuple(Xtr.shape)} 类计数 {dict(zip(ECG_CLASSES, cnt.tolist()))} 逆频率类权重 {[round(v, 3) for v in (cnt.sum() / (K * cnt)).tolist()]}")
    print(f"[ECG] 测试 {tuple(Xte.shape)} 类计数 {dict(zip(ECG_CLASSES, torch.bincount(yte, minlength=K).tolist()))}；导联顺序 {LEADS}（逐文件头核对通过）；"
          f"逐导联 |μ|max={Xtr[:500].mean(-1).abs().max():.2e} σ≈{Xtr[:500].std(-1).mean():.3f}")
    yn = label_noise(ytr, 0.4, 0); print(f"[ECG] ρ=0.4 实际翻转比例 {(yn != ytr).float().mean():.4f}，翻转后类计数 {torch.bincount(yn, minlength=K).tolist()}")
    for T in (25, 12000):
        f = [lr_factor(s, T) for s in range(T)]; print(f"[调度] T={T}: warmup {max(1, round(0.05 * T))} 步；lr 系数 首 {f[0]:.4f} 峰 {max(f):.4f} 末 {f[-1]:.2e}")
    for m in ARCHS:
        for cin, L in ((1, 1024), (12, 1000)):
            print(f"[形状] {m} in={cin}×{L} → {tuple(make_model(m, cin, 0)[0](torch.randn(2, cin, L)).shape)}")
    # 原始文件哈希核对
    reg = pd.read_csv(os.path.join(R, "registry/raw_hashes_c6p.csv")); reg = reg[reg.dataset == "ptbxl"]; bad = 0
    for p, h in zip(reg.path, reg.sha256):
        if hashlib.sha256(open(os.path.join(PTBXL, p), "rb").read()).hexdigest() != h: bad += 1
    print(f"[哈希] PTB-XL {len(reg)} 个文件，不符 {bad}")
    reg = pd.read_csv(os.path.join(R, "registry/raw_hashes.csv")); reg = reg[reg.dataset == "CWRU40"]; bad = miss = 0
    for p, h in zip(reg.path, reg.sha256):
        q = os.path.join(CWRU40, os.path.basename(p))
        if not os.path.exists(q): miss += 1
        elif hashlib.sha256(open(q, "rb").read()).hexdigest() != h: bad += 1
    print(f"[哈希] CWRU40 {len(reg)} 个登记文件，不符 {bad}，缺失 {miss}")
    print("[SMOKE] 完成（未训练）")


def calib(domains):
    """§4：只计算干净准确率。按网格从小到大训练，满足规则即停（与"取最小值"等价）；不满足则跑完整个网格取最接近 0.60。"""
    out_f = os.path.join(R, "t2_calibration.json"); res = json.load(open(out_f)) if os.path.exists(out_f) else {}
    for dom in domains:
        if dom == "cwru": pool, Xte, yte = cwru_data(); cin = 1
        else: Xtr, ytr, Xte, yte, _, _ = ecg_data(); cin = 12
        def data(seed):
            return cwru_train(pool, seed) if dom == "cwru" else (Xtr, ytr)
        for m in ARCHS:
            key = f"{dom}|{m}"
            if key in res and "chosen" in res[key]: continue
            rec = res.setdefault(key, {}); t0 = time.time()
            def run(T, rho=None):
                accs = []
                for s in CAL_SEEDS:
                    X, y = data(s); y = label_noise(y, rho, s) if rho else y
                    net, _ = make_model(m, cin, s); net, _, _ = train_steps(net, X, y, s, T); accs.append(clean_acc(net, Xte, yte))
                return dict(T=T, rho=rho, accs=accs, median=float(np.median(accs)))
            # 强档
            if dom == "cwru":
                rec["strong"] = [run(CWRU_STRONG)]; strong = CWRU_STRONG
            else:
                rec["strong"] = []; strong = None
                for T in ECG_STRONG_GRID:
                    r = run(T); rec["strong"].append(r); print(f"  {key} strong T={T} median={r['median']:.3f}", flush=True)
                    if r["median"] >= 0.80: strong = T; break
                if strong is None: strong = ECG_STRONG_GRID[-1]
            def pick(rows, key_):
                ok = [r for r in rows if 0.55 <= r["median"] <= 0.65]
                return (ok[0][key_], True) if ok else (min(rows, key=lambda r: abs(r["median"] - 0.60))[key_], False)
            rec["weak_steps"] = []
            for T in WEAK_STEPS_GRID:
                r = run(T); rec["weak_steps"].append(r); print(f"  {key} weak_steps T={T} median={r['median']:.3f}", flush=True)
                if 0.55 <= r["median"] <= 0.65: break
            rec["weak_label"] = []
            for rho in RHO_GRID:
                r = run(strong, rho); rec["weak_label"].append(r); print(f"  {key} weak_label ρ={rho} median={r['median']:.3f}", flush=True)
                if 0.55 <= r["median"] <= 0.65: break
            ws, ws_in = pick(rec["weak_steps"], "T"); wl, wl_in = pick(rec["weak_label"], "rho")
            sm = [r for r in rec["strong"] if r["T"] == strong][0]["median"]
            rec["chosen"] = dict(strong_steps=strong, strong_median=sm, weak_steps=ws, weak_steps_in_range=ws_in, rho=wl, rho_in_range=wl_in,
                                 strength_gap=bool(sm < (0.95 if dom == "cwru" else 0.80)), wall_s=round(time.time() - t0))
            print(f"[校准] {key} → {rec['chosen']}", flush=True)
            json.dump(res, open(out_f, "w"), indent=1)
    return res


def main_run(domains):
    cal = json.load(open(os.path.join(R, "t2_calibration.json")))
    ckdir = os.path.join(R, "ckpt_t2"); os.makedirs(ckdir, exist_ok=True)
    for dom in domains:
        out = os.path.join(R, f"t2_{dom}.json"); res = json.load(open(out)) if os.path.exists(out) else []
        done = {(r["tier"], r["model"], r["seed"]) for r in res}
        if dom == "cwru": pool, Xte, yte = cwru_data(); cin = 1
        else: Xtr, ytr, Xte, yte, _, _ = ecg_data(); cin = 12
        Z = torch.randn(N_PROBE, *Xte.shape[1:], generator=torch.Generator().manual_seed(PROBE_SEED))
        total = len(SEEDS) * len(ARCHS) * 3
        for s in SEEDS:
            X, y = cwru_train(pool, s) if dom == "cwru" else (Xtr, ytr)
            for m in ARCHS:
                c = cal[f"{dom}|{m}"]["chosen"]
                for tier, T, rho in (("strong", c["strong_steps"], None), ("weak_steps", c["weak_steps"], None), ("weak_label", c["strong_steps"], c["rho"])):
                    if (tier, m, s) in done: continue
                    yt = label_noise(y, rho, s) if rho else y
                    net, ish = make_model(m, cin, s); t0 = time.time()
                    net, steps, cw = train_steps(net, X, yt, s, T); assert steps == T
                    ck = os.path.join(ckdir, f"{dom}_{tier}_{m}_s{s}.pt"); torch.save(net.state_dict(), ck)
                    probe, color = probe_and_color(net, Xte, Z)
                    res.append(dict(domain=dom, tier=tier, model=m, seed=s, steps=steps, rho=rho, n_train=int(len(X)),
                                    flip_frac=float((yt != y).float().mean()), class_counts=torch.bincount(yt, minlength=K).tolist(), class_weights=cw,
                                    init_sha256=ish, ckpt=os.path.relpath(ck, _ROOT), ckpt_sha256=hashlib.sha256(open(ck, "rb").read()).hexdigest(),
                                    clean_acc=clean_acc(net, Xte, yte), observed=evaluate(net, Xte, yte, s), probe=probe, color=color,
                                    wall_s=round(time.time() - t0, 1)))
                    tmp = out + ".tmp"; json.dump(res, open(tmp, "w")); os.replace(tmp, out)
                    print(f"    {dom} done {len(res)}/{total} {tier} {m} s{s} steps={steps} wall={time.time() - t0:.0f}s", flush=True)   # 不打印靶标
    print("=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["smoke", "calib", "main"], required=True)
    ap.add_argument("--domain", nargs="+", default=["cwru", "ecg"]); a = ap.parse_args()
    check_prereg()
    {"smoke": smoke, "calib": lambda: calib(a.domain), "main": lambda: main_run(a.domain)}[a.stage]()
