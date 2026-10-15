"""T3：强度对齐后的领域比较。严格按 results/PREREG_T3_matched_strength.md（已冻结，SHA256 见 results/registry/prereg_t3_sha256.txt）。
从 code/ 目录运行：python t3.py --stage smoke | calib | main
- smoke（§8）：只构造数据，不训练
- calib（§4）：校准种子 900–902，只计算干净测试准确率；§4.1 选定新领域 N；结果写 results/t3_calibration.json
- main（§2、§3）：150 个实例；运行中不打印任何主导类别或预测分布（§11）
数据与训练函数沿用 t2.py（CWRU、ECG 构造与 T2 完全相同；按步调度相同）。"""
from __future__ import annotations
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, os, json, math, time, hashlib, argparse, glob, tarfile, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "pbx")
import numpy as np, pandas as pd, torch, torch.nn as nn
from data.noise_augment import add_awgn_tensor
from models.wdcnn import WDCNN
from models.drsn import DRSN
from models.lstm_model import LSTMClassifier
from posthoc_t1_margin import add_colored
import t2   # 只复用数据构造（cwru_data / cwru_train / ecg_data）与调度 lr_factor；t2 的哈希校验不会在导入时运行

R = os.path.join(_ROOT, "results"); PREREG = os.path.join(R, "PREREG_T3_matched_strength.md")
SPEECH_TGZ = os.environ.get("SPEECH_TGZ", os.path.expanduser("~/论文CNS/data/speech_commands_v0.02.tar.gz"))
SLC_ZIP = os.environ.get("SLC_ZIP", os.path.expanduser("~/论文CNS/data/StarLightCurves.zip"))
CACHE = os.path.expanduser("~/论文CNS/data/t3_cache")
ARCHS = ("wdcnn", "drsn", "lstm"); SEEDS = range(10, 20); CAL_SEEDS = (900, 901, 902)
SNRS = [10.0, 0.0, -4.0, -6.0, -8.0, -10.0, -12.0]
CWRU_HI = 3000; ECG_NAT = 12000; N_HI_GRID = [3000, 6000, 12000]; RHO_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
LO_TARGET, LO_BAND, HI_MIN = 0.78, (0.75, 0.81), 0.95
PROBE_SEED, N_PROBE, N_COLOR = 424242, 1000, 512
WORDS = ["yes", "no", "up", "down"]
tsha = lambda sd: hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in sd.values())).hexdigest()


def check_prereg():
    reg = open(os.path.join(R, "registry/prereg_t3_sha256.txt")).read().split()[0]
    assert hashlib.sha256(open(PREREG, "rb").read()).hexdigest() == reg, "预注册哈希不符，拒绝运行"


def make_model(m, cin, K, seed):
    torch.manual_seed(30000 + seed)
    net = {"wdcnn": lambda: WDCNN(in_channels=cin, d_model=64, num_classes=K),
           "drsn": lambda: DRSN(in_channels=cin, d_model=64, num_classes=K, num_blocks=4),
           "lstm": lambda: LSTMClassifier(in_channels=cin, d_model=64, num_classes=K, hidden=128)}[m]()
    return net, tsha(net.state_dict())


# ---------------- 新领域数据 ----------------
def speech_data():
    """N1：Speech Commands v0.02，四个词；官方 testing_list 为测试，validation_list 不用，其余为训练；16→8 kHz，补零到 8000，逐片段 z-score。"""
    f = os.path.join(CACHE, "speech4.npz")
    if os.path.exists(f):
        z = np.load(f, allow_pickle=True)
        return torch.from_numpy(z["Xtr"]), torch.from_numpy(z["ytr"]).long(), torch.from_numpy(z["Xte"]), torch.from_numpy(z["yte"]).long(), z["meta"].item()
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    lists, wavs = {}, {}
    with tarfile.open(SPEECH_TGZ, "r:gz") as tf:
        for mem in tf:
            nm = mem.name[2:] if mem.name.startswith("./") else mem.name
            if nm in ("validation_list.txt", "testing_list.txt"):
                lists[nm] = set(tf.extractfile(mem).read().decode().split())
            elif mem.isfile() and nm.endswith(".wav") and nm.split("/")[0] in WORDS:
                wavs[nm] = tf.extractfile(mem).read()
    import io
    out = {"tr": ([], [], []), "te": ([], [], [])}; n_pad = 0
    for nm in sorted(wavs):
        if nm in lists["validation_list.txt"]: continue
        part = "te" if nm in lists["testing_list.txt"] else "tr"
        sr, x = wavfile.read(io.BytesIO(wavs[nm])); assert sr == 16000, nm
        x = resample_poly(x.astype(np.float32), 1, 2)
        if len(x) < 8000: n_pad += 1; x = np.pad(x, (0, 8000 - len(x)))
        x = x[:8000]; x = (x - x.mean()) / (x.std() + 1e-8)
        out[part][0].append(x[None].astype(np.float32)); out[part][1].append(WORDS.index(nm.split("/")[0]))
        out[part][2].append(os.path.basename(nm).split("_nohash_")[0])
    meta = dict(n_padded=n_pad, spk_tr=sorted(set(out["tr"][2])), spk_te=sorted(set(out["te"][2])))
    os.makedirs(CACHE, exist_ok=True)
    np.savez(f, Xtr=np.stack(out["tr"][0]), ytr=np.array(out["tr"][1]), Xte=np.stack(out["te"][0]), yte=np.array(out["te"][1]), meta=np.array(meta, dtype=object))
    return speech_data()


def slc_data():
    """N2：StarLightCurves，UCR 官方训练 / 测试划分，K = 3，长度 1024；标签 1..3 → 0..2。"""
    with zipfile.ZipFile(SLC_ZIP) as zf:
        def rd(part):
            # 单数据集包（StarLightCurves_TRAIN.txt）或 UCR 2018 整包（StarLightCurves/StarLightCurves_TRAIN.tsv，口令为档案公开的 someone）；两者是同一份数据
            nm = [n for n in zf.namelist() if n.endswith((f"StarLightCurves_{part}.txt", f"StarLightCurves_{part}.tsv"))][0]
            a = np.loadtxt(zf.open(nm, pwd=b"someone")); return torch.from_numpy(a[:, 1:][:, None].astype(np.float32)), torch.from_numpy(a[:, 0].astype(int) - 1).long()
        Xtr, ytr = rd("TRAIN"); Xte, yte = rd("TEST")
    return Xtr, ytr, Xte, yte, {}


NEW = {"speech": dict(load=speech_data, K=4, cin=1), "slc": dict(load=slc_data, K=3, cin=1)}


def domain(dom):
    """返回 (训练数据函数 seed→(X,y), Xte, yte, cin, K)。"""
    if dom == "cwru":
        pool, Xte, yte = t2.cwru_data(); return (lambda s: t2.cwru_train(pool, s)), Xte, yte, 1, 4
    if dom == "ecg":
        Xtr, ytr, Xte, yte, _, _ = t2.ecg_data(); return (lambda s: (Xtr, ytr)), Xte, yte, 12, 4
    Xtr, ytr, Xte, yte, _ = NEW[dom]["load"](); return (lambda s: (Xtr, ytr)), Xte, yte, NEW[dom]["cin"], NEW[dom]["K"]


def label_noise(y, rho, seed, K):
    """与 T2 相同（DEVIATIONS_T2.md 第 3 条）：rng([seed, 7]) 无放回选 round(ρ·n) 个，均匀改成其余 K−1 类之一。"""
    rng = np.random.default_rng([seed, 7]); y = y.clone(); n = len(y)
    sel = rng.choice(n, int(round(rho * n)), replace=False); shift = rng.integers(1, K, size=len(sel))
    y[torch.from_numpy(sel)] = (y[torch.from_numpy(sel)] + torch.from_numpy(shift)) % K
    return y


def majority(y, K):
    """训练集（干净标签）多数类；并列（如 CWRU 均衡）时返回 −1，不算“多数类”（§5 BASE_DOMINATED）。"""
    c = torch.bincount(y, minlength=K); return int(c.argmax()) if int((c == c.max()).sum()) == 1 else -1


# ---------------- 训练（与 T2 §3 相同） ----------------
def train_steps(net, X, y, seed, T, K, bs=64):
    torch.manual_seed(seed); np.random.seed(seed)
    net = net.cuda(); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: t2.lr_factor(s, T))
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
def evaluate(net, Xte, yte, seed, K, n_reps=5):
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
def probe_and_color(net, Xte, Z, K):
    z = predict(net, Z); t = z.topk(2, 1).values; p = z.argmax(1).numpy()
    Xp = Xte[:N_COLOR]; torch.manual_seed(31337)
    white = int(np.bincount(predict(net, add_awgn_tensor(Xp.cuda(), -8.0).cpu()).argmax(1).numpy(), minlength=K).argmax())
    col = {}
    for kind in ("low", "high"):
        g = torch.Generator().manual_seed(4040); col[kind] = int(np.bincount(predict(net, add_colored(Xp, -8.0, kind, g, False)).argmax(1).numpy(), minlength=K).argmax())
    return dict(preds=p.astype(int).tolist(), mode=int(np.bincount(p, minlength=K).argmax()), dist=np.bincount(p, minlength=K).tolist(),
                margin_median=float((t[:, 0] - t[:, 1]).median())), dict(white=white, low=col["low"], high=col["high"])


# ---------------- 阶段 ----------------
def smoke(news):
    print("=== §8 冒烟：只构造数据，不训练 ===")
    for dom in ["cwru", "ecg"] + news:
        data, Xte, yte, cin, K = domain(dom); X, y = data(0); cnt = torch.bincount(y, minlength=K)
        print(f"[{dom}] 训练 {tuple(X.shape)} 类计数 {cnt.tolist()} 类权重 {[round(v, 3) for v in (cnt.sum() / (K * cnt)).tolist()]}；测试 {tuple(Xte.shape)} {torch.bincount(yte, minlength=K).tolist()}")
        yn = label_noise(y, 0.3, 0, K); print(f"[{dom}] ρ=0.3 实际翻转比例 {(yn != y).float().mean():.4f}，翻转后类计数 {torch.bincount(yn, minlength=K).tolist()}")
        for m in ARCHS:
            print(f"[形状] {dom} {m} in={cin}×{Xte.shape[-1]} → {tuple(make_model(m, cin, K, 0)[0](torch.randn(2, cin, Xte.shape[-1])).shape)}")
        if dom == "speech":
            meta = NEW["speech"]["load"]()[4]; inter = set(meta["spk_tr"]) & set(meta["spk_te"])
            print(f"[speech] 说话人 训练 {len(meta['spk_tr'])} 测试 {len(meta['spk_te'])} 交叉 {len(inter)}；补零片段 {meta['n_padded']}")
    for T in (3000, 12000):
        f = [t2.lr_factor(s, T) for s in range(T)]; print(f"[调度] T={T}: lr 系数 首 {f[0]:.4f} 峰 {max(f):.4f} 末 {f[-1]:.2e}")
    reg = os.path.join(R, "registry/raw_hashes_t3.csv")
    rows = [dict(dataset=d, path=os.path.basename(p), sha256=hashlib.sha256(open(p, "rb").read()).hexdigest())
            for d, p in (("speech", SPEECH_TGZ), ("slc", SLC_ZIP)) if d in news and os.path.exists(p)]
    if os.path.exists(reg):
        old = pd.read_csv(reg); bad = [r["path"] for r in rows if r["path"] in set(old.path) and old.set_index("path").sha256[r["path"]] != r["sha256"]]
        print(f"[哈希] 新领域原始文件与登记表不符：{bad or '无'}")
        add = [r for r in rows if r["path"] not in set(old.path)]
        if add: pd.concat([old, pd.DataFrame(add)]).to_csv(reg, index=False); print(f"[哈希] 新增登记：{add}")
    else:
        pd.DataFrame(rows).to_csv(reg, index=False); print(f"[哈希] 首次登记 {reg}：{rows}")
    print("[SMOKE] 完成（未训练）")


def calib(cands):
    """§4：只计算干净准确率。"""
    out_f = os.path.join(R, "t3_calibration.json"); res = json.load(open(out_f)) if os.path.exists(out_f) else {}
    save = lambda: json.dump(res, open(out_f, "w"), indent=1)

    def run(dom, m, T, rho=None):
        data, Xte, yte, cin, K = D[dom]; accs = []
        for s in CAL_SEEDS:
            X, y = data(s); y = label_noise(y, rho, s, K) if rho else y
            net, _ = make_model(m, cin, K, s); net, _, _ = train_steps(net, X, y, s, T, K); accs.append(clean_acc(net, Xte, yte))
        r = dict(T=T, rho=rho, accs=accs, median=float(np.median(accs))); print(f"  {dom}|{m} T={T} ρ={rho} median={r['median']:.3f}", flush=True); return r

    D = {}
    # §4.1 新领域可行性：按顺序，第一个在 ≥ 2 个架构上达到 0.95 的候选选定为 N
    feas = res.setdefault("feasibility", {})
    if "selected" not in res:
        sel = None
        for c in cands:
            D[c] = domain(c); fc = feas.setdefault(c, {})
            for m in ARCHS:
                if m in fc: continue
                rows = []
                for T in N_HI_GRID:
                    rows.append(run(c, m, T))
                    if rows[-1]["median"] >= HI_MIN: break
                ok = [r for r in rows if r["median"] >= HI_MIN]
                fc[m] = dict(grid=rows, hi_steps=ok[0]["T"] if ok else N_HI_GRID[-1], hi_median=(ok[0] if ok else rows[-1])["median"], reached=bool(ok)); save()
            if sum(fc[m]["reached"] for m in ARCHS) >= 2: sel = c; break
        res["selected"] = sel; save()
    N = res["selected"]
    if N is None:
        print("[§4.1] 两个候选都不满足：T3 不进入主实验", flush=True); return res
    D.setdefault(N, domain(N)); D.setdefault("cwru", domain("cwru"))
    ch = res.setdefault("chosen", {})
    for m in ARCHS:
        f = feas[N][m]; ch[f"{N}|{m}"] = dict(hi_steps=f["hi_steps"], hi_median=f["hi_median"], hi_gap=not f["reached"])
    for m in ARCHS:   # §4.2 CWRU 高档 3000 步
        k = f"cwru|{m}"
        if k in ch and "hi_median" in ch[k]: continue
        r = run("cwru", m, CWRU_HI); ch[k] = dict(hi_steps=CWRU_HI, hi_median=r["median"], hi_gap=r["median"] < HI_MIN, hi_grid=[r]); save()
    for dom in ("cwru", N):   # §4.3 低档 ρ
        for m in ARCHS:
            k = f"{dom}|{m}"
            if "rho" in ch[k]: continue
            rows = []
            for rho in RHO_GRID:
                rows.append(run(dom, m, ch[k]["hi_steps"], rho))
                if rows[-1]["median"] < LO_BAND[0]: break   # 实现说明：ρ 增大准确率单调下降时，此后各点离 0.78 更远（见 DEVIATIONS_T3.md）
            best = min(rows, key=lambda r: (abs(r["median"] - LO_TARGET), r["rho"]))
            ch[k].update(lo_grid=rows, rho=best["rho"], lo_median=best["median"], lo_gap=not (LO_BAND[0] <= best["median"] <= LO_BAND[1])); save()
            print(f"[校准] {k} → hi {ch[k]['hi_steps']} ({ch[k]['hi_median']:.3f}) ρ={ch[k]['rho']} ({ch[k]['lo_median']:.3f})", flush=True)
    return res


def main_run():
    cal = json.load(open(os.path.join(R, "t3_calibration.json"))); N = cal["selected"]; ch = cal["chosen"]
    assert N is not None, "§4.1 无可行新领域，不跑主实验"
    ckdir = os.path.join(R, "ckpt_t3"); os.makedirs(ckdir, exist_ok=True)
    out = os.path.join(R, "t3_main.json"); res = json.load(open(out)) if os.path.exists(out) else []
    done = {(r["domain"], r["tier"], r["model"], r["seed"]) for r in res}; total = 150
    for dom in ("cwru", N, "ecg"):
        data, Xte, yte, cin, K = domain(dom)
        Z = torch.randn(N_PROBE, *Xte.shape[1:], generator=torch.Generator().manual_seed(PROBE_SEED))
        for s in SEEDS:
            X, y = data(s)
            for m in ARCHS:
                tiers = [("natural", ECG_NAT, None)] if dom == "ecg" else [("hi", ch[f"{dom}|{m}"]["hi_steps"], None), ("lo", ch[f"{dom}|{m}"]["hi_steps"], ch[f"{dom}|{m}"]["rho"])]
                for tier, T, rho in tiers:
                    if (dom, tier, m, s) in done: continue
                    yt = label_noise(y, rho, s, K) if rho else y
                    net, ish = make_model(m, cin, K, s); t0 = time.time()
                    net, steps, cw = train_steps(net, X, yt, s, T, K); assert steps == T
                    ck = os.path.join(ckdir, f"{dom}_{tier}_{m}_s{s}.pt"); torch.save(net.state_dict(), ck)
                    probe, color = probe_and_color(net, Xte, Z, K)
                    res.append(dict(domain=dom, tier=tier, model=m, seed=s, K=K, steps=steps, rho=rho, n_train=int(len(X)),
                                    flip_frac=float((yt != y).float().mean()), class_counts=torch.bincount(yt, minlength=K).tolist(),
                                    train_majority=majority(y, K), class_weights=cw,
                                    init_sha256=ish, ckpt=os.path.relpath(ck, _ROOT), ckpt_sha256=hashlib.sha256(open(ck, "rb").read()).hexdigest(),
                                    clean_acc=clean_acc(net, Xte, yte), observed=evaluate(net, Xte, yte, s, K), probe=probe, color=color,
                                    wall_s=round(time.time() - t0, 1)))
                    tmp = out + ".tmp"; json.dump(res, open(tmp, "w")); os.replace(tmp, out)
                    print(f"    done {len(res)}/{total} {dom} {tier} {m} s{s} steps={steps} wall={time.time() - t0:.0f}s", flush=True)   # 不打印靶标
    print("=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["smoke", "calib", "main"], required=True)
    ap.add_argument("--new", nargs="+", default=["speech", "slc"], help="§4.1 候选顺序；smoke 时只检查已下载的候选"); a = ap.parse_args()
    check_prereg()
    {"smoke": lambda: smoke(a.new), "calib": lambda: calib(a.new), "main": main_run}[a.stage]()
