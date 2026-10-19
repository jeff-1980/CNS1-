"""【事后 / 推理，不训练】IEEE Access 版修改所需的修正后证据（2026-10-19，模拟审稿意见 1、4、6）。
只用修正加载器训练的权重：C-INIT-FIX 2×2（80）、修正后 E5 = C-INIT draws 0、2 + C-INIT-FIX draw 1（189）。
  part A  无噪声测试性能：accuracy、macro-F1、逐类 recall、按测试文件损伤尺寸分层的 recall（全部 269 个实例）
  part B  B2 置信度拒识（同 wp5_consequence.py：τ = 干净 val 窗口 MSP 的 5% 分位；噪声种子 5000+17·seed+rep；SNR 网格同 WP5）
          只在 2×2 上做：2×2 只用 train 划分训练，val 未参与训练；E5 用 train∪val 文件池，val 不独立，故不做
          另算 −8 dB 的 risk–coverage（逐实例、置信度并列按期望风险）与 B2 工作点
  part C  噪声预处理对照（2×2，−8 dB）：(i) 白噪声原协议；(ii) 加噪后逐窗口重新 z-score；(iii) 只乘增益 sqrt(1+10^0.8)、不加噪
从 code/ 目录运行（修正加载器；不得设置 CWRU_LEGACY_FIRST_KEY）。"""
import sys, os, json, hashlib, time, collections
assert os.environ.get("CWRU_LEGACY_FIRST_KEY") != "1", "must use the corrected loader"
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "pbx")
import numpy as np, torch
from data.splits_v2 import build_cwru_splits
from data.noise_augment import add_awgn_tensor
from prior_screen import REGISTRY
from arm2x2 import ROOT40, parse
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"); OUT = R
dev = torch.device("cuda"); SNRS_B2 = [10.0, 4.0, 0.0, -4.0, -6.0, -8.0]; NREP = 5
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()

b = build_cwru_splits(ROOT40, loads=(0, 1, 2, 3), segment_len=1024, overlap=0.5, split_seed=42)
H = lambda a: hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()
trainN = {H(x) for x, y in zip(b.X["train"], b.y["train"]) if y == 0}
assert not any(H(x) in trainN for x, y in zip(b.X["test"], b.y["test"]) if y == 0), "Normal test windows found in training split: legacy loader?"
Xt = torch.from_numpy(np.asarray(b.X["test"], np.float32)).to(dev); yt = torch.from_numpy(np.asarray(b.y["test"], np.int64)).to(dev)
Xc = torch.from_numpy(np.asarray(b.X["val"], np.float32)).to(dev)
tf = [w.file_id for w in b.windows if w.split == "test"]
tsize = np.array([("N" if parse(f)[1] is None else parse(f)[1]) for f in tf]); assert len(tsize) == len(yt)
print("test", tuple(Xt.shape), "val", tuple(Xc.shape), "test sizes", dict(collections.Counter(tsize)), flush=True)

def load(r):
    p = r["ckpt"] if os.path.isabs(r["ckpt"]) else os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), r["ckpt"])); assert sha(p) == r["ckpt_sha256"], p
    net = REGISTRY[r["model"]](4); net.load_state_dict(torch.load(p, map_location="cpu")); return net.to(dev).eval()

@torch.no_grad()
def fwd(net, X):
    ps, cs = [], []
    for i in range(0, len(X), 512):
        pr = torch.softmax(net(X[i:i + 512]).float(), 1); c, p = pr.max(1); ps.append(p); cs.append(c)
    return torch.cat(ps), torch.cat(cs)

def clean_metrics(p):
    y = yt; out = dict(acc=float((p == y).float().mean()))
    rec = [float((p[y == k] == k).float().mean()) for k in range(4)]
    f1 = []
    for k in range(4):
        tp = ((p == k) & (y == k)).sum().item(); fp = ((p == k) & (y != k)).sum().item(); fn = ((p != k) & (y == k)).sum().item()
        f1.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    out.update(macro_f1=float(np.mean(f1)), recall=rec)
    pn = p.cpu().numpy(); yn = y.cpu().numpy()
    out["recall_by_test_size"] = {s: float((pn[tsize == s] == yn[tsize == s]).mean()) for s in sorted(set(tsize))}
    return out

def metrics(pred, conf, y, acc):   # identical to wp5_consequence.metrics, plus coverage and selective risk
    Hm = y == 0; Fm = ~Hm; err = pred != y
    far = (acc[Hm] & (pred[Hm] != 0)).float().mean().item(); mfr = (acc[Fm] & (pred[Fm] == 0)).float().mean().item()
    wtr = (acc[Fm] & (pred[Fm] != 0) & (pred[Fm] != y[Fm])).float().mean().item()
    ne = err.sum().item(); sil = (acc & err).sum().item() / ne if ne else float("nan")
    na = acc.sum().item(); srisk = (acc & err).sum().item() / na if na else float("nan")
    return dict(FAR=far, MFR=mfr, HER=0.5 * (far + mfr), WTR=wtr, REJ=1 - acc.float().mean().item(), SILENT=sil,
                COVERAGE=acc.float().mean().item(), SEL_RISK=srisk, N_ACCEPTED=na, N_ERR_ACCEPTED=(acc & err).sum().item())

def rc_curve(conf, err, covs):
    """tie-aware: at coverage k/n accept the k most confident windows; within a tie group, expected errors are proportional."""
    c = conf.cpu().numpy(); e = err.cpu().numpy().astype(float); n = len(c); order = np.argsort(-c, kind="stable"); cs, es = c[order], e[order]
    out = []
    for cv in covs:
        k = int(round(cv * n))
        if k == 0: out.append(float("nan")); continue
        t = cs[k - 1]; above = cs > t; tie = cs == t; n_above = above.sum(); n_tie = tie.sum()
        exp_err = es[above].sum() + (k - n_above) * (es[tie].mean() if n_tie else 0)
        out.append(float(exp_err / k))
    return out

def zscore(x): m = x.mean(-1, keepdim=True); s = x.std(-1, keepdim=True); return (x - m) / (s + 1e-8)

two = json.load(open(f"{R}/cinit_fix_2x2.json"))
e5 = [r for r in json.load(open(f"{R}/cinit_e5.json")) if r["draw"] in (0, 2)] + json.load(open(f"{R}/cinit_fix_e5.json"))
assert len(two) == 80 and len(e5) == 189
covs = list(np.round(np.arange(0.05, 1.0001, 0.05), 2)); res = dict(A=[], B=[], C=[]); t0 = time.time()
GAIN = float(np.sqrt(1 + 10 ** 0.8))
for exp, rows in (("2x2", two), ("E5", e5)):
    for i, r in enumerate(rows):
        net = load(r); key = dict(exp=exp, model=r["model"], seed=r["seed"], arm=r.get("arm"), set=r.get("set"), draw=r.get("draw"), ckpt_sha256=r["ckpt_sha256"])
        p, _ = fwd(net, Xt); res["A"].append(dict(**key, **clean_metrics(p)))
        if exp == "2x2":
            _, cc = fwd(net, Xc); tau = torch.quantile(cc, 0.05).item(); rb = dict(**key, tau=tau, by_snr={})
            for snr in SNRS_B2:
                agg = collections.defaultdict(list); tops, concs = [], []
                for rep in range(NREP):
                    torch.manual_seed(5000 + 17 * r["seed"] + rep); Xn = add_awgn_tensor(Xt, snr); pp, c = fwd(net, Xn)
                    cnt = torch.bincount(pp, minlength=4); tops.append(int(cnt.argmax())); concs.append(cnt.max().item() / len(pp))
                    agg["B0"].append(metrics(pp, c, yt, torch.ones_like(pp, dtype=torch.bool))); agg["B2"].append(metrics(pp, c, yt, c >= tau))
                    if snr == -8.0: agg["rc"].append(rc_curve(c, pp != yt, covs))
                d = dict(top_mode=int(np.bincount(tops, minlength=4).argmax()), conc=float(np.mean(concs)))
                for bname in ("B0", "B2"):
                    d[bname] = {k: float(np.nanmean([m[k] for m in agg[bname]])) if not all(np.isnan(m[k]) for m in agg[bname]) else None for k in agg[bname][0]}
                if snr == -8.0: d["rc"] = list(np.nanmean(np.array(agg["rc"], float), 0)); d["rc_cov"] = covs
                rb["by_snr"][f"{snr:.1f}"] = d
            res["B"].append(rb)
            rc_ = dict(**key, white=[], renorm=[], gain=None)
            for rep in range(5):
                torch.manual_seed(1000 + 17 * r["seed"] + rep); Xn = add_awgn_tensor(Xt, -8.0)
                for nm, X in (("white", Xn), ("renorm", zscore(Xn))):
                    pp, _ = fwd(net, X); cnt = torch.bincount(pp, minlength=4); rc_[nm].append(dict(top=int(cnt.argmax()), conc=cnt.max().item() / len(pp), acc=float((pp == yt).float().mean())))
            pp, _ = fwd(net, Xt * GAIN); cnt = torch.bincount(pp, minlength=4); rc_["gain"] = dict(top=int(cnt.argmax()), conc=cnt.max().item() / len(pp), acc=float((pp == yt).float().mean()))
            res["C"].append(rc_)
        del net; torch.cuda.empty_cache()
        if (i + 1) % 20 == 0: print(f"  {exp} {i+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
os.makedirs(OUT, exist_ok=True)
json.dump(dict(note="post hoc inference on corrected-loader weights; no training; IEEE Access revision 2026-10-19", gain=GAIN, test_sizes=dict(collections.Counter(tsize)),
               n_val=len(Xc), **res), open(f"{OUT}/postfix_inference.json", "w"))
print("=== DONE", {k: len(v) for k, v in res.items()}, f"{time.time()-t0:.0f}s", flush=True)
