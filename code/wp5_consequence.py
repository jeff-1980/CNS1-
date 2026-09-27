"""WP5：设计选择的样本级决策后果（纯推理，复用已有权重）。严格按 results/PREREG_WP5_decision_consequence.md（SHA256 3d785ddd…）。
默认环境跑非 Mamba 实例；--mamba 在 mamba-cu12 环境跑 vibrmamba / mamba2。"""
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PAPERB_CODE = _os.environ.get("PAPERB_CODE", _os.path.join(_ROOT, "external/paperB_code"))
_PAPERB_RESULTS = _os.environ.get("PAPERB_RESULTS", _os.path.join(_ROOT, "external/paperB_results"))
_CWRU40 = _os.environ.get("CWRU40_DIR", _os.path.join(_ROOT, "data/cwru_12k_de"))
_CWRU16 = _os.environ.get("CWRU16_DIR", _os.path.join(_ROOT, "data/cwru_16"))
_PU = _os.environ.get("PU_DIR", _os.path.join(_ROOT, "data/paderborn"))
_JNU = _os.environ.get("JNU_DIR", _os.path.join(_ROOT, "data/jnu"))
_JNU_RAW = _os.environ.get("JNU_RAW_DIR", _JNU)
import sys, os, json, time, hashlib, argparse, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "pbx")
try:
    import mamba_ssm  # noqa
except Exception:
    stub = types.ModuleType("mamba_ssm"); stub.Mamba = stub.Mamba2 = None; sys.modules["mamba_ssm"] = stub
import numpy as np, torch
from data.splits_v2 import build_cwru_splits as build40, _segment_and_window, GAP_SAMPLES
from data.noise_augment import add_awgn_tensor
from prior_screen import REGISTRY
import pu_wp2 as W
sys.path.insert(0, _PAPERB_CODE)
from pbsrc.train import trainer as T
from pbsrc.train.config import TrainConfig
from pbsrc.data.splits_v2 import build_cwru_splits as build16
R = (_ROOT + "/results"); PRE = f"{R}/PREREG_WP5_decision_consequence.md"
assert hashlib.sha256(open(PRE, "rb").read()).hexdigest() == open(f"{R}/registry/prereg_wp5_sha256.txt").read().split()[0], "prereg hash mismatch"
SNRS = [10.0, 4.0, 0.0, -4.0, -6.0, -8.0]; NREP = 5; MAMBA = {"vibrmamba", "mamba2"}
dev = torch.device("cuda")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()

def sf(x):  # 谱平坦度，x: (N,1,L) GPU 张量
    P = torch.fft.rfft(x.squeeze(1).double(), dim=-1).abs().pow(2)[:, 1:] + 1e-20
    return (P.log().mean(-1).exp() / P.mean(-1)).float()

def pu_calib(block):
    avail = W._codes_present(W.PU_ROOT); X = []
    for c in W.block_codes(block):
        for f in [f for f in avail[c] if f.name.startswith(W.OP)]:
            sig = W._load_pu_signal(f); n = len(sig); a = int(n * 0.6) + GAP_SAMPLES; z = a + int(n * 0.2)
            y, w = [], []; _segment_and_window(sig[a:z], f.stem, "val", 0, W.SEG, W.STEP, True, a, X, y, w)
    return np.stack(X).astype(np.float32)

def forward(net, X):
    ps, cs = [], []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            pr = torch.softmax(net(X[i:i + 512]).float(), 1); c, p = pr.max(1); ps.append(p); cs.append(c)
    return torch.cat(ps), torch.cat(cs)

def metrics(pred, conf, y, acc):
    Hm = y == 0; Fm = ~Hm; err = pred != y
    far = (acc[Hm] & (pred[Hm] != 0)).float().mean().item(); mfr = (acc[Fm] & (pred[Fm] == 0)).float().mean().item()
    wtr = (acc[Fm] & (pred[Fm] != 0) & (pred[Fm] != y[Fm])).float().mean().item()
    ne = err.sum().item(); sil = (acc & err).sum().item() / ne if ne else float("nan")
    return dict(FAR=far, MFR=mfr, HER=0.5 * (far + mfr), WTR=wtr, REJ=1 - acc.float().mean().item(), SILENT=sil)

def build_net(fam, m, ncls):
    if fam == "proj": return REGISTRY[m](ncls).to(dev)
    net = T.build_model(TrainConfig(model_name=m)).to(dev)
    if m == "mamba2":
        import mamba_ssm.modules.mamba2 as M2
        for mod in net.modules():
            if isinstance(mod, M2.Mamba2): mod.use_mem_eff_path = False
        M2.causal_conv1d_fn = None
    return net

def ck(p): return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), p))

def jobs(mamba):
    J = []
    for fn, lab in (("arm2x2.json", "F1"), ("arm2x2_lt.json", "F1")):
        for r in json.load(open(f"{R}/{fn}")):
            J.append(dict(group="cwru40", fam="proj", ncls=4, factor="F1", src=fn, cond=r["arm"], cov=r["coverage"], model=r["model"], seed=r["seed"],
                          ckpt=ck(r["ckpt"]), sha=r["ckpt_sha256"], old_top=r["observed"]["-8.0"]["top_mode"]))
    for fn, fac in (("e4_paperB_harness.json", "F2"), ("wp1_cwru_clean.json", "F2"), ("wp1_cwru_clean_M.json", "F2"), ("wp1_cwru_m2.json", "F2"), ("wp1_cwru_awgn.json", "F3"), ("wp1_cwru_awgn_M.json", "F3")):
        for r in json.load(open(f"{R}/{fn}")):
            for tag in ("best", "final"):
                J.append(dict(group="cwru16", fam="pb", ncls=4, factor=fac, src=fn, cond=tag, model=r["model"], seed=r["seed"], best_epoch=r.get("best_epoch"),
                              ckpt=ck(r[f"ckpt_{tag}"]), sha=r[f"ckpt_{tag}_sha256"], old_top=r[tag]["-8.0"]["top"] if tag in r else None))
    for fn in ("wp2_R.json", "wp2_R_single.json", "wp2_A.json"):
        for r in json.load(open(f"{R}/{fn}")):
            J.append(dict(group="pu" + r["block"], fam="proj", ncls=3, factor="PU", src=fn, cond=r["arm"], model=r["model"], seed=r["seed"],
                          ckpt=ck(r["ckpt"]), sha=r["ckpt_sha256"], old_top=r["observed"]["-8.0"]["top_mode"]))
    return [j for j in J if (j["model"] in MAMBA) == mamba]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mamba", action="store_true"); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--out")
    a = ap.parse_args(); out = a.out or f"{R}/wp5_{'mamba' if a.mamba else 'main'}.json"
    J = jobs(a.mamba)
    if a.smoke: J = [j for j in J if j["seed"] == 0][::7][:8] if not os.environ.get("SMOKE_PU") else [j for j in J if j["group"].startswith("pu") and j["seed"] == 0][::6][:4]
    print("jobs", len(J), {g: sum(j["group"] == g for j in J) for g in sorted({j["group"] for j in J})}, flush=True)
    data = {}
    def get(g):
        if g in data: return data[g]
        if g == "cwru40": b = build40(_CWRU40, loads=(0, 1, 2, 3), segment_len=1024, overlap=0.5, split_seed=42); Xt, yt, Xc = b.X["test"], b.y["test"], b.X["val"]
        elif g == "cwru16": b = build16(_CWRU16, loads=(0, 1, 2, 3), segment_len=1024, overlap=0.5, split_seed=42); Xt, yt, Xc = b.X["test"], b.y["test"], b.X["val"]
        else:
            blk = g[2:]; pools = W.load_pools(W.block_codes(blk)); Xt, yt = W.build_test(pools, blk); Xc = pu_calib(blk)
        Xt = torch.from_numpy(np.asarray(Xt, np.float32)).to(dev); Xc = torch.from_numpy(np.asarray(Xc, np.float32)).to(dev)
        if Xt.dim() == 2: Xt = Xt[:, None]
        if Xc.dim() == 2: Xc = Xc[:, None]
        yt = torch.from_numpy(np.asarray(yt, np.int64)).to(dev)
        sf_thr = torch.quantile(sf(Xc), 0.99).item()
        data[g] = dict(Xt=Xt, yt=yt, Xc=Xc, sf_thr=sf_thr, sfcache={}, counts=torch.bincount(yt).tolist(), n_cal=len(Xc))
        print(f"  {g}: test {tuple(Xt.shape)} counts {data[g]['counts']} calib {len(Xc)} SF99={sf_thr:.4f}", flush=True); return data[g]
    res = json.load(open(out)) if os.path.exists(out) and not a.smoke else []
    done = {(r["src"], r["cond"], r["model"], r["seed"]) for r in res}; t0 = time.time()
    for j in J:
        if (j["src"], j["cond"], j["model"], j["seed"]) in done: continue
        D = get(j["group"]); assert sha(j["ckpt"]) == j["sha"], ("sha mismatch", j["ckpt"])
        net = build_net(j["fam"], j["model"], j["ncls"]); st = torch.load(j["ckpt"], map_location=dev)
        net.load_state_dict(st); net.eval()
        _, cc = forward(net, D["Xc"]); tau = torch.quantile(cc, 0.05).item()
        rec = {k: v for k, v in j.items() if k not in ("fam",)}; rec.update(tau=tau, sf_thr=D["sf_thr"], test_counts=D["counts"], by_snr={})
        for snr in SNRS:
            agg = {b: [] for b in ("B0", "B1", "B2")}; tops, concs = [], []
            for rep in range(NREP):
                torch.manual_seed(5000 + 17 * j["seed"] + rep); Xn = add_awgn_tensor(D["Xt"], snr)
                key = (j["seed"], snr, rep)
                if key not in D["sfcache"]: D["sfcache"][key] = sf(Xn) <= D["sf_thr"]
                p, c = forward(net, Xn); cnt = torch.bincount(p, minlength=j["ncls"]); tops.append(int(cnt.argmax())); concs.append(cnt.max().item() / len(p))
                ones = torch.ones_like(p, dtype=torch.bool)
                agg["B0"].append(metrics(p, c, D["yt"], ones)); agg["B1"].append(metrics(p, c, D["yt"], D["sfcache"][key])); agg["B2"].append(metrics(p, c, D["yt"], c >= tau))
            rec["by_snr"][f"{snr:.1f}"] = dict(top_mode=int(np.bincount(tops, minlength=j["ncls"]).argmax()), conc=float(np.mean(concs)),
                                               **{b: {k: float(np.nanmean([m[k] for m in v])) if not all(np.isnan(m[k]) for m in v) else float("nan") for k in v[0]} for b, v in agg.items()})
        res.append(rec); del net; torch.cuda.empty_cache()
        if not a.smoke or True:
            tmp = out + ".tmp"; json.dump(res, open(tmp, "w")); os.replace(tmp, out)
        if len(res) % 20 == 0 or a.smoke: print(f"    done {len(res)} ({time.time()-t0:.0f}s)", flush=True)
    print("=== ALL DONE ===", len(res), flush=True)

if __name__ == "__main__":
    main()
