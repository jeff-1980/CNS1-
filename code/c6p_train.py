"""C6-P 训练与评估。严格按 results/PREREG_C6_crossdomain_pilot.md（已冻结，SHA256 596b04ac…）§2–§4、§7.1。
数据构造复用 c6p_smoke.py；训练用 arm2x2.train_last（200 epoch = 3000 步，最后一轮权重）；评估与 arm2x2.evaluate 同口径（SNR 网格按 §4）。
从 code/ 目录运行：python c6p_train.py --domain d1|d2 [--smoke]。运行中不打印任何主导类别（§11）。"""
from __future__ import annotations
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, os, json, time, hashlib, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "pbx")
import numpy as np, torch, torch.nn as nn
from data.noise_augment import add_awgn_tensor
from prior_screen import REGISTRY
from arm2x2 import train_last
import c6p_smoke as S

PREREG = os.path.join(_ROOT, "results/PREREG_C6_crossdomain_pilot.md")
SNRS = [10.0, 0.0, -4.0, -6.0, -8.0, -10.0, -12.0]
EPOCHS = 200                                   # 920 / 64 → 15 步/epoch → 3000 步
PROBE_N, PROBE_SEED = 1000, 7000               # §7.1 纯噪声探针（种子为实现选择，预注册未指定）
tsha = lambda sd: hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in sd.values())).hexdigest()


# ---------------- D2 架构（§2：4 层 CNN、CIFAR 版 ResNet-18、ViT-tiny patch 4 / depth 6 / dim 192） ----------------
class CNN4(nn.Module):
    def __init__(self, K=4):
        super().__init__(); L = []
        for i, (ci, co) in enumerate([(3, 32), (32, 64), (64, 128), (128, 256)]):
            L += [nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True)]
            if i > 0: L.append(nn.MaxPool2d(2))
        self.f = nn.Sequential(*L, nn.AdaptiveAvgPool2d(1), nn.Flatten()); self.fc = nn.Linear(256, K)
    def forward(self, x): return self.fc(self.f(x))


class Block(nn.Module):
    def __init__(self, ci, co, s):
        super().__init__()
        self.c1 = nn.Conv2d(ci, co, 3, s, 1, bias=False); self.b1 = nn.BatchNorm2d(co)
        self.c2 = nn.Conv2d(co, co, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(co)
        self.sc = nn.Sequential() if s == 1 and ci == co else nn.Sequential(nn.Conv2d(ci, co, 1, s, bias=False), nn.BatchNorm2d(co))
    def forward(self, x):
        return torch.relu(self.b2(self.c2(torch.relu(self.b1(self.c1(x))))) + self.sc(x))


class ResNet18C(nn.Module):
    """CIFAR stem：3×3 卷积、步长 1、无 maxpool；BasicBlock [2, 2, 2, 2]。"""
    def __init__(self, K=4):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        L, ci = [], 64
        for co, s in [(64, 1), (128, 2), (256, 2), (512, 2)]:
            L += [Block(ci, co, s), Block(co, co, 1)]; ci = co
        self.layers = nn.Sequential(*L); self.pool = nn.AdaptiveAvgPool2d(1); self.fc = nn.Linear(512, K)
    def forward(self, x): return self.fc(self.pool(self.layers(self.stem(x))).flatten(1))


class ViTTiny(nn.Module):
    """patch 4（32×32 → 64 个 token）、dim 192、depth 6、3 头、MLP 768、cls token、可学习位置编码、pre-norm、无 dropout。"""
    def __init__(self, K=4, dim=192, depth=6, heads=3, patch=4):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, patch, patch); n = (32 // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim)); self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02); nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim); self.head = nn.Linear(dim, K)
    def forward(self, x):
        t = self.patch(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1) + self.pos
        return self.head(self.norm(self.enc(t))[:, 0])


ARCHS = {"d1": {m: REGISTRY[m] for m in ("wdcnn", "drsn", "lstm")},
         "d2": {"cnn4": CNN4, "resnet18": ResNet18C, "vit_tiny": ViTTiny}}
NOISE = {"d1": add_awgn_tensor, "d2": S.add_awgn_image}


def make_model(dom, m, seed):
    torch.manual_seed(20000 + seed); net = ARCHS[dom][m](4); return net, tsha(net.state_dict())


@torch.no_grad()
def evaluate(model, Xte, yte, device, seed, noise, n_reps=5):
    """与 arm2x2.evaluate 相同的口径；另存 5 次抽样的完整类分布。"""
    model.eval(); out = {}
    for snr in SNRS:
        reps = []
        for r in range(n_reps):
            torch.manual_seed(1000 + 17 * seed + r); preds = []
            for i in range(0, len(Xte), 256):
                preds.append(model(noise(Xte[i:i + 256].to(device), snr)).argmax(1).cpu())
            p = torch.cat(preds).numpy(); c = np.bincount(p, minlength=4)
            reps.append(dict(top=int(c.argmax()), conc=float(c.max() / len(p)), acc=float((p == yte.numpy()).mean()), dist=c.tolist()))
        tops = [r["top"] for r in reps]
        out[f"{snr:.1f}"] = dict(top_mode=int(np.bincount(tops, minlength=4).argmax()), tops=tops,
                                 conc_mean=float(np.mean([r["conc"] for r in reps])),
                                 acc_mean=float(np.mean([r["acc"] for r in reps])), dist_rep0=reps[0]["dist"],
                                 dists=[r["dist"] for r in reps])
    return out


@torch.no_grad()
def clean_and_probe(model, Xte, yte, device, seed):
    model.eval()
    p = torch.cat([model(Xte[i:i + 256].to(device)).argmax(1).cpu() for i in range(0, len(Xte), 256)]).numpy()
    g = torch.Generator().manual_seed(PROBE_SEED + seed)
    Z = torch.randn(PROBE_N, *Xte.shape[1:], generator=g)                     # 单位功率高斯噪声
    q = torch.cat([model(Z[i:i + 250].to(device)).argmax(1).cpu() for i in range(0, PROBE_N, 250)]).numpy()
    c = np.bincount(q, minlength=4)
    return float((p == yte.numpy()).mean()), dict(mode=int(c.argmax()), dist=c.tolist())


def arm_data(dom, ctx, arms, arm):
    groups = [(c, s, list(map(int, ids))) for c in range(4) for s, ids in arms[arm][c]]
    y = np.concatenate([[c] * len(ids) for c, _, ids in groups]).astype(np.int64)
    if dom == "d1":
        X = torch.tensor(np.stack([ctx["sig"][i] for _, _, ids in groups for i in ids])[:, None, :])
        Xte = torch.tensor(np.stack([ctx["sig"][i] for i in ctx["test_ids"]])[:, None, :]); norm = None
    else:
        raw = ctx["Xtrain"][[i for _, _, ids in groups for i in ids]]
        mu, sd = raw.mean((0, 2, 3)), raw.std((0, 2, 3))                    # 按本臂训练集的通道均值 / 标准差
        X = (raw - mu[None, :, None, None]) / sd[None, :, None, None]
        Xte = (ctx["Xtest"] - mu[None, :, None, None]) / sd[None, :, None, None]
        norm = dict(mean=mu.tolist(), std=sd.tolist())
    alloc = {f"{ctx['names'][c]}|{s}": len(ids) for c, s, ids in groups}
    idx_sha = hashlib.sha256(json.dumps([[c, s, ids] for c, s, ids in groups]).encode()).hexdigest()
    return X, torch.from_numpy(y), Xte, torch.tensor(ctx["test_y"]), alloc, idx_sha, norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["d1", "d2"], required=True); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--ptbxl", default=os.path.expanduser("~/论文CNS/data/ptbxl_raw/ptb-xl-1.0.3"))
    ap.add_argument("--cifar", default=os.path.expanduser("~/计算方法探究/data/cifar-100-python"))
    a = ap.parse_args(); dom = a.domain
    reg = open(os.path.join(_ROOT, "results/registry/prereg_c6p_sha256.txt")).read().split()[0]
    assert hashlib.sha256(open(PREREG, "rb").read()).hexdigest() == reg, "预注册哈希不符，拒绝运行"
    out = os.path.join(_ROOT, f"results/c6p_{dom}{'_smoke' if a.smoke else ''}.json")
    ckdir = os.path.join(_ROOT, "results/ckpt_c6p" + ("_smoke" if a.smoke else "")); os.makedirs(ckdir, exist_ok=True)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):                          # 数据构造（与冒烟相同），不重复打印
        _, _, ctx = (S.d1(a.ptbxl, S.N_PER) if dom == "d1" else S.d2(a.cifar, S.N_PER))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = json.load(open(out)) if os.path.exists(out) else []
    done = {(r["arm"], r["model"], r["seed"]) for r in res}
    seeds = a.seeds[:1] if a.smoke else a.seeds; models = list(ARCHS[dom]); arms_ = ["S_ALL"] if a.smoke else list(S.ARMS)
    total = len(seeds) * len(models) * len(arms_)
    for s in seeds:
        arms = S.build_arms(ctx["base_pool"], ctx["manip"], s, ctx["budget"]); S.check_arms(arms, ctx["budget"])
        for m in models:
            for arm in arms_:
                if (arm, m, s) in done: continue
                X, y, Xte, yte, alloc, idx_sha, norm = arm_data(dom, ctx, arms, arm)
                print(f"--- {dom} {arm} {m} seed{s} --- X={tuple(X.shape)} test={tuple(Xte.shape)} cls={np.bincount(y.numpy(), minlength=4).tolist()}", flush=True)
                net, ish = make_model(dom, m, s); t0 = time.time()
                net, steps, cw = train_last(net, X, y, dev, s, epochs=1 if a.smoke else EPOCHS, log=lambda x: print(x, flush=True))
                if not a.smoke: assert steps == 3000, steps
                ck = os.path.join(ckdir, f"{dom}_{arm}_{m}_s{s}.pt"); torch.save(net.state_dict(), ck)
                ev = evaluate(net, Xte, yte, dev, s, NOISE[dom]); clean, probe = clean_and_probe(net, Xte, yte, dev, s)
                res.append(dict(domain=dom, arm=arm, model=m, seed=s, classes=ctx["names"], n_train=int(len(y)),
                                class_counts=np.bincount(y.numpy(), minlength=4).tolist(), subtype_alloc=alloc, idx_sha256=idx_sha,
                                n_test=int(len(yte)), norm=norm, steps=steps, class_weights=cw, init_sha256=ish, ckpt=os.path.relpath(ck, _ROOT),
                                ckpt_sha256=hashlib.sha256(open(ck, "rb").read()).hexdigest(), n_params=sum(p.numel() for p in net.parameters()),
                                wall_s=round(time.time() - t0, 1), clean_acc=clean, probe=probe, observed=ev))
                tmp = out + ".tmp"; json.dump(res, open(tmp, "w"), indent=1); os.replace(tmp, out)
                print(f"    done {len(res)}/{total} steps={steps} wall={time.time()-t0:.0f}s", flush=True)   # 不打印靶标
    print("=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()
