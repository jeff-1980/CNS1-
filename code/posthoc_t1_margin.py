"""【探索性 / 事后】T1：噪声区域的决策是否被训练数据"钉住"。只做推理，不训练，不改变任何预注册判定。
模型：CWRU E5 的 189 个权重（wdcnn/drsn/lstm）+ C6-P 的 90 个权重（D1 心电、D2 图像）。权重 SHA256 先与结果记录逐一核对。

【事先固定的比较（写于看任何 T1 数字之前，2026-10-12）】
  每个模型：
  M1 纯噪声 margin：1000 个单位功率高斯噪声输入（种子 424242，同一领域所有模型用同一批噪声），logit top1 − top2 的中位数；
     归一化 margin = M1 / 该模型在干净测试集上的 top1 − top2 中位数（消除不同架构/领域的 logit 尺度差异）。
  M2 种子一致性：同一格（CWRU：集合 × 抽取，3 个种子；C6-P：臂，5 个种子）的模型在同一批纯噪声输入上的逐样本预测两两一致率的均值。
  M3 路径：+10 → −12 dB（1 dB 步长，23 点）、固定噪声，数据集级主导类别在 ≤ 0 dB 段的翻转次数；稳定 SNR = 自此往下到 −12 dB 主导类别不再改变的最高 SNR。
  M4 有色噪声：−8 dB，功率与白噪声相同的低通（频带下半）与高通（频带上半）高斯噪声；记录主导类别是否与白噪声相同。
  主比较（逐领域 × 架构，报告模型级中位数）：归一化 margin 的中位数、归一化 margin ≥ 0.5 的模型比例、M2 的中位数。
  "钉住 / 未钉住"清楚分开的判据（事先定）：
     (i) CWRU 每个架构的归一化 margin 中位数都高于 D1、D2 每个架构的中位数；且
     (ii) CWRU 每个架构的 M2 中位数 ≥ 0.90，D1、D2 每个架构的 M2 中位数 < 0.90。
  两条都满足 → CLEAN_SEPARATION；只满足一条 → PARTIAL；都不满足 → NO_SEPARATION。
从 code/ 目录运行：CWRU40_DIR=... python posthoc_t1_margin.py"""
from __future__ import annotations
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, os, json, hashlib, itertools, collections, contextlib, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "pbx")
import numpy as np, torch
from data.noise_augment import add_awgn_tensor
from data.splits_v2 import build_cwru_splits
from prior_screen import REGISTRY
import c6p_smoke as S, c6p_train as T

R = os.path.join(_ROOT, "results"); CK_E5 = os.environ.get("E5_CKPT_ROOT", os.path.expanduser("~/论文CNS/results"))
CWRU40 = os.environ.get("CWRU40_DIR", os.path.expanduser("~/论文CNS/data/cwru_12k_de"))
dev = torch.device("cuda"); NOISE_SEED, N_NOISE, N_PATH = 424242, 1000, 512
PATH_SNRS = [float(s) for s in range(10, -13, -1)]


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


@torch.no_grad()
def logits(net, X, bs=250): return torch.cat([net(X[i:i + bs].to(dev)).float().cpu() for i in range(0, len(X), bs)])


def margin(z): t = z.topk(2, 1).values; return (t[:, 0] - t[:, 1])


def band_noise(shape, kind, g):
    """与输入同形状的高斯噪声，按最后一维（一维）或空间二维 FFT 保留低 / 高半带，再缩放到单位功率。"""
    n = torch.randn(*shape, generator=g)
    if len(shape) == 3:
        F = torch.fft.rfft(n, dim=-1); k = F.shape[-1]; m = torch.zeros(k)
        m[: k // 2] = 1 if kind == "low" else 0; m[k // 2:] = 0 if kind == "low" else 1; n = torch.fft.irfft(F * m, n=shape[-1], dim=-1)
    else:
        F = torch.fft.fft2(n); fy = torch.fft.fftfreq(shape[-2]).abs(); fx = torch.fft.fftfreq(shape[-1]).abs()
        r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2); m = (r <= 0.25).float() if kind == "low" else (r > 0.25).float()
        n = torch.fft.ifft2(F * m).real
    return n / n.pow(2).mean(dim=tuple(range(1, n.dim())), keepdim=True).sqrt()


def add_colored(x, snr, kind, g, image):
    p = x.pow(2).mean(dim=tuple(range(1, x.dim())) if image else -1, keepdim=True)
    return x + band_noise(tuple(x.shape), kind, g) * (p / 10 ** (snr / 10)).sqrt()


@torch.no_grad()
def measure(net, Xte, Z, noise, image):
    net.eval(); out = {}
    zc = logits(net, Xte); zz = logits(net, Z)
    mc, mz = margin(zc).median().item(), margin(zz)
    pz = zz.argmax(1); cnt = np.bincount(pz.numpy(), minlength=4); top2 = np.argsort(-zz.mean(0).numpy())[:2]
    out.update(clean_margin_med=mc, noise_margin_med=mz.median().item(), noise_margin_q=[float(q) for q in mz.quantile(torch.tensor([.1, .25, .5, .75, .9]))],
               norm_margin=mz.median().item() / max(mc, 1e-6), noise_pred=pz.numpy().astype(np.int8), noise_dist=cnt.tolist(),
               noise_top2_by_mean_logit=top2.tolist())
    Xp = Xte[:N_PATH]; doms = []
    for snr in PATH_SNRS:
        torch.manual_seed(31337); doms.append(int(np.bincount(logits(net, noise(Xp.to(dev), snr).cpu()).argmax(1).numpy(), minlength=4).argmax()))
    low = [d for s, d in zip(PATH_SNRS, doms) if s <= 0]
    settle = next(s for i, s in enumerate(PATH_SNRS) if len(set(doms[i:])) == 1)
    out.update(path_dominant=doms, flips_le0=int(sum(a != b for a, b in zip(low, low[1:]))), settle_snr=settle)
    torch.manual_seed(31337); white = int(np.bincount(logits(net, noise(Xp.to(dev), -8.0).cpu()).argmax(1).numpy(), minlength=4).argmax())
    col = {}
    for kind in ("low", "high"):
        g = torch.Generator().manual_seed(4040); y = add_colored(Xp, -8.0, kind, g, image)
        col[kind] = int(np.bincount(logits(net, y).argmax(1).numpy(), minlength=4).argmax())
    out.update(dom_white_m8=white, dom_low_m8=col["low"], dom_high_m8=col["high"])
    return out


def run_domain(name, models, Xte, noise, image):
    g = torch.Generator().manual_seed(NOISE_SEED); Z = torch.randn(N_NOISE, *Xte.shape[1:], generator=g)
    res = []
    for meta, net in models:
        r = measure(net.to(dev), Xte, Z, noise, image); r.update(meta); res.append(r); net.cpu()
    return res


def load_e5():
    rows = json.load(open(os.path.join(R, "e5_size_id.json"))); out = []; bad = 0
    for r in rows:
        p = os.path.join(CK_E5, "ckpt_e5", os.path.basename(r["ckpt"]))
        if sha(p) != r["ckpt_sha256"]: bad += 1; continue
        net = REGISTRY[r["model"]](4); net.load_state_dict(torch.load(p, map_location="cpu"))
        out.append((dict(domain="cwru", model=r["model"], cell=f"{r['set']}|d{r['draw']}", seed=r["seed"], set=r["set"]), net))
    return out, bad


def load_c6p(dom):
    rows = json.load(open(os.path.join(R, f"c6p_{dom}.json"))); out = []; bad = 0
    for r in rows:
        p = os.path.join(_ROOT, r["ckpt"])
        if sha(p) != r["ckpt_sha256"]: bad += 1; continue
        net = T.ARCHS[dom][r["model"]](4); net.load_state_dict(torch.load(p, map_location="cpu"))
        out.append((dict(domain=dom, model=r["model"], cell=r["arm"], seed=r["seed"], norm=r["norm"]), net))
    return out, bad


def seed_agreement(res):
    by = collections.defaultdict(list)
    for r in res: by[(r["model"], r["cell"])].append(r["noise_pred"])
    ag = {}
    for k, ps in by.items():
        ag[k] = float(np.mean([np.mean(a == b) for a, b in itertools.combinations(ps, 2)]))
    return ag


def main():
    print(__doc__.split("【事先固定的比较")[1].split("从 code/")[0])
    out = {}; verified = {}
    with contextlib.redirect_stdout(io.StringIO()):
        b = build_cwru_splits(CWRU40, loads=(0, 1, 2, 3), segment_len=1024, overlap=0.5, split_seed=42)
        _, _, c1 = S.d1(os.path.expanduser("~/论文CNS/data/ptbxl_raw/ptb-xl-1.0.3"), S.N_PER)
        _, _, c2 = S.d2(os.path.expanduser("~/计算方法探究/data/cifar-100-python"), S.N_PER)
    Xc = torch.from_numpy(b.X["test"]).float()
    m, bad = load_e5(); verified["cwru"] = (len(m), bad)
    out["cwru"] = run_domain("cwru", m, Xc, add_awgn_tensor, False)
    X1 = torch.tensor(np.stack([c1["sig"][i] for i in c1["test_ids"]])[:, None, :])
    m, bad = load_c6p("d1"); verified["d1"] = (len(m), bad)
    out["d1"] = run_domain("d1", m, X1, add_awgn_tensor, False)
    m, bad = load_c6p("d2"); verified["d2"] = (len(m), bad); res = []
    for meta, net in m:                                                   # D2：每个模型按自己臂的标准化统计量
        mu, sd = torch.tensor(meta["norm"]["mean"]), torch.tensor(meta["norm"]["std"])
        Xt = (c2["Xtest"] - mu[None, :, None, None]) / sd[None, :, None, None]
        meta = {k: v for k, v in meta.items() if k != "norm"}
        res += run_domain("d2", [(meta, net)], Xt, S.add_awgn_image, True)
    out["d2"] = res
    for d in out:
        for r in out[d]: r.pop("norm", None)
    print("权重核对（通过, 不符）：", verified)
    summ = {}
    for d, res in out.items():
        ag = seed_agreement(res); summ[d] = {}
        for mdl in dict.fromkeys(r["model"] for r in res):
            rs = [r for r in res if r["model"] == mdl]; nm = np.array([r["norm_margin"] for r in rs])
            summ[d][mdl] = dict(n=len(rs), norm_margin_median=float(np.median(nm)), frac_norm_margin_ge_0_5=float(np.mean(nm >= 0.5)),
                                raw_noise_margin_median=float(np.median([r["noise_margin_med"] for r in rs])),
                                seed_agreement_median=float(np.median([v for (mm, _), v in ag.items() if mm == mdl])),
                                flips_le0_mean=float(np.mean([r["flips_le0"] for r in rs])), settle_snr_median=float(np.median([r["settle_snr"] for r in rs])),
                                low_same_as_white=float(np.mean([r["dom_low_m8"] == r["dom_white_m8"] for r in rs])),
                                high_same_as_white=float(np.mean([r["dom_high_m8"] == r["dom_white_m8"] for r in rs])))
    cw = summ["cwru"]; oth = [v for d in ("d1", "d2") for v in summ[d].values()]
    c_i = min(v["norm_margin_median"] for v in cw.values()) > max(v["norm_margin_median"] for v in oth)
    c_ii = all(v["seed_agreement_median"] >= 0.90 for v in cw.values()) and all(v["seed_agreement_median"] < 0.90 for v in oth)
    sep = "CLEAN_SEPARATION" if c_i and c_ii else ("PARTIAL" if c_i or c_ii else "NO_SEPARATION")
    for d, s in summ.items():
        for mdl, v in s.items():
            print(f"{d:5s} {mdl:9s} n={v['n']:3d} 归一化margin中位 {v['norm_margin_median']:.3f}  ≥0.5比例 {v['frac_norm_margin_ge_0_5']:.2f}  "
                  f"原始margin {v['raw_noise_margin_median']:.3f}  种子一致 {v['seed_agreement_median']:.3f}  ≤0dB翻转 {v['flips_le0_mean']:.2f}  "
                  f"稳定SNR {v['settle_snr_median']:+.0f}  低通同白 {v['low_same_as_white']:.2f}  高通同白 {v['high_same_as_white']:.2f}")
    print(f"判据 (i) margin 分开：{c_i}；(ii) 种子一致性分开：{c_ii} → {sep}")
    for d in out:
        for r in out[d]: r["noise_pred"] = r["noise_pred"].tolist()
    json.dump(dict(note="探索性/事后，不改变任何预注册判定", criteria=__doc__.split("【事先固定的比较")[1].split("从 code/")[0],
                   verified=verified, summary=summ, criterion_i=c_i, criterion_ii=c_ii, separation=sep, per_model=out),
              open(os.path.join(R, "posthoc_t1_margin.json"), "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
