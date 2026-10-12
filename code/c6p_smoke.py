"""C6-P §9 冒烟检查（results/PREREG_C6_crossdomain_pilot.md 草案）：只构造数据与各臂抽样，不训练、不评估、不看任何模型输出。
从 code/ 目录运行：python c6p_smoke.py [--ptbxl DIR] [--cifar DIR] [--cwru DIR]"""
from __future__ import annotations
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, os, ast, json, pickle, argparse, hashlib, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "pbx")
import numpy as np, pandas as pd, torch
from data.noise_augment import add_awgn_tensor

N_PER = 230; SEEDS = range(5); ARMS = ("S_a", "S_b", "S_ALL")
D1_BASE, D1_MANIP, D1_SPARE = "NORM", ("MI", "STTC", "CD"), "HYP"
D2_BASE, D2_MANIP = "large_natural_outdoor_scenes", ("fish", "flowers", "vehicles_1")


def add_awgn_image(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    """§4 D2：逐样本功率，按全部非批维 (C, H, W) 求均值；其余与 add_awgn_tensor 相同。"""
    sig_power = x.pow(2).mean(dim=tuple(range(1, x.dim())), keepdim=True)
    snr_lin = 10 ** (snr_db / 10.0)
    noise_std = (sig_power / (snr_lin + 1e-12)).sqrt()
    return x + torch.randn_like(x) * noise_std


def split_quota(n, k):
    return [n // k + (1 if i < n % k else 0) for i in range(k)]


def build_arms(base_pool, manip, seed, n_per):
    """manip: {cls_idx: (subtype pools dict name->ids, [a, b, (c...)])}。返回 {arm: {cls_idx: [(subtype, ids)]}}。
    基准类 rng([seed,0])；被操控类 rng([seed,class,subtype]) 无放回，S_a 与 S_ALL 中子类 a 为同一排列的前缀。"""
    base = list(np.random.default_rng([seed, 0]).permutation(base_pool)[:n_per])
    arms = {a: {0: [("ALL", base)]} for a in ARMS}
    for ci, (pools, order) in manip.items():
        perm = {s: np.random.default_rng([seed, ci, si]).permutation(pools[s]) for si, s in enumerate(order)}
        arms["S_a"][ci] = [(order[0], list(perm[order[0]][:n_per]))]
        arms["S_b"][ci] = [(order[1], list(perm[order[1]][:n_per]))]
        arms["S_ALL"][ci] = [(s, list(perm[s][:q])) for s, q in zip(order, split_quota(n_per, len(order)))]
    return arms


def check_arms(arms, n_per, n_cls=4):
    for a, d in arms.items():
        cnt = [sum(len(ids) for _, ids in d[c]) for c in range(n_cls)]
        assert cnt == [n_per] * n_cls, (a, cnt)
        for c in range(n_cls):
            ids = [i for _, g in d[c] for i in g]; assert len(set(ids)) == len(ids), (a, c, "dup")
    for c in range(1, n_cls):   # 前缀配对
        s_a = arms["S_a"][c][0]; s_all = dict(arms["S_ALL"][c])
        assert list(s_all[s_a[0]]) == list(s_a[1][:len(s_all[s_a[0]])]), (c, "prefix")


def fmt_alloc(arms, names):
    out = []
    for a in ARMS:
        cnt = [sum(len(ids) for _, ids in arms[a][c]) for c in range(4)]
        alloc = "; ".join(f"{names[c]}: " + "/".join(f"{s}={len(ids)}" for s, ids in arms[a][c]) for c in range(4))
        out.append(f"    {a:5s} counts={cnt}  {alloc}")
    return "\n".join(out)


# ---------------- D1：PTB-XL ----------------
def read_lead2(root, rel):
    hea = open(os.path.join(root, rel + ".hea")).read().split("\n")
    n_sig, fs, n_samp = (int(v) for v in hea[0].split()[1:4])
    sig = [l.split() for l in hea[1:1 + n_sig]]
    assert n_sig == 12 and fs == 100 and n_samp == 1000 and all(s[1] == "16" for s in sig), rel
    assert sig[1][-1] == "II", (rel, sig[1][-1])
    g = sig[1][2].split("/")[0]                                     # 形如 1000.0(0)/mV：增益(基线)
    gain, base = float(g.split("(")[0]), int(g.split("(")[1].rstrip(")")) if "(" in g else int(sig[1][4])
    raw = np.fromfile(os.path.join(root, rel + ".dat"), dtype="<i2").reshape(n_samp, n_sig)
    x = (raw[:, 1].astype(np.float32) - base) / gain
    return (x - x.mean()) / (x.std() + 1e-8)


def d1(root, n_per):
    db = pd.read_csv(os.path.join(root, "ptbxl_database.csv"), index_col="ecg_id")
    st = pd.read_csv(os.path.join(root, "scp_statements.csv"), index_col=0); st = st[st.diagnostic == 1]
    db["codes"] = db.scp_codes.apply(ast.literal_eval)
    db["diag"] = db.codes.apply(lambda d: {k: v for k, v in d.items() if k in st.index})
    db["supers"] = db.diag.apply(lambda d: sorted({st.loc[k, "diagnostic_class"] for k in d}))
    db["subs"] = db.diag.apply(lambda d: sorted({st.loc[k, "diagnostic_subclass"] for k in d}))
    single = db[db.supers.apply(len) == 1].copy(); single["sup"] = single.supers.str[0]
    # 子类型资格：单一超类、诊断语句只属于一个子类，且该子类至少有一条 likelihood = 100 的语句
    single["sub_ok"] = single.apply(lambda r: r.subs[0] if len(r.subs) == 1 and any(
        v == 100 for k, v in r.diag.items()) else None, axis=1)
    pool, test = single[single.strat_fold <= 8], single[single.strat_fold >= 9]
    print(f"[D1] 单一超类记录：训练池 {len(pool)}，测试 {len(test)}；超类计数（训练池）："
          f"{pool.sup.value_counts().to_dict()}")
    avail, ranks = {}, {}
    for c in (*D1_MANIP, D1_SPARE):
        vc = pool[pool.sup == c].sub_ok.value_counts(); ranks[c] = list(vc.index); avail[c] = vc.to_dict()
        print(f"  {c:5s} 子类（单一子类、含 likelihood=100）训练池可用数，按降序：{avail[c]}")
    deviations = []

    def choose(c, need_single, quota_c):
        ok = [s for s in ranks[c] if avail[c][s] >= need_single]
        if len(ok) < 2: return None
        a, b = ok[:2]; rest = [s for s in ranks[c] if s not in (a, b)]
        if ranks[c][:2] != [a, b]: deviations.append(f"{c}: 前两名 {ranks[c][:2]} 不足 {need_single}，顺延为 a={a}, b={b}")
        cands = [s for s in rest if avail[c][s] >= quota_c]
        if not cands: deviations.append(f"{c}: 无满足配额 {quota_c} 的子类 c，S_ALL 只用 a/b"); return [a, b]
        if cands[0] != (rest[0] if rest else None): deviations.append(f"{c}: 子类 c 顺延为 {cands[0]}")
        return [a, b, cands[0]]

    for budget in (n_per, 150):
        q = split_quota(budget, 3)[-1]
        orders = {c: choose(c, budget, q) for c in (*D1_MANIP, D1_SPARE)}
        failed = [c for c in D1_MANIP if orders[c] is None]
        if not failed: classes = list(D1_MANIP); break
        if len(failed) == 1 and orders[D1_SPARE] is not None:
            deviations.append(f"{failed[0]} 不足两个满足 {budget} 的子类 → 用 HYP 替换")
            classes = [D1_SPARE if c == failed[0] else c for c in D1_MANIP]; break
        deviations.append(f"预算 {budget}：{failed} 不满足且 HYP 不能全部替换 → 降到下一档")
    else:
        raise SystemExit("[D1] 预算降到 150 仍不满足 §9 规则")
    names = [D1_BASE] + classes
    print(f"[D1] 预算 = {budget}/类；类 = {names}；子类 a/b/c = {{{', '.join(f'{c}: {orders[c]}' for c in classes)}}}")
    for d in deviations: print("  [偏离/顺延]", d)
    base_pool = np.array(sorted(pool[pool.sup == D1_BASE].index))
    manip = {ci: ({s: np.array(sorted(pool[(pool.sup == c) & (pool.sub_ok == s)].index)) for s in orders[c]}, orders[c])
             for ci, c in enumerate(classes, start=1)}
    used = set()
    for seed in SEEDS:
        arms = build_arms(base_pool, manip, seed, budget); check_arms(arms, budget)
        print(f"  seed {seed}:\n" + fmt_alloc(arms, names))
        for a in ARMS:
            for c in range(4): used |= {int(i) for _, g in arms[a][c] for i in g}
    te = test[test.sup.isin(names)]
    print(f"[D1] 测试集（folds 9–10，含全部子类）：{te.sup.value_counts().reindex(names).to_dict()}")
    ids = sorted(used | set(te.index))
    X = np.stack([read_lead2(root, db.loc[i, "filename_lr"]) for i in ids])
    assert X.shape == (len(ids), 1000) and np.isfinite(X).all()
    print(f"[D1] 已读取导联 II（训练臂并集 + 测试）{X.shape}，逐记录 z-score；均值 |μ|max={np.abs(X.mean(1)).max():.2e}, σ≈{X.std(1).mean():.3f}")
    ctx = dict(names=names, budget=budget, base_pool=base_pool, manip=manip, sig=dict(zip(ids, X)),
               test_ids=list(te.index), test_y=[names.index(s) for s in te.sup])
    return dict(budget=budget, classes=names, orders={c: orders[c] for c in classes}, deviations=deviations,
                avail={c: {str(k): int(v) for k, v in avail[c].items()} for c in avail}), X[:64], ctx


# ---------------- D2：CIFAR-100 ----------------
def d2(root, n_per):
    def load(f): return pickle.load(open(os.path.join(root, f), "rb"), encoding="latin1")
    meta, tr, te = load("meta"), load("train"), load("test")
    fine_names, coarse_names = meta["fine_label_names"], meta["coarse_label_names"]
    fine, coarse = np.array(tr["fine_labels"]), np.array(tr["coarse_labels"])
    names = [D2_BASE, *D2_MANIP]; cidx = [coarse_names.index(n) for n in names]
    subs = {n: sorted(set(fine[coarse == ci])) for n, ci in zip(names, cidx)}
    for n in names:
        print(f"  {n:30s} 细类(编号升序) = {[f'{f}:{fine_names[f]}({(fine == f).sum()})' for f in subs[n]]}")
    base_pool = np.where(coarse == cidx[0])[0]
    manip = {}
    for ci, n in enumerate(D2_MANIP, start=1):
        order = [fine_names[f] for f in subs[n]]                    # a、b = 第 1、2 个；S_ALL 用全部 5 个
        manip[ci] = ({fine_names[f]: np.where(fine == f)[0] for f in subs[n]}, order)
        assert all(len(manip[ci][0][s]) >= n_per for s in order[:2])
    print(f"[D2] 类 = {names}；a/b = {{{', '.join(f'{n}: {manip[i][1][:2]}' for i, n in enumerate(D2_MANIP, 1))}}}")
    Xall = torch.tensor(tr["data"].reshape(-1, 3, 32, 32), dtype=torch.float32) / 255.0
    for seed in SEEDS:
        arms = build_arms(base_pool, manip, seed, n_per); check_arms(arms, n_per)
        print(f"  seed {seed}:\n" + fmt_alloc(arms, names))
        if seed == 0:
            idx = [i for a in ["S_ALL"] for c in range(4) for _, g in arms[a][c] for i in g]
            mu, sd = Xall[idx].mean((0, 2, 3)), Xall[idx].std((0, 2, 3))
            print(f"    seed0 S_ALL 训练集通道均值 {mu.numpy().round(4)}，标准差 {sd.numpy().round(4)}（标准化用；各臂各自计算）")
    tc = np.array(te["coarse_labels"])
    print(f"[D2] 测试集：{ {n: int((tc == ci).sum()) for n, ci in zip(names, cidx)} }")
    Xb = (Xall[base_pool[:256]] - mu[None, :, None, None]) / sd[None, :, None, None]
    sel = np.isin(tc, cidx)
    ctx = dict(names=names, budget=n_per, base_pool=base_pool, manip=manip, Xtrain=Xall,
               Xtest=torch.tensor(te["data"].reshape(-1, 3, 32, 32)[sel], dtype=torch.float32) / 255.0,
               test_y=[cidx.index(c) for c in tc[sel]])
    return dict(classes=names, ab={n: manip[i][1][:2] for i, n in enumerate(D2_MANIP, 1)}), Xb, ctx


# ---------------- AWGN ----------------
def cwru_windows(cwru_dir, n=64, L=1024):
    if cwru_dir and os.path.isdir(cwru_dir):
        import scipy.io as sio
        f = sorted(p for p in os.listdir(cwru_dir) if p.endswith(".mat"))[0]
        m = sio.loadmat(os.path.join(cwru_dir, f)); key = [k for k in m if k.endswith("_DE_time")][0]
        s = m[key].ravel().astype(np.float32); w = np.stack([s[i * L // 2: i * L // 2 + L] for i in range(n)])
        w = (w - w.mean(1, keepdims=True)) / w.std(1, keepdims=True)
        return torch.tensor(w[:, None, :]), f"CWRU {f}:{key}（真实窗口，逐窗 z-score）"
    g = torch.Generator().manual_seed(0)
    return torch.randn(n, 1, L, generator=g), "随机数据（未找到 CWRU 原始文件）"


def awgn_checks(cwru_dir, ecg_x, img_x):
    xw, src = cwru_windows(cwru_dir)
    for nm, x in (("CWRU [N,1,1024]", xw), ("ECG [N,1,1000]", torch.tensor(ecg_x[:, None, :]))):
        ok = True
        for snr in (10.0, -8.0, -12.0):
            torch.manual_seed(1234); a = add_awgn_image(x, snr); torch.manual_seed(1234); b = add_awgn_tensor(x, snr)
            ok &= torch.equal(a, b)
        print(f"[AWGN] D2 实现 vs add_awgn_tensor 在 {nm} 上逐值一致：{ok}" + (f"（输入：{src}）" if nm.startswith("CWRU") else ""))
        assert ok
    res = {}
    for snr in (10.0, -8.0):
        torch.manual_seed(1000); y = add_awgn_image(img_x, snr); n = y - img_x
        meas = (10 * torch.log10(img_x.pow(2).mean((1, 2, 3)) / n.pow(2).mean((1, 2, 3)))).mean().item()
        res[snr] = meas; print(f"[AWGN] CIFAR 批（256 张，标准化后）设定 {snr:+.0f} dB → 实测 {meas:+.3f} dB（偏差 {meas - snr:+.3f}）")
        assert abs(meas - snr) < 0.1
    return src, res


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptbxl", default=os.path.expanduser("~/论文CNS/data/ptbxl_raw/ptb-xl-1.0.3"))
    ap.add_argument("--cifar", default=os.path.expanduser("~/计算方法探究/data/cifar-100-python"))
    ap.add_argument("--cwru", default=os.environ.get("CWRU40_DIR", os.path.expanduser("~/论文CNS/data/cwru_12k_de")))
    ap.add_argument("--hashes", default=os.path.join(_ROOT, "results/registry/raw_hashes_c6p.csv"))
    ap.add_argument("--out", default=os.path.join(_ROOT, "results/c6p_smoke.json"))
    a = ap.parse_args()
    print("=" * 20, "D1 PTB-XL", "=" * 20); r1, ecg, _ = d1(a.ptbxl, N_PER)
    print("=" * 20, "D2 CIFAR-100", "=" * 20); r2, img, _ = d2(a.cifar, N_PER)
    print("=" * 20, "AWGN", "=" * 20); src, snr = awgn_checks(a.cwru, ecg, img)
    rows = []
    for dom, root in (("ptbxl", a.ptbxl), ("cifar100", a.cifar)):
        for dp, _, fs in os.walk(root):
            for f in sorted(fs): p = os.path.join(dp, f); rows.append((dom, os.path.relpath(p, root), os.path.getsize(p), sha256(p)))
    os.makedirs(os.path.dirname(a.hashes), exist_ok=True)
    pd.DataFrame(sorted(rows), columns=["dataset", "path", "bytes", "sha256"]).to_csv(a.hashes, index=False)
    print(f"[HASH] {len(rows)} 个原始文件的 SHA256 → {a.hashes}")
    json.dump(dict(d1=r1, d2=r2, awgn_cwru_input=src, awgn_image_measured_snr=snr), open(a.out, "w"), ensure_ascii=False, indent=1)
    print("[SMOKE] 全部检查通过（未训练、未评估）")


if __name__ == "__main__":
    main()
