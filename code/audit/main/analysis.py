"""正式阶段 §4–§5：从 results/main_objects/protocols/*.json 计算 H-A1、H-A2、H-B1、H-B2（Holm）与 S1。
单位 o×g；对象层取组的均值。只用 ≥ 5 个完整种子的组；同一对象内的组按种子序号配对（取各组共有的种子）。
作者规则是否为 last 按对象登记（AUTHOR_IS_LAST，来源 attempts.csv 的 notes）；20、24 用种子 0–9 的正式文件
（*_altseeds.json 不进分析）；73_report_only.json 不进口径比较（jeff 2026-10-24）。
H-A2 的 Holm p 取对象簇 bootstrap 中 ρ ≤ 0 的比例（预注册只写了 CI 判据，此处为实现选择，写入偏离记录）。
敏感性：去掉仿真数据对象（49）；去掉按一行修复或删死 import 纳入的对象（13、42、77、82）；两者都去掉。
在 code/ 下运行：python -m audit.main.analysis --protocols ../results/main_objects/protocols --out ../results/main_analysis"""
import argparse, glob, json, os
import numpy as np
from scipy import stats

AUTHOR_IS_LAST = {  # True = 作者规则就是最后一轮（含 last checkpoint / 最后一次迭代 / 最后测试点）
    "01": True, "08": False, "11": True, "13": True, "14": False, "15": False, "18": False, "19": True,
    "20": True, "21": False, "23": True, "24": True, "25": True, "30": False, "32": False, "34": False,
    "36": False, "37": False, "38": False, "39": False, "42": False, "49": False, "51": False, "53": False,
    "54": True, "58": True, "60": False, "62": True, "64": True, "68": False, "71": True, "73": True,
    "74": True, "76": False, "77": True, "81": True, "82": True, "83": False}
SENSITIVITY = {"main": set(), "no_simulated": {"49"}, "no_oneline": {"13", "42", "77", "82"},
               "no_simulated_no_oneline": {"49", "13", "42", "77", "82"}}
PROTOS = ["author", "last", "best_rep", "minloss", "budget50"]
N_BOOT = N_PERM = 10_000
SEED = 20261017
ALPHA = 0.05


def load(pdir):
    objs = {}
    for f in sorted(glob.glob(os.path.join(pdir, "*.json"))):
        name = os.path.basename(f)[:-5]
        if not name.isdigit(): continue          # *_altseeds, *_report_only
        groups = {}
        for r in json.load(open(f)):
            groups.setdefault(r["group"], {})[r["seed"]] = {k: r["protocols"][k]["acc"] for k in PROTOS}
        groups = {g: s for g, s in groups.items() if len(s) >= 5}
        if not groups: continue
        common = sorted(set.intersection(*(set(s) for s in groups.values())))
        objs[name] = {g: {k: np.array([s[x][k] if s[x][k] is not None else np.nan for x in common], float)
                          for k in PROTOS} for g, s in groups.items()}
    if set(objs) - set(AUTHOR_IS_LAST): raise SystemExit(f"author rule not registered: {sorted(set(objs) - set(AUTHOR_IS_LAST))}")
    return objs


def boot_ci(x, f, rng):
    x = np.asarray(x); b = [f(x[rng.integers(0, len(x), len(x))]) for _ in range(N_BOOT)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def h_a1(objs, rng):
    I = {o: float(np.mean([np.mean(v["author"] - v["last"]) for v in g.values()])) for o, g in objs.items() if not AUTHOR_IS_LAST[o]}
    x = np.array(list(I.values()))
    w = stats.wilcoxon(x, alternative="greater") if np.any(x != 0) else None
    return dict(n_objects=len(x), median=float(np.median(x)), ci95=boot_ci(x, np.median, rng), I_o=I,
                p=float(w.pvalue) if w else 1.0, n_zero=int(np.sum(x == 0)))


def spearman_cluster(objs, keep, diff, rng):
    units = {o: [(float(np.mean(v[diff[0]] - v[diff[1]])), 1 - float(np.mean(v["last"]))) for v in g.values()]
             for o, g in objs.items() if keep(o)}
    def rho(os_):
        u = [p for o in os_ for p in units[o]]
        return stats.spearmanr([a for a, _ in u], [h for _, h in u]).statistic
    names = list(units); obs = float(rho(names))
    b = np.array([rho([names[i] for i in rng.integers(0, len(names), len(names))]) for _ in range(N_BOOT)])
    b = b[~np.isnan(b)]
    ci = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if len(b) else [float("nan")] * 2
    return dict(n_objects=len(names), n_units=sum(map(len, units.values())), rho=obs, ci95=ci,
                p_boot=float(np.mean(b <= 0)) if len(b) else 1.0)


def log_r(a, l):
    sa, sl = np.std(a, ddof=1), np.std(l, ddof=1)
    if sa == 0 and sl == 0: return None
    return float(np.log(max(sa / sl if sl > 0 else np.inf, 0.01))) if sa > 0 else float(np.log(0.01))


def h_b1(objs, keep, proto):
    L = {}
    for o, g in objs.items():
        if not keep(o): continue
        v = [x for x in (log_r(gv[proto], gv["last"]) for gv in g.values()) if x is not None and np.isfinite(x)]
        if v: L[o] = float(np.median(v))
    x = np.array(list(L.values()))
    w = stats.wilcoxon(x, alternative="less") if np.any(x != 0) else None
    return dict(n_objects=len(x), median_log_r=float(np.median(x)), log_r_o=L, p=float(w.pvalue) if w else 1.0)


def verdict(d):
    """d: (..., S) paired differences -> +1 / -1 / 0 (paired t, two-sided α = 0.05)"""
    n = d.shape[-1]; m = d.mean(-1); s = d.std(-1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = m / (s / np.sqrt(n)); p = 2 * stats.t.sf(np.abs(t), n - 1)
    p = np.where(s == 0, np.where(m == 0, 1.0, 0.0), p)   # constant non-zero difference counts as significant
    return np.where(p < ALPHA, np.sign(m), 0).astype(int), p


def pair_verdicts(g, proto):
    names = sorted(g); out = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = g[names[i]][proto] - g[names[j]][proto]
            if np.isnan(d).any(): continue
            v, p = verdict(d[None]); out[(names[i], names[j])] = (int(v[0]), float(p[0]))
    return out


def h_b2(objs, keep, rng):
    obs, null, per_obj = 0, np.zeros(N_PERM, int), {}
    for o, g in objs.items():
        if not keep(o) or len(g) < 2: continue
        names = sorted(g); A = np.array([g[n]["author"] for n in names]); L = np.array([g[n]["last"] for n in names])
        S = A.shape[1]; pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]
        va, _ = verdict(np.array([A[i] - A[j] for i, j in pairs])); vl, _ = verdict(np.array([L[i] - L[j] for i, j in pairs]))
        c = int(np.sum(va != vl)); obs += c; per_obj[o] = dict(pairs=len(pairs), changes=c)
        m = rng.random((N_PERM, S)) < 0.5
        Ap = np.where(m[:, None, :], L[None], A[None]); Lp = np.where(m[:, None, :], A[None], L[None])
        dA = np.stack([Ap[:, i] - Ap[:, j] for i, j in pairs], 1); dL = np.stack([Lp[:, i] - Lp[:, j] for i, j in pairs], 1)
        null += np.sum(verdict(dA)[0] != verdict(dL)[0], 1)
    return dict(n_objects=len(per_obj), observed=obs, null_q95=float(np.percentile(null, 95)),
                p=float((1 + np.sum(null >= obs)) / (1 + N_PERM)), per_object=per_obj)


def s1(objs, keep):
    out = {}
    for alt in ["last", "best_rep", "minloss", "budget50"]:
        rev, ps = 0, []
        for o, g in objs.items():
            if not keep(o) or len(g) < 2: continue
            va, vb = pair_verdicts(g, "author"), pair_verdicts(g, alt)
            for k in va.keys() & vb.keys():
                rev += va[k][0] != 0 and vb[k][0] == -va[k][0]; ps += [va[k][1], vb[k][1]]
        out[alt] = dict(reversals=int(rev), n_tests=len(ps))
    return out


def holm(p):
    k = sorted(p, key=p.get); m = len(k); adj, run = {}, 0
    for i, name in enumerate(k): run = max(run, min(1, (m - i) * p[name])); adj[name] = run
    return adj


def run(objs):
    rng = np.random.default_rng(SEED)
    nl = lambda o: not AUTHOR_IS_LAST[o]; every = lambda o: True
    r = dict(H_A1=h_a1(objs, rng), H_A2=spearman_cluster(objs, every, ("best_rep", "last"), rng),
             H_A2_author=spearman_cluster(objs, nl, ("author", "last"), rng),
             H_B1=h_b1(objs, nl, "author"), H_B1_best_rep=h_b1(objs, every, "best_rep"),
             H_B2=h_b2(objs, nl, rng), S1=s1(objs, nl))
    p = dict(H_A1=r["H_A1"]["p"], H_A2=r["H_A2"]["p_boot"], H_B1=r["H_B1"]["p"], H_B2=r["H_B2"]["p"])
    r["holm"] = {k: dict(p=p[k], p_holm=v, supported=v < ALPHA) for k, v in holm(p).items()}
    r["H_A2"]["supported_ci"] = r["H_A2"]["ci95"][0] > 0
    r["n_objects"] = len(objs); r["n_author_not_last"] = sum(nl(o) for o in objs)
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--protocols", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); allo = load(a.protocols); res = {}
    for name, drop in SENSITIVITY.items():
        res[name] = run({o: g for o, g in allo.items() if o not in drop})
        h = res[name]["holm"]
        print(f"{name}: {res[name]['n_objects']} objects ({res[name]['n_author_not_last']} author != last); "
              + "; ".join(f"{k} p={v['p']:.4f} holm={v['p_holm']:.4f}" for k, v in h.items()))
    json.dump(res, open(os.path.join(a.out, "hypotheses.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
