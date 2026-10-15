"""第 I 部分（论文A 受控测量）的再分析。按 results/PREREG_AUDIT_pilot.md §0b 冻结的规则。

输入：论文A 的 runs 目录（每个 <scenario>_<model>_<seed>.json 一个运行），只读。
方法（与论文A 相同）：每对架构 × 每个噪声档，在各准则下对种子做配对 t 检验（α=0.05），
比较两个准则下的判决：显著+ / 显著- / 不显著。判决改变分为“新增显著 / 失去显著 / 反向显著（反转）”。
§0b 增加的规则：
  2. 反转数同时报未校正与 BH-FDR（每个准则内、对全部 pair×SNR 检验）；给出零假设下的反转期望
     （对每个种子随机翻转配对差的符号，两准则用同一翻转，保留准则间相关；默认 2000 次）；
     并在“架构对”层级（对噪声档取平均后再检验）报告
  1. 种子数 ≥ 20 时，另报 200 个随机 5 种子子集的反转数与不可判定率分布（种子数不足时注明不适用）
在 code/ 下运行：python -m audit.partI_reanalysis RUNS_DIR [RUNS_DIR ...] --scenario jnu --out OUT.json [--md OUT.md]"""
import argparse, glob, itertools, json, math, os, random, re
import numpy as np

CRITERIA = ("first100", "minvalloss", "noisyval", "last")
ALPHA = 0.05


# ---------- 统计 ----------
def _betacf(a, b, x):
    m_max, eps = 300, 3e-16
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-300 else 1e-300); h = d
    for m in range(1, m_max + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c if abs(1 + aa / c) > 1e-300 else 1e-300; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c if abs(1 + aa / c) > 1e-300 else 1e-300
        de = d * c; h *= de
        if abs(de - 1) < eps: break
    return h


def betainc(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2): return math.exp(lbt) * _betacf(a, b, x) / a
    return 1 - math.exp(lbt) * _betacf(b, a, 1 - x) / b


def paired_t(d):
    """配对差 d（各种子）的双侧 t 检验。返回 (均值差, p)。方差为 0 时：均值为 0 → p=1，否则 p=0。"""
    d = np.asarray(d, float); n = len(d); m = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    if sd == 0: return m, (1.0 if m == 0 else 0.0)
    t = m / (sd / math.sqrt(n)); df = n - 1
    return m, betainc(df / 2, 0.5, df / (df + t * t))


def bh(pvals, q=ALPHA):
    """Benjamini–Hochberg：返回每个检验是否被拒绝。"""
    p = np.asarray(pvals, float); n = len(p); order = np.argsort(p)
    thr = q * np.arange(1, n + 1) / n
    passed = p[order] <= thr
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    rej = np.zeros(n, bool); rej[order[:k]] = True
    return rej


def verdict(m, sig):
    return 0 if not sig else (1 if m > 0 else -1)


# ---------- 读取论文A 结果 ----------
def _acc_table(rec):
    """返回 {criterion: {snr(float): accuracy}}。兼容 snr_sweep[crit][snr] 与 snr_sweep[snr][crit] 两种结构。"""
    sw = rec.get("snr_sweep") or rec.get("cross_domain")   # 论文A 的跨载结果把同结构的数据存在 cross_domain
    if sw is None: raise KeyError("no snr_sweep/cross_domain; keys=" + ",".join(rec))
    def acc(v): return float(v["accuracy"] if isinstance(v, dict) else v)
    out = {}
    if set(sw) & set(CRITERIA):
        for c in CRITERIA:
            if c in sw: out[c] = {float(s): acc(v) for s, v in sw[c].items()}
    else:
        for s, row in sw.items():
            for c in CRITERIA:
                if c in row: out.setdefault(c, {})[float(s)] = acc(row[c])
    return out


def load(runs_dirs, scenario):
    """runs_dirs：一个或多个目录（如论文A 的 seed 0–4 与 5–19 两代）。同一 (model, seed) 出现两次即报错。"""
    pat = re.compile(rf"^{re.escape(scenario)}_(.+)_(\d+)\.json$")
    data = {}
    files = sorted(f for d in runs_dirs for f in glob.glob(os.path.join(d, "*.json")))
    for f in files:
        m = pat.match(os.path.basename(f))
        if not m: continue
        rec = json.load(open(f))
        model, seed = rec.get("model", m.group(1)), int(rec.get("seed", m.group(2)))
        if (model, seed) in data: raise SystemExit(f"duplicate run {(model, seed)}: {f}")
        data[(model, seed)] = _acc_table(rec)
    if not data: raise SystemExit(f"no runs for scenario {scenario} in {runs_dirs}")
    models = sorted({k[0] for k in data}); seeds = sorted({k[1] for k in data})
    missing = [(m, s) for m in models for s in seeds if (m, s) not in data]
    if missing: raise SystemExit(f"incomplete grid, missing {missing[:5]}…")
    snrs = sorted(next(iter(data.values()))[CRITERIA[0]])
    # A[c][model] -> array seeds × snrs
    A = {c: {m: np.array([[data[(m, s)][c][x] for x in snrs] for s in seeds]) for m in models} for c in CRITERIA}
    return A, models, seeds, snrs


# ---------- 分析 ----------
def pair_diffs(A, models, crit, seed_idx):
    """返回 list[(i,j,snr_k)] 与对应的配对差矩阵 (n_cmp, n_seed)。"""
    rows, keys = [], []
    for i, j in itertools.combinations(range(len(models)), 2):
        d = A[crit][models[i]][seed_idx] - A[crit][models[j]][seed_idx]
        for k in range(d.shape[1]): rows.append(d[:, k]); keys.append((i, j, k))
    return keys, np.array(rows)


def classify(D1, D2, correction):
    """D1, D2: (n_cmp, n_seed) 两准则下的配对差。返回每个比较的 (v1, v2)。"""
    t1 = [paired_t(r) for r in D1]; t2 = [paired_t(r) for r in D2]
    p1 = np.array([p for _, p in t1]); p2 = np.array([p for _, p in t2])
    if correction == "bh": s1, s2 = bh(p1), bh(p2)
    else: s1, s2 = p1 < ALPHA, p2 < ALPHA
    v1 = np.array([verdict(m, s) for (m, _), s in zip(t1, s1)])
    v2 = np.array([verdict(m, s) for (m, _), s in zip(t2, s2)])
    return v1, v2, t1, t2


def summarize(v1, v2):
    n = len(v1)
    changed = int((v1 != v2).sum()); rev = int(((v1 * v2) == -1).sum())
    created = int(((v1 == 0) & (v2 != 0)).sum()); erased = int(((v1 != 0) & (v2 == 0)).sum())
    undecidable = int(((v1 == 0) & (v2 == 0)).sum())
    return dict(n=n, changed=changed, changed_frac=round(changed / n, 4), created=created, erased=erased,
                reversals=rev, undecidable_both=undecidable, undecidable_frac=round(undecidable / n, 4))


def null_reversals(D1, D2, correction, n_perm, rng):
    """零假设（两架构无差别）下的反转数分布：每个种子随机翻转配对差符号，两准则同一翻转。"""
    n_seed = D1.shape[1]; out = []
    for _ in range(n_perm):
        s = rng.choice([-1.0, 1.0], size=n_seed)
        v1, v2, _, _ = classify(D1 * s, D2 * s, correction)
        out.append(int(((v1 * v2) == -1).sum()))
    out = np.array(out)
    return dict(mean=round(float(out.mean()), 4), p95=float(np.percentile(out, 95)), max=int(out.max()), n_perm=n_perm)


def analyse_pair(A, models, seeds, snrs, c1, c2, n_perm, rng, subsets):
    idx = np.arange(len(seeds))
    keys, D1 = pair_diffs(A, models, c1, idx); _, D2 = pair_diffs(A, models, c2, idx)
    res = dict(criteria=[c1, c2])
    for corr in ("none", "bh"):
        v1, v2, t1, t2 = classify(D1, D2, corr)
        r = summarize(v1, v2)
        r["reversal_list"] = [dict(a=models[keys[n][0]], b=models[keys[n][1]], snr=snrs[keys[n][2]],
                                   diff_c1=round(t1[n][0], 5), p_c1=t1[n][1], diff_c2=round(t2[n][0], 5), p_c2=t2[n][1])
                              for n in np.where((v1 * v2) == -1)[0]]
        r["null"] = null_reversals(D1, D2, corr, n_perm, rng) if n_perm else None
        res[corr] = r
    # 架构对层级：对噪声档取平均后的配对差
    P1 = np.array([(A[c1][models[i]] - A[c1][models[j]]).mean(1) for i, j in itertools.combinations(range(len(models)), 2)])
    P2 = np.array([(A[c2][models[i]] - A[c2][models[j]]).mean(1) for i, j in itertools.combinations(range(len(models)), 2)])
    res["arch_pair_level"] = {}
    for corr in ("none", "bh"):
        v1, v2, _, _ = classify(P1, P2, corr); res["arch_pair_level"][corr] = summarize(v1, v2)
    # 5 种子子集分布（规则 1）
    if len(seeds) >= 20:
        revs, und = [], []
        for _ in range(subsets):
            sub = np.sort(rng.choice(len(seeds), 5, replace=False))
            _, S1 = pair_diffs(A, models, c1, sub); _, S2 = pair_diffs(A, models, c2, sub)
            v1, v2, _, _ = classify(S1, S2, "none"); s = summarize(v1, v2)
            revs.append(s["reversals"]); und.append(s["undecidable_frac"])
        res["five_seed_subsets"] = dict(n=subsets, reversals_mean=float(np.mean(revs)),
                                        reversals_dist={int(k): int(v) for k, v in zip(*np.unique(revs, return_counts=True))},
                                        frac_subsets_with_reversal=float(np.mean(np.array(revs) > 0)),
                                        undecidable_frac_median=float(np.median(und)))
    else:
        res["five_seed_subsets"] = f"not applicable: only {len(seeds)} seeds"
    return res


def criterion_effect(A, models, c1, c2):
    """各架构在噪声档上的平均准确率差（c2 − c1，百分点）与单格最大绝对差。"""
    return {m: dict(mean_pp=round(100 * float((A[c2][m] - A[c1][m]).mean()), 3),
                    max_abs_cell_pp=round(100 * float(np.abs((A[c2][m] - A[c1][m]).mean(0)).max()), 3)) for m in models}


def to_md(out):
    L = [f"# 第 I 部分再分析：{out['scenario']}", "",
         f"运行目录 `{out['runs_dir']}`；{len(out['models'])} 个架构 × {len(out['seeds'])} 个种子 × {len(out['snrs'])} 个噪声档。",
         "按冻结预注册 §0b。配对 t 检验，α=0.05；BH-FDR 在每个准则内对全部 pair×SNR 检验做。", ""]
    for r in out["pairs"]:
        c1, c2 = r["criteria"]
        L += [f"## {c1} vs {c2}", "",
              "| 校正 | 比较数 | 判决改变 | 新增显著 | 失去显著 | 反转 | 零假设反转均值 / 95% 分位 | 两准则下都不可判定 |", "|---|---|---|---|---|---|---|---|"]
        for corr in ("none", "bh"):
            s = r[corr]; nl = s["null"] or {}
            L.append(f"| {'未校正' if corr == 'none' else 'BH-FDR'} | {s['n']} | {s['changed']} ({s['changed_frac']:.0%}) | {s['created']} | {s['erased']} | **{s['reversals']}** | {nl.get('mean', '–')} / {nl.get('p95', '–')} | {s['undecidable_both']} ({s['undecidable_frac']:.0%}) |")
        for corr in ("none", "bh"):
            s = r["arch_pair_level"][corr]
            L.append(f"\n架构对层级（{'未校正' if corr == 'none' else 'BH-FDR'}，{s['n']} 对）：判决改变 {s['changed']}，反转 {s['reversals']}，都不可判定 {s['undecidable_both']}。")
        L.append(f"\n5 种子子集：{r['five_seed_subsets']}")
        if r["none"]["reversal_list"]:
            L += ["", "未校正反转清单：", "", "| A | B | SNR | 差（%s） | p | 差（%s） | p |" % (c1, c2), "|---|---|---|---|---|---|---|"]
            for x in r["none"]["reversal_list"]:
                L.append(f"| {x['a']} | {x['b']} | {x['snr']} | {100*x['diff_c1']:.2f} pp | {x['p_c1']:.3g} | {100*x['diff_c2']:.2f} pp | {x['p_c2']:.3g} |")
        L.append("")
    L += ["## 准则对报告准确率的影响（last − first100，噪声档平均，百分点）", "", "| 架构 | 平均 | 单档最大绝对值 |", "|---|---|---|"]
    for m, e in out["criterion_effect_last_vs_first100"].items():
        L.append(f"| {m} | {e['mean_pp']} | {e['max_abs_cell_pp']} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dirs", nargs="+"); ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--md")
    ap.add_argument("--n-perm", type=int, default=2000); ap.add_argument("--subsets", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261015)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    A, models, seeds, snrs = load(a.runs_dirs, a.scenario)
    primary = [("first100", "last")]
    others = [p for p in itertools.combinations(CRITERIA, 2) if p != ("first100", "last")]
    out = dict(scenario=a.scenario, runs_dir=[os.path.abspath(d) for d in a.runs_dirs], models=models, seeds=seeds, snrs=snrs,
               rng_seed=a.seed, primary=list(primary[0]), pairs=[])
    for c1, c2 in primary + others:
        out["pairs"].append(analyse_pair(A, models, seeds, snrs, c1, c2, a.n_perm if (c1, c2) in primary else 0, rng, a.subsets))
    out["criterion_effect_last_vs_first100"] = criterion_effect(A, models, "first100", "last")
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    md = to_md(out)
    if a.md: open(a.md, "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
