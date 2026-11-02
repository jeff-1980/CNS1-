"""选模虚高定律（计划见 results/inflation_law/PLAN.md，运行前提交 90c7b49）。
G = E[max_k c_k − c_E] ≈ σ·e(K)，σ ≈ φ·sqrt(h(1−h)/n)。检验 A（σ、K 已知，种子样本外）、B（σ 对余量）、C（全局 φ，按对象留一）。
测试集大小 n 由指标取值的最小间隔反推：所有取值都是间隔整数倍（吻合率 ≥ 0.99）且 n < 50 000 才算可反推（对象 37 的 1e-5 间隔是小数位截断，不算）。
在 code/ 下运行：python -m audit.main.inflation_law --protocols ../results/main_objects/protocols --out ../results/inflation_law"""
import argparse, glob, json, os
import numpy as np
from scipy import stats, integrate

EPS = 1e-4                       # 0.01 pp，log 时加在 G 上
SENS = {"main": set(), "no_simulated_no_oneline": {"49", "13", "42", "77", "82"}}
AUTHOR_NOT_LAST = {"08", "14", "15", "18", "21", "30", "32", "34", "36", "37", "38", "39", "42", "49", "51", "53",
                   "60", "68", "76", "83"}


def e_max(K):
    """K 个独立标准正态的期望最大值"""
    if K <= 1: return 0.0
    f = lambda x: x * K * stats.norm.pdf(x) * stats.norm.cdf(x) ** (K - 1)
    return integrate.quad(f, -10, 10)[0]


def infer_n(curves):
    v = np.unique(np.round(np.concatenate(curves), 9)); d = np.diff(v); d = d[d > 1e-7]
    if not len(d): return None
    g = d.min(); fit = np.mean(np.abs(v / g - np.round(v / g)) < 1e-3)
    return int(round(1 / g)) if fit >= 0.99 and 1 / g < 50000 else None


def run_stats(c, frac):
    c = np.asarray(c, float); E = len(c); p = c[int(E * (1 - frac)):]
    if len(p) < 3: return None
    sig = np.std(np.diff(p), ddof=1) / np.sqrt(2) if frac else None
    return dict(G=c.max() - c[-1], sig=sig, sig_direct=np.std(p, ddof=1), K=len(p), mu=p.mean())


def load(pdir, frac):
    out = {}
    for f in sorted(glob.glob(os.path.join(pdir, "*.json"))):
        o = os.path.basename(f)[:-5]
        if not o.isdigit(): continue
        R = json.load(open(f)); n = infer_n([r["acc_curve"] for r in R if r["acc_curve"]])
        groups = {}
        for r in R:
            s = run_stats(r["acc_curve"], frac) if r["acc_curve"] else None
            if s: groups.setdefault(r["group"], {})[r["seed"]] = s
        for g, runs in groups.items():
            if len(runs) >= 5: out[(o, g)] = dict(obj=o, n=n, runs=runs)
    return out


def calib(pred, obs):
    x, y = np.log10(np.asarray(pred) + EPS), np.log10(np.asarray(obs) + EPS)
    lr = stats.linregress(x, y)
    return dict(n_groups=len(x), slope=float(lr.slope), intercept=float(lr.intercept), r2=float(lr.rvalue ** 2),
                spearman=float(stats.spearmanr(pred, obs).statistic))


def test_a(G, sigkey):
    pred, obs = [], []
    for v in G.values():
        seeds = sorted(v["runs"]); R = v["runs"]
        if len(seeds) >= 10:
            est, tgt = seeds[:len(seeds) // 2], seeds[len(seeds) // 2:]
            s = np.mean([R[x][sigkey] for x in est]); K = round(np.mean([R[x]["K"] for x in est]))
            pred.append(s * e_max(K)); obs.append(np.mean([R[x]["G"] for x in tgt]))
        else:
            p = [np.mean([R[y][sigkey] for y in seeds if y != x]) * e_max(R[x]["K"]) for x in seeds]
            pred.append(np.mean(p)); obs.append(np.mean([R[x]["G"] for x in seeds]))
    return calib(pred, obs)


def group_level(v, sigkey):
    R = v["runs"].values()
    return dict(sig=np.mean([r[sigkey] for r in R]), h=1 - np.mean([r["mu"] for r in R]),
                K=round(np.mean([r["K"] for r in R])), G=np.mean([r["G"] for r in R]))


def test_b(G, sigkey):
    x, y, skipped = [], [], 0
    for v in G.values():
        a = group_level(v, sigkey)
        if a["sig"] <= 0 or not 0 < a["h"] < 1: skipped += 1; continue
        x.append(np.log10(np.sqrt(a["h"] * (1 - a["h"])))); y.append(np.log10(a["sig"]))
    lr = stats.linregress(x, y)
    return dict(n_groups=len(x), skipped=skipped, slope=float(lr.slope), r2=float(lr.rvalue ** 2))


def test_c(G, sigkey):
    rows = []
    for v in G.values():
        if v["n"] is None: continue
        a = group_level(v, sigkey); s_s = np.sqrt(max(a["h"] * (1 - a["h"]), 0) / v["n"])
        rows.append(dict(obj=v["obj"], phi=a["sig"] / s_s if s_s > 0 else np.nan, s_s=s_s, K=a["K"], G=a["G"]))
    objs = sorted({r["obj"] for r in rows}); pred, obs = [], []
    for o in objs:
        phi = np.nanmedian([r["phi"] for r in rows if r["obj"] != o])
        for r in rows:
            if r["obj"] == o: pred.append(phi * r["s_s"] * e_max(r["K"])); obs.append(r["G"])
    phis = np.array([r["phi"] for r in rows]); phis = phis[np.isfinite(phis)]
    out = calib(pred, obs)
    out.update(n_objects=len(objs), phi_median=float(np.median(phis)),
               phi_iqr=[float(np.percentile(phis, 25)), float(np.percentile(phis, 75))])
    return out


def verdict(a, c):
    ok = lambda t, r2: 0.8 <= t["slope"] <= 1.25 and t["r2"] >= r2
    A, C = ok(a, 0.5), ok(c, 0.4)
    return dict(A_pass=A, C_pass=C, conclusion="law" if A and C else ("formula_only" if A else "model_fails"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--protocols", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); res = {}
    variants = {"main": (0.5, "sig"), "plateau_last_third": (1 / 3, "sig"), "sigma_direct": (0.5, "sig_direct")}
    for vname, (frac, sigkey) in variants.items():
        G_all = load(a.protocols, frac)
        subsets = {"all": G_all,
                   "no_simulated_no_oneline": {k: v for k, v in G_all.items() if v["obj"] not in SENS["no_simulated_no_oneline"]},
                   "author_not_last": {k: v for k, v in G_all.items() if v["obj"] in AUTHOR_NOT_LAST}}
        for sname, G in subsets.items():
            if vname != "main" and sname != "all": continue
            A, B, C = test_a(G, sigkey), test_b(G, sigkey), test_c(G, sigkey)
            res[f"{vname}/{sname}"] = dict(n_groups=len(G), A=A, B=B, C=C, verdict=verdict(A, C))
            print(f"{vname}/{sname}: groups={len(G)} A slope={A['slope']:.2f} R2={A['r2']:.2f} | "
                  f"B slope={B['slope']:.2f} R2={B['r2']:.2f} | C slope={C['slope']:.2f} R2={C['r2']:.2f} "
                  f"phi={C['phi_median']:.2f} | {res[f'{vname}/{sname}']['verdict']['conclusion']}")
    n_rows = sorted({(v["obj"], v["n"]) for v in load(a.protocols, 0.5).values()})
    with open(os.path.join(a.out, "n_infer.csv"), "w") as f:
        f.write("obj,n_test_inferred\n"); [f.write(f"{o},{n if n else ''}\n") for o, n in n_rows]
    json.dump(res, open(os.path.join(a.out, "law_tests.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
