"""第 II 部分汇总：读 epoch_protocols 输出的 JSON，按预注册 §3.3 与 §4 给出每个对象的
  (1) H1 早饱和：报告划分准确率在总 epoch 的前 5%（ceil(0.05·E)）内达到 ≥ 0.999 的运行数；
  (2) 各组（模型 / 任务×方法）在 4 个协议下的均值 ± 标准差，以及 author − last 差；
  (3) 结论翻转：组间成对差的配对 t 检验（按 rep 序号配对，α=0.05，另报协议内 BH-FDR），
      相对作者默认协议的 created / erased / reversed，以及均值排名是否改变。
作者代码无种子（P2/P3）或固定种子（P4）时，rep 序号配对只是约定，报告里写明。
在 code/ 下运行：python -m audit.ext.partII_summary P2=p2.json P3=p3.json P4=p4.json --strata P3='^t(\\d+)_' --out OUT.md"""
import argparse, itertools, json, math, re
from collections import defaultdict
import numpy as np
from audit.partI_reanalysis import paired_t, bh

PROTOCOLS = ("author", "last", "minvalloss", "budget50")
ALPHA = 0.05


def load(path):
    G = defaultdict(dict)
    for r in json.load(open(path)):
        G[r["group"]][r["rep"]] = r
    return G


def h1(G):
    rows = {}
    for g, reps in sorted(G.items()):
        hits, firsts = 0, []
        for r in reps.values():
            n = r["n_epochs"]; lim = math.ceil(0.05 * n)
            first = next((i + 1 for i, a in enumerate(r["acc_curve"]) if a >= 0.999), None)
            firsts.append(first); hits += first is not None and first <= lim
        rows[g] = dict(n=len(reps), early=hits, limit=math.ceil(0.05 * next(iter(reps.values()))["n_epochs"]), first=firsts)
    return rows


def table(G):
    T = {}
    for g, reps in sorted(G.items()):
        T[g] = {p: np.array([reps[k]["protocols"][p]["acc"] for k in sorted(reps)]) for p in PROTOCOLS}
        T[g]["_epochs"] = {p: [reps[k]["protocols"][p]["epoch"] for k in sorted(reps)] for p in PROTOCOLS}
    return T


def ranking(T, p, groups):
    m = {g: round(float(T[g][p].mean()), 9) for g in groups}
    return {g: 1 + sum(m[h] > m[g] for h in groups) for g in groups}


def flips(T, strata_re):
    groups = sorted(T)
    if strata_re:
        key = lambda g: (re.search(strata_re, g) or [None, g])[1]
        pairs = [(a, b) for a, b in itertools.combinations(groups, 2) if key(a) == key(b)]
    else:
        pairs = list(itertools.combinations(groups, 2))
    res = {}
    for p in PROTOCOLS:
        stats = []
        for a, b in pairs:
            n = min(len(T[a][p]), len(T[b][p]))
            m, pv = paired_t(T[a][p][:n] - T[b][p][:n])
            stats.append((m, pv))
        rej = bh([s[1] for s in stats]) if stats else []
        res[p] = [dict(pair=f"{a} − {b}", diff=m, p=pv,
                       v=0 if pv >= ALPHA else (1 if m > 0 else -1),
                       v_bh=0 if not rj else (1 if m > 0 else -1))
                  for (a, b), (m, pv), rj in zip(pairs, stats, rej)]
    out = {}
    for p in PROTOCOLS[1:]:
        c = dict(created=0, erased=0, reversed=0, created_bh=0, erased_bh=0, reversed_bh=0)
        for x, y in zip(res["author"], res[p]):
            for sfx, k in (("", "v"), ("_bh", "v_bh")):
                if x[k] == 0 and y[k] != 0: c["created" + sfx] += 1
                elif x[k] != 0 and y[k] == 0: c["erased" + sfx] += 1
                elif x[k] * y[k] == -1: c["reversed" + sfx] += 1
        out[p] = c
    rank = {p: ranking(T, p, groups) for p in PROTOCOLS}
    # 排名改变 = 某对组在作者协议下严格有序、在该协议下严格反序（并列变有序不算）
    mean = {p: {g: round(float(T[g][p].mean()), 9) for g in groups} for p in PROTOCOLS}
    rank_changed = {p: [f"{a} / {b}" for a, b in pairs
                        if (mean["author"][a] - mean["author"][b]) * (mean[p][a] - mean[p][b]) < 0] for p in PROTOCOLS[1:]}
    return res, out, rank, rank_changed, len(pairs)


def fmt(x): return f"{x:.4f}"


def to_md(name, G, strata_re):
    H, T = h1(G), table(G)
    res, cnt, rank, rch, npairs = flips(T, strata_re)
    L = [f"## {name}", "", f"组数 {len(T)}，每组 rep 数 {sorted({len(v) for v in G.values()})}，比较对数 {npairs}"
         + (f"（只比同一分层 `{strata_re}` 内的组）" if strata_re else "（全部组两两比较）"), "",
         "### H1 早饱和（前 5% epoch 内准确率 ≥ 0.999）", "", "| 组 | 早饱和运行 / 总数 | 前 5% 上限 epoch | 各 rep 首次 ≥0.999 的 epoch |", "|---|---|---|---|"]
    for g, r in H.items():
        L.append(f"| {g} | {r['early']}/{r['n']} | {r['limit']} | {', '.join('—' if f is None else str(f) for f in r['first'])} |")
    L += ["", "### 各协议下的报告值（均值 ± 标准差；括号内为排名）", "",
          "| 组 | " + " | ".join(PROTOCOLS) + " | author − last | author 所选 epoch |", "|---|" + "---|" * (len(PROTOCOLS) + 2)]
    for g in T:
        cells = [f"{fmt(T[g][p].mean())} ± {fmt(T[g][p].std(ddof=1))} ({rank[p][g]})" for p in PROTOCOLS]
        L.append(f"| {g} | " + " | ".join(cells) + f" | {fmt((T[g]['author'] - T[g]['last']).mean())} | {', '.join(map(str, T[g]['_epochs']['author']))} |")
    L += ["", "### 相对作者默认协议的结论变化（配对 t，α=0.05；括号内为协议内 BH-FDR 后）", "",
          "| 协议 | 新增显著 | 显著消失 | 方向反转 | 均值排名反序的组对 |", "|---|---|---|---|---|"]
    for p, c in cnt.items():
        L.append(f"| {p} | {c['created']} ({c['created_bh']}) | {c['erased']} ({c['erased_bh']}) | {c['reversed']} ({c['reversed_bh']}) | {'；'.join(rch[p]) if rch[p] else '否'} |")
    L += ["", "<details><summary>逐对检验</summary>", "", "| 比较 | " + " | ".join(PROTOCOLS) + " |", "|---|" + "---|" * len(PROTOCOLS)]
    for i in range(npairs):
        L.append(f"| {res['author'][i]['pair']} | " + " | ".join(
            f"{res[p][i]['diff']:+.4f} (p={res[p][i]['p']:.3f})" for p in PROTOCOLS) + " |")
    L += ["", "</details>", ""]
    any_h1 = any(r["early"] > 0 for r in H.values())
    maj_h1 = any(r["early"] * 2 > r["n"] for r in H.values())
    any_flip = any(c["created"] or c["erased"] or c["reversed"] for c in cnt.values()) or any(rch.values())
    return "\n".join(L), dict(h1_any=any_h1, h1_majority=maj_h1, flip=any_flip)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("objects", nargs="+", help="NAME=protocols.json")
    ap.add_argument("--strata", nargs="*", default=[], help="NAME=REGEX：只比较该正则第 1 组相同的组")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    strata = dict(s.split("=", 1) for s in a.strata)
    parts, verdicts = [], {}
    for o in a.objects:
        name, path = o.split("=", 1)
        md, v = to_md(name, load(path), strata.get(name))
        parts.append(md); verdicts[name] = v
    head = ["# 第 II 部分重跑汇总（预注册 §3.3、§4）", "", "rep 序号配对：作者代码无种子或固定种子，配对只是约定。", "",
            "| 对象 | H1（任一运行） | H1（某组过半运行） | §3.3 结论变化 |", "|---|---|---|---|"]
    for k, v in verdicts.items():
        head.append(f"| {k} | {'是' if v['h1_any'] else '否'} | {'是' if v['h1_majority'] else '否'} | {'是' if v['flip'] else '否'} |")
    n_run = len(verdicts)
    pos = sum(v["h1_majority"] or v["flip"] for v in verdicts.values())
    gate = "STOP_FEASIBILITY" if n_run < 3 else ("STOP_EFFECT" if pos == 0 else "GO")
    head += ["", f"§4 试点判定（P2–P5 可运行 {n_run} 个；出现 H1（过半口径）或结论变化的 {pos} 个）：**{gate}**", ""]
    open(a.out, "w").write("\n".join(head + parts))
    print("\n".join(head))


if __name__ == "__main__":
    main()
