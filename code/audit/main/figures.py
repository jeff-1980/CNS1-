"""正式阶段结果图：(a) 作者规则不是 last 的 20 个对象的虚高 I_o（种子 bootstrap 95% CI）；
(b) 全部 88 个 o×g 的 best_rep − last 对任务余量 h。数据与 analysis.py 相同（load、AUTHOR_IS_LAST）。
作者规则分类（来源 attempts.csv 的 notes）：选在验证集上 / 选在报告集（测试或目标域）上。
在 code/ 下运行：python -m audit.main.figures --protocols ../results/main_objects/protocols --out ../results/main_analysis"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .analysis import AUTHOR_IS_LAST, SEED, load

SELECTS_ON_REPORT = {"18", "34", "36", "37", "39", "42", "53", "76", "83"}   # 其余 11 个作者非 last 对象选在验证集上
COL = {"last": "#7B5CB5", "val": "#1F6FA8", "report": "#C2502C"}
LAB = {"last": "author rule = last epoch", "val": "author selects on validation split",
       "report": "author selects on reported split"}
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d9d9d9"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def kind(o): return "last" if AUTHOR_IS_LAST[o] else ("report" if o in SELECTS_ON_REPORT else "val")


def forest(objs, ax, rng):
    rows = []
    for o, g in objs.items():
        if AUTHOR_IS_LAST[o]: continue
        D = np.array([v["author"] - v["last"] for v in g.values()]) * 100   # groups × seeds
        b = [D[:, rng.integers(0, D.shape[1], D.shape[1])].mean() for _ in range(2000)]
        rows.append((D.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5), o))
    rows.sort()
    for i, (m, lo, hi, o) in enumerate(rows):
        c = COL[kind(o)]
        ax.plot([lo, hi], [i, i], color=c, lw=1.5, solid_capstyle="round")
        ax.plot(m, i, "o", ms=5, color=c, mec="white", mew=1)
    ax.axvline(0, color=MUTED, lw=0.8)
    med = float(np.median([r[0] for r in rows]))
    ax.axvline(med, color=INK, lw=0.8, ls=(0, (3, 2)))
    ax.text(med, -0.9, f" median {med:.2f} pp", color=INK, va="top", ha="left")
    ax.set_ylim(-1.6, len(rows) - 0.5)
    ax.set_yticks(range(len(rows)), [f"#{r[3]}" for r in rows])
    ax.set_xlabel("inflation, author − last (percentage points)")
    ax.set_title("a  Inflation by object (author rule ≠ last, n = 20)", loc="left", color=INK, fontsize=9)
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)


def scatter(objs, ax, res):
    for k in ["last", "val", "report"]:
        pts = [(1 - v["last"].mean(), (v["best_rep"] - v["last"]).mean() * 100)
               for o, g in objs.items() if kind(o) == k for v in g.values()]
        x, y = np.array(pts).T
        ax.scatter(x * 100, y, s=22, color=COL[k], edgecolor="white", linewidth=0.8, label=LAB[k], zorder=3)
    h = res["H_A2"]
    ax.text(0.03, 0.97, f"Spearman ρ = {h['rho']:.2f}\nobject-cluster 95% CI {h['ci95'][0]:.2f}–{h['ci95'][1]:.2f}\n"
            f"{h['n_units']} method groups, {h['n_objects']} objects", transform=ax.transAxes, va="top", color=INK)
    ax.set_xscale("symlog", linthresh=1); ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("task headroom h = 100 − last-epoch accuracy (%)")
    ax.set_ylabel("best_rep − last (percentage points)")
    ax.set_title("b  Counterfactual gap grows with headroom", loc="left", color=INK, fontsize=9)
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--protocols", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); objs = load(a.protocols)
    res = json.load(open(os.path.join(a.out, "hypotheses.json")))["main"]
    fig, (l, r) = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw=dict(width_ratios=[1, 1.25]))
    forest(objs, l, np.random.default_rng(SEED)); scatter(objs, r, res)
    fig.legend(*r.get_legend_handles_labels(), frameon=False, loc="lower center", ncol=3, fontsize=7.5, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("pdf", "png"): fig.savefig(os.path.join(a.out, f"fig_inflation.{ext}"), dpi=300)


if __name__ == "__main__":
    main()
