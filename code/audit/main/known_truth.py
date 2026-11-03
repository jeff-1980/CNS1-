"""已知真值检验汇总（计划见 results/known_truth/PLAN.md，678548b）。
输入：results/known_truth/curves/<obj>.json，每项 {group, seed, acc_curve_S, acc_curve_T}（逐轮，等长）。
在 code/ 下运行：python -m audit.main.known_truth --curves ../results/known_truth/curves --out ../results/known_truth"""
import argparse, glob, json, os
import numpy as np
from scipy import stats

B, RNG = 2000, np.random.default_rng(20261102)


def run(S, T):
    S, T = np.asarray(S, float), np.asarray(T, float); e = int(np.argmax(S))   # 并列取最早
    return dict(Gs=S[e] - S[-1], Gt=T[e] - T[-1], O=S[e] - T[e], T_sel=T[e], T_last=T[-1], e_star=e + 1, E=len(S))


def reading(r):
    return "mostly_inflation" if r <= 0.3 else ("mostly_real" if r >= 0.7 else "mixed")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--curves", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); groups = {}
    for f in sorted(glob.glob(os.path.join(a.curves, "*.json"))):
        obj = os.path.basename(f)[:-5]
        for r in json.load(open(f)):
            groups.setdefault((obj, r["group"]), []).append(run(r["acc_curve_S"], r["acc_curve_T"]))
    rows = []
    for (obj, g), R in groups.items():
        Gs, Gt = sum(x["Gs"] for x in R), sum(x["Gt"] for x in R)
        w = stats.wilcoxon([x["T_sel"] for x in R], [x["T_last"] for x in R]) if len(R) >= 5 and any(x["Gt"] for x in R) else None
        rows.append(dict(obj=obj, group=g, n_seeds=len(R), Gs_mean=Gs / len(R), Gt_mean=Gt / len(R),
                         R=Gt / Gs if Gs > 0 else None, O_mean=float(np.mean([x["O"] for x in R])),
                         wilcoxon_p=float(w.pvalue) if w else None))
    valid = [r for r in rows if r["R"] is not None]; objs = sorted({r["obj"] for r in valid})
    med = float(np.median([r["R"] for r in valid]))
    boot = []
    for _ in range(B):
        pick = RNG.choice(objs, len(objs)); boot.append(np.median([r["R"] for o in pick for r in valid if r["obj"] == o]))
    out = dict(n_groups=len(rows), n_groups_with_R=len(valid), n_objects=len(objs), R_median=med,
               R_ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
               O_median=float(np.median([r["O_mean"] for r in rows])), reading=reading(med), groups=rows)
    json.dump(out, open(os.path.join(a.out, "known_truth.json"), "w"), indent=1)
    print(f"groups={len(rows)} R median={med:.2f} CI={out['R_ci95']} O median={out['O_median']*100:.2f} pp -> {out['reading']}")


if __name__ == "__main__":
    main()
