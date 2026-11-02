"""Exploratory (not preregistered): a law for best-epoch inflation, from the 880 recorded test-accuracy curves.

Idea: if the reported-split accuracy fluctuates around its plateau with epoch-to-epoch spread s over K epochs,
picking the best epoch gains about the expected maximum of K draws, s * z_K (z_K from Blom's approximation).
s itself grows with headroom h = 1 - last-epoch accuracy, s ~ k h^beta. Both steps are checked with
leave-one-object-out (LOOO) prediction at the method-group level (the H-A2 unit, 88 groups).

usage (in code/): python3 -m audit.main.headroom_law --protocols ../results/main_objects/protocols --out ../results/main_analysis
"""
import argparse, glob, json, os
import numpy as np
from scipy import stats

from audit.main.analysis import load as load_main   # same object/group/seed filter as the main analysis

PLATEAU = 0.5       # last half of the recorded curve


def z_k(k):
    return stats.norm.ppf((k - 0.375) / (k + 0.25)) if k > 1 else 0.0


def run_stats(r):
    c = np.array([x for x in r["acc_curve"] if x is not None], float)
    if len(c) < 10:
        return None
    tail = c[int(len(c) * (1 - PLATEAU)):]
    last = r["protocols"]["last"]["acc"]; best = r["protocols"]["best_rep"]["acc"]
    if last is None or best is None:
        return None
    w = max(3, len(c) // 20); sm = np.convolve(c, np.ones(w) / w, mode="valid")
    decline = max(0.0, float(sm.max() - sm[-1]))   # sustained peak-then-decline of the smoothed curve
    return dict(gap=best - last, h=1 - last, s=float(np.std(tail, ddof=1)), K=len(c), z=z_k(len(tail)), decline=decline)


def r2(y, yhat):
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--protocols"); ap.add_argument("--out"); a = ap.parse_args()
    keep = {o: set(g) for o, g in load_main(a.protocols).items()}
    G = []                                            # one row per object x group
    for f in sorted(glob.glob(os.path.join(a.protocols, "*.json"))):
        o = os.path.basename(f)[:-5]
        if o not in keep:
            continue
        by = {}
        for r in json.load(open(f)):
            if r["group"] in keep[o] and (s := run_stats(r)):
                by.setdefault(r["group"], []).append(s)
        for g, rs in by.items():
            G.append(dict(obj=o, group=g, n_runs=len(rs), **{k: float(np.mean([x[k] for x in rs])) for k in ("gap", "h", "s", "K", "z", "decline")}))
    obj = np.array([g["obj"] for g in G]); gap = np.array([g["gap"] for g in G]); h = np.array([g["h"] for g in G])
    s = np.array([g["s"] for g in G]); z = np.array([g["z"] for g in G])
    out = {"n_groups": len(G), "n_objects": len(set(obj)), "plateau_fraction": PLATEAU}

    # Step 1: gap = c * s * z_K  (one constant)
    x1 = s * z
    pred1 = np.empty_like(gap)
    for o in set(obj):
        m = obj != o; c = np.sum(x1[m] * gap[m]) / np.sum(x1[m] ** 2); pred1[~m] = c * x1[~m]
    c_all = float(np.sum(x1 * gap) / np.sum(x1 ** 2))
    out["step1_gap_vs_s_zK"] = dict(c=c_all, looo_r2=r2(gap, pred1), spearman=float(stats.spearmanr(x1, gap).statistic),
                                    median_abs_err_pp=float(np.median(np.abs(gap - pred1)) * 100))

    # Step 2: s = k * h^beta  (log-log, groups with h > 0 and s > 0)
    ok = (h > 0) & (s > 0)
    lb = stats.linregress(np.log(h[ok]), np.log(s[ok]))
    pred_s = np.full_like(s, np.nan)
    for o in set(obj[ok]):
        m = ok & (obj != o); t = ok & (obj == o); fit = stats.linregress(np.log(h[m]), np.log(s[m]))
        pred_s[t] = np.exp(fit.intercept) * h[t] ** fit.slope
    out["step2_s_vs_h"] = dict(beta=float(lb.slope), k=float(np.exp(lb.intercept)), r2_log=float(lb.rvalue ** 2),
                               looo_r2_log=r2(np.log(s[ok]), np.log(pred_s[ok])), n=int(ok.sum()))

    # Combined: gap predicted from final accuracy and epoch count only, gap = c * k h^beta * z_K
    pred2 = np.full_like(gap, np.nan)
    for o in set(obj[ok]):
        m = ok & (obj != o); t = ok & (obj == o); fit = stats.linregress(np.log(h[m]), np.log(s[m]))
        xs = np.exp(fit.intercept) * h ** fit.slope * z
        c = np.sum(xs[m] * gap[m]) / np.sum(xs[m] ** 2); pred2[t] = c * xs[t]
    out["combined_gap_from_h_and_K"] = dict(looo_r2=r2(gap[ok], pred2[ok]), spearman=float(stats.spearmanr(pred2[ok], gap[ok]).statistic),
                                            median_abs_err_pp=float(np.median(np.abs(gap[ok] - pred2[ok])) * 100), n=int(ok.sum()))

    # Baseline for comparison: gap = a * h^b directly
    pred3 = np.full_like(gap, np.nan); okg = ok & (gap > 0)
    for o in set(obj[okg]):
        m = okg & (obj != o); t = ok & (obj == o); fit = stats.linregress(np.log(h[m]), np.log(gap[m]))
        pred3[t] = np.exp(fit.intercept) * h[t] ** fit.slope
    out["baseline_gap_power_law_in_h"] = dict(looo_r2=r2(gap[ok], pred3[ok]), median_abs_err_pp=float(np.nanmedian(np.abs(gap[ok] - pred3[ok])) * 100))

    # Thresholds: headroom above which the predicted gap exceeds 1 pp / 2 pp at the median K
    Kmed = float(np.median(z)); hs = np.logspace(-4, 0, 2000)
    g_h = c_all * out["step2_s_vs_h"]["k"] * hs ** out["step2_s_vs_h"]["beta"] * Kmed
    out["threshold_h_for_1pp"] = float(hs[np.argmax(g_h >= 0.01)]) if np.any(g_h >= 0.01) else None
    out["threshold_h_for_2pp"] = float(hs[np.argmax(g_h >= 0.02)]) if np.any(g_h >= 0.02) else None
    out["median_zK"] = Kmed

    # Decomposition: sustained decline (smoothed peak minus smoothed end) vs the rest (epoch-to-epoch luck)
    dec = np.array([g["decline"] for g in G]); bins = [(-1, 0.02), (0.02, 0.10), (0.10, 0.30), (0.30, 1.01)]
    out["by_headroom"] = []
    for lo, hi in bins:
        m = (h > lo) & (h <= hi)
        out["by_headroom"].append(dict(h_range=[max(lo, 0), hi], n_groups=int(m.sum()), n_objects=len(set(obj[m])),
                                       gap_median_pp=float(np.median(gap[m]) * 100), decline_median_pp=float(np.median(dec[m]) * 100),
                                       decline_share=float(dec[m].sum() / gap[m].sum()) if gap[m].sum() > 0 else None,
                                       share_gap_gt_1pp=float(np.mean(gap[m] > 0.01)), share_gap_gt_5pp=float(np.mean(gap[m] > 0.05))))
    out["spearman_h_decline"] = float(stats.spearmanr(h, dec).statistic)
    out["spearman_h_rest"] = float(stats.spearmanr(h, gap - dec).statistic)

    # Rule of thumb, LOOO: does "last-epoch accuracy < 98% => gap > 1 pp" hold on held-out objects?
    hits = [(g["h"] > 0.02) == (g["gap"] > 0.01) for g in G]
    out["rule_h_gt_0.02_predicts_gap_gt_1pp"] = dict(accuracy=float(np.mean(hits)),
        object_level_accuracy=float(np.mean([np.mean([hh for hh, gg in zip(hits, G) if gg["obj"] == o]) for o in set(obj)])))

    json.dump(dict(summary=out, groups=G), open(os.path.join(a.out, "headroom_law.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
