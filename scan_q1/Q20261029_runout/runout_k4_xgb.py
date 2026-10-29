"""Q-20261029-1 K4 (exploratory, not decisive): does a data-driven fatigue model
inherit the run-out handling bias?

DOI-grouped 5-fold CV over all S-N points. Models (xgboost, monotone in log S):
  XGB-DROP : squared-error regression on log10 N, failures only
  XGB-FAIL : squared-error regression, run-outs treated as failures
  XGB-AFT  : survival:aft (normal), run-outs right-censored at their stop count
For every held-out eligible series (rows.json from runout_killtest.py) the model's
stress at N = 2e6 is read off a 60-point stress grid and compared with that
series' own censored references (Tobit Basquin and fixed fatigue-limit MLE).

Usage: python runout_k4_xgb.py S-N.json parameter.json OUTDIR   (needs OUTDIR/rows.json)
"""
import json
import os
import re
import sys

import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold

SEED = 20261029
LEAKY = re.compile(r"(fat|strength|slope|intercept|^m$|^c$|class|doi|title|author|year|ref|"
                   r"id$|url|note|comment|source|life|cycle|n_?f|runout|stress_range|basquin)", re.I)
PARAMS = dict(max_depth=6, eta=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
              tree_method="hist", seed=SEED)
ROUNDS = 500


def _load(path):
    d = json.load(open(path))
    return d["data"] if isinstance(d, dict) else d


def build_features(par):
    by_id = {}
    for p in par:
        by_id.setdefault(str(p["dataset_id"]), p)
    keys = sorted({k for p in by_id.values() for k in p if k not in ("dataset_id",)})
    num, cat, dropped = [], [], []
    for k in keys:
        if LEAKY.search(k):
            dropped.append(k)
            continue
        vals = [p.get(k) for p in by_id.values()]
        nonnull = [v for v in vals if v not in (None, "", "NA", "nan")]
        if len(nonnull) < 0.2 * len(vals):
            dropped.append(k)
            continue
        fl = 0
        for v in nonnull:
            try:
                float(v)
                fl += 1
            except (TypeError, ValueError):
                pass
        if fl >= 0.8 * len(nonnull):
            num.append(k)
        elif isinstance(nonnull[0], str):
            levels = {}
            for v in nonnull:
                levels[v] = levels.get(v, 0) + 1
            top = [lv for lv, _ in sorted(levels.items(), key=lambda t: -t[1])[:30]]
            cat.append((k, top))
        else:
            dropped.append(k)

    def vec(ds):
        p = by_id.get(ds, {})
        out = []
        for k in num:
            try:
                out.append(float(p.get(k)))
            except (TypeError, ValueError):
                out.append(np.nan)
        for k, top in cat:
            v = p.get(k)
            out.extend([1.0 if v == lv else 0.0 for lv in top])
        return out

    names = num + [f"{k}={lv}" for k, top in cat for lv in top]
    return by_id, vec, names, dropped


def main(sn_path, par_path, outdir):
    sn, par = _load(sn_path), _load(par_path)
    rows = {r["id"]: r for r in json.load(open(os.path.join(outdir, "rows.json")))}
    by_id, vec, names, dropped = build_features(par)
    X, logS, N, ro, ds, grp = [], [], [], [], [], []
    fcache = {}
    for r in sn:
        try:
            n, s, c = float(r["life_n"]), float(r["stress_range"]), int(r["runout"])
        except (TypeError, ValueError, KeyError):
            continue
        if not (n > 0 and s > 0 and c in (0, 1)):
            continue
        k = str(r["dataset_id"])
        if k not in fcache:
            fcache[k] = vec(k)
        X.append(fcache[k])
        logS.append(np.log10(s))
        N.append(n)
        ro.append(c)
        ds.append(k)
        grp.append(by_id.get(k, {}).get("doi") or f"ds:{k}")
    X = np.column_stack([np.array(logS), np.array(X, float)])
    N, ro, ds = np.array(N), np.array(ro), np.array(ds)
    feat = ["log10_S"] + names
    mono = "(" + ",".join(["-1"] + ["0"] * len(names)) + ")"
    gid = {g: i for i, g in enumerate(sorted(set(grp)))}
    groups = np.array([gid[g] for g in grp])

    out = {m: {} for m in ("drop", "fail", "aft")}
    flag = {m: {} for m in out}
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, groups=groups)):
        p = dict(PARAMS, monotone_constraints=mono)
        f_tr = tr[ro[tr] == 0]
        m_drop = xgb.train(dict(p, objective="reg:squarederror"),
                           xgb.DMatrix(X[f_tr], label=np.log10(N[f_tr])), ROUNDS)
        m_fail = xgb.train(dict(p, objective="reg:squarederror"),
                           xgb.DMatrix(X[tr], label=np.log10(N[tr])), ROUNDS)
        d_aft = xgb.DMatrix(X[tr])
        d_aft.set_float_info("label_lower_bound", N[tr])
        d_aft.set_float_info("label_upper_bound", np.where(ro[tr] == 1, np.inf, N[tr]))
        m_aft = xgb.train(dict(p, objective="survival:aft", eval_metric="aft-nloglik",
                               aft_loss_distribution="normal", aft_loss_distribution_scale=1.0),
                          d_aft, ROUNDS)
        for k in sorted(set(ds[te]) & set(rows)):
            r = rows[k]
            grid = np.linspace(np.log10(0.3 * r["smin_fail"]), np.log10(3 * r["smax"]), 60)
            Xg = np.repeat(X[np.where(ds == k)[0][:1]], len(grid), 0)
            Xg[:, 0] = grid
            dg = xgb.DMatrix(Xg)
            preds = {"drop": m_drop.predict(dg), "fail": m_fail.predict(dg),
                     "aft": np.log10(np.maximum(m_aft.predict(dg), 1e-30))}
            for m, lp in preds.items():
                # lp non-increasing in log S; find log S where lp crosses log10(2e6)
                t = np.log10(2e6)
                # no crossing inside the grid: clamp to the grid end (bound) and flag it
                if lp[0] < t:
                    out[m][k], flag[m][k] = 10 ** grid[0], "below"
                    continue
                if lp[-1] > t:
                    out[m][k], flag[m][k] = 10 ** grid[-1], "above"
                    continue
                flag[m][k] = "ok"
                j = np.argmax(lp <= t)
                x0, x1, y0, y1 = grid[j - 1], grid[j], lp[j - 1], lp[j]
                out[m][k] = 10 ** (x0 + (t - y0) * (x1 - x0) / (y1 - y0)) if y1 != y0 else 10 ** x1
        print(f"fold {fold} done", flush=True)

    res = {"features_used": feat, "features_dropped": dropped, "n_points": int(len(N)),
           "n_groups": int(len(gid))}
    for m in out:
        f = list(flag[m].values())
        res[f"{m}_crossing"] = {lab: f.count(lab) for lab in ("ok", "below", "above")}
    for ref in ("S_tobit_2e+06", "S_fl_2e+06"):
        for m in out:
            v = np.array([out[m][k] / rows[k][ref] - 1 for k in out[m]
                          if rows[k].get(ref) and np.isfinite(out[m][k])])
            res[f"{m}_vs_{ref}"] = {"n": int(len(v)), "median": float(np.median(v)) if len(v) else None,
                                    "median_abs": float(np.median(np.abs(v))) if len(v) else None,
                                    "q25": float(np.percentile(v, 25)) if len(v) else None,
                                    "q75": float(np.percentile(v, 75)) if len(v) else None}
        both = [k for k in out["drop"] if np.isfinite(out["drop"][k]) and np.isfinite(out["aft"].get(k, np.nan))
                and rows[k].get(ref)]
        res[f"share_aft_closer_than_drop_vs_{ref}"] = float(np.mean(
            [abs(out["aft"][k] / rows[k][ref] - 1) < abs(out["drop"][k] / rows[k][ref] - 1) for k in both])) \
            if both else None
        res[f"median_aft_over_drop_{ref}"] = float(np.median([out["aft"][k] / out["drop"][k] for k in both])) \
            if both else None
    json.dump(res, open(os.path.join(outdir, "k4_result.json"), "w"), indent=1)
    json.dump({m: {k: float(v) for k, v in d.items()} for m, d in out.items()},
              open(os.path.join(outdir, "k4_series.json"), "w"))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
