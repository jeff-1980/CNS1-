#!/usr/bin/env python3
"""Mechanical topic-pair forecaster v0: backtest (K1, K3) + forecast list.

Task (Y1): for a mechanical topic a and any topic b whose cumulative co-occurrence up to year T
is < NEW_MAX (=3) articles, predict whether cumulative co-occurrence reaches >= HIT (=10) by T+3.

Splits (frozen before the test set is scored):
  train  T in 2011..2015 (labels end <= 2018)
  val    T = 2018 (labels 2019-2021)           -> early stopping / model check
  test   T = 2021 (labels 2022-2024), model retrained on T in 2011..2018 (labels end <= 2021)
  K3     model trained on T in 2011..2017 (labels end <= 2020), scores pairs at T = 2020;
         where do topic pairs of NC/SA 2023-2026 mechanical papers fall in that ranking?
  final  model trained on T in 2011..2022, scores pairs at T = 2025 -> forecast list for 2026-2028

Gate K1: test AUC(model) >= max(AUC of preferential-attachment baselines) + 0.03 AND
         precision@100(model) > the best baseline precision@100.

Variants (v1, 2026-11-04):
  --target ns   label = first co-paper in a Nature/Science-family journal: candidates are pairs with
                >= 1 co-paper anywhere but 0 in the NS venue set up to T; positive if >= 1 NS co-paper by T+3.
                Needs cooc_ns/ (fetch_openalex.py --ns-cooc). K3 then counts NC/SA pairs with no NS co-paper by 2020.
  --scidata     adds Scientific Data paper counts per topic (window T-2..T) and their 3-year growth as features
                (dataset leading indicator; needs scidata/ from fetch_openalex.py --scidata).
  Every forecast also gets b_generic_ai (partner in AI / computer vision) and a second list without those partners.

Usage: python model.py --data DIR --out DIR [--ext] [--target cooc|ns] [--scidata]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata

NEW_MAX = 3      # "not yet connected": cumulative co-occurrence < 3
HIT = 10         # "emerged": cumulative co-occurrence >= 10 by T+3
H = 3            # horizon in years
MIN_N = 20       # node must have >= 20 articles in year T to be a candidate end
SEED = 20261102
GENERIC_AI_SUBFIELDS = {"1702", "1707"}  # Artificial Intelligence; Computer Vision and Pattern Recognition


def load(data, ext=False, ns_pairs=False, scidata=False):
    topics = json.load(open(os.path.join(data, "topics_all.json"), encoding="utf-8"))
    mech = json.load(open(os.path.join(data, "mech_topics.json"), encoding="utf-8"))
    if ext:  # energy/transport extension (fetch_openalex.py --ext): these topics become source nodes too
        fx = os.path.join(data, "ext_topics.json")
        if os.path.exists(fx):
            mech = sorted(set(mech) | set(json.load(open(fx, encoding="utf-8"))))
        else:
            print("WARNING --ext given but ext_topics.json missing; using mechanical topics only", flush=True)
    years = sorted(int(f[:-5]) for f in os.listdir(os.path.join(data, "totals")) if f.endswith(".json"))
    ids = sorted(topics)
    idx = {t: i for i, t in enumerate(ids)}
    N = len(ids)
    tot = np.zeros((len(years), N))
    ns = np.zeros((len(years), N))
    C = []
    for k, y in enumerate(years):
        for t, c in json.load(open(os.path.join(data, "totals", f"{y}.json"))).items():
            if t in idx:
                tot[k, idx[t]] = c
        fns = os.path.join(data, "ns", f"{y}.json")
        if os.path.exists(fns):
            for t, c in json.load(open(fns)).items():
                if t in idx:
                    ns[k, idx[t]] = c
        rows, cols, vals = [], [], []
        missing = 0
        for a in mech:
            f = os.path.join(data, "cooc", str(y), f"{a}.json")
            if not os.path.exists(f):
                missing += 1
                continue
            ia = idx[a]
            for b, c in json.load(open(f)).items():
                if b in idx and b != a:
                    rows.append(ia); cols.append(idx[b]); vals.append(c)
        if missing:
            print(f"WARNING year {y}: {missing} mech topics without co-occurrence file", flush=True)
        M_ = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
        # symmetrise; mech-mech pairs are reported from both ends -> take the max
        M_ = M_.maximum(M_.T).tocsr()
        C.append(M_)
    mech_idx = np.array(sorted(idx[t] for t in mech if t in idx))
    Cns = []
    if ns_pairs:
        for y in years:
            rows, cols, vals = [], [], []
            for a in mech:
                f = os.path.join(data, "cooc_ns", str(y), f"{a}.json")
                if not os.path.exists(f):
                    raise SystemExit(f"cooc_ns/{y}/{a}.json missing: run fetch_openalex.py --ns-cooc first")
                for b, c in json.load(open(f)).items():
                    if b in idx and b != a:
                        rows.append(idx[a]); cols.append(idx[b]); vals.append(c)
            M_ = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
            Cns.append(M_.maximum(M_.T).tocsr())
    sd = np.zeros((len(years), N))
    if scidata:
        for k, y in enumerate(years):
            f = os.path.join(data, "scidata", f"{y}.json")
            if not os.path.exists(f):
                raise SystemExit(f"scidata/{y}.json missing: run fetch_openalex.py --scidata first")
            for t, c in json.load(open(f)).items():
                if t in idx:
                    sd[k, idx[t]] = c
    return dict(topics=topics, ids=ids, idx=idx, years=years, tot=tot, ns=ns, C=C, Cns=Cns, sd=sd,
                use_sd=scidata, mech_idx=mech_idx)


def cum(D, T, key="C"):
    k = D["years"].index(T)
    out = D[key][0].copy()
    for j in range(1, k + 1):
        out = out + D[key][j]
    return out.tocsr()


def recent(D, T, w=3):
    k = D["years"].index(T)
    out = D["C"][k].copy()
    for j in range(max(0, k - w + 1), k):
        out = out + D["C"][j]
    return out.tocsr()


def snapshot(D, T, with_label=True, target="cooc"):
    """Candidate pairs + features at year T (and label at T+H if available)."""
    yrs = D["years"]
    k = yrs.index(T)
    mi = D["mech_idx"]
    N = len(D["ids"])
    cumT = cum(D, T)
    A = (cumT >= NEW_MAX).astype(np.float64).tocsr()
    A.setdiag(0); A.eliminate_zeros()
    deg = np.asarray(A.sum(1)).ravel()
    A3 = (cum(D, yrs[k - 3]) >= NEW_MAX).astype(np.float64).tocsr()
    deg3 = np.asarray(A3.sum(1)).ravel()
    Am = A[mi]
    CN = (Am @ A).toarray()
    w = np.where(deg > 1, 1.0 / np.log(np.maximum(deg, 2)), 0.0)
    AA = (Am @ sp.diags(w) @ A).toarray()
    R = recent(D, T)
    Rn = np.sqrt(np.asarray(R.multiply(R).sum(1)).ravel()) + 1e-9
    COS = (R[mi] @ R.T).toarray() / (Rn[mi][:, None] * Rn[None, :])
    cumM = cumT[mi].toarray()
    curM = D["C"][k][mi].toarray()
    cumNSM = cum(D, T, "Cns")[mi].toarray() if target == "ns" else None

    n = D["tot"]
    nT = n[k]
    tot_all = n.sum(1)
    share = n / tot_all[:, None]
    g1 = np.log((nT + 1) / (n[k - 1] + 1))
    g3 = np.log((nT + 1) / (n[k - 3] + 1))
    gs3 = np.log((share[k] + 1e-9) / (share[k - 3] + 1e-9))
    nsw = D["ns"][max(0, k - 2):k + 1].sum(0)
    nw = n[max(0, k - 2):k + 1].sum(0)
    ns_share = nsw / np.maximum(nw, 1)
    dg = np.log((deg + 1) / (deg3 + 1))

    active = nT >= MIN_N
    mech_pos = -np.ones(N, dtype=int)
    mech_pos[mi] = np.arange(len(mi))
    ia_list, ib_list = [], []
    for r, a in enumerate(mi):
        if not active[a]:
            continue
        if target == "ns":  # known in the literature, not yet in the NS venue set
            ok = active & (cumM[r] >= 1) & (cumNSM[r] == 0)
        else:
            ok = active & (cumM[r] < NEW_MAX)
        ok[a] = False
        # mech-mech pairs: keep one orientation
        ok &= ~((mech_pos >= 0) & (mech_pos < r))
        b = np.nonzero(ok)[0]
        ia_list.append(np.full(len(b), r)); ib_list.append(b)
    r_ = np.concatenate(ia_list); b_ = np.concatenate(ib_list)
    a_ = mi[r_]
    fields = np.array([int(D["topics"][t]["field"] or 0) for t in D["ids"]])
    X = pd.DataFrame({
        "cum_ab": cumM[r_, b_], "cur_ab": curM[r_, b_],
        "logn_a": np.log1p(nT[a_]), "logn_b": np.log1p(nT[b_]),
        "g1_a": g1[a_], "g1_b": g1[b_], "g3_a": g3[a_], "g3_b": g3[b_],
        "gs3_a": gs3[a_], "gs3_b": gs3[b_],
        "deg_a": deg[a_], "deg_b": deg[b_], "dg_a": dg[a_], "dg_b": dg[b_],
        "cn": CN[r_, b_], "aa": AA[r_, b_],
        "jac": CN[r_, b_] / np.maximum(deg[a_] + deg[b_] - CN[r_, b_], 1),
        "lpa": np.log1p(deg[a_] * deg[b_]), "cos": COS[r_, b_],
        "ns_a": ns_share[a_], "ns_b": ns_share[b_],
        "b_is_mech": (mech_pos[b_] >= 0).astype(int), "field_b": fields[b_],
    })
    if D.get("use_sd"):
        sdw = D["sd"][max(0, k - 2):k + 1].sum(0)
        sdw3 = D["sd"][max(0, k - 5):max(0, k - 2)].sum(0)
        X["sd_a"], X["sd_b"] = np.log1p(sdw[a_]), np.log1p(sdw[b_])
        X["sdg_a"] = np.log((sdw[a_] + 1) / (sdw3[a_] + 1))
        X["sdg_b"] = np.log((sdw[b_] + 1) / (sdw3[b_] + 1))
    num = [c for c in X.columns if c != "field_b"]
    X[num] = X[num].astype(np.float32)
    X["field_b"] = X["field_b"].astype("category")
    meta = pd.DataFrame({"a": [D["ids"][i] for i in a_], "b": [D["ids"][i] for i in b_], "T": T})
    y = None
    if with_label and T + H in yrs:
        if target == "ns":
            y = (cum(D, T + H, "Cns")[mi].toarray()[r_, b_] >= 1).astype(int)
        else:
            cumH = cum(D, T + H)[mi].toarray()
            y = (cumH[r_, b_] >= HIT).astype(int)
    return X, y, meta


def auc(y, s):
    y = np.asarray(y); r = rankdata(s)
    p = y.sum(); q = len(y) - p
    return float((r[y == 1].sum() - p * (p + 1) / 2) / (p * q)) if p and q else float("nan")


def ap(y, s):
    o = np.argsort(-s, kind="stable"); yy = np.asarray(y)[o]
    hits = np.cumsum(yy); prec = hits / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / max(yy.sum(), 1))


def p_at(y, s, k):
    o = np.argsort(-s, kind="stable")[:k]
    return float(np.asarray(y)[o].mean())


def baselines(X):
    return {"PA_degree": X["lpa"].values,
            "PA_size": (X["logn_a"] + X["logn_b"]).values,
            "AdamicAdar": X["aa"].values + 1e-6 * X["lpa"].values,
            "growth": (X["g3_a"] + X["g3_b"]).values}


def train(Xs, ys, Xv=None, yv=None):
    import lightgbm as lgb
    rng = np.random.default_rng(SEED)
    X = pd.concat(Xs, ignore_index=True); y = np.concatenate(ys)
    keep = (y == 1) | (rng.random(len(y)) < 0.2)       # subsample negatives 1:5
    X, y = X[keep], y[keep]
    w = np.where(y == 1, 1.0, 5.0)                        # undo the subsampling in the loss
    params = dict(n_estimators=2000, learning_rate=0.03, num_leaves=63, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=SEED, verbose=-1)
    m = lgb.LGBMClassifier(**params)
    if Xv is not None:
        m.fit(X, y, sample_weight=w, eval_set=[(Xv, yv)], eval_metric="auc",  # noqa
              callbacks=[lgb.early_stopping(100, verbose=False)])
        return m, m.best_iteration_
    m.fit(X, y, sample_weight=w)
    return m, params["n_estimators"]


def evaluate(name, y, scores):
    rows = []
    for k, s in scores.items():
        rows.append({"split": name, "scorer": k, "AUC": auc(y, s), "AP": ap(y, s),
                     "P@100": p_at(y, s, 100), "P@1000": p_at(y, s, 1000)})
    return rows


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--data", required=True)
    ap_.add_argument("--out", required=True)
    ap_.add_argument("--ext", action="store_true", help="include energy/transport topics (ext_topics.json) as source nodes")
    ap_.add_argument("--target", choices=["cooc", "ns"], default="cooc",
                     help="cooc: co-occurrence emergence (v0); ns: first Nature/Science-family co-paper")
    ap_.add_argument("--scidata", action="store_true", help="add Scientific Data leading-indicator features")
    ap_.add_argument("--save-all", action="store_true", help="also write every candidate pair of the final forecast (gzip) with its percentile")
    a = ap_.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    D = load(a.data, ext=a.ext, ns_pairs=a.target == "ns", scidata=a.scidata)
    yrs = D["years"]
    print(f"loaded {len(D['ids'])} topics, {len(D['mech_idx'])} mech, years {yrs[0]}-{yrs[-1]}", flush=True)
    S = {}

    def snap(T):
        if T not in S:
            S[T] = snapshot(D, T, target=a.target)
            X, y, _ = S[T]
            print(f"  T={T}: {len(X):,} candidate pairs, positives {int(y.sum()) if y is not None else 'n/a'}"
                  f" ({time.time() - t0:.0f}s)", flush=True)
        return S[T]

    first = yrs[0] + 3
    tr_T = [T for T in range(first, 2016)]
    res = []
    # validation
    Xv, yv, _ = snap(2018)
    m, best_it = train([snap(T)[0] for T in tr_T], [snap(T)[1] for T in tr_T], Xv, yv)
    sv = m.predict_proba(Xv)[:, 1]
    res += evaluate("val_T2018", yv, {"model": sv, **baselines(Xv)})
    n_it = max(int(best_it or 300), 100)
    print(f"val done, best_iter={n_it}", flush=True)

    # retrain with fixed iteration count for test / K3 / final
    def fit(Ts):
        import lightgbm as lgb
        rng = np.random.default_rng(SEED)
        X = pd.concat([snap(T)[0] for T in Ts], ignore_index=True)
        y = np.concatenate([snap(T)[1] for T in Ts])
        keep = (y == 1) | (rng.random(len(y)) < 0.2)
        X, y = X[keep], y[keep]
        mm = lgb.LGBMClassifier(n_estimators=n_it, learning_rate=0.03, num_leaves=63, min_child_samples=50,
                                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                random_state=SEED, verbose=-1)
        mm.fit(X, y, sample_weight=np.where(y == 1, 1.0, 5.0))
        return mm

    # test (scored once)
    m_test = fit(list(range(first, 2019)))
    Xt, yt, mt = snap(2021)
    st = m_test.predict_proba(Xt)[:, 1]
    res += evaluate("test_T2021", yt, {"model": st, **baselines(Xt)})
    imp = pd.Series(m_test.booster_.feature_importance("gain"), index=Xt.columns).sort_values(ascending=False)
    imp.to_csv(os.path.join(a.out, "feature_importance_gain.csv"))
    R = pd.DataFrame(res)
    R.to_csv(os.path.join(a.out, "metrics.csv"), index=False)
    te = R[R.split == "test_T2021"].set_index("scorer")
    pa_best_auc = te.loc[["PA_degree", "PA_size"], "AUC"].max()
    base_best_p100 = te.drop(index="model")["P@100"].max()
    k1 = bool(te.loc["model", "AUC"] >= pa_best_auc + 0.03 and te.loc["model", "P@100"] > base_best_p100)
    print(R.to_string(index=False), flush=True)
    print(f"K1 {'PASS' if k1 else 'FAIL'}: model AUC {te.loc['model','AUC']:.3f} vs best PA {pa_best_auc:.3f};"
          f" P@100 {te.loc['model','P@100']:.2f} vs best baseline {base_best_p100:.2f}", flush=True)
    tops_t = mt.assign(score=st, hit=yt).sort_values("score", ascending=False).head(300)
    tops_t["a_name"] = tops_t.a.map(lambda t: D["topics"][t]["name"])
    tops_t["b_name"] = tops_t.b.map(lambda t: D["topics"][t]["name"])
    tops_t.to_csv(os.path.join(a.out, "test_T2021_top300.csv"), index=False)

    # K3: NC/SA 2023-2026 mechanical papers vs T=2020 ranking
    k3 = {}
    fk3 = os.path.join(a.data, "k3_ncsa_works.jsonl")
    if os.path.exists(fk3) and 2020 in yrs:
        m_k3 = fit(list(range(first, 2018)))
        X20, y20, meta20 = snap(2020)
        s20 = m_k3.predict_proba(X20)[:, 1]
        pct = rankdata(s20) / len(s20)
        pct_pa = rankdata(X20["lpa"].values) / len(s20)
        key = {(r.a, r.b): i for i, r in enumerate(meta20.itertuples())}
        mech = set(D["ids"][i] for i in D["mech_idx"])
        cum20 = cum(D, 2020, "Cns") if a.target == "ns" else cum(D, 2020)
        new_max = 1 if a.target == "ns" else NEW_MAX
        n_pairs = n_new = 0
        new_pct, new_pct_pa = [], []
        seen = set()
        for line in open(fk3, encoding="utf-8"):
            w = json.loads(line)
            ts = [t for t in w["topics"] if t in D["idx"]]
            for i_ in range(len(ts)):
                for j_ in range(i_ + 1, len(ts)):
                    a_, b_ = ts[i_], ts[j_]
                    if a_ not in mech and b_ not in mech:
                        continue
                    pr = tuple(sorted((a_, b_)))
                    if pr in seen:
                        continue
                    seen.add(pr)
                    n_pairs += 1
                    if cum20[D["idx"][a_], D["idx"][b_]] < new_max:
                        n_new += 1
                        for k_ in ((a_, b_), (b_, a_)):
                            if k_ in key:
                                new_pct.append(pct[key[k_]]); new_pct_pa.append(pct_pa[key[k_]])
                                break
        k3 = {"distinct_pairs": n_pairs, "new_at_2020": n_new, "new_in_candidate_set": len(new_pct),
              "mean_percentile_model": float(np.mean(new_pct)) if new_pct else None,
              "share_top10pct_model": float(np.mean(np.array(new_pct) >= 0.9)) if new_pct else None,
              "mean_percentile_PA": float(np.mean(new_pct_pa)) if new_pct_pa else None,
              "share_top10pct_PA": float(np.mean(np.array(new_pct_pa) >= 0.9)) if new_pct_pa else None}
        print("K3:", k3, flush=True)

    # final forecast at the last complete year
    T_last = yrs[-1]
    m_fin = fit([T for T in range(first, T_last - H + 1)])
    Xf, _, mf = snapshot(D, T_last, with_label=False, target=a.target)
    sf = m_fin.predict_proba(Xf)[:, 1]
    fc = mf.assign(p_emerge=sf, cum_ab=Xf["cum_ab"].values, n_a=np.expm1(Xf["logn_a"].values).round(),
                   n_b=np.expm1(Xf["logn_b"].values).round(), g3_a=Xf["g3_a"].values, g3_b=Xf["g3_b"].values,
                   ns_a=Xf["ns_a"].values, ns_b=Xf["ns_b"].values)
    fc["a_name"] = fc.a.map(lambda t: D["topics"][t]["name"])
    fc["b_name"] = fc.b.map(lambda t: D["topics"][t]["name"])
    fc["b_subfield"] = fc.b.map(lambda t: D["topics"][t]["subfield_name"])
    fc = fc.sort_values("p_emerge", ascending=False)
    fc["b_generic_ai"] = fc.b.map(lambda t: str(D["topics"][t]["subfield"]) in GENERIC_AI_SUBFIELDS).astype(int)
    fc.head(2000).to_csv(os.path.join(a.out, f"forecast_T{T_last}_top2000.csv"), index=False)
    fc[fc.b_generic_ai == 0].head(2000).to_csv(os.path.join(a.out, f"forecast_T{T_last}_top2000_nonAI.csv"), index=False)
    if a.save_all:
        fc["pct"] = (rankdata(fc.p_emerge.values) / len(fc)).round(4)
        fc.to_csv(os.path.join(a.out, f"forecast_T{T_last}_all.csv.gz"), index=False, compression="gzip")

    summary = {"K1_pass": k1, "best_iter": n_it, "K3": k3, "n_topics": len(D["ids"]),
               "n_mech": int(len(D["mech_idx"])), "years": [yrs[0], yrs[-1]],
               "rules": {"NEW_MAX": NEW_MAX, "HIT": HIT, "H": H, "MIN_N": MIN_N},
               "target": a.target, "scidata_features": bool(a.scidata),
               "share_generic_ai_top100": float(fc.head(100).b_generic_ai.mean()),
               "runtime_s": round(time.time() - t0)}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
