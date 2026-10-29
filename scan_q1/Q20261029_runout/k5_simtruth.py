"""Q-20261029-1 K5: simulated-truth check of the run-out handling bias.

Pre-registered in PREREG_K5_simtruth.md. Parametric simulation that keeps every
eligible real series' design (stress levels, which points were run-outs, stop counts)
and replaces the lives with draws from a known truth curve. Three truths per series:
  T1 Basquin       : log10 N = a + b log10 S (the series' own Tobit fit)
  T2 fatigue limit : log10 N = a + b log10(S - S0) (the series' own FL-MLE fit)
  T3 IIW-type knee : T1 above S_k = S_T1(1e7); slope -22 below S_k
Scatter sigma = series Tobit sigma clipped to [0.10, 0.40]. Failures are censored at
the series' largest run-out stop count; run-out points keep their own stop count.
Estimators DROP, FAIL, TOBIT, FL (same code as runout_killtest.py) are compared with
the true median strength at 2e6 (decisive) and 1e7 (reported).

Also (report only, real data): decomposition of the K3 characteristic-strength gap
(mean / k / sigma / slope terms), expected class-flip rate from discretisation alone,
and DOI-cluster bootstrap intervals for the K1-K3 statistics.

Usage: python k5_simtruth.py S-N.json parameter.json OUTDIR
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runout_killtest as K  # noqa: E402

SEED = 20261030
REPS = 20
N2, N7 = 2e6, 1e7
SIG_LO, SIG_HI = 0.10, 0.40
M2_KNEE = 22.0
FAT_STEP = np.log10(1.122)


def truth_logn(kind, p, S):
    if kind == "T1":
        return p["a"] + p["b"] * np.log10(S)
    if kind == "T2":
        out = np.full(len(S), np.inf)
        ok = S > p["s0"]
        out[ok] = p["a"] + p["b"] * np.log10(S[ok] - p["s0"])
        return out
    # T3: Basquin above the knee stress, slope -M2_KNEE below it
    sk = p["sk"]
    up = p["a"] + p["b"] * np.log10(S)
    lo = np.log10(N7) - M2_KNEE * (np.log10(S) - np.log10(sk))
    return np.where(S >= sk, up, lo)


def truth_s(kind, p, n):
    if kind == "T2":
        return p["s0"] + 10 ** ((np.log10(n) - p["a"]) / p["b"])
    if kind == "T3" and n > N7:
        return p["sk"] * 10 ** ((np.log10(N7) - np.log10(n)) / M2_KNEE)
    return 10 ** ((np.log10(n) - p["a"]) / p["b"])


def estimates(S, y, c):
    x = np.log10(S)
    if (c == 0).sum() < 3 or (c == 1).sum() < 1 or np.ptp(x[c == 0]) <= 0:
        return None
    ad, bd, sdd = K.ols(x[c == 0], y[c == 0])
    af, bf, _ = K.ols(x, y)
    at, bt, sdt, ok, _ = K.tobit(x, y, c, (ad, bd, sdd))
    if not ok:
        return None
    fl = K.fit_fl(S, y, c, (at, bt, sdt))
    out = {}
    for n in (N2, N7):
        out[("drop", n)] = K.s_basquin(ad, bd, n)
        out[("fail", n)] = K.s_basquin(af, bf, n)
        out[("tobit", n)] = K.s_basquin(at, bt, n)
        out[("fl", n)] = fl["S"][n] if fl else np.nan
    return out


def sim_series(args):
    idx, sid, pts = args
    np.seterr(all="ignore")
    rng = np.random.default_rng(np.random.SeedSequence([SEED, idx]))
    p = np.array(pts)
    S, y, c = p[:, 0], np.log10(p[:, 1]), p[:, 2].astype(int)
    x = np.log10(S)
    if (c == 0).sum() < 3 or (c == 1).sum() < 1 or np.ptp(x[c == 0]) <= 0:
        return None
    ad, bd, sdd = K.ols(x[c == 0], y[c == 0])
    at, bt, sdt, ok, _ = K.tobit(x, y, c, (ad, bd, sdd))
    if not ok or bt >= 0:
        return None
    fl = K.fit_fl(S, y, c, (at, bt, sdt))
    sig = float(np.clip(sdt, SIG_LO, SIG_HI))
    stop_all = p[c == 1, 1].max()
    stop = np.where(c == 1, p[:, 1], stop_all)
    truths = {"T1": {"a": at, "b": bt}, "T3": {"a": at, "b": bt, "sk": K.s_basquin(at, bt, N7)}}
    if fl:
        truths["T2"] = {"a": fl["a"], "b": fl["b"], "s0": fl["s0"]}
    res = {"id": sid}
    for kind, tp in truths.items():
        ts = {n: truth_s(kind, tp, n) for n in (N2, N7)}
        mu = truth_logn(kind, tp, S)
        err = {(m, n): [] for m in ("drop", "fail", "tobit", "fl") for n in (N2, N7)}
        gap = []
        n_ok = 0
        for _ in range(REPS):
            ly = mu + rng.normal(0, sig, len(S))
            cc = (ly >= np.log10(stop)).astype(int)
            yy = np.where(cc == 1, np.log10(stop), ly)
            e = estimates(S, yy, cc)
            if e is None:
                continue
            n_ok += 1
            for (m, n), v in e.items():
                err[(m, n)].append(v / ts[n] - 1)
            gap.append(e[("drop", N2)] / e[("tobit", N2)] - 1)
        res[kind] = {"n_ok": n_ok, "gap_drop_tobit_2e6": float(np.nanmedian(gap)) if gap else np.nan}
        for (m, n), v in err.items():
            v = np.array(v, float)
            res[kind][f"{m}_{n:.0e}_med"] = float(np.nanmedian(v)) if len(v) else np.nan
            res[kind][f"{m}_{n:.0e}_mae"] = float(np.nanmedian(np.abs(v))) if len(v) else np.nan
    return res


def decomposition(series, doi):
    """Real data: split log10(Sc_DROP / Sc_TOBIT) at 2e6 into mean, k, sigma, slope terms."""
    rows = []
    for sid, pts in series.items():
        p = np.array(pts)
        S, y, c = p[:, 0], np.log10(p[:, 1]), p[:, 2].astype(int)
        x = np.log10(S)
        nf, nr = int((c == 0).sum()), int((c == 1).sum())
        if nf < 3 or nr < 1 or np.ptp(x[c == 0]) <= 0:
            continue
        ad, bd, sdd = K.ols(x[c == 0], y[c == 0])
        at, bt, sdt, ok, _ = K.tobit(x, y, c, (ad, bd, sdd))
        if not ok or bd >= 0 or bt >= 0:
            continue
        rec = {"id": sid, "doi": doi.get(sid)}
        a3d, b3d, s3d, n3d = K.fixed_m3(x[c == 0], y[c == 0], c[c == 0], False)
        a3t, b3t, s3t, n3t = K.fixed_m3(x, y, c, True)
        for lab, (A, B, SD, KK), (A2, B2, SD2, KK2) in (
                ("free", (ad, bd, sdd, K.k_tol(nf)), (at, bt, sdt, K.k_tol(nf + nr))),
                ("m3", (a3d, b3d, s3d, K.k_tol(n3d)), (a3t, b3t, s3t, K.k_tol(n3t)))):
            L = np.log10(N2)
            mean_d, mean_t = (L - A) / B, (L - A2) / B2
            rec[f"{lab}_mean"] = mean_d - mean_t
            rec[f"{lab}_k"] = (KK - KK2) * SD2 / B2
            rec[f"{lab}_sigma"] = KK * (SD - SD2) / B2
            rec[f"{lab}_slope"] = KK * SD * (1 / B - 1 / B2)
            rec[f"{lab}_total"] = sum(rec[f"{lab}_{t}"] for t in ("mean", "k", "sigma", "slope"))
            fd = K.fat_class(10 ** (mean_d + KK * SD / B))
            ft = K.fat_class(10 ** (mean_t + KK2 * SD2 / B2))
            rec[f"{lab}_flip"] = (fd != ft) if np.isfinite(fd) and np.isfinite(ft) else np.nan
        rows.append(rec)
    out = {"n": len(rows)}
    for lab in ("free", "m3"):
        for t in ("mean", "k", "sigma", "slope", "total"):
            v = np.array([r[f"{lab}_{t}"] for r in rows], float)
            v = v[np.isfinite(v)]
            out[f"{lab}_{t}_median_pct"] = float(np.median(10 ** v - 1))
            out[f"{lab}_{t}_share_neg"] = float(np.mean(v < 0))
        tot = np.array([r[f"{lab}_total"] for r in rows], float)
        tot = tot[np.isfinite(tot)]
        out[f"{lab}_expected_flip_from_gap"] = float(np.mean(np.minimum(np.abs(tot) / FAT_STEP, 1)))
        fl = np.array([r[f"{lab}_flip"] for r in rows], float)
        out[f"{lab}_observed_flip"] = float(np.nanmean(fl))
    return out, rows


def doi_bootstrap(rows_path, n_boot=2000):
    rows = json.load(open(rows_path))
    rng = np.random.default_rng(SEED)
    by = {}
    for r in rows:
        by.setdefault(r.get("doi") or f"ds:{r['id']}", []).append(r)
    keys = list(by)

    def stats_of(sample):
        d = np.array([r["S_drop_2e+06"] / r["S_tobit_2e+06"] - 1 for r in sample], float)
        f = np.array([r["S_drop_2e+06"] / r["S_fl_2e+06"] - 1 if r.get("S_fl_2e+06") else np.nan
                      for r in sample], float)
        ch = [K.fat_class(r["Sc_drop"]) != K.fat_class(r["Sc_tobit"]) for r in sample
              if np.isfinite(K.fat_class(r["Sc_drop"])) and np.isfinite(K.fat_class(r["Sc_tobit"]))]
        return (np.nanmedian(d), np.nanmedian(np.abs(f)), np.mean(ch) if ch else np.nan)

    bs = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys), replace=True)
        bs.append(stats_of([r for i in pick for r in by[keys[i]]]))
    bs = np.array(bs, float)
    names = ("drop_vs_tobit_2e6_median", "K1_drop_vs_fl_2e6_median_abs", "K3_share_changed")
    return {nm: {"point": float(v), "ci95": [float(np.nanpercentile(bs[:, j], 2.5)),
                                             float(np.nanpercentile(bs[:, j], 97.5))]}
            for j, (nm, v) in enumerate(zip(names, stats_of(rows)))} | {"n_clusters": len(keys)}


def main(sn_path, par_path, outdir):
    np.seterr(all="ignore")
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    log = open(os.path.join(outdir, "k5_progress.log"), "a")
    series, doi, _ = K.read_series(sn_path, par_path)
    keys = sorted(series)
    tasks = [(i, k, series[k]) for i, k in enumerate(keys)]
    procs = max(1, (os.cpu_count() or 2) - 2)
    log.write(f"start {len(tasks)} series, {procs} procs\n")
    log.flush()
    sims = []
    with mp.Pool(procs) as pool:
        for j, r in enumerate(pool.imap_unordered(sim_series, tasks, chunksize=4)):
            if r is not None:
                sims.append(r)
            if j % 200 == 0:
                log.write(f"{time.time() - t0:.0f}s {j}/{len(tasks)} sims {len(sims)}\n")
                log.flush()
    json.dump(sims, open(os.path.join(outdir, "k5_series.json"), "w"), default=float)

    res = {"n_series_simulated": len(sims), "reps": REPS, "procs": procs}
    for kind in ("T1", "T2", "T3"):
        sub = [r[kind] for r in sims if kind in r and r[kind]["n_ok"] > 0]
        d = {"n_series": len(sub)}
        for m in ("drop", "fail", "tobit", "fl"):
            for n in ("2e+06", "1e+07"):
                d[f"{m}_{n}_median_bias"] = float(np.nanmedian([s[f"{m}_{n}_med"] for s in sub]))
                d[f"{m}_{n}_median_abs_err"] = float(np.nanmedian([s[f"{m}_{n}_mae"] for s in sub]))
        d["sim_gap_drop_vs_tobit_2e6"] = float(np.nanmedian([s["gap_drop_tobit_2e6"] for s in sub]))
        d["K5b_drop_worse_than_tobit"] = d["drop_2e+06_median_abs_err"] > d["tobit_2e+06_median_abs_err"]
        d["K5b_drop_worse_than_fl"] = d["drop_2e+06_median_abs_err"] > d["fl_2e+06_median_abs_err"]
        res[kind] = d
    k5a = abs(res["T3"]["drop_2e+06_median_bias"])
    res["K5a"] = {"stat_abs_median_drop_bias_T3_2e6": k5a, "threshold": 0.03, "pass": bool(k5a >= 0.03)}
    k5b_fail = [t for t in ("T1", "T2", "T3")
                if not (res[t]["K5b_drop_worse_than_tobit"] or res[t]["K5b_drop_worse_than_fl"])]
    res["K5b"] = {"truths_where_censored_not_better": k5b_fail, "pass": not k5b_fail}
    res["verdict_K5"] = "PASS" if res["K5a"]["pass"] and res["K5b"]["pass"] else "KILL"
    res["observed_gap_drop_vs_tobit_2e6_real"] = -0.0599

    dec, dec_rows = decomposition(series, doi)
    res["decomposition_real"] = dec
    json.dump(dec_rows, open(os.path.join(outdir, "k5_decomposition_rows.json"), "w"), default=float)
    rows_path = os.path.join(outdir, "rows.json")
    if os.path.exists(rows_path):
        res["doi_bootstrap_real"] = doi_bootstrap(rows_path)
    res["runtime_s"] = round(time.time() - t0)
    json.dump(res, open(os.path.join(outdir, "k5_result.json"), "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
    log.write("DONE\n")


if __name__ == "__main__":
    main(*sys.argv[1:4])
