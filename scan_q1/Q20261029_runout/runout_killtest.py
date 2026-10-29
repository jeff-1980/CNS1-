"""Q-20261029-1 kill test: run-out handling bias in published weld S-N data.

Pre-registered in PREREG_Q20261029_runout.md (frozen, SHA-256 recorded before any
run on real data). Implements K1-K3 (decisive) on the Deng et al. 2025 Sci Data
weld fatigue database (figshare 10.6084/m9.figshare.29254265, CC BY 4.0).
K4 (exploratory) lives in runout_k4_xgb.py.

Usage: python runout_killtest.py S-N.json parameter.json OUTDIR
"""
import json
import os
import sys
import time

import numpy as np
from scipy import optimize, special, stats

SEED = 20261029
N_PLACEBO = 50
NS = (2e6, 1e7)
FL_GRID = np.arange(0.0, 0.951, 0.05)          # S0 / min failure stress
_trapz = getattr(np, "trapezoid", None) or np.trapz
GL_X, GL_W = np.polynomial.legendre.leggauss(48)
# IIW FAT classes (MPa), extended geometrically (R20-like) at both ends
FAT = np.array([10, 11.2, 12.5, 14, 16, 18, 20, 22.5, 25, 28, 32, 36, 40, 45, 50, 56, 63,
                71, 80, 90, 100, 112, 125, 140, 160, 180, 200, 225, 250, 280, 315, 355,
                400, 450, 500, 560, 630, 710, 800, 900, 1000, 1120, 1250])


def _load(path):
    d = json.load(open(path))
    return (d["data"], d.get("units")) if isinstance(d, dict) else (d, None)


def read_series(sn_path, par_path):
    sn, units = _load(sn_path)
    par, _ = _load(par_path)
    doi = {}
    for p in par:
        doi.setdefault(str(p["dataset_id"]), p.get("doi"))
    series = {}
    for r in sn:
        try:
            n, s, ro = float(r["life_n"]), float(r["stress_range"]), int(r["runout"])
        except (TypeError, ValueError, KeyError):
            continue
        if n > 0 and s > 0 and ro in (0, 1):
            series.setdefault(str(r["dataset_id"]), []).append((s, n, ro))
    return series, doi, units


# ---------------------------------------------------------------- fits
def ols(x, y):
    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    sd = np.sqrt((res ** 2).sum() / max(len(y) - 2, 1))
    return a, b, sd


def tobit_ll(p, x, y, c):
    a, b, ls = p
    sd = np.exp(ls)
    mu = a + b * x
    return np.where(c == 0, stats.norm.logpdf(y, mu, sd), stats.norm.logsf(y, mu, sd)).sum()


def tobit(x, y, c, start):
    a0, b0, sd0 = start
    r = optimize.minimize(lambda p: -tobit_ll(p, x, y, c), [a0, b0, np.log(max(sd0, 0.05))],
                          method="Nelder-Mead",
                          options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8})
    return r.x[0], r.x[1], np.exp(r.x[2]), r.success, -r.fun


def s_basquin(a, b, n):
    return 10 ** ((np.log10(n) - a) / b) if b < 0 else np.nan


def fit_fl(S, y, c, start):
    """Fixed fatigue-limit censored MLE: log10 N = a + b log10(S - S0), S0 profiled.
    Run-outs at S <= S0 have survival 1 (contribute 0); failures need S > S0."""
    smin = S[c == 0].min()
    best = None
    for f in FL_GRID:
        s0 = f * smin
        keep = S > s0
        x = np.log10(S[keep] - s0)
        cc, yy = c[keep], y[keep]
        if (cc == 0).sum() < 3 or np.ptp(x[cc == 0]) <= 0:
            continue
        a0, b0, sd0 = ols(x[cc == 0], yy[cc == 0])
        a, b, sd, ok, ll = tobit(x, yy, cc, (a0, b0, sd0))
        if ok and b < 0 and (best is None or ll > best[-1]):
            best = (f, s0, a, b, sd, ll)
    if best is None:
        return None
    f, s0, a, b, sd, ll = best
    return {"f": f, "s0": s0, "a": a, "b": b, "sd": sd, "ll": ll,
            "boundary": bool(f >= FL_GRID[-1] - 1e-9),
            "S": {n: s0 + 10 ** ((np.log10(n) - a) / b) for n in NS}}


def rflm_nll(p, S, y, c):
    """Pascual & Meeker (1999) random fatigue limit model, natural logs.
    ln N = b0 + b1 ln(S - V), ln V ~ N(mu_g, sg), ln N | V ~ N(., s); V >= S -> no failure."""
    b0, b1, ls, mg, lsg = p
    s, sg = np.exp(ls), np.exp(lsg)
    lnS = np.log(S)
    ln_n = y * np.log(10.0)
    lo = mg - 8 * sg
    hi = np.maximum(lnS, lo + 1e-9)
    v = 0.5 * (hi - lo)[:, None] * GL_X[None, :] + 0.5 * (hi + lo)[:, None]
    w = 0.5 * (hi - lo)[:, None] * GL_W[None, :]
    mu = b0 + b1 * np.log(np.maximum(S[:, None] - np.exp(v), 1e-300))
    gv = stats.norm.pdf(v, mg, sg)
    fail = (w * gv * stats.norm.pdf(ln_n[:, None], mu, s)).sum(1)
    surv = (w * gv * stats.norm.sf(ln_n[:, None], mu, s)).sum(1) + stats.norm.sf(lnS, mg, sg)
    ll = np.where(c == 0, np.log(np.maximum(fail, 1e-300)), np.log(np.maximum(surv, 1e-300)))
    return -ll.sum()


def fit_rflm(S, y, c, fl):
    if fl is None:
        return None
    ln10 = np.log(10.0)
    smin, smax = S[c == 0].min(), S.max()
    bounds = [(None, None), (-30.0, -0.3), (np.log(0.01), np.log(5.0)),
              (np.log(0.05 * smin), np.log(1.5 * smax)), (np.log(0.01), np.log(1.0))]
    r = None
    for f0 in (0.3, 0.7, 0.95):          # multi-start over the fatigue-limit location
        p0 = [fl["a"] * ln10, min(max(fl["b"], -29.0), -0.4), np.log(max(fl["sd"] * ln10, 0.02)),
              np.log(f0 * smin), np.log(0.1)]
        try:
            ri = optimize.minimize(rflm_nll, p0, args=(S, y, c), method="L-BFGS-B", bounds=bounds,
                                   options={"maxiter": 2000})
        except (FloatingPointError, ValueError):
            continue
        if np.isfinite(ri.fun) and (r is None or ri.fun < r.fun):
            r = ri
    if r is None:
        return None
    b0, b1, ls, mg, lsg = r.x
    if not r.success or b1 >= 0:
        return None
    at_bound = any(lo is not None and abs(v - lo) < 1e-6 or hi is not None and abs(v - hi) < 1e-6
                   for v, (lo, hi) in zip(r.x, bounds))
    out = {}
    for n in NS:
        # median-curve strength at life n: solve P(N <= n | S) = 0.5
        def g(s_):
            if s_ <= 0:
                return -0.5
            v = np.linspace(mg - 8 * np.exp(lsg), np.log(s_), 400)
            if v[-1] <= v[0]:
                return -0.5
            mu = b0 + b1 * np.log(np.maximum(s_ - np.exp(v), 1e-300))
            pdf = stats.norm.pdf(v, mg, np.exp(lsg))
            cdf_fail = _trapz(pdf * stats.norm.cdf(np.log(n), mu, np.exp(ls)), v)
            return cdf_fail - 0.5
        try:
            out[n] = optimize.brentq(g, 0.2 * smin, 20 * S.max(), xtol=1e-6)
        except ValueError:
            out[n] = np.nan
    return {"S": out, "sg": float(np.exp(lsg)), "mg": float(mg), "boundary": bool(at_bound)}


# ---------------------------------------------------------------- IIW characteristic / FAT
def k_tol(n):
    """One-sided tolerance factor, 95 % survival at 75 % confidence (noncentral t)."""
    if n < 3:
        return np.nan
    return stats.nct.ppf(0.75, df=n - 1, nc=stats.norm.ppf(0.95) * np.sqrt(n)) / np.sqrt(n)


def s_char(a, b, sd, n_pts, n=2e6):
    if b >= 0:
        return np.nan
    return 10 ** ((np.log10(n) - a + k_tol(n_pts) * sd) / b)


def fat_class(s):
    if not np.isfinite(s) or s < FAT[0]:
        return np.nan
    return FAT[FAT <= s + 1e-9].max()


def fixed_m3(x, y, c, censored):
    """Fixed-slope (m = 3, b = -3) fit: intercept + sd only."""
    if censored:
        def nll(p):
            a, ls = p
            sd = np.exp(ls)
            mu = a - 3 * x
            return -np.where(c == 0, stats.norm.logpdf(y, mu, sd), stats.norm.logsf(y, mu, sd)).sum()
        f = (c == 0)
        a0 = np.mean(y[f] + 3 * x[f])
        r = optimize.minimize(nll, [a0, np.log(0.2)], method="Nelder-Mead")
        return r.x[0], -3.0, np.exp(r.x[1]), len(y)
    a = np.mean(y + 3 * x)
    sd = np.sqrt(((y - (a - 3 * x)) ** 2).sum() / max(len(y) - 1, 1))
    return a, -3.0, sd, len(y)


# ---------------------------------------------------------------- main
def summ(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "median": float(np.median(v)),
            "median_abs": float(np.median(np.abs(v))),
            "share_abs_gt5": float(np.mean(np.abs(v) > 0.05)),
            "share_abs_gt10": float(np.mean(np.abs(v) > 0.10)),
            "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75))}


def main(sn_path, par_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    series, doi, units = read_series(sn_path, par_path)
    assert series, "no S-N series parsed"
    log = open(os.path.join(outdir, "progress.log"), "a")
    rows = []
    n_ro = 0
    keys = sorted(series)
    for i, k in enumerate(keys):
        p = np.array(series[k])
        S, y, c = p[:, 0], np.log10(p[:, 1]), p[:, 2].astype(int)
        x = np.log10(S)
        nf, nr = int((c == 0).sum()), int((c == 1).sum())
        n_ro += nr >= 1
        if nf < 3 or nr < 1 or np.ptp(x[c == 0]) <= 0:
            continue
        ad, bd, sdd = ols(x[c == 0], y[c == 0])
        af, bf, _ = ols(x, y)
        at, bt, sdt, ok, _ = tobit(x, y, c, (ad, bd, sdd))
        if not ok:
            continue
        rec = {"id": k, "doi": doi.get(k), "nf": nf, "nr": nr,
               "smin_fail": float(S[c == 0].min()), "smax": float(S.max()),
               "ro_stop_median": float(np.median(p[c == 1, 1]))}
        for N in NS:
            st = s_basquin(at, bt, N)
            rec[f"S_tobit_{N:.0e}"] = st
            rec[f"S_drop_{N:.0e}"] = s_basquin(ad, bd, N)
            rec[f"S_fail_{N:.0e}"] = s_basquin(af, bf, N)
        # K1 knee-aware references
        fl = fit_fl(S, y, c, (at, bt, sdt))
        rec["fl_ok"] = fl is not None
        if fl:
            rec["fl_f"], rec["fl_boundary"] = fl["f"], fl["boundary"]
            for N in NS:
                rec[f"S_fl_{N:.0e}"] = fl["S"][N]
        rf = fit_rflm(S, y, c, fl)
        rec["rflm_ok"] = rf is not None
        if rf:
            rec["rflm_boundary"] = rf["boundary"]
            for N in NS:
                rec[f"S_rflm_{N:.0e}"] = rf["S"][N]
        # K2 placebo: delete nr random failures (run-outs kept), refit Tobit
        if nf - nr >= 3:
            fidx = np.where(c == 0)[0]
            pl = []
            for _ in range(N_PLACEBO):
                drop = rng.choice(fidx, nr, replace=False)
                keep = np.ones(len(c), bool)
                keep[drop] = False
                xx, yy, cc = x[keep], y[keep], c[keep]
                if np.ptp(xx[cc == 0]) <= 0:
                    pl.append([np.nan, np.nan])
                    continue
                a1, b1, s1 = ols(xx[cc == 0], yy[cc == 0])
                ap, bp, _, okp, _ = tobit(xx, yy, cc, (a1, b1, s1))
                pl.append([s_basquin(ap, bp, N) / rec[f"S_tobit_{N:.0e}"] - 1 if okp else np.nan
                           for N in NS])
            rec["placebo"] = pl
        # K3 IIW characteristic strength at 2e6 and FAT class
        rec["Sc_drop"] = s_char(ad, bd, sdd, nf)
        rec["Sc_tobit"] = s_char(at, bt, sdt, nf + nr)
        rec["Sm_drop"], rec["Sm_tobit"] = rec["S_drop_2e+06"], rec["S_tobit_2e+06"]
        a3d, b3d, s3d, n3d = fixed_m3(x[c == 0], y[c == 0], c[c == 0], False)
        a3t, b3t, s3t, n3t = fixed_m3(x, y, c, True)
        rec["Sc_m3_drop"] = s_char(a3d, b3d, s3d, n3d)
        rec["Sc_m3_tobit"] = s_char(a3t, b3t, s3t, n3t)
        rows.append(rec)
        if i % 200 == 0:
            log.write(f"{time.time() - t0:.0f}s series {i}/{len(keys)} eligible {len(rows)}\n")
            log.flush()

    def bias(num, den):
        return [r.get(num, np.nan) / r.get(den, np.nan) - 1 if r.get(den) else np.nan for r in rows]

    res = {"units": units, "n_series": len(series), "n_with_runout": int(n_ro),
           "n_eligible": len(rows), "n_doi": len({r["doi"] for r in rows}),
           "runtime_s": round(time.time() - t0)}
    # baseline replication
    for N in ("2e+06", "1e+07"):
        res[f"drop_vs_tobit_{N}"] = summ(bias(f"S_drop_{N}", f"S_tobit_{N}"))
        res[f"fail_vs_tobit_{N}"] = summ(bias(f"S_fail_{N}", f"S_tobit_{N}"))
        res[f"drop_vs_fl_{N}"] = summ(bias(f"S_drop_{N}", f"S_fl_{N}"))
        res[f"fail_vs_fl_{N}"] = summ(bias(f"S_fail_{N}", f"S_fl_{N}"))
        res[f"drop_vs_rflm_{N}"] = summ(bias(f"S_drop_{N}", f"S_rflm_{N}"))
        nb = [r.get(f"S_drop_{N}", np.nan) / r[f"S_fl_{N}"] - 1
              for r in rows if r.get("fl_ok") and not r.get("fl_boundary")]
        res[f"drop_vs_fl_nonboundary_{N}"] = summ(nb)
    res["fl_ok"] = int(sum(r["fl_ok"] for r in rows))
    res["fl_boundary"] = int(sum(r.get("fl_boundary", False) for r in rows))
    res["fl_f_median"] = float(np.median([r["fl_f"] for r in rows if r["fl_ok"]]))
    res["rflm_ok"] = int(sum(r["rflm_ok"] for r in rows))
    res["rflm_boundary"] = int(sum(r.get("rflm_boundary", False) for r in rows))

    # K1 decision (primary: fixed fatigue-limit censored MLE at 2e6)
    k1 = res["drop_vs_fl_2e+06"]["median_abs"]
    res["K1"] = {"stat_median_abs_drop_vs_fl_2e6": k1, "threshold": 0.03,
                 "pass": bool(k1 >= 0.03)}

    # K2 decision
    sub = [r for r in rows if "placebo" in r]
    drop_sub = [r["S_drop_2e+06"] / r["S_tobit_2e+06"] - 1 for r in sub]
    d_med_abs = float(np.nanmedian(np.abs(drop_sub)))
    d_med = float(np.nanmedian(drop_sub))
    P = np.array([r["placebo"] for r in sub], float)        # series x draws x N
    pl_med_abs = np.nanmedian(np.abs(P[:, :, 0]), axis=0)    # per draw
    pl_med = np.nanmedian(P[:, :, 0], axis=0)
    p95_abs = float(np.percentile(pl_med_abs, 95))
    res["K2"] = {"n_series": len(sub), "drop_median_abs_2e6": d_med_abs, "drop_median_2e6": d_med,
                 "placebo_median_abs_p95": p95_abs,
                 "placebo_median_abs_mean": float(np.mean(pl_med_abs)),
                 "placebo_median_signed_p2.5_p97.5": [float(np.percentile(pl_med, 2.5)),
                                                      float(np.percentile(pl_med, 97.5))],
                 "drop_1e7_median_abs": float(np.nanmedian(np.abs(
                     [r["S_drop_1e+07"] / r["S_tobit_1e+07"] - 1 for r in sub]))),
                 "placebo_1e7_median_abs_p95": float(np.percentile(
                     np.nanmedian(np.abs(P[:, :, 1]), axis=0), 95)),
                 "pass": bool(d_med_abs > p95_abs)}

    # K3 decision
    def fat_change(a_key, b_key):
        ch, n = 0, 0
        steps = []
        for r in rows:
            fa, fb = fat_class(r.get(a_key, np.nan)), fat_class(r.get(b_key, np.nan))
            if np.isfinite(fa) and np.isfinite(fb):
                n += 1
                ch += fa != fb
                steps.append(int(np.searchsorted(FAT, fa) - np.searchsorted(FAT, fb)))
            r[f"fat_{a_key}"], r[f"fat_{b_key}"] = fa, fb
        steps = np.array(steps)
        return {"n": n, "share_changed": ch / n if n else np.nan,
                "share_lower": float(np.mean(steps < 0)) if n else np.nan,
                "share_higher": float(np.mean(steps > 0)) if n else np.nan,
                "share_ge2_classes": float(np.mean(np.abs(steps) >= 2)) if n else np.nan}
    res["K3_char_free"] = fat_change("Sc_drop", "Sc_tobit")
    res["K3_mean_free"] = fat_change("Sm_drop", "Sm_tobit")
    res["K3_char_m3"] = fat_change("Sc_m3_drop", "Sc_m3_tobit")
    res["K3"] = {"stat_share_changed": res["K3_char_free"]["share_changed"], "threshold": 0.15,
                 "pass": bool(res["K3_char_free"]["share_changed"] >= 0.15)}
    res["verdict"] = "PASS" if all(res[k]["pass"] for k in ("K1", "K2", "K3")) else "KILL"
    res["runtime_s"] = round(time.time() - t0)

    for r in rows:
        r.pop("placebo", None)
    json.dump(rows, open(os.path.join(outdir, "rows.json"), "w"), default=float)
    json.dump(res, open(os.path.join(outdir, "result.json"), "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
    log.write("DONE\n")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main(*sys.argv[1:4])
