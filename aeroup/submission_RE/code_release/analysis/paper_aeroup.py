"""Paper analysis for "AeroUp vortex-generator retrofit at Hill of Towie: drivetrain load cost of an energy gain".
Descriptive extension of the frozen kill tests (scan_daily_q1/kill_tests/Q20261030_aeroup, PLAN_v1_FROZEN,
DEVIATIONS D1/D2). No new pass/fail gates. Estimators are copied from the frozen scripts; this file only
(i) adds outcomes (torque DEL for several S-N slopes, rotor speed, within-10-min power std, all temperature
channels), (ii) writes figure-ready CSVs, (iii) adds a sensitivity grid, an event-study series, a power-bin
decomposition, a torque-spectrum comparison, and a pure-scaling benchmark for the load cost ratio.

Usage: python paper_aeroup.py <ROOT with 2020..2023.zip + Hill_of_Towie_ShutdownDuration.zip> <OUTDIR> [NBOOT=2000]
Read-only on ROOT. Writes CSVs and log lines to OUTDIR.
"""
import glob
import io
import os
import re
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT, OUT = sys.argv[1], sys.argv[2]
NBOOT = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
os.makedirs(OUT, exist_ok=True)
if os.path.exists(f"{OUT}/log.txt"):
    os.remove(f"{OUT}/log.txt")  # fresh log per run (only our own output file)
RATED = 2300.0
T13 = 2304522
STATIONS = list(range(2304510, 2304531))
NAME = {s: f"T{s - 2304509:02d}" for s in STATIONS}
PRE = ("2020-01-01", "2021-09-22 23:59")
POST = ("2021-09-30", "2022-07-13 23:59")
MS = [3, 10 / 3, 4, 6, 8, 10]          # S-N / bearing exponents for the torque DEL proxy
MK = {m: ("D" + (f"{m:g}" if m != 10 / 3 else "10_3")) for m in MS}
rng = np.random.default_rng(20261104)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(f"{OUT}/log.txt", "a") as fh:
        fh.write(s + "\n")


# ------------------------------------------------------------------ inventory
def read_small(fh):
    raw = fh.read()
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(io.StringIO(raw.decode(enc)))
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    return pd.DataFrame({"unreadable": [len(raw)]})


log("ROOT files:", sorted(os.path.basename(p) for p in glob.glob(f"{ROOT}/*")))
for p in sorted(glob.glob(f"{ROOT}/*")):
    if p.lower().endswith(".csv") and os.path.getsize(p) < 2_000_000:
        log(f"\n--- small csv {os.path.basename(p)} ---")
        with open(p, "rb") as fh:
            log(read_small(fh).to_string(max_rows=80))
    if p.lower().endswith(".zip") and not re.search(r"/20\d\d\.zip$", p):
        with zipfile.ZipFile(p) as z:
            log(f"zip {os.path.basename(p)}: {z.namelist()[:20]}")
            for n in z.namelist():
                if n.lower().endswith(".csv") and z.getinfo(n).file_size < 200_000:
                    with z.open(n) as fh:
                        log(f"--- {n} ---\n" + read_small(fh).to_string(max_rows=80))


def list_cols(table, year=2021):
    with zipfile.ZipFile(f"{ROOT}/{year}.zip") as z:
        n = sorted(n for n in z.namelist() if n.split("/")[-1].startswith(table + "_"))[0]
        with z.open(n) as f:
            return list(pd.read_csv(f, nrows=0).columns)


def read_table(table, cols, years=(2020, 2021, 2022, 2023)):
    out = []
    for y in years:
        with zipfile.ZipFile(f"{ROOT}/{y}.zip") as z:
            names = sorted(n for n in z.namelist() if n.split("/")[-1].startswith(table + "_"))
            if not names:
                raise SystemExit(f"{table} not in {y}.zip")
            for n in names:
                with z.open(n) as f:
                    head = pd.read_csv(f, nrows=0).columns
                use = [c for c in ["TimeStamp", "StationId"] + cols if c in head]
                with z.open(n) as f:
                    out.append(pd.read_csv(f, usecols=use, low_memory=False))
    df = pd.concat(out, ignore_index=True)
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], utc=True, format="mixed")
    return df.drop_duplicates(["TimeStamp", "StationId"])


cols_grid, cols_tur, cols_tmp = list_cols("tblSCTurGrid"), list_cols("tblSCTurbine"), list_cols("tblSCTurTemp")
log("\ntblSCTurGrid columns:", cols_grid)
log("tblSCTurbine columns:", cols_tur)
log("tblSCTurTemp columns:", cols_tmp)
SD_COL = "wtc_ActPower_stddev" if "wtc_ActPower_stddev" in cols_grid else None
PITCH = [c for c in cols_tur if "pitch" in c.lower() and c.endswith("_mean")][:3]
TEMPS = [c for c in cols_tmp if c.endswith("_mean") and c != "wtc_AmbieTmp_mean"]
log("SD_COL", SD_COL, "PITCH", PITCH, "TEMPS", TEMPS)

grid = read_table("tblSCTurGrid", ["wtc_ActPower_mean"] + ([SD_COL] if SD_COL else []))
tur = read_table("tblSCTurbine", ["wtc_MainSRpm_mean", "wtc_YawPos_mean", "wtc_PowerRef_endvalue"] + PITCH)
tmp = read_table("tblSCTurTemp", ["wtc_AmbieTmp_mean"] + TEMPS)
df = grid.merge(tur, on=["TimeStamp", "StationId"], how="inner").merge(tmp, on=["TimeStamp", "StationId"], how="left")
del grid, tur, tmp
with zipfile.ZipFile(f"{ROOT}/Hill_of_Towie_ShutdownDuration.zip") as z:
    name = [n for n in z.namelist() if n.endswith(".csv")][0]
    with z.open(name) as f:
        sd = pd.read_csv(f)
sd["TimeStamp"] = pd.to_datetime(sd["TimeStamp_StartFormat"], utc=True, format="mixed") + pd.Timedelta(minutes=10)
sd["StationId"] = sd["TurbineName"].str[1:].astype(int) + 2304509
sd = sd.drop_duplicates(["TimeStamp", "StationId"])
df = df.merge(sd[["TimeStamp", "StationId", "ShutdownDuration"]], on=["TimeStamp", "StationId"], how="left")
log("rows merged:", len(df), " share with shutdown info:", round(df["ShutdownDuration"].notna().mean(), 3))

# data-quality counts before filtering
df["year"] = df["TimeStamp"].dt.year
f_sd = df["ShutdownDuration"].fillna(1) == 0
f_p = df["wtc_ActPower_mean"] > 0
f_cur = df["wtc_PowerRef_endvalue"].fillna(RATED) >= 2290
q = pd.DataFrame({"all": df.groupby(["StationId", "year"]).size(),
                  "no_shutdown": df[f_sd].groupby(["StationId", "year"]).size(),
                  "no_shutdown_p_pos": df[f_sd & f_p].groupby(["StationId", "year"]).size(),
                  "kept": df[f_sd & f_p & f_cur].groupby(["StationId", "year"]).size()})
q.to_csv(f"{OUT}/data_quality.csv")
df = df[f_sd & f_p & f_cur].copy()
df["omega"] = df["wtc_MainSRpm_mean"] * 2 * np.pi / 60
df = df[df["omega"] > 0.3]
df["T"] = df["wtc_ActPower_mean"] * 1e3 / df["omega"] / 1e6  # MNm, electrical torque proxy at main shaft
log("normal-operation rows:", len(df), " per turbine median:", int(df.groupby("StationId").size().median()))

PV = {k: df.pivot_table(index="TimeStamp", columns="StationId", values=c) for k, c in
      [("p", "wtc_ActPower_mean"), ("yaw", "wtc_YawPos_mean"), ("t", "T"), ("w", "omega"),
       ("amb", "wtc_AmbieTmp_mean")] + ([("sd", SD_COL)] if SD_COL else []) +
      [(f"pitch{i}", c) for i, c in enumerate(PITCH)] + [(c, c) for c in TEMPS]}
P, Y = PV["p"], PV["yaw"]
EXTRA = [k for k in PV if k not in ("p", "yaw")]
del df


# ------------------------------------------------------------------ frozen estimator (generalised outcomes)
def frame_for(target, exclude=(T13,), pbin=50, nsec=8, min_n=10, rated_frac=0.95):
    refs = [s for s in P.columns if s != target and s not in exclude]
    rp = P[refs]
    n = rp.notna().sum(axis=1)
    yaw = np.deg2rad(Y[refs])
    ang = np.rad2deg(np.arctan2(np.sin(yaw).mean(axis=1), np.cos(yaw).mean(axis=1))) % 360
    f = pd.DataFrame({"p": P[target], "rp": rp.mean(axis=1), "sec": ang // (360 / nsec), "n": n})
    for k in EXTRA:
        f[k] = PV[k].get(target)
    f = f[(f["n"] >= min_n) & f["p"].notna()].dropna(subset=["rp", "sec", "t"])
    f["cell"] = (f["rp"] // pbin).astype(int).astype(str) + "_" + f["sec"].astype(int).astype(str)
    f["rated"] = (f["p"] >= rated_frac * RATED).astype(float)
    f["week"] = f.index.to_period("W").astype(str)
    return f


def frame_ref(target, ref):  # frozen K3/K3b single-reference frame
    f = pd.DataFrame({"p": P[target], "t": PV["t"][target], "rp": P[ref], "yaw": Y[ref]})
    f = f.dropna(subset=["p", "t", "rp", "yaw"])
    f["cell"] = (f["rp"] // 50).astype(int).astype(str) + "_" + ((f["yaw"] % 360) // 45).astype(int).astype(str)
    f["rated"] = (f["p"] >= 0.95 * RATED).astype(float)
    return f


def vals_of(d):
    v = {"p": d["p"], "rated": d["rated"]}
    for m in MS:
        v[MK[m]] = d["t"] ** m
    for k in ("w", "sd", "pitch0"):
        if k in d:
            v[k] = d[k]
    return pd.DataFrame(v)


def effects(pre, post, w=None):
    vals = vals_of(pre)
    if w is None:
        cm = vals.groupby(pre["cell"]).mean()
    else:
        num = vals.mul(w, axis=0).groupby(pre["cell"]).sum(min_count=1)
        den = vals.notna().mul(w, axis=0).groupby(pre["cell"]).sum()
        cm = (num / den.replace(0, np.nan))[den["p"] > 0]
    po = post[post["cell"].isin(cm.index)]
    e = cm.loc[po["cell"]].mean()
    o = vals_of(po).mean()
    r = {"E": o.p / e.p - 1, "R_pp": 100 * (o.rated - e.rated), "coverage": len(po) / max(len(post), 1),
         "n_pre": len(pre), "n_post": len(po)}
    for m in MS:
        r[MK[m]] = (o[MK[m]] / e[MK[m]]) ** (1 / m) - 1
    if "w" in o:
        r["W"] = o.w / e.w - 1
    if "sd" in o and np.isfinite(e.get("sd", np.nan)) and e["sd"] > 0:
        r["SD"] = o.sd / e.sd - 1
    if "pitch0" in o:
        r["PITCH"] = o.pitch0 - e.pitch0
    return r


def mweights(pre, post):
    mp = pre.index.month.value_counts(normalize=True)
    mq = post.index.month.value_counts(normalize=True)
    r = (mq / mp).reindex(range(1, 13)).fillna(0)
    return pd.Series(r.loc[pre.index.month].to_numpy(), index=pre.index)


def split(f, a, b):
    return f.loc[a[0]:a[1]].copy(), f.loc[b[0]:b[1]].copy()


def temp_resid(f, col, a, b):
    g = f.dropna(subset=[col, "amb"])
    if len(g) < 2000:
        return np.nan
    c2 = (g["p"] // 100).astype(int).astype(str) + "_" + (g["amb"] // 5).astype(int).astype(str)
    g = g.assign(c2=c2)
    x, y = split(g, a, b)
    m = x.groupby("c2")[col].mean()
    y = y[y["c2"].isin(m.index)]
    return float((y[col] - m.loc[y["c2"]].to_numpy()).mean()) if len(y) else np.nan


KEYS = ["E", "R_pp"] + [MK[m] for m in MS] + ["W", "SD", "PITCH"]

# ------------------------------------------------------------------ A. T13 window, all 21 turbines
log("\n=== A. T13 clean window, common reference set (all except T13 and target) ===")
fullf = {s: frame_for(s) for s in STATIONS if s in P.columns}
frames = {s: split(fullf[s], PRE, POST) for s in fullf}
others = [s for s in frames if s != T13 and len(frames[s][0]) > 1000 and len(frames[s][1]) > 1000]
tab = pd.DataFrame({s: effects(*frames[s]) for s in others + [T13]}).T
for col in TEMPS:
    tab["TMP_" + col] = [temp_resid(fullf[s], col, PRE, POST) for s in tab.index]
tab.index.name = "station"
tab.insert(0, "turbine", [NAME[s] for s in tab.index])
tab.to_csv(f"{OUT}/A_per_turbine_T13window.csv")
keys = [k for k in KEYS if k in tab] + [c for c in tab if c.startswith("TMP_")]
med = tab.loc[others, keys].median()
corr = tab.loc[T13, keys] - med
log(tab[["turbine"] + keys[:6] + ["coverage"]].round(4).to_string())
log("\nT13 corrected (minus placebo median):")
log(corr.round(4).to_string())
rank = {k: int((tab[k] >= tab.loc[T13, k]).sum()) for k in keys}
log("T13 rank (1 = largest) among", len(tab), ":", rank)

pre13, post13 = frames[T13]
pg, qg = dict(tuple(pre13.groupby("week"))), dict(tuple(post13.groupby("week")))
pw, qw = list(pg), list(qg)
bs = []
for i in range(NBOOT):
    a = pd.concat([pg[w] for w in rng.choice(pw, len(pw))])
    b = pd.concat([qg[w] for w in rng.choice(qw, len(qw))])
    e = effects(a, b)
    bs.append({k: e[k] - med[k] for k in KEYS if k in e})
bs = pd.DataFrame(bs)
for m in MS:
    bs["C_" + MK[m]] = bs[MK[m]] / bs["E"]
bs.to_csv(f"{OUT}/A_T13_bootstrap.csv", index=False)
summ = []
for k in bs.columns:
    pt = corr[k] if k in corr else corr[k[2:]] / corr["E"]
    summ.append({"quantity": k, "estimate": pt, "lo": bs[k].quantile(0.025), "hi": bs[k].quantile(0.975),
                 "rank_of_21": rank.get(k, np.nan)})
summ = pd.DataFrame(summ)
summ.to_csv(f"{OUT}/A_T13_corrected_summary.csv", index=False)
log("\nT13 corrected with week-bootstrap 95% CI (placebo median fixed):")
log(summ.round(4).to_string(index=False))
log("reproducibility check vs frozen K2: E +0.048, R_pp +1.67, D3 +0.023, C 0.48")

# alternative drift corrections
alt = {"median_of_20": med, "mean_of_20": tab.loc[others, keys].mean(),
       "trimmed_mean_of_20": tab.loc[others, keys].apply(lambda c: c.sort_values().iloc[2:-2].mean()),
       "no_correction": med * 0}
alt = pd.DataFrame({k: tab.loc[T13, keys] - v for k, v in alt.items()}).T
alt["C_D3"] = alt["D3"] / alt["E"]
alt.to_csv(f"{OUT}/A_alt_corrections.csv")
log(alt[["E", "R_pp", "D3", "C_D3"]].round(4).to_string())

# ------------------------------------------------------------------ B. where does the gain come from (power-bin decomposition)
log("\n=== B. decomposition by reference-power bin ===")
rows = []
for s in others + [T13]:
    pre, post = frames[s]
    vals = vals_of(pre)
    cm = vals.groupby(pre["cell"]).mean()
    po = post[post["cell"].isin(cm.index)]
    ex = cm.loc[po["cell"]].set_index(po.index)
    ob = vals_of(po)
    b = (po["rp"] // 200 * 200).clip(upper=2200)
    g = pd.DataFrame({"bin": b, "n": 1, "p_obs": ob["p"], "p_exp": ex["p"], "r_obs": ob["rated"], "r_exp": ex["rated"],
                      "t3_obs": ob["D3"], "t3_exp": ex["D3"]}).groupby("bin").sum()
    g["station"] = s
    rows.append(g.reset_index())
B = pd.concat(rows)
B.to_csv(f"{OUT}/B_power_bin_decomposition.csv", index=False)
b13 = B[B.station == T13].set_index("bin")
log("T13 share of energy gain by ref-power bin:",
    ((b13.p_obs - b13.p_exp) / (b13.p_obs - b13.p_exp).sum()).round(3).to_dict())
log("T13 share of T^3 gain by ref-power bin:",
    ((b13.t3_obs - b13.t3_exp) / (b13.t3_obs - b13.t3_exp).sum()).round(3).to_dict())

# ------------------------------------------------------------------ C. torque spectrum (observed vs expected) for T13 and others
log("\n=== C. torque spectrum ===")
tmax = float(np.nanquantile(PV["t"].to_numpy(), 0.9995)) * 1.02
edges = np.linspace(0, tmax, 41)
rows = []
for s in others + [T13]:
    pre, post = frames[s]
    hb = pd.cut(pre["t"], edges, labels=False, include_lowest=True)
    ct = pd.crosstab(pre["cell"], hb, normalize="index")
    po = post[post["cell"].isin(ct.index)]
    expd = ct.loc[po["cell"]].sum().reindex(range(40), fill_value=0) / len(po)
    obsd = pd.cut(po["t"], edges, labels=False, include_lowest=True).value_counts(normalize=True).reindex(range(40), fill_value=0)
    rows.append(pd.DataFrame({"station": s, "bin": range(40), "t_lo": edges[:-1], "t_hi": edges[1:],
                              "obs": obsd.to_numpy(), "exp": expd.to_numpy()}))
pd.concat(rows).to_csv(f"{OUT}/C_torque_spectrum.csv", index=False)

# ------------------------------------------------------------------ D. known truth (frozen K1) + pure-scaling benchmark of C
log("\n=== D. K1 known-truth injection and pure power-scaling benchmark ===")


def inject(post, k=1.04):
    q = post.copy()
    p2 = np.minimum(q["p"] * k, RATED)
    q["t"] = q["t"] * p2 / q["p"]
    q["p"] = p2
    q["rated"] = (q["p"] >= 0.95 * RATED).astype(float)
    return q


def truth(post, k=1.04):
    q = inject(post, k)
    r = {"E": q["p"].mean() / post["p"].mean() - 1, "R_pp": 100 * (q["rated"].mean() - post["rated"].mean())}
    for m in MS:
        r[MK[m]] = ((q["t"] ** m).mean() / (post["t"] ** m).mean()) ** (1 / m) - 1
    return r


rows = []
for s in others:
    pre, post = frames[s]
    est = effects(pre, inject(post))
    mo = tab.loc[[o for o in others if o != s], keys].median()
    tr = truth(post)
    r = {"station": s}
    for k in ["E", "R_pp", "D3", "D10_3"]:
        r[k + "_true"], r[k + "_rec"] = tr[k], est[k] - mo[k]
    rows.append(r)
k1 = pd.DataFrame(rows)
k1["C_true"], k1["C_rec"] = k1["D3_true"] / k1["E_true"], k1["D3_rec"] / k1["E_rec"]
k1.to_csv(f"{OUT}/D_known_truth.csv", index=False)
log(k1.median().round(4).to_string())
rows = []
for k in [1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.08]:
    for lab, d in [("T13_pre", pre13), ("T13_post", post13)]:
        r = truth(d, k)
        r.update({"k": k, "data": lab})
        rows.append(r)
sc = pd.DataFrame(rows)
for m in MS:
    sc["C_" + MK[m]] = sc[MK[m]] / sc["E"]
sc.to_csv(f"{OUT}/D_pure_scaling_benchmark.csv", index=False)
log(sc.round(4).to_string(index=False))

# ------------------------------------------------------------------ E. sensitivity grid (one at a time from default)
log("\n=== E. sensitivity grid (T13 corrected) ===")
GRID = [("default", {}), ("pbin25", {"pbin": 25}), ("pbin100", {"pbin": 100}), ("nsec4", {"nsec": 4}),
        ("nsec12", {"nsec": 12}), ("min_n8", {"min_n": 8}), ("min_n15", {"min_n": 15}),
        ("rated0.90", {"rated_frac": 0.90}), ("rated0.98", {"rated_frac": 0.98})]
rows = []


def corrected(fr, w_fun=None, post_trim=None):
    t = {}
    for s, (pre, post) in fr.items():
        if post_trim:
            post = post.loc[post_trim:]
        t[s] = effects(pre, post, w_fun(pre, post) if w_fun else None)
    t = pd.DataFrame(t).T
    oth = [s for s in t.index if s != T13]
    c = t.loc[T13, ["E", "R_pp", "D3", "D10_3"]] - t.loc[oth, ["E", "R_pp", "D3", "D10_3"]].median()
    c["C_D3"] = c["D3"] / c["E"]
    c["rank_R"] = int((t["R_pp"] >= t.loc[T13, "R_pp"]).sum())
    c["rank_E"] = int((t["E"] >= t.loc[T13, "E"]).sum())
    c["rank_D3"] = int((t["D3"] >= t.loc[T13, "D3"]).sum())
    return c


for lab, kw in GRID:
    fr = {s: split(frame_for(s, **kw), PRE, POST) for s in others + [T13]}
    c = corrected(fr)
    c["config"] = lab
    rows.append(c)
fr0 = {s: frames[s] for s in others + [T13]}
for lab, kw in [("month_reweight", {"w_fun": mweights}), ("post_from_2021-11-01", {"post_trim": "2021-11-01"})]:
    c = corrected(fr0, **kw)
    c["config"] = lab
    rows.append(c)
pre_short = {s: (frames[s][0].loc["2020-10-01":], frames[s][1]) for s in others + [T13]}
c = corrected(pre_short)
c["config"] = "pre_from_2020-10-01"
rows.append(c)
S = pd.DataFrame(rows)
S.to_csv(f"{OUT}/E_sensitivity.csv", index=False)
log(S.round(4).to_string(index=False))

# ------------------------------------------------------------------ F. event study: monthly effects vs pre cells (all 21)
log("\n=== F. monthly event study ===")
rows = []
for s in others + [T13]:
    f = fullf[s]
    pre = f.loc[PRE[0]:PRE[1]]
    for per, g in f.loc["2020-01-01":"2022-06-30 23:59"].groupby(f.loc["2020-01-01":"2022-06-30 23:59"].index.to_period("M")):
        if len(g) < 300:
            continue
        e = effects(pre, g)
        rows.append({"station": s, "month": str(per), "E": e["E"], "R_pp": e["R_pp"], "D3": e["D3"], "n": e["n_post"]})
pd.DataFrame(rows).to_csv(f"{OUT}/F_monthly_event_study.csv", index=False)

# ------------------------------------------------------------------ G. 2022-23 rollout replication (frozen K3, D1) and null check (K3b, D2)
log("\n=== G. replication K3 (ref T13) and null K3b (ref T01, T21) ===")


def rep(ref, pre_w, post_w, months, excl, pre_months=False):
    rows = []
    for s in STATIONS:
        if s in excl or s not in P.columns:
            continue
        f = frame_ref(s, ref)
        pre = f.loc[pre_w[0]:pre_w[1]]
        post = f.loc[post_w[0]:post_w[1]]
        if months is not None:
            post = post[post.index.month.isin(months)]
            if pre_months:  # K3b filtered both windows to the same months
                pre = pre[pre.index.month.isin(months)]
        if len(pre) < 2000 or len(post) < 2000:
            log(f"skip {NAME[s]} ref {NAME[ref]}: pre {len(pre)} post {len(post)}")
            continue
        e = effects(pre, post, mweights(pre, post))
        e["station"] = s
        rows.append(e)
    return pd.DataFrame(rows).set_index("station")


def pooled(t, lab):
    k = ["E", "R_pp", "D3", "D10_3"]
    pt = t[k].mean()
    b = np.array([t[k].iloc[rng.integers(0, len(t), len(t))].mean().to_numpy() for _ in range(NBOOT)])
    out = {"set": lab, "n": len(t), "share_R_pos": (t["R_pp"] > 0).mean()}
    for i, kk in enumerate(k):
        out[kk], out[kk + "_lo"], out[kk + "_hi"] = pt[kk], np.percentile(b[:, i], 2.5), np.percentile(b[:, i], 97.5)
    cr = b[:, 2] / b[:, 0]
    out["C_D3"], out["C_lo"], out["C_hi"] = pt["D3"] / pt["E"], np.percentile(cr, 2.5), np.percentile(cr, 97.5)
    return out


k3 = rep(T13, ("2021-10-01", "2022-07-13 23:59"), ("2023-06-01", "2023-12-31 23:59"), [6, 7, 10, 11, 12], (T13,))
k3.to_csv(f"{OUT}/G_K3_per_turbine.csv")
pools = [pooled(k3, "K3_rollout_ref_T13")]
for ref in (2304510, 2304530):
    nb = rep(ref, ("2020-06-01", "2020-12-31 23:59"), ("2021-06-01", "2021-12-31 23:59"), [6, 7, 10, 11, 12], (T13, ref), pre_months=True)
    nb.to_csv(f"{OUT}/G_K3b_null_ref_{NAME[ref]}.csv")
    pools.append(pooled(nb, f"K3b_null_ref_{NAME[ref]}"))
pools = pd.DataFrame(pools)
pools.to_csv(f"{OUT}/G_replication_pooled.csv", index=False)
log(pools.round(4).to_string(index=False))
log("reproducibility check vs frozen K3: R +1.82, E +0.052, D3 +0.028, 19/20; K3b T01 R +0.33 E +0.010; T21 R +0.05 E +0.023")
log("\nDONE")
