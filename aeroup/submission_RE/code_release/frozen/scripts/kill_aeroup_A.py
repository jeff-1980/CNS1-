"""Q-20261030-1 formal kill tests K1, K2, K4, K5 (plan PLAN_v1_FROZEN.md, sha256 f453f19b...).
Read-only on Hill of Towie v2.1.0 zips. Usage: python kill_aeroup_A.py <ROOT> [NBOOT=500]
Common reference set for every turbine = all turbines except T13 and the target.
Formal estimator = target effect minus median of the 20 non-T13 placebo effects (drift correction).

K1 known truth: for each non-T13 turbine s, multiply post power by 1.04 capped at 2300 kW (torque scaled
   by the same factor, speed unchanged); estimate = eff(s injected) - median(eff of the other 19, not injected).
   PASS if median recovered E in [+3.0%, +5.0%] AND median |R_pp bias| < 0.3 pp AND median |D3 bias| < 0.005.
K2 PASS if T13 corrected R_pp >= 1.0 AND T13 raw R_pp ranks 1st of 21.
K4 pre stamps reweighted by month-of-year so pre month shares equal post month shares (all turbines);
   PASS if min(T13 raw reweighted R_pp, T13 corrected reweighted R_pp) >= 0.8.
K5 informational: lists within-10-min std/max/min columns; if wtc_ActPower_stddev exists reports the T13
   corrected change in mean power std (ratio) within the same cells.
Order K1 -> K2 -> K4; if K1 fails the run stops with VERDICT KILL (K1).
"""
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = sys.argv[1]
NBOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 500
RATED = 2300.0
T13 = 2304522
STATIONS = list(range(2304510, 2304531))
PRE = ("2020-01-01", "2021-09-22 23:59")
POST = ("2021-09-30", "2022-07-13 23:59")
FAKE_SPLIT = "2020-12-01"
rng = np.random.default_rng(20261102)


def read_table(table, cols, years=(2020, 2021, 2022)):
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
                missing = set(cols) - set(use)
                if missing:
                    print(f"WARNING {n}: missing {sorted(missing)}")
                with z.open(n) as f:
                    out.append(pd.read_csv(f, usecols=use, low_memory=False))
    df = pd.concat(out, ignore_index=True)
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], utc=True, format="mixed")
    return df.drop_duplicates(["TimeStamp", "StationId"])


def list_cols(table, year=2021):
    with zipfile.ZipFile(f"{ROOT}/{year}.zip") as z:
        n = sorted(n for n in z.namelist() if n.split("/")[-1].startswith(table + "_"))[0]
        with z.open(n) as f:
            return list(pd.read_csv(f, nrows=0).columns)


K5_COLS = {}
for tb in ["tblSCTurGrid", "tblSCTurbine"]:
    cols = list_cols(tb)
    hit = [c for c in cols if any(k in c.lower() for k in ["std", "max", "min"]) and any(
        k in c.lower() for k in ["power", "rpm", "torq"])]
    K5_COLS[tb] = hit
    print(f"K5 columns in {tb}: {hit}")
SD_COL = "wtc_ActPower_stddev" if "wtc_ActPower_stddev" in K5_COLS["tblSCTurGrid"] else None
grid = read_table("tblSCTurGrid", ["wtc_ActPower_mean"] + ([SD_COL] if SD_COL else []))
tur = read_table("tblSCTurbine", ["wtc_MainSRpm_mean", "wtc_YawPos_mean", "wtc_PowerRef_endvalue"])
tmp = read_table("tblSCTurTemp", ["wtc_HSGenTmp_mean", "wtc_AmbieTmp_mean"])
df = grid.merge(tur, on=["TimeStamp", "StationId"], how="inner").merge(
    tmp, on=["TimeStamp", "StationId"], how="left")

with zipfile.ZipFile(f"{ROOT}/Hill_of_Towie_ShutdownDuration.zip") as z:
    name = [n for n in z.namelist() if n.endswith(".csv")][0]
    with z.open(name) as f:
        sd = pd.read_csv(f)
sd["TimeStamp"] = pd.to_datetime(sd["TimeStamp_StartFormat"], utc=True, format="mixed") + pd.Timedelta(minutes=10)
sd["StationId"] = sd["TurbineName"].str[1:].astype(int) + 2304509
sd = sd.drop_duplicates(["TimeStamp", "StationId"])
df = df.merge(sd[["TimeStamp", "StationId", "ShutdownDuration"]], on=["TimeStamp", "StationId"], how="left")
print("rows merged:", len(df), " share with shutdown info:", round(df["ShutdownDuration"].notna().mean(), 3))

ok = (df["ShutdownDuration"].fillna(1) == 0) & (df["wtc_ActPower_mean"] > 0)
if "wtc_PowerRef_endvalue" in df:
    ok &= df["wtc_PowerRef_endvalue"].fillna(RATED) >= 2290
df = df[ok].copy()
df["omega"] = df["wtc_MainSRpm_mean"] * 2 * np.pi / 60
df = df[df["omega"] > 0.3]
df["T"] = df["wtc_ActPower_mean"] * 1e3 / df["omega"] / 1e6  # MNm, electrical torque proxy
print("normal-operation rows:", len(df), " per turbine median:", int(df.groupby("StationId").size().median()))

P = df.pivot_table(index="TimeStamp", columns="StationId", values="wtc_ActPower_mean")
Y = df.pivot_table(index="TimeStamp", columns="StationId", values="wtc_YawPos_mean")
TQ = df.pivot_table(index="TimeStamp", columns="StationId", values="T")
SD = df.pivot_table(index="TimeStamp", columns="StationId", values=SD_COL) if SD_COL else None
HS = df.pivot_table(index="TimeStamp", columns="StationId", values="wtc_HSGenTmp_mean")
AMB = df.pivot_table(index="TimeStamp", columns="StationId", values="wtc_AmbieTmp_mean")


def frame_for(target, exclude=()):
    refs = [s for s in P.columns if s != target and s not in exclude]
    rp = P[refs]
    n = rp.notna().sum(axis=1)
    yaw = np.deg2rad(Y[refs])
    ang = np.rad2deg(np.arctan2(np.sin(yaw).mean(axis=1), np.cos(yaw).mean(axis=1))) % 360
    f = pd.DataFrame({
        "p": P[target], "t": TQ[target], "hs": HS.get(target), "sd": SD.get(target) if SD is not None else np.nan, "amb": AMB.get(target),
        "rp": rp.mean(axis=1), "sec": ang // 45, "n": n})
    f = f[(f["n"] >= 10) & f["p"].notna()].dropna(subset=["rp", "sec", "t"])
    f["cell"] = (f["rp"] // 50).astype(int).astype(str) + "_" + f["sec"].astype(int).astype(str)
    f["rated"] = (f["p"] >= 0.95 * RATED).astype(float)
    f["week"] = f.index.to_period("W").astype(str)
    return f




def cell_means(pre, w=None):
    vals = pd.DataFrame({"p": pre["p"], "rated": pre["rated"], "t3": pre["t"] ** 3,
                         "t103": pre["t"] ** (10 / 3), "sd": pre["sd"]})
    if w is None:
        return vals.groupby(pre["cell"]).mean()
    vw = vals.mul(w, axis=0)
    num = vw.groupby(pre["cell"]).sum(min_count=1)
    den = vals.notna().mul(w, axis=0).groupby(pre["cell"]).sum()
    cm = num / den.replace(0, np.nan)
    return cm[den["p"] > 0]


def effects(pre, post, w=None):
    cm = cell_means(pre, w)
    po = post[post["cell"].isin(cm.index)]
    exp = cm.loc[po["cell"]]
    e = exp.mean()
    o = pd.Series({"p": po["p"].mean(), "rated": po["rated"].mean(), "t3": (po["t"] ** 3).mean(),
                   "t103": (po["t"] ** (10 / 3)).mean(), "sd": po["sd"].mean()})
    return {"E": o.p / e.p - 1, "R_pp": 100 * (o.rated - e.rated),
            "D3": (o.t3 / e.t3) ** (1 / 3) - 1, "D10_3": (o.t103 / e.t103) ** (3 / 10) - 1,
            "SD": (o.sd / e.sd - 1) if np.isfinite(e.sd) and e.sd > 0 else np.nan,
            "coverage": len(po) / max(len(post), 1)}


def split(f, a, b):
    return f.loc[a[0]:a[1]].copy(), f.loc[b[0]:b[1]].copy()


def inject(post, k=1.04):
    q = post.copy()
    p2 = np.minimum(q["p"] * k, RATED)
    q["t"] = q["t"] * p2 / q["p"]
    q["p"] = p2
    q["rated"] = (q["p"] >= 0.95 * RATED).astype(float)
    return q


def truth(post):
    q = inject(post)
    return {"E": q["p"].mean() / post["p"].mean() - 1,
            "R_pp": 100 * (q["rated"].mean() - post["rated"].mean()),
            "D3": ((q["t"] ** 3).mean() / (post["t"] ** 3).mean()) ** (1 / 3) - 1}


def month_weights(pre, post):
    mp = pre.index.month.value_counts(normalize=True)
    mq = post.index.month.value_counts(normalize=True)
    r = (mq / mp).reindex(range(1, 13)).fillna(0)
    return pd.Series(r.loc[pre.index.month].to_numpy(), index=pre.index)


KEYS = ["E", "R_pp", "D3", "D10_3", "SD"]
frames = {}
for s in STATIONS:
    if s in P.columns:
        frames[s] = split(frame_for(s, exclude=(T13,)), PRE, POST)
others = [s for s in frames if s != T13 and len(frames[s][0]) > 1000 and len(frames[s][1]) > 1000]
print(f"turbines with frames: {len(frames)}; non-T13 usable: {len(others)}")
eff = {s: effects(*frames[s]) for s in others + [T13]}
tab = pd.DataFrame(eff).T
tab.index.name = "station"
print("\nRaw effects, common reference set (all except T13):")
print(tab[KEYS + ["coverage"]].round(4).to_string())
med = tab.loc[others, KEYS].median()

# ---- K1
rows = []
for s in others:
    pre, post = frames[s]
    est = effects(pre, inject(post))
    m_other = tab.loc[[o for o in others if o != s], KEYS].median()
    tr = truth(post)
    rows.append({"station": s, "E_true": tr["E"], "E_rec": est["E"] - m_other["E"],
                 "R_true": tr["R_pp"], "R_rec": est["R_pp"] - m_other["R_pp"],
                 "D3_true": tr["D3"], "D3_rec": est["D3"] - m_other["D3"]})
k1 = pd.DataFrame(rows)
k1["R_bias"] = k1["R_rec"] - k1["R_true"]
k1["D3_bias"] = k1["D3_rec"] - k1["D3_true"]
print("\nK1 known-truth injection (+4% power, capped):")
print(k1.round(4).to_string(index=False))
k1_E = k1["E_rec"].median()
k1_R = k1["R_bias"].abs().median()
k1_D = k1["D3_bias"].abs().median()
K1 = (0.03 <= k1_E <= 0.05) and k1_R < 0.3 and k1_D < 0.005
print(f"K1: median E_rec {k1_E:+.4f}; median |R bias| {k1_R:.3f} pp; median |D3 bias| {k1_D:.4f} -> {'PASS' if K1 else 'FAIL'}")
if not K1:
    print("VERDICT: KILL (K1)")
    sys.exit(0)

# ---- K2
corr = tab.loc[T13, KEYS] - med
rank = int((tab["R_pp"] >= tab.loc[T13, "R_pp"]).sum())
pre13, post13 = frames[T13]
bs = []
pg, qg = dict(tuple(pre13.groupby("week"))), dict(tuple(post13.groupby("week")))
pw, qw = list(pg), list(qg)
for _ in range(NBOOT):
    a = pd.concat([pg[w] for w in rng.choice(pw, len(pw))])
    b = pd.concat([qg[w] for w in rng.choice(qw, len(qw))])
    e = effects(a, b)
    bs.append({k: e[k] - med[k] for k in KEYS})
ci = pd.DataFrame(bs).quantile([0.025, 0.975])
print("\nK2 T13 drift-corrected (T13 minus placebo median; CI = T13 week bootstrap, placebo median fixed):")
for k in KEYS:
    print(f"  {k}: {corr[k]:+.4f}  [{ci.loc[0.025, k]:+.4f}, {ci.loc[0.975, k]:+.4f}]")
print(f"  T13 raw R_pp rank among {len(tab)}: {rank}  (one-sided permutation p = {rank / len(tab):.3f})")
C = corr["D3"] / corr["E"] if corr["E"] != 0 else np.nan
C103 = corr["D10_3"] / corr["E"] if corr["E"] != 0 else np.nan
print(f"  load cost ratio C = D3/E = {C:.3f};  D10_3/E = {C103:.3f}")
cb = pd.DataFrame(bs)
cb = (cb["D3"] / cb["E"]).replace([np.inf, -np.inf], np.nan).dropna()
print(f"  C 95% CI [{cb.quantile(0.025):.3f}, {cb.quantile(0.975):.3f}]")
K2 = corr["R_pp"] >= 1.0 and rank == 1
print(f"K2 -> {'PASS' if K2 else 'FAIL'}")

# ---- K4
effw = {}
for s in others + [T13]:
    pre, post = frames[s]
    effw[s] = effects(pre, post, month_weights(pre, post))
tw = pd.DataFrame(effw).T
raw_w = tw.loc[T13, "R_pp"]
corr_w = raw_w - tw.loc[others, "R_pp"].median()
print(f"\nK4 month-reweighted: T13 raw R_pp {raw_w:+.3f}, corrected {corr_w:+.3f}; "
      f"E raw {tw.loc[T13, 'E']:+.4f}, D3 raw {tw.loc[T13, 'D3']:+.4f}; placebo median R_pp {tw.loc[others, 'R_pp'].median():+.3f}")
K4 = min(raw_w, corr_w) >= 0.8
print(f"K4 -> {'PASS' if K4 else 'FAIL'}")

# ---- K5
print(f"\nK5 (informational): SD_COL = {SD_COL}; T13 corrected change in mean within-10-min power std: {corr['SD']:+.4f}")

print("\nSUMMARY K1 PASS; K2", "PASS" if K2 else "FAIL", "; K4", "PASS" if K4 else "FAIL")
print("VERDICT:", "CONTINUE TO K3" if (K2 and K4) else "KILL (" + ",".join(n for n, ok in [("K2", K2), ("K4", K4)] if not ok) + ")")
