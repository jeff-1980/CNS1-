"""Q-20261030-1 kill test K3 (replication on the 2022-23 AeroUp rollout), deviation D1 from PLAN_v1_FROZEN.
Reason (written before running): the install-date file shows all 20 other turbines started works 2022-07-14..09-20
and 8 finished only 2023-04/05, so no turbine is in a constant state across any pre/post pair of the other
turbines; the frozen K3 reference sets (not-yet-started >= 6, or finished > 30 days) are empty.
Replacement: T13 (upgraded 2021-09, constant state afterwards) is the sole reference turbine.
  pre  = 2021-10-01 .. 2022-07-13 (only T13 upgraded), post = 2023-06..2023-12 minus Aug-Sep (all upgraded).
  cell = T13 power 50 kW bin x T13 yaw 8 sectors; pre reweighted to post month shares.
  Effect for each of the 20 turbines = observed - expected in post (same outcomes as baseline).
Gate unchanged from plan: pooled (mean over turbines) R_pp > 0 with 95% turbine-bootstrap CI excluding 0,
and >= 60% of turbines with R_pp > 0. Informational: within-pre placebo split 2022-02-16, E, D3, C = D3/E.
Usage: python kill_aeroup_K3.py <ROOT> [NBOOT=2000]
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
PRE = ("2021-10-01", "2022-07-13 23:59")
POST = ("2023-06-01", "2023-12-31 23:59")
POST_MONTHS = [6, 7, 10, 11, 12]  # Aug-Sep absent from PRE, dropped from POST
FAKE_SPLIT = "2020-12-01"
rng = np.random.default_rng(20261103)


def read_table(table, cols, years=(2021, 2022, 2023)):
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


def list_cols(table, year=2022):
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




def frame_ref(target):
    f = pd.DataFrame({"p": P[target], "t": TQ[target], "rp": P[T13], "yaw": Y[T13]})
    f = f.dropna(subset=["p", "t", "rp", "yaw"])
    f["cell"] = (f["rp"] // 50).astype(int).astype(str) + "_" + ((f["yaw"] % 360) // 45).astype(int).astype(str)
    f["rated"] = (f["p"] >= 0.95 * RATED).astype(float)
    return f


def effects(pre, post, w):
    vals = pd.DataFrame({"p": pre["p"], "rated": pre["rated"], "t3": pre["t"] ** 3, "t103": pre["t"] ** (10 / 3)})
    num = vals.mul(w, axis=0).groupby(pre["cell"]).sum()
    den = w.groupby(pre["cell"]).sum()
    cm = num.div(den, axis=0)[den > 0]
    po = post[post["cell"].isin(cm.index)]
    e = cm.loc[po["cell"]].mean()
    o = {"p": po["p"].mean(), "rated": po["rated"].mean(), "t3": (po["t"] ** 3).mean(), "t103": (po["t"] ** (10 / 3)).mean()}
    return {"E": o["p"] / e.p - 1, "R_pp": 100 * (o["rated"] - e.rated),
            "D3": (o["t3"] / e.t3) ** (1 / 3) - 1, "D10_3": (o["t103"] / e.t103) ** (3 / 10) - 1,
            "n_pre": len(pre), "n_post": len(po), "coverage": len(po) / max(len(post), 1)}


def mweights(pre, post):
    mp = pre.index.month.value_counts(normalize=True)
    mq = post.index.month.value_counts(normalize=True)
    r = (mq / mp).reindex(range(1, 13)).fillna(0)
    return pd.Series(r.loc[pre.index.month].to_numpy(), index=pre.index)


rows, plc = [], []
for s in STATIONS:
    if s == T13 or s not in P.columns:
        continue
    f = frame_ref(s)
    pre = f.loc[PRE[0]:PRE[1]]
    post = f.loc[POST[0]:POST[1]]
    post = post[post.index.month.isin(POST_MONTHS)]
    if len(pre) < 2000 or len(post) < 2000:
        print(f"skip {s}: pre {len(pre)} post {len(post)}")
        continue
    e = effects(pre, post, mweights(pre, post))
    e["station"] = s
    rows.append(e)
    a, b = pre.loc[:"2022-02-15 23:59"], pre.loc["2022-02-16":]
    if len(a) > 1000 and len(b) > 1000:
        q = effects(a, b, mweights(a, b))
        plc.append({"station": s, "E": q["E"], "R_pp": q["R_pp"], "D3": q["D3"]})
tab = pd.DataFrame(rows).set_index("station")
print("\nK3 per-turbine effects (reference = T13 only, month-reweighted):")
print(tab.round(4).to_string())
keys = ["E", "R_pp", "D3", "D10_3"]
pooled = tab[keys].mean()
bs = np.array([tab[keys].iloc[rng.integers(0, len(tab), len(tab))].mean().to_numpy() for _ in range(NBOOT)])
lo, hi = np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)
print(f"\nPooled over {len(tab)} turbines (95% turbine bootstrap):")
for i, k in enumerate(keys):
    print(f"  {k}: {pooled[k]:+.4f}  [{lo[i]:+.4f}, {hi[i]:+.4f}]")
share = (tab["R_pp"] > 0).mean()
print(f"  share of turbines with R_pp > 0: {share:.2f};  share with E > 0: {(tab['E'] > 0).mean():.2f};  share with D3 > 0: {(tab['D3'] > 0).mean():.2f}")
cr = bs[:, 2] / bs[:, 0]
print(f"  load cost ratio C = D3/E pooled: {pooled['D3'] / pooled['E']:.3f}  [{np.percentile(cr, 2.5):.3f}, {np.percentile(cr, 97.5):.3f}]")
pl = pd.DataFrame(plc)
print("\nPlacebo (informational): split inside pre at 2022-02-16")
print(pl.round(4).to_string(index=False))
print(f"  placebo median R_pp {pl['R_pp'].median():+.3f}, median |R_pp| {pl['R_pp'].abs().median():.3f}, median E {pl['E'].median():+.4f}")
K3 = pooled["R_pp"] > 0 and lo[1] > 0 and share >= 0.6
print("\nK3 ->", "PASS" if K3 else "FAIL")
print("VERDICT:", "K3 PASS" if K3 else "KILL (K3)")
