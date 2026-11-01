"""Q-20261030-1 K3b: known-null check of the single-reference-turbine design used in K3 (deviation D2).
Added 2026-11-01 AFTER K3 ran, because K3's informational placebo (split inside pre at 2022-02-16) was
degenerate (pre/post months barely overlap, so month reweighting kept only February) and gave +2.9 pp.
K3b: no turbine changes state in either window. Reference = T01 (2304510) and, separately, T21 (2304530).
  pre = 2020 Jun, Jul, Oct, Nov, Dec;  post = 2021 same months (T13 excluded as target and reference).
  Same estimator as K3 (cells = reference power 50 kW x reference yaw 8 sectors, month reweighting).
Gate (written before running): for BOTH references, pooled |R_pp| < 0.91 (half of K3 pooled 1.82) and
pooled |E| < 0.026 (half of K3 pooled 0.052). Otherwise K3 is downgraded to KILL (single-reference
year-on-year drift as large as the claimed effect).
Usage: python kill_aeroup_K3b.py <ROOT> [NBOOT=2000]
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
PRE = ("2020-06-01", "2020-12-31 23:59")
POST = ("2021-06-01", "2021-12-31 23:59")
POST_MONTHS = [6, 7, 10, 11, 12]  # same months as K3 post, applied to both windows
FAKE_SPLIT = "2020-12-01"
rng = np.random.default_rng(20261103)


def read_table(table, cols, years=(2020, 2021)):
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




def frame_ref(target, REF):
    f = pd.DataFrame({"p": P[target], "t": TQ[target], "rp": P[REF], "yaw": Y[REF]})
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



res = {}
keys = ["E", "R_pp", "D3", "D10_3"]
for REF in (2304510, 2304530):
    rows = []
    for s in STATIONS:
        if s in (T13, REF) or s not in P.columns:
            continue
        f = frame_ref(s, REF)
        f = f[f.index.month.isin(POST_MONTHS)]
        pre = f.loc[PRE[0]:PRE[1]]
        post = f.loc[POST[0]:POST[1]]
        if len(pre) < 2000 or len(post) < 2000:
            print(f"ref {REF} skip {s}: pre {len(pre)} post {len(post)}")
            continue
        e = effects(pre, post, mweights(pre, post))
        e["station"] = s
        rows.append(e)
    tab = pd.DataFrame(rows).set_index("station")
    print(f"\nK3b reference {REF}: per-turbine null effects (2020 vs 2021, same months)")
    print(tab.round(4).to_string())
    pooled = tab[keys].mean()
    bs = np.array([tab[keys].iloc[rng.integers(0, len(tab), len(tab))].mean().to_numpy() for _ in range(NBOOT)])
    lo, hi = np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)
    for i, k in enumerate(keys):
        print(f"  pooled {k}: {pooled[k]:+.4f}  [{lo[i]:+.4f}, {hi[i]:+.4f}]")
    print(f"  median |R_pp| {tab['R_pp'].abs().median():.3f}; share R_pp>0 {(tab['R_pp'] > 0).mean():.2f}")
    res[REF] = abs(pooled["R_pp"]) < 0.91 and abs(pooled["E"]) < 0.026
    print(f"  gate for ref {REF}: {'PASS' if res[REF] else 'FAIL'}")
ok = all(res.values())
print("\nK3b ->", "PASS" if ok else "FAIL")
print("VERDICT:", "K3 STANDS" if ok else "KILL (K3b: single-reference drift too large)")
